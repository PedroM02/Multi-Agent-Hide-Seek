from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import agent_utils as au
from action_resolution import resolve_actions
from agent import Agent, build_agents_for_env
from decision_making import select_team_roles
from environment import Environment
from observation_definition import build_observation
from reward_attribution import attribute_rewards


def generate_walls(
    width: int,
    height: int,
    num_walls: int,
    wall_size: int,
    rng: random.Random,
    existing_walls: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    """Generate `num_walls` straight wall segments (horizontal or vertical),
    each with `wall_size` cells. Walls are placed randomly but never fill
    an entire row/column (to keep the map passable). Already-occupied cells
    (from `existing_walls`) are skipped.

    Returns the combined list of wall cell coordinates.
    """
    occupied: set[Tuple[int, int]] = set(existing_walls) if existing_walls else set()
    result: List[Tuple[int, int]] = list(occupied)

    for _ in range(num_walls):
        # Try up to 50 random placements before giving up on this wall.
        for _attempt in range(50):
            horizontal = rng.choice([True, False])
            if horizontal:
                max_x = width - wall_size
                if max_x < 0:
                    continue
                ox = rng.randint(0, max_x)
                oy = rng.randint(0, height - 1)
                cells = [(ox + i, oy) for i in range(wall_size)]
                # Don't block the full row
                if len(cells) >= width:
                    continue
            else:
                max_y = height - wall_size
                if max_y < 0:
                    continue
                ox = rng.randint(0, width - 1)
                oy = rng.randint(0, max_y)
                cells = [(ox, oy + i) for i in range(wall_size)]
                # Don't block the full column
                if len(cells) >= height:
                    continue

            # Skip if any cell already occupied
            if any(c in occupied for c in cells):
                continue

            for c in cells:
                occupied.add(c)
            result.extend(cells)
            break

    return result


def _exchange_team_messages(
    raw_obs: Dict[int, dict],
) -> Dict[int, Tuple[Tuple[int, int, int], ...]]:
    """Speaker-centric, single-hop, synchronous team comms.

    Every agent broadcasts the enemies it directly sees this step to the
    teammates inside its own vision radius (its visible_allies). The
    receiver collects everything it was told, deduped by enemy_id, sorted
    deterministically. The receiver's own direct sightings are not
    filtered out here — priority handling lives in Agent.decide.
    """
    shared: Dict[int, list] = {aid: [] for aid in raw_obs}
    for sender_obs in raw_obs.values():
        sightings = sender_obs["visible_enemies"]
        if not sightings:
            continue
        for _ax, _ay, ally_id in sender_obs["visible_allies"]:
            if ally_id in shared:
                shared[ally_id].extend(sightings)

    out: Dict[int, Tuple[Tuple[int, int, int], ...]] = {}
    for aid, lst in shared.items():
        seen_ids: set[int] = set()
        deduped: List[Tuple[int, int, int]] = []
        for ex, ey, eid in lst:
            if eid in seen_ids:
                continue
            seen_ids.add(eid)
            deduped.append((ex, ey, eid))
        deduped.sort(key=lambda t: (t[2], t[0], t[1]))
        out[aid] = tuple(deduped)
    return out


class SimulationConfig:
    def __init__(self) -> None:
        self.width = 10
        self.height = 8
        self.timesteps = 200
        self.vision_radius_predator = 2
        self.vision_radius_prey = 2
        self.num_predators = 1
        self.num_prey = 1
        self.seed = 0
        self.walls: Optional[Sequence[Tuple[int, int]]] = None
        self.num_walls: int = 0
        self.wall_size: int = 3
        self.enable_comms: bool = False


def copy_config(base: SimulationConfig, **overrides) -> SimulationConfig:
    c = SimulationConfig()
    c.width = base.width
    c.height = base.height
    c.timesteps = base.timesteps
    c.vision_radius_predator = base.vision_radius_predator
    c.vision_radius_prey = base.vision_radius_prey
    c.num_predators = base.num_predators
    c.num_prey = base.num_prey
    c.seed = base.seed
    c.walls = base.walls
    c.num_walls = base.num_walls
    c.wall_size = base.wall_size
    c.enable_comms = base.enable_comms
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


class EpisodeSummary:
    def __init__(
        self,
        outcome: str,
        steps: int,
        cumulative_rewards: Dict[int, float],
        episode_seed: int,
    ) -> None:
        self.outcome = outcome
        self.steps = steps
        self.cumulative_rewards = cumulative_rewards
        self.episode_seed = episode_seed


class SimulationState:
    def __init__(self, config: SimulationConfig, rng: random.Random) -> None:
        self.config = config
        self.rng = rng

        # Resolve wall cells: manual config.walls + auto-generated walls.
        walls = config.walls
        if config.num_walls > 0:
            walls = generate_walls(
                config.width,
                config.height,
                config.num_walls,
                config.wall_size,
                rng,
                existing_walls=walls,
            )

        self.env = Environment(config.width, config.height, walls)
        self.agents: List[Agent] = []
        self.step_index = 0
        self.outcome = au.OUTCOME_ONGOING
        self.cumulative_rewards: Dict[int, float] = {}
        self.reset_episode()

    def reset_episode(self) -> None:
        self.env.set_agent_positions(
            self.config.num_predators,
            self.config.num_prey,
            self.rng,
        )
        self.agents = build_agents_for_env(self.env, self.rng)
        for a in self.agents:
            a.reset_memory()
        self.step_index = 0
        self.outcome = au.OUTCOME_ONGOING
        self.cumulative_rewards = {bid: 0.0 for bid in self.env.agent_bodies}

    def step_once(self) -> bool:
        if self.outcome != au.OUTCOME_ONGOING:
            return False

        # Phase 1: build raw observations for every alive agent.
        raw_obs: Dict[int, dict] = {}
        for agent in self.agents:
            body = self.env.agent_bodies[agent.agent_id]
            if not body.alive:
                continue
            radius = (
                self.config.vision_radius_predator
                if body.team == au.TEAM_PREDATOR
                else self.config.vision_radius_prey
            )
            raw_obs[agent.agent_id] = build_observation(
                self.env, agent.agent_id, radius,
            )

        # Phase 2: synchronous, single-hop, speaker-centric team comms.
        # Every agent broadcasts its directly-visible enemies to teammates
        # within its own vision radius (i.e. its visible_allies). Skipped
        # entirely when comms are disabled, so receivers see empty reports
        # and Agent.decide collapses to direct-sight + memory.
        if self.config.enable_comms:
            shared_enemies = _exchange_team_messages(raw_obs)
        else:
            shared_enemies = {aid: tuple() for aid in raw_obs}

        # Phase 2b: each agent fuses its direct sightings with the
        # teammate reports addressed to it, producing the
        # priority-resolved `active_enemies` set on its own observation.
        # The fusion lives on the Agent (delegated to its Perception) so
        # that the simulation only orchestrates — it doesn't decide what
        # the agent "knows".
        agents_by_id = {a.agent_id: a for a in self.agents}
        for aid, obs in raw_obs.items():
            agents_by_id[aid].prepare_observation(obs, shared_enemies[aid])

        # Phase 2c: per-team role assignment. The selector currently
        # reduces to "predators -> CHASER, prey -> FLEE" and is kept as
        # the single hook for future hunting / protection strategies.
        # Roles + role_target are written back onto each Agent and
        # threaded into the obs so a richer choose_action can dispatch
        # on them later without further plumbing.
        for team in (au.TEAM_PREDATOR, au.TEAM_PREY):
            team_ids = [
                aid for aid, ob in raw_obs.items() if ob["team"] == team
            ]
            assignments = select_team_roles(
                team, team_ids, raw_obs, agents_by_id, self.env,
            )
            for aid, (role, target) in assignments.items():
                a = agents_by_id[aid]
                a.role = role
                a.role_target = target

        # Phase 3: agents decide using direct sightings, then teammate
        # reports, then their own memory — now within their assigned role.
        intentions: Dict[int, str] = {}
        for agent in self.agents:
            body = self.env.agent_bodies[agent.agent_id]
            if not body.alive:
                continue
            obs = dict(raw_obs[agent.agent_id])
            obs["role"] = agent.role
            obs["role_target"] = agent.role_target
            intentions[agent.agent_id] = agent.decide(obs)

        resolve_actions(self.env, intentions, self.rng)
        captured = self.env.apply_captures()
        rews = attribute_rewards(self.env, captured)
        for aid, r in rews.items():
            self.cumulative_rewards[aid] = self.cumulative_rewards.get(aid, 0.0) + r

        self.step_index += 1

        if not self.env.any_prey_alive():
            self.outcome = au.OUTCOME_PREDATORS_WIN
            return False
        if self.step_index >= self.config.timesteps:
            if self.env.any_prey_alive():
                self.outcome = au.OUTCOME_PREY_WIN
            else:
                self.outcome = au.OUTCOME_PREDATORS_WIN
            return False
        return True

    def status_line(self) -> str:
        return f"Timestep {self.step_index}/{self.config.timesteps}  Current Outcome={self.outcome}"


def run_episode(config: SimulationConfig, rng: random.Random) -> EpisodeSummary:
    sim = SimulationState(config, rng)
    while sim.step_once():
        pass
    return EpisodeSummary(
        outcome=sim.outcome,
        steps=sim.step_index,
        cumulative_rewards=dict(sim.cumulative_rewards),
        episode_seed=config.seed,
    )


class BatchSummary:
    def __init__(self) -> None:
        self.predator_wins = 0
        self.prey_timeout_wins = 0
        self.total_steps = 0
        self.runs = 0


def run_batch(config: SimulationConfig, num_runs: int) -> BatchSummary:
    acc = BatchSummary()
    for i in range(num_runs):
        cfg = copy_config(config, seed=config.seed + i)
        rng = random.Random(cfg.seed)
        summary = run_episode(cfg, rng)
        acc.runs += 1
        acc.total_steps += summary.steps
        if summary.outcome == au.OUTCOME_PREDATORS_WIN:
            acc.predator_wins += 1
        elif summary.outcome == au.OUTCOME_PREY_WIN:
            acc.prey_timeout_wins += 1
    return acc