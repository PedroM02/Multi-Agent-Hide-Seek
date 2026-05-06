from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import game_types as gt
from action_resolution import resolve_actions
from agent import Agent, build_agents_for_env
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


class SimulationConfig:
    def __init__(self) -> None:
        self.width = 10
        self.height = 8
        self.timesteps = 200
        self.vision_radius = 1
        self.num_predators = 1
        self.num_prey = 1
        self.seed = 0
        self.walls: Optional[Sequence[Tuple[int, int]]] = None
        self.num_walls: int = 0
        self.wall_size: int = 3


def copy_config(base: SimulationConfig, **overrides) -> SimulationConfig:
    c = SimulationConfig()
    c.width = base.width
    c.height = base.height
    c.timesteps = base.timesteps
    c.vision_radius = base.vision_radius
    c.num_predators = base.num_predators
    c.num_prey = base.num_prey
    c.seed = base.seed
    c.walls = base.walls
    c.num_walls = base.num_walls
    c.wall_size = base.wall_size
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

        # Resolve wall cells: manual --obstacles + auto-generated walls.
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
        self.outcome = gt.OUTCOME_ONGOING
        self.cumulative_rewards: Dict[int, float] = {}
        self.reset_episode()

    def reset_episode(self) -> None:
        self.env.reset_agent_positions_random(
            self.config.num_predators,
            self.config.num_prey,
            self.rng,
        )
        self.agents = build_agents_for_env(self.env, self.rng)
        for a in self.agents:
            a.reset_memory()
        self.step_index = 0
        self.outcome = gt.OUTCOME_ONGOING
        self.cumulative_rewards = {bid: 0.0 for bid in self.env.bodies}

    def step_once(self) -> bool:
        if self.outcome != gt.OUTCOME_ONGOING:
            return False

        intentions: Dict[int, str] = {}
        for agent in self.agents:
            body = self.env.bodies[agent.agent_id]
            if not body.alive:
                continue
            obs = build_observation(
                self.env,
                agent.agent_id,
                self.config.vision_radius,
            )
            intentions[agent.agent_id] = agent.decide(obs)

        resolve_actions(self.env, intentions, self.rng)
        captured = self.env.apply_captures()
        rews = attribute_rewards(self.env, captured)
        for aid, r in rews.items():
            self.cumulative_rewards[aid] = self.cumulative_rewards.get(aid, 0.0) + r

        self.step_index += 1

        if not self.env.any_prey_alive():
            self.outcome = gt.OUTCOME_PREDATORS_WIN
            return False
        if self.step_index >= self.config.timesteps:
            if self.env.any_prey_alive():
                self.outcome = gt.OUTCOME_PREY_WIN_TIMEOUT
            else:
                self.outcome = gt.OUTCOME_PREDATORS_WIN
            return False
        return True

    def status_line(self) -> str:
        return f"step {self.step_index}/{self.config.timesteps}  outcome={self.outcome}"


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
        if summary.outcome == gt.OUTCOME_PREDATORS_WIN:
            acc.predator_wins += 1
        elif summary.outcome == gt.OUTCOME_PREY_WIN_TIMEOUT:
            acc.prey_timeout_wins += 1
    return acc