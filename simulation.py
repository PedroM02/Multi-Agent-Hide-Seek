from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import agent_utils as au
from action_resolution import resolve_actions
from agent import Agent, build_agents_for_env
from decision_making import select_team_roles
from distances import chebyshev
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
        # Cooperative-knockout (prey-defend) mechanic. When enabled,
        # groups of Chebyshev-adjacent prey can defeat predators that
        # are Chebyshev-1 of >=2 group members. Two modes:
        #   - None  : mechanic disabled (default).
        #   - "stun": predators are stunned for `stun_duration` steps;
        #             they recover and rejoin the chase.
        #   - "kill": predators are removed from the run permanently;
        #             when all predators are dead, prey win.
        # See `SimulationState._resolve_knockouts` for the full rules.
        self.prey_defend: Optional[str] = None
        self.stun_duration: int = 3


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
    c.prey_defend = base.prey_defend
    c.stun_duration = base.stun_duration
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

        # Phase 4: cooperative-knockout system (opt-in via
        # --prey-defend). Up to two passes, both gated on the
        # configured mode:
        #   4a. Existing stuns (only in "stun" mode): any predator
        #       with stun_remaining > 0 from a previous step is
        #       locked to STAY this step. We still let it call
        #       decide() above so its perception and memory
        #       bookkeeping stay consistent across the window; only
        #       the action is overridden.
        #   4b. Knockout: groups of Chebyshev-adjacent prey defeat
        #       up to n-1 unclaimed predators per group (see
        #       _resolve_knockouts for the exact rules). In "stun"
        #       mode each defeated predator gets a stun timer;
        #       in "kill" mode it is marked dead. Either way the
        #       predator's action is overridden to STAY and the
        #       "sandwicher" prey are also locked to STAY for this
        #       step.
        if self.config.prey_defend == "stun":
            for body in self.env.agent_bodies.values():
                if (
                    body.team == au.TEAM_PREDATOR
                    and body.alive
                    and body.stun_remaining > 0
                ):
                    intentions[body.agent_id] = au.STAY
        if self.config.prey_defend is not None:
            self._resolve_knockouts(intentions, self.config.prey_defend)

        resolve_actions(self.env, intentions, self.rng)
        captured = self.env.apply_captures()
        rews = attribute_rewards(self.env, captured)
        for aid, r in rews.items():
            self.cumulative_rewards[aid] = self.cumulative_rewards.get(aid, 0.0) + r

        # Phase 5: decrement stun timers at end of step ("stun" mode
        # only; in "kill" mode no predator ever carries a timer). A
        # stun of duration D applied on step T overrides the
        # predator on steps T, T+1, ..., T+D-1 (D forced-STAY steps
        # in total) and the predator is back online on step T+D.
        # This is what decrementing after the action+capture passes
        # gives us: stun_remaining is D at the start of step T
        # (override applied, capture skipped, then decremented to
        # D-1) and 0 at the start of step T+D.
        if self.config.prey_defend == "stun":
            for body in self.env.agent_bodies.values():
                if body.team == au.TEAM_PREDATOR and body.stun_remaining > 0:
                    body.stun_remaining -= 1

        self.step_index += 1

        # Termination checks. Prey-extinction is checked first so
        # that a same-step "last prey captured" outcome is recorded
        # as a predator win even when --prey-defend kill has just
        # eliminated the last predator (this matches the existing
        # convention that all-prey-dead always means predator win).
        # Predator-extinction is checked second and is only
        # reachable in "kill" mode but is correct as a general
        # rule, so it isn't gated.
        if not self.env.any_prey_alive():
            self.outcome = au.OUTCOME_PREDATORS_WIN
            return False
        if not self.env.any_predator_alive():
            self.outcome = au.OUTCOME_PREY_WIN
            return False
        if self.step_index >= self.config.timesteps:
            if self.env.any_prey_alive():
                self.outcome = au.OUTCOME_PREY_WIN
            else:
                self.outcome = au.OUTCOME_PREDATORS_WIN
            return False
        return True

    def _resolve_knockouts(self, intentions: Dict[int, str], mode: str) -> None:
        """Cooperative knockout pass for the prey-defend mechanic.

        Identifies groups of alive prey by connected components on
        the prey-prey Chebyshev-1 adjacency graph. For each group of
        size n >= 2:

        * Candidate predators are alive, not currently stunned
          (stun_remaining == 0; relevant only in stun mode), not
          already defeated by an earlier group this same step, and
          Chebyshev-1 of at least 2 distinct group members
          ("sandwich" condition, the natural n>=2 generalisation of
          "Cheb-1 of both prey").
        * Up to n - 1 candidates are defeated, picked in ascending
          predator agent id order (fully deterministic — no RNG
          flows through this pass).
        * The defeat effect depends on `mode`:
            - "stun": predator's `stun_remaining` is set to
              `cfg.stun_duration`; the predator is locked to STAY
              this step and the capture pass will skip it for the
              duration of the stun.
            - "kill": predator's `alive` is set to False;
              the predator is locked to STAY this step (cosmetic,
              since the resolver and capture pass already skip
              dead bodies) and stays dead for the rest of the run.
        * The "sandwichers" — prey from this group at Cheb-1 of any
          defeated predator — are forced to STAY for this step.
          Other group members are free to act normally.

        Groups are processed in ascending order of their lowest
        member's agent id, so when multiple groups could claim the
        same predator the order is reproducible. Predators defeated
        by an earlier group are skipped by later groups (no double-
        counting against later groups' n-1 caps either: caps are
        per-group, candidates are per-group filtered).
        """
        assert mode in ("stun", "kill"), f"unexpected prey_defend mode {mode!r}"
        alive_prey_by_id = {
            b.agent_id: b
            for b in self.env.agent_bodies.values()
            if b.alive and b.team == au.TEAM_PREY
        }
        if len(alive_prey_by_id) < 2:
            return

        alive_pred_by_id = {
            b.agent_id: b
            for b in self.env.agent_bodies.values()
            if b.alive and b.team == au.TEAM_PREDATOR
        }
        if not alive_pred_by_id:
            return

        # Connected components on prey-prey Chebyshev-1 adjacency.
        prey_ids = sorted(alive_prey_by_id)
        adj: Dict[int, List[int]] = {pid: [] for pid in prey_ids}
        for i, pid1 in enumerate(prey_ids):
            b1 = alive_prey_by_id[pid1]
            for pid2 in prey_ids[i + 1:]:
                b2 = alive_prey_by_id[pid2]
                if chebyshev(b1.x, b1.y, b2.x, b2.y) <= 1:
                    adj[pid1].append(pid2)
                    adj[pid2].append(pid1)

        visited: set[int] = set()
        groups: List[List[int]] = []
        for pid in prey_ids:
            if pid in visited:
                continue
            stack = [pid]
            comp: List[int] = []
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                comp.append(cur)
                for nb in adj[cur]:
                    if nb not in visited:
                        stack.append(nb)
            comp.sort()
            groups.append(comp)
        groups.sort(key=lambda g: g[0])

        defeated_this_step: set[int] = set()
        pred_ids_sorted = sorted(alive_pred_by_id)

        for group in groups:
            n = len(group)
            if n < 2:
                continue
            cap = n - 1

            candidates: List[int] = []
            for pid_pred in pred_ids_sorted:
                if pid_pred in defeated_this_step:
                    continue
                pb = alive_pred_by_id[pid_pred]
                if pb.stun_remaining > 0:
                    continue
                count = 0
                for prey_id in group:
                    pry = alive_prey_by_id[prey_id]
                    if chebyshev(pb.x, pb.y, pry.x, pry.y) <= 1:
                        count += 1
                        if count >= 2:
                            break
                if count >= 2:
                    candidates.append(pid_pred)

            if not candidates:
                continue
            chosen = candidates[:cap]
            for pred_id in chosen:
                defeated_this_step.add(pred_id)
                pb = alive_pred_by_id[pred_id]
                if mode == "stun":
                    pb.stun_remaining = self.config.stun_duration
                else:  # mode == "kill"
                    pb.alive = False
                intentions[pred_id] = au.STAY
                for prey_id in group:
                    pry = alive_prey_by_id[prey_id]
                    if chebyshev(pb.x, pb.y, pry.x, pry.y) <= 1:
                        intentions[prey_id] = au.STAY

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
        # Counts every prey-victory episode (timeout OR `--prey-defend
        # kill` elimination of all predators). The old name
        # `prey_timeout_wins` was inaccurate after the kill mode
        # landed.
        self.prey_wins = 0
        self.total_steps = 0
        # Per-outcome step totals so the reporter can show "mean
        # steps among predator wins" vs "mean steps among prey wins"
        # separately. The pooled mean is still `total_steps / runs`.
        self.predator_win_steps = 0
        self.prey_win_steps = 0
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
            acc.predator_win_steps += summary.steps
        elif summary.outcome == au.OUTCOME_PREY_WIN:
            acc.prey_wins += 1
            acc.prey_win_steps += summary.steps
    return acc


def format_batch_summary(summary: BatchSummary, config: SimulationConfig) -> str:
    """Render a `BatchSummary` as a one-data-row markdown table.

    The layout is designed for the writeup workflow: each command
    contributes a single row to a master table that aggregates
    results across many configurations. The function always emits
    `header + separator + data row`, so the output of a single
    command is self-describing; when concatenating multiple
    commands' outputs into one big table, keep the header from the
    first command and drop the header+separator from every
    subsequent one (or pipe the output through `Select-Object
    -Last 1` / `tail -n 1` to get just the data row).

    Columns mirror the visualisation the user works from:

    | Number of Predators | Number of Prey | Predator Wins | Prey Wins
      | Mean Run Timesteps | Mean Run Timesteps in Predator-won
      | Mean Run Timesteps in Prey-won |

    Means are formatted with two decimals. Cells whose underlying
    count is 0 (e.g. `predator_wins == 0`) render as an empty cell
    rather than `n/a` to match the visual style of the source
    spreadsheet — markdown renderers display them as blank.
    """
    def fmt_mean(total: int, count: int) -> str:
        if count <= 0:
            return ""
        return f"{total / count:.2f}"

    header_cells = [
        ("Number of Predators", 19),
        ("Number of Prey", 14),
        ("Predator Wins", 13),
        ("Prey Wins", 9),
        ("Mean Run Timesteps", 18),
        ("Mean Run Timesteps in Predator-won", 34),
        ("Mean Run Timesteps in Prey-won", 30),
    ]
    data_cells = [
        str(config.num_predators),
        str(config.num_prey),
        str(summary.predator_wins),
        str(summary.prey_wins),
        fmt_mean(summary.total_steps, summary.runs),
        fmt_mean(summary.predator_win_steps, summary.predator_wins),
        fmt_mean(summary.prey_win_steps, summary.prey_wins),
    ]

    header = "| " + " | ".join(label for label, _w in header_cells) + " |"
    sep = "| " + " | ".join("-" * w for _label, w in header_cells) + " |"
    row = "| " + " | ".join(
        f"{value:>{w}}" for value, (_label, w) in zip(data_cells, header_cells)
    ) + " |"
    return "\n".join([header, sep, row])