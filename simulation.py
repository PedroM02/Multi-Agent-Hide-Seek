import random

import agent_utils as au
from action_resolution import resolve_actions
from agent import Agent, build_agents_for_env
from decision_making import select_pack_prey_id
from distances import chebyshev
from environment import Environment
from observation_definition import build_observation
from reward_attribution import attribute_rewards


def generate_walls(
    width,
    height,
    num_walls,
    wall_size,
    rng,
    existing_walls=None,
):
    """Generate `num_walls` straight wall segments (horizontal or vertical),
    each with `wall_size` cells. Walls are placed randomly but never fill
    an entire row/column (to keep the map passable). Already-occupied cells
    (from `existing_walls`) are skipped.

    Returns the combined list of wall cell coordinates.
    """
    occupied = set(existing_walls) if existing_walls else set()
    result = list(occupied)

    for _ in range(num_walls):
        # Try up to 50 random placements before giving up on this wall.
        for _attempt in range(50):
            horizontal = rng.choice([True, False])
            if horizontal:
                max_x = width - wall_size
                if max_x < 0:
                    continue
                origin_x = rng.randint(0, max_x)
                origin_y = rng.randint(0, height - 1)
                cells = [(origin_x + i, origin_y) for i in range(wall_size)]
                # Don't block the full row
                if len(cells) >= width:
                    continue
            else:
                max_y = height - wall_size
                if max_y < 0:
                    continue
                origin_x = rng.randint(0, width - 1)
                origin_y = rng.randint(0, max_y)
                cells = [(origin_x, origin_y + i) for i in range(wall_size)]
                # Don't block the full column
                if len(cells) >= height:
                    continue

            # Skip if any cell already occupied
            if any(cell in occupied for cell in cells):
                continue

            for cell in cells:
                occupied.add(cell)
            result.extend(cells)
            break

    return result


def comms_enabled_for_team(config, team):
    """True when `config.comms` enables intra-team messaging for `team`."""
    mode = config.comms
    if mode is None:
        return False
    if mode == "both":
        return True
    if mode == "predators":
        return team == au.TEAM_PREDATOR
    if mode == "prey":
        return team == au.TEAM_PREY
    return False


def _alive_oracle_prey(env):
    """True positions of all alive prey (Level 6 oracle channel)."""
    prey = []
    for body in env.agent_bodies.values():
        if body.alive and body.team == au.TEAM_PREY:
            prey.append((body.x, body.y, body.agent_id))
    prey.sort(key=lambda t: (t[2], t[0], t[1]))
    return tuple(prey)


def _tag_enemy_reports(sender_id, sightings):
    """Attach sender_id to each direct enemy sighting for comms."""
    return [
        (sender_id, enemy_x, enemy_y, enemy_id)
        for enemy_x, enemy_y, enemy_id in sightings
    ]


def _dedupe_enemy_messages(messages):
    seen_keys = set()
    deduped = []
    for message in messages:
        sender_id, enemy_x, enemy_y, enemy_id = message
        key = (sender_id, enemy_id)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append((sender_id, enemy_x, enemy_y, enemy_id))
    deduped.sort(key=lambda t: (t[0], t[3], t[1], t[2]))
    return tuple(deduped)


def _dedupe_ally_messages(messages):
    seen_ids = set()
    deduped = []
    for ally_x, ally_y, ally_id in messages:
        if ally_id in seen_ids:
            continue
        seen_ids.add(ally_id)
        deduped.append((ally_x, ally_y, ally_id))
    deduped.sort(key=lambda t: (t[2], t[0], t[1]))
    return tuple(deduped)


def _comms_round(raw_obs, config, enemy_payloads, ally_payloads):
    """One synchronous fan-out round for enemies and/or allies."""
    shared_enemies = {agent_id: [] for agent_id in raw_obs}
    shared_allies = {agent_id: [] for agent_id in raw_obs}
    for sender_obs in raw_obs.values():
        if not comms_enabled_for_team(config, sender_obs["team"]):
            continue
        sender_id = sender_obs["agent_id"]
        enemies = enemy_payloads.get(sender_id, ())
        allies = ally_payloads.get(sender_id, ())
        if not enemies and not allies:
            continue
        for _, _, ally_id in sender_obs["visible_allies"]:
            if ally_id not in shared_enemies:
                continue
            if enemies:
                shared_enemies[ally_id].extend(enemies)
            if allies:
                shared_allies[ally_id].extend(allies)

    return (
        {
            agent_id: _dedupe_enemy_messages(messages)
            for agent_id, messages in shared_enemies.items()
        },
        {
            agent_id: _dedupe_ally_messages(messages)
            for agent_id, messages in shared_allies.items()
        },
    )


def _exchange_team_messages(raw_obs, config):
    """Speaker-centric, single-hop, synchronous team comms.

    Every agent on a comms-enabled team broadcasts the enemies it directly
    sees this step to the teammates inside its own vision radius (its
    visible_allies). Each report is tagged with the sender's agent id:
    ``(sender_id, enemy_x, enemy_y, enemy_id)``. The receiver collects
    everything it was told, deduped by ``(sender_id, enemy_id)``, sorted
    deterministically. The receiver's own direct sightings are not filtered
    out here — priority handling lives in Agent.decide. Agents on teams with
    comms disabled do not send.
    """
    enemy_payloads = {
        obs["agent_id"]: _tag_enemy_reports(
            obs["agent_id"], obs["visible_enemies"],
        )
        for obs in raw_obs.values()
    }
    shared_enemies, _ = _comms_round(
        raw_obs, config, enemy_payloads, {},
    )
    return shared_enemies


def _exchange_pack_messages(raw_obs, config):
    """Two-pass comms for `--mode pack`: relay enemies and ally positions.

    Pass 1 broadcasts direct sightings. Pass 2 rebroadcasts the union of
    direct and pass-1 reports, reaching teammates two hops away (enough
    for a three-predator chain to share all prey and ally positions).
    """
    pass1_enemies = {
        obs["agent_id"]: _tag_enemy_reports(
            obs["agent_id"], obs["visible_enemies"],
        )
        for obs in raw_obs.values()
    }
    pass1_allies = {
        obs["agent_id"]: obs["visible_allies"] for obs in raw_obs.values()
    }
    shared_enemies_p1, shared_allies_p1 = _comms_round(
        raw_obs, config, pass1_enemies, pass1_allies,
    )

    pass2_enemies = {}
    pass2_allies = {}
    for obs in raw_obs.values():
        agent_id = obs["agent_id"]
        pass2_enemies[agent_id] = (
            _tag_enemy_reports(agent_id, obs["visible_enemies"])
            + list(shared_enemies_p1[agent_id])
        )
        pass2_allies[agent_id] = (
            obs["visible_allies"] + shared_allies_p1[agent_id]
        )

    return _comms_round(raw_obs, config, pass2_enemies, pass2_allies)


class SimulationConfig:
    def __init__(self):
        self.width = 10
        self.height = 8
        self.timesteps = 200
        self.vision_radius_predator = 2
        self.vision_radius_prey = 2
        self.num_predators = 1
        self.num_prey = 1
        self.seed = 0
        self.walls = None
        self.num_walls = 0
        self.wall_size = 3
        # Intra-team communication mode. None = disabled (default).
        # "prey" | "predators" | "both" — see comms_enabled_for_team.
        self.comms = None
        # Cooperative-knockout (prey-defend) mechanic. When enabled,
        # groups of Chebyshev-adjacent prey can defeat predators that
        # are Chebyshev-1 of >=2 group members. Two modes:
        #   - None  : mechanic disabled (default).
        #   - "stun": predators are stunned for `stun_duration` steps;
        #             they recover and rejoin the chase.
        #   - "kill": predators are removed from the run permanently;
        #             when all predators are dead, prey win.
        # See `SimulationState._resolve_knockouts` for the full rules.
        self.prey_defend = None
        self.stun_duration = 3
        # Predator decision mode — see agent_utils.MODE_*.
        self.mode = au.MODE_CHASE
        # In `--mode roles`, assign ROLE_SEARCHER when allies are visible
        # but no prey is known. Off by default; enable with --searcher.
        self.roles_searcher = False


def copy_config(base, **overrides):
    new_config = SimulationConfig()
    new_config.width = base.width
    new_config.height = base.height
    new_config.timesteps = base.timesteps
    new_config.vision_radius_predator = base.vision_radius_predator
    new_config.vision_radius_prey = base.vision_radius_prey
    new_config.num_predators = base.num_predators
    new_config.num_prey = base.num_prey
    new_config.seed = base.seed
    new_config.walls = base.walls
    new_config.num_walls = base.num_walls
    new_config.wall_size = base.wall_size
    new_config.comms = base.comms
    new_config.prey_defend = base.prey_defend
    new_config.stun_duration = base.stun_duration
    new_config.mode = base.mode
    new_config.roles_searcher = base.roles_searcher
    for key, value in overrides.items():
        setattr(new_config, key, value)
    return new_config


class EpisodeSummary:
    def __init__(self, outcome, steps, cumulative_rewards, episode_seed):
        self.outcome = outcome
        self.steps = steps
        self.cumulative_rewards = cumulative_rewards
        self.episode_seed = episode_seed


class SimulationState:
    def __init__(self, config, rng):
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
        self.agents = []
        self.step_index = 0
        self.outcome = au.OUTCOME_ONGOING
        self.cumulative_rewards = {}
        self.reset_episode()

    def reset_episode(self):
        self.env.set_agent_positions(
            self.config.num_predators,
            self.config.num_prey,
            self.rng,
        )
        self.agents = build_agents_for_env(self.env, self.rng, self.config)
        for agent in self.agents:
            agent.reset_memory()
        self.step_index = 0
        self.outcome = au.OUTCOME_ONGOING
        self.cumulative_rewards = {
            agent_id: 0.0 for agent_id in self.env.agent_bodies
        }
        self._pack_focus_prey_id = None

    def _inject_oracle_obs(self, raw_obs):
        """Level 6: clairvoyant prey positions + optional shared pack target."""
        if self.config.mode != au.MODE_OPTIMAL:
            return

        oracle = _alive_oracle_prey(self.env)
        walls = set(self.env.wall_cells)
        grid_width, grid_height = self.env.width, self.env.height

        predator_positions = []
        for obs in raw_obs.values():
            if obs["team"] == au.TEAM_PREDATOR:
                predator_positions.append((obs["agent_x"], obs["agent_y"]))

        pack_target = None
        alive_ids = {prey_id for _, _, prey_id in oracle}
        if self._pack_focus_prey_id not in alive_ids:
            self._pack_focus_prey_id = None
        if oracle and self._pack_focus_prey_id is None:
            self._pack_focus_prey_id = select_pack_prey_id(
                predator_positions, oracle, grid_width, grid_height, walls,
            )
        if self._pack_focus_prey_id is not None:
            for prey_x, prey_y, prey_id in oracle:
                if prey_id == self._pack_focus_prey_id:
                    pack_target = (prey_x, prey_y)
                    break

        for obs in raw_obs.values():
            if obs["team"] != au.TEAM_PREDATOR:
                continue
            obs["wall_cells"] = walls
            obs["grid_width"] = grid_width
            obs["grid_height"] = grid_height
            if pack_target is not None:
                obs["pack_target"] = pack_target

    def step_once(self):
        if self.outcome != au.OUTCOME_ONGOING:
            return False

        # Phase 1: build raw observations for every alive agent.
        raw_obs = {}
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

        # Phase 2: synchronous, speaker-centric team comms.
        # Pack mode uses two passes so a three-predator chain can relay
        # prey and ally positions; all other modes stay single-hop.
        if self.config.comms is not None:
            if self.config.mode == au.MODE_PACK:
                shared_enemies, shared_allies = _exchange_pack_messages(
                    raw_obs, self.config,
                )
            else:
                shared_enemies = _exchange_team_messages(
                    raw_obs, self.config,
                )
                shared_allies = {
                    agent_id: tuple() for agent_id in raw_obs
                }
        else:
            shared_enemies = {agent_id: tuple() for agent_id in raw_obs}
            shared_allies = {agent_id: tuple() for agent_id in raw_obs}

        # Phase 2b: each agent fuses its direct sightings with the
        # teammate reports addressed to it, producing the
        # priority-resolved `active_enemies` set on its own observation.
        # The fusion lives on the Agent (delegated to its Perception) so
        # that the simulation only orchestrates — it doesn't decide what
        # the agent "knows".
        agents_by_id = {agent.agent_id: agent for agent in self.agents}
        for agent_id, obs in raw_obs.items():
            if comms_enabled_for_team(self.config, obs["team"]):
                enemy_reports = shared_enemies[agent_id]
                ally_reports = shared_allies[agent_id]
            else:
                enemy_reports = tuple()
                ally_reports = tuple()
            agents_by_id[agent_id].prepare_observation(
                obs, enemy_reports, ally_reports,
            )

        # Phase 2.5: oracle fields for Level 6 optimal modes.
        self._inject_oracle_obs(raw_obs)

        # Phase 3: agents decide using direct sightings, then teammate
        # reports, then their own memory. Roles are derived locally
        # inside each agent in `--mode roles`.
        intentions = {}
        for agent in self.agents:
            body = self.env.agent_bodies[agent.agent_id]
            if not body.alive:
                continue
            intentions[agent.agent_id] = agent.decide(
                raw_obs[agent.agent_id],
            )

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

        resolve_actions(self.env, intentions)
        captured = self.env.apply_captures()
        rewards = attribute_rewards(self.env, captured)
        for agent_id, reward in rewards.items():
            self.cumulative_rewards[agent_id] = (
                self.cumulative_rewards.get(agent_id, 0.0) + reward
            )

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

    def _resolve_knockouts(self, intentions, mode):
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
            body.agent_id: body
            for body in self.env.agent_bodies.values()
            if body.alive and body.team == au.TEAM_PREY
        }
        if len(alive_prey_by_id) < 2:
            return

        alive_pred_by_id = {
            body.agent_id: body
            for body in self.env.agent_bodies.values()
            if body.alive and body.team == au.TEAM_PREDATOR
        }
        if not alive_pred_by_id:
            return

        # Connected components on prey-prey Chebyshev-1 adjacency.
        prey_ids = sorted(alive_prey_by_id)
        adjacency = {prey_id: [] for prey_id in prey_ids}
        for i, prey_id_1 in enumerate(prey_ids):
            body_1 = alive_prey_by_id[prey_id_1]
            for prey_id_2 in prey_ids[i + 1:]:
                body_2 = alive_prey_by_id[prey_id_2]
                if chebyshev(body_1.x, body_1.y, body_2.x, body_2.y) <= 1:
                    adjacency[prey_id_1].append(prey_id_2)
                    adjacency[prey_id_2].append(prey_id_1)

        visited = set()
        groups = []
        for prey_id in prey_ids:
            if prey_id in visited:
                continue
            stack = [prey_id]
            component = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                for neighbor_id in adjacency[current]:
                    if neighbor_id not in visited:
                        stack.append(neighbor_id)
            component.sort()
            groups.append(component)
        groups.sort(key=lambda group: group[0])

        defeated_this_step = set()
        predator_ids_sorted = sorted(alive_pred_by_id)

        for group in groups:
            group_size = len(group)
            if group_size < 2:
                continue
            cap = group_size - 1

            candidates = []
            for predator_id in predator_ids_sorted:
                if predator_id in defeated_this_step:
                    continue
                predator_body = alive_pred_by_id[predator_id]
                if predator_body.stun_remaining > 0:
                    continue
                count = 0
                for prey_id in group:
                    prey_body = alive_prey_by_id[prey_id]
                    if chebyshev(
                        predator_body.x, predator_body.y,
                        prey_body.x, prey_body.y,
                    ) <= 1:
                        count += 1
                        if count >= 2:
                            break
                if count >= 2:
                    candidates.append(predator_id)

            if not candidates:
                continue
            chosen = candidates[:cap]
            for predator_id in chosen:
                defeated_this_step.add(predator_id)
                predator_body = alive_pred_by_id[predator_id]
                if mode == "stun":
                    predator_body.stun_remaining = self.config.stun_duration
                else:  # mode == "kill"
                    predator_body.alive = False
                intentions[predator_id] = au.STAY
                for prey_id in group:
                    prey_body = alive_prey_by_id[prey_id]
                    if chebyshev(
                        predator_body.x, predator_body.y,
                        prey_body.x, prey_body.y,
                    ) <= 1:
                        intentions[prey_id] = au.STAY

    def status_line(self):
        return f"Timestep {self.step_index}/{self.config.timesteps}  Current Outcome={self.outcome}"


def run_episode(config, rng):
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
    def __init__(self):
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


def run_batch(config, num_runs):
    accumulator = BatchSummary()
    for i in range(num_runs):
        run_config = copy_config(config, seed=config.seed + i)
        rng = random.Random(run_config.seed)
        summary = run_episode(run_config, rng)
        accumulator.runs += 1
        accumulator.total_steps += summary.steps
        if summary.outcome == au.OUTCOME_PREDATORS_WIN:
            accumulator.predator_wins += 1
            accumulator.predator_win_steps += summary.steps
        elif summary.outcome == au.OUTCOME_PREY_WIN:
            accumulator.prey_wins += 1
            accumulator.prey_win_steps += summary.steps
    return accumulator


def format_batch_summary(summary, config):
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
    def fmt_mean(total, count):
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

    header = "| " + " | ".join(label for label, _width in header_cells) + " |"
    sep = "| " + " | ".join("-" * width for _label, width in header_cells) + " |"
    row = "| " + " | ".join(
        f"{value:>{width}}" for value, (_label, width) in zip(data_cells, header_cells)
    ) + " |"
    return "\n".join([header, sep, row])
