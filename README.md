# PredPrey Sim

A grid-world predator-prey simulation built as a small multi-agent
system. Predators (purple) chase prey (green); prey try to escape until
the timestep budget runs out. The baseline behaviour is plain chase
vs. flee with partial information; opt-in flags layer on intra-team
communication (`--comms`) and a team strategy role system
(`--roles`) that adds pickable obstacles with team ownership and
role-based use of them.

The codebase favours readability over performance: the per-step pipeline
is explicit, every public field is small and documented, and behaviour
that touches the rules of the game lives in clearly named modules
rather than inside large dispatch tables.

---

## Quickstart

```bash
python main.py                    # batch run, defaults
python main.py --gui              # pygame window
python main.py --runs 5 --seed 0  # five seeded runs, text summary
python main.py --help             # full CLI reference
```

Python 3.10+, `pygame` for the GUI only.

---

## Game model

### Grid

A rectangle of `--width` x `--height` cells. Cells can be empty, walls
(impassable, immutable) or movable obstacles. Walls come from two
sources combined into the same set:

- `--walls N --wall-size K` auto-generates `N` straight wall segments of
  length `K`. The generator never blocks a full row or column.
- Programmatic `SimulationConfig.walls` for tests.

### Agents and movement

Agents act simultaneously every step. Each step every alive agent
chooses one of:

- Cardinal moves: `up`, `down`, `left`, `right`.
- `stay`.
- `pickup` or `drop` (obstacle, see below).

Movement rules at the action-resolution stage
([`action_resolution._target_cell`](action_resolution.py)):

- Off-grid and walls always block.
- **Same-team agents never share a cell.** If a teammate is currently on
  the target cell and is staying this step (`STAY`, or `PICKUP` / `DROP`,
  which don't change position), the entry is denied. If the teammate is
  itself moving away, the entry falls through to the same-team
  collision pass, which uses a random winner with forced-stay
  propagation in case the teammate's own move ends up blocked.
- **Cross-team co-location is one-directional.** A predator stepping
  onto a prey cell is the capture mechanic and is allowed (subject to
  the obstacle / simultaneity rules below). Prey moving onto a
  predator cell is always denied — there is no symmetric "prey
  suicide" path.
- Obstacles add the further conditions described under
  "[Co-location with a dropped obstacle](#co-location-with-a-dropped-obstacle)".

### Vision and distance

Each team has an independent Chebyshev vision radius
(`--vision-predator`, `--vision-prey`, both default to 2). Within that
square an agent sees all alive enemies, alive teammates, and unheld
obstacles (with their ownership). Outside that square the world is dark.

Two distance metrics show up in the codebase:

- **Chebyshev** (`distances.chebyshev`) — used only by vision (an L_inf
  square is the right shape for "everything within k of me on a grid
  with 8-neighbour visibility, even though we only move on 4").
- **Manhattan** (`distances.manhattan`) — used for everything tactical:
  chase scoring, flee scoring, pickup capture-imminent guard. With pure
  4-cardinal movement the predator's true step-cost to prey is exactly
  Manhattan, so Chebyshev would be over-permissive (e.g. a diagonal
  enemy at Chebyshev 1 is at Manhattan 2 and is one step further out
  than the metric suggests).

### Capture

After actions resolve, any cell that contains at least one alive
predator and at least one alive prey marks every prey on that cell as
captured. The episode ends the moment every prey is dead (predators
win) or when the timestep budget elapses (prey wins on timeout).

---

## Perception, memory and team communication

The per-step pipeline lives in [`simulation.py`](simulation.py)
`SimulationState.step_once`:

1. **Raw observations.** Each alive agent gets an obs dict from
   [`observation_definition.py`](observation_definition.py): its own
   cell, legal actions, vision radius, visible enemies, visible
   allies, visible obstacles (with `locked_team`), what it is holding,
   whether it is standing on an unheld obstacle, and a 4-tuple of
   cardinal-blocked flags.
2. **Team comms** (optional, opt-in via `--comms`). Each speaker
   broadcasts its directly-visible enemies to the teammates inside its
   own vision radius (the `visible_allies` it currently sees). Comms
   are single-hop, synchronous, and speaker-centric — see
   [`simulation._exchange_team_messages`](simulation.py). The fan-out
   means a receiver may know about an enemy slightly outside its own
   square, but only when a teammate literally sees the enemy *this
   step*.
3. **Priority resolution.** Each agent fuses direct sight with the
   teammate reports addressed to it via
   [`Agent.prepare_observation`](agent.py), which delegates the actual
   logic to
   [`Perception.compute_active_enemies`](perception.py). The fusion
   rule is strict priority:
   - if any direct sightings, use those;
   - else if any teammate reports, use those;
   - else empty.
   The simulation only orchestrates the call — the agent (through its
   perception module) decides what it "knows". The result lives on
   `obs["active_enemies"]` and both the selector and decide consume
   it, so they cannot disagree about who the threat is.
   When `active_enemies` is non-empty, the receiver's `last_seen_enemy`
   memory is refreshed via
   [`Perception.update_last_seen_enemy`](perception.py) — the fresher
   signal (own eyes or a teammate's eyes) wipes out any older memory.
4. **Role assignment** (optional, opt-in via `--roles`). When the flag
   is on, the team role selector runs once per team (see "Roles"
   below) and writes a `(role, role_target)` onto every Agent. When
   off, every agent keeps the default role it was constructed with
   (`ROLE_CHASER` for predators, `ROLE_FLEE` for prey) and the
   decision layer additionally suppresses every obstacle interaction
   so the game collapses to plain chase vs. flee on top of the
   walls / obstacles that happen to exist on the map.
5. **Per-agent decision.** Each agent's `Agent.decide` calls
   `DecisionMaking.choose_action` with the comms-augmented obs plus
   `last_seen_enemy`. The chosen action goes back to action resolution.

This ordering makes the contract simple: the selector and the agent
both consume the same `active_enemies` set, so they cannot disagree
about who the threat is or where it is.

---

## Obstacles and ownership

Obstacles are unit-sized movable cells defined in
[`environment.Obstacle`](environment.py). The model has three knobs:

1. **Pickup.** An agent's `PICKUP` action is legal iff there is at
   least one unheld obstacle within Chebyshev 1 of the agent that the
   agent's team is *allowed* to take. An unclaimed obstacle is always
   takeable; a locked obstacle is takeable only by its owning team.
   The agent picks up the first matching obstacle, which now travels
   invisibly with it.
2. **Drop.** The agent's `DROP` action puts the held obstacle on the
   agent's current cell and stamps `obstacle.locked_team` with the
   dropper's team. Locks persist across re-pickups: a prey-locked
   obstacle that gets re-picked-up by a prey teammate and re-dropped
   stays prey-locked.
3. **Movement blocking.** An unheld obstacle is a movement obstacle.
   The default `--lock-mode symmetric` blocks both teams from entering
   the cell. The opt-in `--lock-mode owner-passable` makes own-locked
   obstacles passable for the owning team, while still blocking the
   enemy. Lock-mode is wired through
   [`environment.Environment`](environment.py),
   [`action_resolution._target_cell`](action_resolution.py),
   [`environment.Environment.legal_actions`](environment.py), and
   [`observation_definition._is_blocked_for_team`](observation_definition.py)
   so the cardinal-blocked observation, the navigation choices and the
   actual movement resolution always agree.

### Capture-imminent pickup guard

`PICKUP` forfeits the agent's movement that step. With 4-cardinal
movement, an enemy at Manhattan 1 can step into the agent's cell next
turn. `DecisionMaking.choose_action` therefore skips `PICKUP` if any
agent in the `active_enemies` set is at Manhattan ≤ 1 — see the
"Capture-imminent pickup guard" comment in
[`decision_making.py`](decision_making.py). The guard uses
`active_enemies` (direct sight + teammate reports), not stale memory:
pickup is a now-or-never call and we want it conditioned on the
freshest threat the team can produce.

This is why "Manhattan" and not "Chebyshev": Chebyshev would flag a
diagonal enemy as adjacent, but a diagonal enemy needs *two* cardinal
steps to capture, not one — by the time it reaches the agent the agent
will have already moved, so picking up is still safe.

### Co-location with a dropped obstacle

An obstacle on the ground is *next to* an agent that shares its cell,
not a force field around it. The cell is treated as enterable for
movement only when at least one live agent at the cell is going to
*stay* this step. Because actions resolve simultaneously,
[`action_resolution._target_cell`](action_resolution.py) consults the
full move-intention dict before granting entry:

- The step a prey drops an obstacle, the prey is co-located with it
  and intends to stay (or is itself the mover). A predator that steps
  in lands on the shared cell and the capture is resolved normally.
- The step a prey moves off the obstacle, the cell is on its way to
  being empty of agents. A predator that tried to enter the cell that
  same step is denied — the new wall is taking effect, and the
  predator stays put rather than parking on an enemy-locked obstacle.
- The step *after* the prey has fully moved off, the cell is empty
  with the obstacle still on it. Now it behaves like a wall (subject
  to lock-mode), blocking predators until someone picks it up.

[`observation_definition._is_blocked_for_team`](observation_definition.py)
applies the same agent-on-cell exception but without the move-intent
lookahead — it reports the *current* snapshot for the `cardinal_blocked`
field of an observation. The simulation's movement rule is the
authoritative one.

This is why "drop and squat" is not a winning prey strategy and why
"drop on top of me, flee, the predator follows me into the obstacle"
is also not exploitable. The useful pattern is "drop now, flee next
step": the prey leaves behind a one-cell wall that the chasing
predator has to detour around, while the prey itself never enjoyed
any invulnerability frames.

---

## Roles

Roles are an opt-in extension enabled with `--roles`. **Default
behaviour is plain chase vs. flee with no obstacle interaction.** When
the flag is off, the team role selector is skipped, every agent keeps
its construction-time role (`ROLE_CHASER` for predators, `ROLE_FLEE`
for prey), and `DecisionMaking.choose_action` short-circuits the
obstacle pipeline: `PICKUP` is never chosen, so no agent ever holds an
obstacle, so `DROP` never becomes legal either, and the FLANKER / NET /
SHIELDER / BREADCRUMB / BUNKER branches are unreachable.

The rest of this section describes what those branches do **when the
flag is on**. Roles are assigned every step by a team-level selector
(`decision_making.select_team_roles`) and consumed by the per-agent
behaviour in `DecisionMaking.choose_action`. Each agent gets a
`(role, role_target)` pair:

- **Role** decides the action style (chase, flank, drop, flee, ...).
- **Role target** is the optional grid cell the agent is heading
  toward (or already on). Role targets are intentionally *sticky*: if
  the previous step's target is still valid (still a cardinal of the
  current target prey for a flanker; still in the cardinal slot list
  for a net agent), the selector keeps it. This prevents jerky target
  flapping when the prey moves by one cell.

### Predator roles

Selector inputs: the team's aggregated view of prey (union of
`active_enemies` across team members) and the per-agent
`held_obstacle` flag.

1. Pick the **target prey** — the one closest to the predator centroid
   (Manhattan), deterministic tiebreak by enemy id.
2. Split predators into **carriers** (holding an obstacle) and
   **non-carriers** (empty hands).
3. **Non-carriers** always get `ROLE_CHASER`. CHASER runs the existing
   greedy min-Manhattan chase against `last_seen_enemy`
   (refreshed-from-active-this-step via `Agent.decide`). When stale
   memory takes them to a cell where the enemy is no longer present,
   they wipe the memory and explore.
4. **Single carrier** gets `ROLE_FLANKER`. Target cell is the cardinal
   of the prey furthest from the predator centroid (intuition: take
   the far side so the prey gets squeezed back through its pursuers).
   FLANKER's per-step behaviour is "if I'm already on the target cell,
   `DROP`; otherwise greedy navigation toward the target".
5. **Two or more carriers** get `ROLE_NET`. The selector greedily
   assigns each carrier to one of the (up to four) walkable cardinals
   of the prey, honouring stickiness on previous slot assignments.
   Carriers that don't fit (more carriers than slots) fall back to
   `ROLE_CHASER`. NET's per-step behaviour is identical to FLANKER's:
   navigate, then drop on arrival.

The reason CHASER is allowed to `PICKUP` (subject to the Manhattan-1
guard) is that this is how predators acquire obstacles to flank with
in the first place. The selector picks them up by being a CHASER, and
the next step they become a FLANKER or NET because they now hold
something.

### Prey roles

Selector inputs: `held_obstacle` and `active_enemies`. The cascade
chooses the role according to the distance to the nearest tracked
threat and whether dropping the obstacle right now will produce a
wall that actually intersects the chase. Because a dropped obstacle
takes effect only after the prey leaves the cell, the "useful" drop
window is when the nearest predator is one to two steps further than
adjacent — too close and the prey is captured before the wall lands,
too far and the wall is unlikely to be on the predator's actual path.

| Trigger | Role | Per-step behaviour |
|---------|------|--------------------|
| Not holding | `ROLE_FLEE` | Existing flee logic: head away from the primary threat while avoiding moves that close distance to any other tracked predator. |
| Holding, no known threat | `ROLE_BUNKER` | Keep carrying the obstacle (no drop). Movement is the same flee logic; with no active threat that degenerates to a random walk among legal cells. |
| Holding, nearest threat at Manhattan ≤ 1 | `ROLE_FLEE` | Drop is wasted — the predator captures next step regardless of what's on the cell. Flee instead. |
| Holding, nearest threat at Manhattan == 2 | `ROLE_SHIELDER` | `DROP` on the current cell. After the prey moves away next step, the wall lands exactly between it and the chasing predator. |
| Holding, nearest threat at Manhattan == 3 | `ROLE_BREADCRUMB` | `DROP` on the current cell to start a chase-extending trail. |
| Holding, nearest threat at Manhattan ≥ 4 | `ROLE_BUNKER` | Still too far for a drop to pay off. Save the obstacle. |

`ROLE_FUNNELER` is currently retired by the selector — its
"navigate-into-the-predator's-predicted-cell" tactic only worked under
the old "drop is a force field" rule, and under the new rule walking
toward the predator just hands them an easy capture. The constant
and the predict-next-cell helper remain in the codebase for future
experiments.

---

## Visualisation

`python main.py --gui` opens a pygame window driven by
[`visualization.py`](visualization.py).

| Element | Meaning |
|---------|---------|
| Purple rounded rect | Predator body |
| Green rounded rect | Prey body |
| Single white letter inside a body | Current role (`C` chaser, `F` flanker, `N` net, `-` flee, `R` breadcrumb, `S` shielder, `K` bunker; `U` funneler is reserved and not currently assigned) |
| Light grey filled circle | Unclaimed obstacle |
| Purple-tinted filled circle | Predator-locked obstacle |
| Green-tinted filled circle | Prey-locked obstacle |
| Black ring on an agent body | That agent is carrying an obstacle |
| Dark grey filled cell | Wall |

Controls: `space` / `right` step once, `a` toggles auto-run, `r` resets
the current run with its seed, `n` advances to the next seeded run when
the current one is over, `esc` quits.

The GUI mirrors the headless logging: every time a run finishes it
prints a one-line summary (`run=i/N  seed=S  outcome=...  steps=K`)
and on quit it prints the final `runs= predator_wins= prey_timeout_wins=`
plus `mean_steps=` block, identical in format to the batch runner.
Closing the window before every run has finished prints an explicit
`simulation aborted (X/N runs completed)` line before the partial
summary, so the printed totals never silently misrepresent what
actually ran.

---

## CLI reference

All flags are kebab-case; full list available via `python main.py --help`.

| Flag | Default | Meaning |
|------|---------|---------|
| `--gui` | off | Open the pygame window instead of running headless |
| `--width N` | 10 | Grid width in cells |
| `--height N` | 8 | Grid height in cells |
| `--timestep N` (alias `--timesteps`) | 200 | Per-run step cap; prey wins on cap |
| `--runs N` | 1 | Number of seeded runs in the batch |
| `--seed N` | 0 | Base seed; run `i` uses `seed + i` |
| `--vision-predator N` | 2 | Chebyshev radius for predators |
| `--vision-prey N` | 2 | Chebyshev radius for prey |
| `--predators N` | 1 | Number of predators per run |
| `--prey N` | 1 | Number of prey per run |
| `--walls N` | 2 | Random wall segments to generate |
| `--wall-size N` | 2 | Length of each generated wall segment |
| `--obstacles N` | 0 | Number of pickable obstacles to spawn |
| `--comms` | off | Enable speaker-centric, single-hop team communication |
| `--roles` | off | Enable team strategy roles + obstacle interaction (without it agents only chase / flee and ignore obstacles) |
| `--lock-mode {symmetric, owner-passable}` | symmetric | Obstacle locking semantics (no effect when `--roles` is off, since nothing ever gets locked) |

Determinism: a given `(seed, run index, all other flags)` reproduces
exactly the same episode trace, because every stochastic choice flows
through `random.Random(cfg.seed)` constructed per run.

---

## Architecture map

| File | Responsibility |
|------|----------------|
| [`main.py`](main.py) | CLI parsing, builds `SimulationConfig`, dispatches to batch or GUI |
| [`simulation.py`](simulation.py) | `SimulationConfig`, `SimulationState`, the three-phase `step_once`, batch driver |
| [`environment.py`](environment.py) | Grid, `AgentBody`, `Obstacle`, `pickup_obstacle` / `drop_obstacle`, `legal_actions`, capture resolution |
| [`action_resolution.py`](action_resolution.py) | Same-step intention resolution with lock-mode-aware `_target_cell` |
| [`observation_definition.py`](observation_definition.py) | Per-agent obs dict including `on_obstacle_cell` and `cardinal_blocked` |
| [`perception.py`](perception.py) | Pure perception helpers: `compute_active_enemies` (direct + shared fusion), `update_last_seen_enemy` memory refresh |
| [`decision_making.py`](decision_making.py) | Geometry helpers, team role selector, role-dispatched `choose_action` |
| [`agent.py`](agent.py) | `Agent` class: holds memory, role, role_target; `prepare_observation` perception fusion; `decide` glue |
| [`agent_utils.py`](agent_utils.py) | Constants: actions, teams, outcomes, role names + display letters |
| [`distances.py`](distances.py) | `chebyshev`, `manhattan` |
| [`reward_attribution.py`](reward_attribution.py) | Per-step reward shaping (each alive predator gets +1 on every step that a prey is captured) |
| [`visualization.py`](visualization.py) | Pygame renderer with the legend above |
