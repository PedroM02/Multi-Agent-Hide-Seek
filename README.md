# PredPrey Sim

A grid-world predator-prey simulation built as a small multi-agent
system. Predators (purple) chase prey (green); prey try to escape until
the timestep budget runs out. Both teams have partial information and
optional intra-team communication. A team-level role selector sits in
the per-step pipeline as the hook for future hunting / protection
strategies; right now it simply tags predators as `CHASER` and prey as
`FLEE`.

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

A rectangle of `--width` x `--height` cells. Cells are either empty or
walls (impassable, immutable). Walls come from two sources combined
into the same set:

- `--walls N --wall-size K` auto-generates `N` straight wall segments of
  length `K`. The generator never blocks a full row or column.
- Programmatic `SimulationConfig.walls` for tests.

### Agents and movement

Agents act simultaneously every step. Each step every alive agent
chooses one of:

- Cardinal moves: `up`, `down`, `left`, `right`.
- `stay`.

Movement rules at the action-resolution stage
([`action_resolution._target_cell`](action_resolution.py)):

- Off-grid and walls always block.
- **Same-team agents never share a cell.** If a teammate is currently on
  the target cell and is staying this step (any zero-delta action), the
  entry is denied. If the teammate is itself moving away, the entry
  falls through to the same-team collision pass, which uses a random
  winner with forced-stay propagation in case the teammate's own move
  ends up blocked.
- **Cross-team co-location is one-directional.** A predator stepping
  onto a prey cell is the capture mechanic and is allowed. Prey moving
  onto a predator cell is always denied — there is no symmetric "prey
  suicide" path.

### Vision and distance

Each team has an independent Chebyshev vision radius
(`--vision-predator`, `--vision-prey`, both default to 2). Within that
square an agent sees all alive enemies and alive teammates. Outside
that square the world is dark.

Two distance metrics show up in the codebase:

- **Chebyshev** (`distances.chebyshev`) — used only by vision (an L_inf
  square is the right shape for "everything within k of me on a grid
  with 8-neighbour visibility, even though we only move on 4").
- **Manhattan** (`distances.manhattan`) — used for everything tactical:
  chase scoring, flee scoring. With pure 4-cardinal movement the
  predator's true step-cost to prey is exactly Manhattan, so Chebyshev
  would be over-permissive (e.g. a diagonal enemy at Chebyshev 1 is at
  Manhattan 2 and is one step further out than the metric suggests).

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
   cell, legal actions, vision radius, visible enemies and visible
   allies.
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
   `obs["active_enemies"]` and downstream consumers use it, so they
   cannot disagree about who the threat is.
   When `active_enemies` is non-empty, the receiver's `last_seen_enemy`
   memory is refreshed via
   [`Perception.update_last_seen_enemy`](perception.py) — the fresher
   signal (own eyes or a teammate's eyes) wipes out any older memory.
4. **Role assignment.** The team role selector runs once per team and
   writes a `(role, role_target)` onto every Agent. The current
   selector is trivial — predators always get `ROLE_CHASER`, prey
   always get `ROLE_FLEE` — and exists as the hook for the next round
   of hunting / protection strategies (see "Roles" below).
5. **Per-agent decision.** Each agent's `Agent.decide` calls
   `DecisionMaking.choose_action` with the comms-augmented obs plus
   `last_seen_enemy`. `choose_action` currently dispatches on team
   alone (predators chase, prey flee). The chosen action goes back to
   action resolution.

This ordering makes the contract simple: the selector and the agent
both consume the same `active_enemies` set, so they cannot disagree
about who the threat is or where it is.

---

## Roles

Roles are written every step by
[`decision_making.select_team_roles`](decision_making.py) onto each
`Agent` (`agent.role`, `agent.role_target`) and threaded into the per-
agent observation so the decision layer can dispatch on them. Two
roles are currently defined:

| Role | Team | Per-step behaviour |
|------|------|--------------------|
| `ROLE_CHASER` | predator | Greedy min-Manhattan chase against the closest target in `active_enemies`, falling back to `last_seen_enemy` when no enemy is visible. Stale memory at ego's cell is wiped and the agent explores. |
| `ROLE_FLEE` | prey | Head away from the primary threat (the Manhattan-closest tracked predator) while avoiding moves that close distance to any other tracked predator. Same `last_seen_enemy` semantics as the chaser. |

The selector and `choose_action` are deliberately small right now: the
role API (function shape, sticky `role_target`, role letters in the
GUI) is preserved so a richer taxonomy of hunting / protection
strategies can be reintroduced without re-plumbing the per-step
pipeline.

---

## Visualisation

`python main.py --gui` opens a pygame window driven by
[`visualization.py`](visualization.py).

| Element | Meaning |
|---------|---------|
| Purple rounded rect | Predator body |
| Green rounded rect | Prey body |
| Single white letter inside a body | Current role (`C` chaser, `-` flee) |
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
| `--comms` | off | Enable speaker-centric, single-hop team communication |

Determinism: a given `(seed, run index, all other flags)` reproduces
exactly the same episode trace, because every stochastic choice flows
through `random.Random(cfg.seed)` constructed per run.

---

## Architecture map

| File | Responsibility |
|------|----------------|
| [`main.py`](main.py) | CLI parsing, builds `SimulationConfig`, dispatches to batch or GUI |
| [`simulation.py`](simulation.py) | `SimulationConfig`, `SimulationState`, the per-step pipeline, batch driver |
| [`environment.py`](environment.py) | Grid, `AgentBody`, agent placement, `legal_actions`, capture resolution |
| [`action_resolution.py`](action_resolution.py) | Same-step intention resolution with the same-team and cross-team movement rules |
| [`observation_definition.py`](observation_definition.py) | Per-agent observation dict |
| [`perception.py`](perception.py) | Pure perception helpers: `compute_active_enemies` (direct + shared fusion), `update_last_seen_enemy` memory refresh |
| [`decision_making.py`](decision_making.py) | Team role selector and per-agent `choose_action` (chase / flee) |
| [`agent.py`](agent.py) | `Agent` class: holds memory, role, role_target; `prepare_observation` perception fusion; `decide` glue |
| [`agent_utils.py`](agent_utils.py) | Constants: actions, teams, outcomes, role names + display letters |
| [`distances.py`](distances.py) | `chebyshev`, `manhattan` |
| [`reward_attribution.py`](reward_attribution.py) | Per-step reward shaping (each alive predator gets +1 on every step that a prey is captured) |
| [`visualization.py`](visualization.py) | Pygame renderer with the legend above |
