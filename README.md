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
*active* predator and at least one alive prey marks every prey on
that cell as captured. "Active" here means `stun_remaining == 0` —
stunned predators (see "Prey-defend (cooperative knockout)" below)
cannot capture, so a prey co-located only with stunned predators
survives the step. The episode ends the moment one of three
conditions becomes true:

- All prey are dead → **predators win**.
- All predators are dead (only reachable with `--prey-defend kill`)
  → **prey win**.
- The timestep budget elapses with at least one prey alive
  → **prey win** (timeout); otherwise predators win.

The "all-prey-dead" check is evaluated first, so if the last prey
and the last predator both die on the same step the run is logged
as a predator win — the existing convention is preserved when the
new kill mode adds a second extinction path.

---

## Prey-defend (cooperative knockout)

Opt-in via `--prey-defend {stun,kill}`. Omit the flag for the
default behaviour, in which case the knockout system never fires
and the rest of the rules are unchanged.

When enabled, groups of Chebyshev-adjacent prey can defeat
predators that they collectively sandwich. The intent is to give
prey a cooperative tool that pays off the cohesion bias in
`_flee` / `_wander_with_cohesion`: pairs and packs become a real
defensive asset and not just less-scattered targets.

The two modes share **all** geometric and capacity rules; they
differ only in what happens to a defeated predator:

| Mode | Effect on defeated predator | Episode termination consequence |
|------|------------------------------|----------------------------------|
| `stun` | `stun_remaining` set to `stun_duration` (= 3 steps). Predator locked to STAY for the window; capture pass skips it. Comes back online after the timer reaches 0. | None directly — predators recover. |
| `kill` | `alive` set to `False`. Predator is removed from the run permanently; resolver and capture pass automatically skip it (they already iterate over `alive_bodies` only). | When all predators are dead the episode ends with a prey win on the spot. |

### Shared mechanic

1. **Groups.** Alive prey are partitioned by connected components
   on the prey-prey Chebyshev-1 adjacency graph. So a chain
   A–B–C (each pair Cheb-1 of the next) is one group of size 3
   even when A and C are Cheb-2 apart.
2. **Defeatable predator.** A predator P is defeatable by a group
   G iff P is alive, is not already stunned
   (`stun_remaining == 0`; relevant only in `stun` mode), has not
   been defeated by an earlier group this same step, and is
   Chebyshev-1 of at least 2 distinct members of G. This is the
   natural generalisation of the n = 2 "Cheb-1 of both prey" rule.
3. **Cap.** A group of size n can defeat at most `n - 1`
   predators per step. When there are more candidates than the
   cap, the lowest predator agent id wins (fully deterministic —
   no RNG flows through the knockout pass at all).
4. **Forced STAY.** For every newly defeated predator, the prey
   from the group that are Cheb-1 of that predator (the
   "sandwichers") have their pending action overridden to STAY
   for this step. Other group members can move normally — the
   freeze is the price paid by the prey actually doing the
   sandwiching, not by the whole pack.
5. **No post-stun immunity** (`stun` mode). As soon as
   `stun_remaining` reaches 0 the predator is a fresh candidate
   for the next group's knockout, if the geometry still holds.

### Pipeline ordering

`SimulationState.step_once` runs the knockout system between agent
decisions and action resolution:

1. Phase 1–3: observations → comms → role assignment → `decide`,
   exactly as without `--prey-defend`. Stunned predators still
   call `decide` so their `last_seen_enemy` and perception
   bookkeeping stay consistent across the stun window; only the
   *action* is overridden.
2. **Phase 4a** (only when `prey_defend == "stun"`): every predator
   with `stun_remaining > 0` has its intention overridden to STAY.
3. **Phase 4b** (only when `prey_defend` is set):
   `SimulationState._resolve_knockouts(intentions, mode)` builds
   groups, picks defeatable predators per the rules above, applies
   the mode-specific effect (`stun_remaining = stun_duration` or
   `alive = False`), and overrides sandwicher prey to STAY.
4. Resolver and capture pass run as usual; the capture pass reads
   `stun_remaining` and `alive` directly (no extra plumbing
   needed — `apply_captures` already iterates `alive_bodies`).
5. **Phase 5** (only when `prey_defend == "stun"`): decrement
   every positive `stun_remaining` by 1.
6. **Termination checks.** Prey-extinction → predators win;
   else predator-extinction (only reachable in `kill` mode)
   → prey win; else timeout.

This ordering means defeat is *prophylactic* — a predator that
was going to capture this very step gets its action overwritten
(stun) or its body removed (kill) before the resolver runs, so
the prey survives.

### Design notes

- Prey observations do not change. Prey still see stunned
  predators as enemies; their flee logic still treats them as
  threats. This is intentional: a stunned predator's stun window
  is finite, so prey keeping distance is the right move. It also
  means we don't need to plumb a new observation field, and
  `Perception` is unchanged. In `kill` mode dead predators
  naturally drop out of `visible_enemies` (the observation loop
  already filters on `alive`).
- The mechanic interacts predictably with multi-predator games:
  an unstunned/uninvolved predator can still capture a
  sandwicher prey on the same step that prey is forced to STAY.
  The cap rule (n - 1, not n) bakes this trade-off in by design,
  and applies symmetrically in both modes.
- Reward attribution is unchanged. Defeated predators do not
  generate any new reward signal — `attribute_rewards` still
  only tracks per-step predator captures.
- Determinism: the knockout pass is RNG-free in both modes.
  Groups are processed in ascending min-id order; within a group,
  predator candidates are sorted by id. Replays with identical
  seeds + flags produce identical traces.

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
6. **Knockout system** (optional, opt-in via
   `--prey-defend {stun,kill}`). Sits between decisions and the
   resolver. Two phases:
   - 4a (`stun` mode only): predators with `stun_remaining > 0`
     from a previous step have their action overridden to STAY.
   - 4b: `_resolve_knockouts` runs the cooperative-knockout pass,
     defeating up to `n - 1` predators per prey group and forcing
     the sandwicher prey to STAY. The defeat effect is mode-
     specific: `stun` writes `stun_remaining`, `kill` flips
     `alive` to False. See "Prey-defend (cooperative knockout)"
     above for the full rules.
7. **Resolution, capture, and stun decrement.** Action resolver
   runs over the (possibly overridden) intentions; the capture
   pass skips stunned and dead predators (both branches handled
   by `apply_captures` reading `stun_remaining` and `alive`);
   finally positive `stun_remaining` values are decremented by 1
   (`stun` mode only). The post-step termination check then adds
   a "no predators left → prey win" path on top of the existing
   "no prey left → predators win" and "timeout" rules.

This ordering makes the contract simple: the selector and the
agent both consume the same `active_enemies` set, so they cannot
disagree about who the threat is or where it is. The knockout
system is purely additive — when `--prey-defend` is omitted, step
6 and the stun-aware parts of step 7 collapse to no-ops and the
rest of the pipeline is unchanged.

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
| `ROLE_FLEE` | prey | Head away from the primary threat (the Manhattan-closest tracked predator) while avoiding cardinal moves that close distance to any other tracked predator. Same `last_seen_enemy` semantics as the chaser. See "Prey flee scoring" below for the exact selection rule. |

The selector and `choose_action` are deliberately small right now: the
role API (function shape, sticky `role_target`, role letters in the
GUI) is preserved so a richer taxonomy of hunting / protection
strategies can be reintroduced without re-plumbing the per-step
pipeline.

### Prey flee scoring

`DecisionMaking._flee` works in two stages: building the candidate
set, then scoring it. Both stages are cohesion-aware: prey prefer to
stay Chebyshev-1 of a visible teammate, which is the geometry that
enables the cooperative-knockout mechanic (planned) and, even
without it, reduces stragglers.

**Stage 1 — candidate set.** Cardinal moves are checked against every
*secondary* threat (every active predator other than the primary):
a cardinal is "safe" iff it does not strictly close distance to any
secondary predator. STAY is split out of this check from the start,
because STAY has a zero delta and so trivially "doesn't close on
anyone" — treating it as just another safe action used to let prey
freeze whenever every real cardinal looked unsafe. Cardinals are
then pruned by one guard:

1. **Suicide guard.** Drop any cardinal whose target cell is
   currently occupied by an active enemy (a predator the prey sees
   or has been told about). `legal_actions` only checks bounds and
   walls, so without this guard a prey adjacent to a predator can
   have the step-into-capture cardinal end up in `safe_cardinals`
   (it moves further from every *other* predator) and the scorer
   would happily pick it over STAY on the Manhattan tiebreak.

Cardinals that land on a visible teammate's *current* cell
("stack" moves) are deliberately **not** pruned. Action resolution
is simultaneous, so when the teammate is itself moving away this
step the step-1 cell is free by the time the resolver lays down
final positions; the prey just cascade-follows into the vacated
spot. Earlier versions of `_flee` did prune stack moves up front,
but that throws away the cascade-follow path in exactly the
situations where it's most valuable — cornered prey whose only
"safe" cardinal points through a fleeing ally. The cost of leaving
stack moves in is one wasted step when the ally actually decides
to STAY (the resolver then blocks the stack and the prey is forced
to STAY too), which is strictly cheaper than refusing the cascade
and walking into a predator. See "Stage 2 — scoring" below for
how the cohesion term is shaped so the scorer doesn't pick stack
moves *for their own sake*.

The candidate-set rules are then:

- If at least one cardinal survives the suicide guard and is also
  safe, the candidate set is `safe_cardinals + [STAY]`. STAY is
  included because in narrow situations (e.g. corner prey with a
  single diagonal predator at Manhattan 2) it is genuinely the
  highest-distance option.
- Otherwise, if any cardinals survive the suicide guard, the
  candidate set is those cardinals **without STAY**. This is the
  rule that prevents the freeze cascade: rather than letting the
  primary walk in for free, the prey accepts closing on a
  secondary threat and tries to outrun the primary instead.
- Otherwise (every cardinal is a wall or a suicide) the candidate
  set is `[STAY]`. We don't fall back to the unpruned `legal`
  here — we just decided every cardinal in it was bad, so the
  scorer would only pick the least-bad of a bad bunch.

**Stage 2 — scoring.** Candidates are scored lexicographically on
`(-manhattan_to_primary, |d_a - 1|, jitter)`:

- `-manhattan_to_primary` — same primary-threat term as before:
  maximise current Manhattan distance to the closest tracked
  predator.
- `|d_a - 1|` — cohesion term. `d_a` is the minimum Chebyshev
  distance from the candidate cell to any visible teammate. The
  absolute-value shape makes `d_a = 1` the unique optimum
  (score 0); `d_a = 0` (a stack move) and `d_a = 2` both score 1
  and tie. This is the key half of the "no ally-stack guard"
  trade-off: stack moves are no longer *uniquely* rewarded by the
  cohesion term, so the only way a stack move gets picked is if
  the primary-distance key strictly favoured it on its own merits
  (e.g. it's the only safe cardinal). With no visible ally the
  term is pinned to 0, so the cohesion rule collapses to the
  previous primary-only scoring for solo prey.
- `jitter` — per-agent RNG tiebreak.

No look-ahead and no map inspection — the prey acts on what it can
observe right now (primary-threat position, visible teammates). The
suicide guard plus the primary-only tactical term are what break
the common multi-predator freeze cascade where one prey gets
pinned by a `safe_actions = {STAY}` situation and its teammates
pile up behind it; the cohesion term is what pulls scattered prey
back into pairs once the immediate threat is handled; and the
removal of the old ally-stack guard is what lets two prey
cascade-flee through each other when geometry forces it.

**Wander mode.** When the prey has no `active_enemies` and no
`last_seen_enemy` (and likewise after stale memory at ego is
cleared), it doesn't pick uniformly at random over `legal_actions`
any more. Instead it drops cardinals that would land on a visible
teammate (the wander path keeps an ally-stack guard locally —
there is no primary-distance imperative in wander, so the
cascade-follow trade-off doesn't apply) and picks the cardinal (or
STAY) that minimises `|d_a - 1|` to the nearest visible teammate,
RNG breaking ties. This way prey preemptively form pairs *before*
a predator shows up, which is what makes the cooperative geometry
reachable in practice rather than only recoverable mid-flee. With
no visible ally, wander remains a pure uniform random pick —
identical to the prior behaviour.

---

## Visualisation

`python main.py --gui` opens a pygame window driven by
[`visualization.py`](visualization.py).

| Element | Meaning |
|---------|---------|
| Purple rounded rect | Predator body |
| Dimmed (~50%) purple rounded rect | Stunned predator (only seen with `--prey-defend stun`; cannot move or capture for `stun_duration` steps) |
| (no rendering) | Killed predator (only with `--prey-defend kill`); the body is dropped from rendering the same way captured prey are |
| Green rounded rect | Prey body |
| Single white letter inside a body | Current role (`C` chaser, `F` flee) — also dimmed when the body is stunned |
| Dark grey filled cell | Wall |

Controls: `space` / `right` step once, `a` toggles auto-run, `r` resets
the current run with its seed, `n` advances to the next seeded run when
the current one is over, `esc` quits.

The GUI mirrors the headless logging: every time a run finishes it
prints a one-line summary (`run=i/N  seed=S  outcome=...  steps=K`)
and on quit it prints the same final markdown-style table as the
batch runner. The table is a single data row with seven columns —
`Number of Predators`, `Number of Prey`, `Predator Wins`,
`Prey Wins`, `Mean Run Timesteps` (pooled across all runs),
`Mean Run Timesteps in Predator-won` (mean computed only over
episodes that ended in a predator win), and `Mean Run Timesteps
in Prey-won` (same, for prey wins, including kill-mode
eliminations). The shape is designed for the writeup workflow: run
many configurations, keep the header from the first command's
output, and append only the data row from each subsequent
command's output to build a master comparison table.
Cells whose underlying count is 0 are left empty so the rendered
table matches the spreadsheet style of the source document.
Closing the window before every run has finished prints an
explicit `simulation aborted (X/N runs completed)` line before
the partial summary, so the printed totals never silently
misrepresent what actually ran.

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
| `--prey-defend {stun,kill}` | off | Enable the cooperative-knockout mechanic. Groups of Cheb-1 prey defeat up to `n-1` sandwiched predators per step (sandwicher prey are forced to STAY that step). `stun` freezes the predator for 3 steps; `kill` removes it from the run and ends the episode early once every predator is gone. See "Prey-defend (cooperative knockout)" above. |

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
