# PredPrey Sim

This project consistes of a grid-world predator–prey multi-agent simulation. Below, you can find how to setup and run the project, including how to replicate our results.

This project was built by:
- João Marques 116509
- Pedro Machado 78657
- Pedro Morais 115744

---

## Requirements

- Python 3.10+
- Dependencies: `pygame`, `torch`, and `numpy`

---

## Setup

To setup the project and install dependencies, at the project root run the commands below in PowerShell.
These will create a virtual environment named "venv" and activate it, and install dependencies into it.


```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Verify the installation with the commands below.

```bash
python main.py --help
python -m rl.train --help
python -m rl.evaluate --help
```

---

## Running the project

The project contains three main points of use and interaction. These are exposed via Command Line Interface (CLI) at the following commands:


- `python main.py` — runs the simulation in several modes.
- `python -m rl.train` — trains agents with a learned policy.
- `python -m rl.evaluate` — evaluates the trained agents' performance in the simulation (using a learned policy).


The CLI flags and parameters for each command are detailed below:

### `python main.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--gui` | off | Open GUI window for visual interface. |
| `--width` | `10` | Playable map cells width. |
| `--height` | `8` | Playable map cells height. |
| `--timestep` / `--timesteps` | `200` | Maximum timesteps per run. Episodes end earlier if all prey are caught. |
| `--runs` | `1` | Number of full runs to execute (each run uses `--timesteps` as the maximum). |
| `--vision-predator` | `2` | Chebyshev vision radius for predators. |
| `--vision-prey` | `2` | Chebyshev vision radius for prey. |
| `--predators` | `1` | Number of predators per run. |
| `--prey` | `1` | Number of prey per run. |
| `--seed` | `0` | Base seed for the simulation. |
| `--walls` | `2` | Number of wall segments to generate randomly. |
| `--wall-size` | `2` | Length (in cells) of each randomly generated wall segment. |
| `--mode` | `chase` | Predator decision mode: `random`, `chase`, `roles`, `rl`, `optimal`. |
| `--checkpoint` | none | Path to a trained RL policy checkpoint (required for `--mode rl`). |
| `--comms` | none | Enable comms for `prey`, `predators`, or `both`. Omit for no comms. |
| `--searcher` | off | In `--mode roles`, use `ROLE_SEARCHER` until prey is seen or reported via comms. |
| `--prey-defend` | none | Cooperative prey defense: `stun` or `kill`. Omit to disable. |

### `python -m rl.train`

| Flag | Default | Description |
|------|---------|-------------|
| `--algo` | `ippo` | Training algorithm: `ippo` (decentralized critic) or `mappo` (centralized critic). |
| `--updates` | `1000` | Number of training updates to perform. |
| `--predators` | `3` | Number of predators per run. |
| `--num-walls` | `2` | Number of wall segments placed on the map randomly. |
| `--wall-size` | `2` | Length of wall segments (in cells). |
| `--rollout-steps` | `4096` | Number of transitions to collect per policy update. |
| `--lr` | `3e-4` | Learning rate for the policy optimizer. |
| `--entropy-coef` | `0.02` | Entropy bonus weight to encourage exploration. |
| `--entropy-floor` | `0.4` | Penalize policy entropy below this value. Set `0` to disable. |
| `--entropy-floor-coef` | `0.05` | Strength of the entropy-floor penalty. |
| `--seed` | `0` | Random seed for training rollouts. |
| `--checkpoint-dir` | `checkpoints/{algo}` | Directory to save policy checkpoints (set automatically if omitted). |
| `--checkpoint` | none | Path to a checkpoint to resume training. |
| `--eval-every` | `25` | Evaluate policy every N updates. |
| `--eval-runs` | `20` | Number of RL runs per prey count during in-training evaluation. |
| `--save-every` | `100` | Save policy checkpoint every N updates. |
| `--curriculum` | off | Prey=2 for updates 0–199, then `{2,3}`, then `{2,3,4}`. |
| `--prey-defend` | none | Prey defense: `stun` or `kill`. Omit to disable. |
| `--search` | off | Enable predator search heuristic when prey are unknown (not learned). |

### `python -m rl.evaluate`

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *(required)* | Path to the checkpoint/policy to evaluate. |
| `--predators` | `3` | Number of predators per run. |
| `--prey` | `2`, `3`, `4` | Number of prey per run. If omitted, evaluate all three counts. |
| `--walls` | `2` | Number of wall segments placed on the map randomly. |
| `--wall-size` | `2` | Length of wall segments (in cells). |
| `--runs` | `50` | Number of RL runs for each prey count. |
| `--seed` | `0` | Base seed for the runs. |
| `--prey-defend` | none | Cooperative prey defense: `stun` or `kill`. Omit to disable. |



---

## Replicating results

To replicate results, run the commands below, changing `--prey` from 2 up to 4. Add `--prey-defend stun` to each command to enable stunning and assess those results, too. Use `--gui` to visualize the runs.

RL runs need a trained checkpoint (`--checkpoint`). Using ours, according to the paths in the commands below, will yield identical results. Additionally, checkpoints trained using search will automatically trigger search with no CLI flag.

**Random**

```bash
python main.py --mode random --comms prey --predators 3 --prey 2 --runs 100
```

**Chase**

```bash
python main.py --mode chase --comms prey --predators 3 --prey 2 --runs 100
```

**Chase with comms**

```bash
python main.py --mode chase --comms both --predators 3 --prey 2 --runs 100
```

**Roles**

```bash
python main.py --mode roles --comms both --predators 3 --prey 2 --runs 100
```

**Roles with search**

```bash
python main.py --mode roles --comms both --searcher --predators 3 --prey 2 --runs 100
```

**IPPO**

```bash
python main.py --mode rl --checkpoint rl/checkpoints/ippo/best_eval.pt --predators 3 --prey 2 --runs 100
```

**MAPPO**

```bash
python main.py --mode rl --checkpoint rl/checkpoints/mappo/best_eval.pt --predators 3 --prey 2 --runs 100
```

**IPPO with search**

```bash
python main.py --mode rl --checkpoint rl/checkpoints/ippo_search/best_eval.pt --predators 3 --prey 2 --runs 100
```

**MAPPO with search**

```bash
python main.py --mode rl --checkpoint rl/checkpoints/mappo_search/best_eval.pt --predators 3 --prey 2 --runs 100
```

**Optimal**

```bash
python main.py --mode optimal --predators 3 --prey 2 --runs 100
```



---

## RL Training

Training agents with a new learned policy is possible using the commands below. The available algorithms are IPPO and MAPPO.
Training can be resumed if a `--checkpoint` is provided and `--algo` matches the algorithm with which the policy was initially trained. Resuming training will continue from the update training was stopped up until `--updates`.
Curriculum learning is possible using `--curriculum`, where training starts with uniformly sampling with 2 prey, moving to 2 and 3 prey, and finally 2, 3, or 4 prey.
Including `--search` will switch policy behavior to non-learned search behavior when prey are not known

**Example with IPPO**

```bash
python -m rl.train --algo ippo --updates 1000 --predators 3 --seed 0 --checkpoint-dir rl/checkpoints/ippo
```

**Resume training with IPPO**

```bash
python -m rl.train --algo ippo --checkpoint rl/checkpoints/ippo/latest.pt --updates 1000 --predators 3 --seed 0 --checkpoint-dir rl/checkpoints/ippo
```

**Training with IPPO and curriculum**

```bash
python -m rl.train --algo ippo --curriculum --updates 1000 --predators 3 --seed 0 --checkpoint-dir rl/checkpoints/ippo_curriculum
```




---

## Evaluating RL policies

Evaluating the performance of agents which sample actions from a learned policy is also possible. A `--checkpoint` containing a policy must be provided. The simulation will run with agents following the policy (and search behavior if it was enabled during training), outputting evaluation metrics such as the number of predator wins, mean run timesteps, and mean run timesteps for runs won by predators. If `--prey` is ommitted, the evaluation will run for each number of prey.

```bash
python -m rl.evaluate --checkpoint rl/checkpoints/ippo/best_eval.pt --predators 3 --runs 100
```

