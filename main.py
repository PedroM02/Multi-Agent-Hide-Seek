from __future__ import annotations
from simulation import BatchSummary, SimulationConfig, run_batch

import argparse


def _parse_walls(raw: str | None) -> list[tuple[int, int]] | None:
    if not raw:
        return None
    out: list[tuple[int, int]] = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        xs, ys = part.split(",")
        out.append((int(xs.strip()), int(ys.strip())))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Predator–prey grid MAS. Default: batch runs without GUI; use --gui for pygame.",)
    parser.add_argument("--gui", action="store_true", help="Open pygame window (otherwise batch text output).")
    parser.add_argument("--width", type=int, default=10, help="Playable cells wide (no separate border layer).")
    parser.add_argument("--height", type=int, default=8, help="Playable cells tall.")
    parser.add_argument("--timestep", "--timesteps", type=int, default=200, dest="timesteps", help="Maximum timesteps per run (one episode ends earlier if all prey are caught).",)
    parser.add_argument("--runs", type=int, default=1, help="How many full runs to execute (each run uses --timesteps as the step cap).",)
    parser.add_argument("--vision-predator", type=int, default=2, help="Chebyshev vision radius.")
    parser.add_argument("--vision-prey", type=int, default=2, help="Chebyshev prey vision radius.")
    parser.add_argument("--predators", type=int, default=1)
    parser.add_argument("--prey", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--walls", type=int, default=2, dest="num_walls", help="Number of wall segments to generate randomly.",)
    parser.add_argument("--wall-size", type=int, default=2, dest="wall_size", help="Length (in cells) of each randomly generated wall segment.",)
    parser.add_argument("--obstacles", type=int, default=0, help="Number of pickable obstacle items (0 = none; default). Set > 0 to enable.",)
    parser.add_argument("--comms", action="store_true", help="Enable team communications (opt-in). Each agent broadcasts directly-visible enemies to teammates inside its vision radius; receivers fall back from direct sight to teammate reports to memory.",)
    parser.add_argument("--lock-mode", choices=["symmetric", "owner-passable"], default="symmetric", dest="lock_mode", help="Obstacle locking semantics. 'symmetric' (default): dropped obstacles block both teams; only the owning team can re-pickup. 'owner-passable': the owning team can walk through its own locked obstacles while the enemy team is still blocked.",)
    args = parser.parse_args()

    config = SimulationConfig()
    config.width = args.width
    config.height = args.height
    config.timesteps = args.timesteps
    config.vision_radius_predator = args.vision_predator
    config.vision_radius_prey = args.vision_prey
    config.num_predators = args.predators
    config.num_prey = args.prey
    config.seed = args.seed
    config.num_walls = args.num_walls
    config.wall_size = args.wall_size
    config.num_obstacles = args.obstacles
    config.enable_comms = args.comms
    config.lock_mode = args.lock_mode

    if args.gui:
        from visualization import run_visualization

        run_visualization(config, args.runs)
        return

    summary: BatchSummary = run_batch(config, args.runs)
    mean_steps = summary.total_steps / max(1, summary.runs)
    print(f"runs={summary.runs}  predator_wins={summary.predator_wins}  prey_timeout_wins={summary.prey_timeout_wins}")
    print(f"mean_steps={mean_steps:.2f}")


if __name__ == "__main__":
    main()