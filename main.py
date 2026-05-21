from simulation import BatchSummary, SimulationConfig, format_batch_summary, run_batch

import argparse


def _parse_walls(raw):
    if not raw:
        return None
    out = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        xs, ys = part.split(",")
        out.append((int(xs.strip()), int(ys.strip())))
    return out


def main():
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
    parser.add_argument(
        "--mode",
        choices=["random", "chase", "optimal"],
        default="chase",
        help=(
            "Predator decision mode. random: Level 1. chase: Level 2 "
            "(add --comms for Level 3). optimal: Level 6 clairvoyant BFS "
            "with shared pack focus. Level 4 will add --mode roles."
        ),
    )
    parser.add_argument(
        "--comms",
        choices=["prey", "predators", "both"],
        default=None,
        metavar="TEAM",
        help=(
            "Enable intra-team communication for the given team(s). "
            "Each enabled agent broadcasts directly-visible enemies to "
            "teammates inside its vision radius; receivers fall back from "
            "direct sight to teammate reports to memory. Values: prey, "
            "predators, both. Omit the flag for no comms."
        ),
    )
    parser.add_argument(
        "--prey-defend",
        choices=["stun", "kill"],
        default=None,
        dest="prey_defend",
        help=(
            "Enable the cooperative-knockout mechanic (opt-in). Groups of "
            "Chebyshev-adjacent prey defeat predators that are Cheb-1 of "
            ">=2 group members: up to n-1 predators per group of size n. "
            "Mode `stun` freezes the predator for 3 timesteps; mode "
            "`kill` removes it from the run entirely. The prey doing the "
            "sandwiching are forced to STAY that step. Omit the flag to "
            "disable the mechanic."
        ),
    )
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
    config.mode = args.mode
    config.comms = args.comms
    config.prey_defend = args.prey_defend

    if args.gui:
        from visualization import run_visualization

        run_visualization(config, args.runs)
        return

    summary = run_batch(config, args.runs)
    print(format_batch_summary(summary, config))


if __name__ == "__main__":
    main()
