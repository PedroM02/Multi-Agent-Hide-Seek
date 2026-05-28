import argparse

from simulation import SimulationConfig, format_batch_summary, run_batch


def _load_rl_policy(checkpoint_path):
    import torch

    from rl.algo import IPPO
    from rl.checkpointing import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, _ppo_cfg, payload = load_checkpoint(checkpoint_path, device)
    policy.eval()
    return policy, device, payload.get("algo", IPPO), payload


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
        choices=["random", "chase", "pack", "roles", "optimal", "rl"],
        default="chase",
        help=(
            "Predator decision mode. random: Level 1. chase: Level 2 "
            "(add --comms for Level 3). pack: Level 4 pack "
            "hunting with shared prey focus (requires --comms predators "
            "or both). roles: Level 5 team roles "
            "with chaser/flanker coordination.  optimal: Level 6 clairvoyant BFS with shared "
            "pack focus."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a trained IPPO checkpoint (required for --mode rl).",
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
        "--searcher",
        action="store_true",
        help=(
            "Enable ROLE_SEARCHER in `--mode roles`. Predators stay "
            "searchers until prey is directly seen or reported via comms, "
            "using a persisted heading (repel visible allies when (re)setting, "
            "then keep going). Re-heads when a new ally enters vision. "
            "Requires `--mode roles`."
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
    config.roles_searcher = args.searcher

    if config.mode == "pack" and config.comms not in ("predators", "both"):
        parser.error("--mode pack requires --comms predators or --comms both")

    if config.roles_searcher and config.mode != "roles":
        parser.error("--searcher requires --mode roles")

    rl_policy = None
    rl_device = None
    rl_algo = None
    rl_use_search = False
    if config.mode == "rl":
        if args.checkpoint is None:
            parser.error("--mode rl requires --checkpoint")
        rl_policy, rl_device, rl_algo, rl_payload = _load_rl_policy(args.checkpoint)
        from rl.checkpointing import checkpoint_use_search

        rl_use_search = checkpoint_use_search(rl_payload)
        if config.comms is None:
            config.comms = "both"

    if args.gui:
        from visualization import run_visualization

        run_visualization(
            config,
            args.runs,
            rl_policy=rl_policy,
            rl_device=rl_device,
            rl_algo=rl_algo,
            rl_use_search=rl_use_search,
        )
        return

    if config.mode == "rl":
        from rl.eval_runner import run_rl_batch

        summary = run_rl_batch(
            config,
            args.runs,
            rl_policy,
            rl_device,
            deterministic=True,
            algo=rl_algo,
            use_search=rl_use_search,
        )
        print(format_batch_summary(summary, config))
        return

    summary = run_batch(config, args.runs)
    print(format_batch_summary(summary, config))


if __name__ == "__main__":
    main()
