import argparse

from simulation import SimulationConfig, format_batch_summary, run_batch


def load_rl_policy(checkpoint_path):
    '''Loads a trained RL policy from a checkpoint file'''
    import torch

    from rl.algo import IPPO
    from rl.checkpointing import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, ppo_config, payload = load_checkpoint(checkpoint_path, device)
    policy.eval()
    return policy, device, payload.get("algo", IPPO), payload


def main():

    # Define argument parser and arguments for CLI inputs
    parser = argparse.ArgumentParser(description="Predator–prey grid MAS. Default: batch runs without GUI; use --gui for visual interface.",)
    parser.add_argument("--gui", action="store_true", help="Open GUI window for visual interface.")
    parser.add_argument("--width", type=int, default=10, help="Playable map cells width.")
    parser.add_argument("--height", type=int, default=8, help="Playable map cells height.")
    parser.add_argument("--timestep", "--timesteps", type=int, default=200, dest="timesteps", help="Maximum timesteps per run. Episodes end earlier if all prey are caught.",)
    parser.add_argument("--runs", type=int, default=1, help="Number of full runs to execute (each run uses --timesteps as the maximum number of timesteps).",)
    parser.add_argument("--vision-predator", type=int, default=2, help="Chebyshev vision radius for predators.")
    parser.add_argument("--vision-prey", type=int, default=2, help="Chebyshev vision radius for prey.")
    parser.add_argument("--predators", type=int, default=1, help="Number of predators per run.")
    parser.add_argument("--prey", type=int, default=1, help="Number of prey per run.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for the simulation.")
    parser.add_argument("--walls", type=int, default=2, dest="num_walls", help="Number of wall segments to generate randomly.",)
    parser.add_argument("--wall-size", type=int, default=2, dest="wall_size", help="Length (in cells) of each randomly generated wall segment.",)
    parser.add_argument("--mode", choices=["random", "chase", "roles", "rl", "optimal"], default="chase", 
                                  help=("Predator decision modes. Random: random action selection. Chase: greedy distance-minimizing pursuit. Roles: team roles with chaser/flanker coordination. RL: reinforcement learning policy. Optimal: full map oracle information with shared prey focus."))
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a trained RL policy checkpoint (required and only used for --mode rl).")
    parser.add_argument("--comms", choices=["prey", "predators", "both"], default=None, 
                                   help=("Enable intra-team communication for the given team(s). Each enabled agent broadcasts directly-visible enemies to teammates inside its vision radius. Omit the flag for no comms."))
    parser.add_argument("--searcher", action="store_true", help=("Enable ROLE_SEARCHER in roles mode. Predators stay searchers until prey is directly seen or reported via comms. Requires roles mode."))
    parser.add_argument("--prey-defend", choices=["stun", "kill"], default=None, dest="prey_defend",
                                         help=("Enable cooperative-knockout mechanic. Groups of adjacent prey attack up to nr_prey-1 adjacent predators. Stun freezes the predator for 3 timesteps while Kill removes it from the run entirely. Prey are forced to STAY that step. Omit the flag to disable the mechanic."))
    args = parser.parse_args()


    # Initialize simulation configuration with CLI inputs
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
    config.roles_searcher = args.searcher
    config.prey_defend = args.prey_defend
 

    # Check if the "roles" mode is enabled to have a "searcher" role
    if config.roles_searcher and config.mode != "roles":
        parser.error("--searcher requires --mode roles")
    
    # Configuration for RL mode
    rl_policy = None
    rl_device = None
    rl_algo = None
    rl_use_search = False
    # Ensure policy path is provided and loaded
    if config.mode == "rl":
        if args.checkpoint is None:
            parser.error("--mode rl requires --checkpoint")
        rl_policy, rl_device, rl_algo, rl_payload = load_rl_policy(args.checkpoint)
        from rl.checkpointing import checkpoint_use_search
        # Use RL with search if policy was trained with search
        rl_use_search = checkpoint_use_search(rl_payload)
        # Default communications to both teams for comms if not specified
        if config.comms is None:
            config.comms = "both"

    # Run GUI if enabled
    if args.gui:
        from visualization import run_visualization

        run_visualization(config, args.runs, rl_policy=rl_policy, rl_device=rl_device, rl_algo=rl_algo, rl_use_search=rl_use_search)
        return

    # Run RL batch if in RL mode, otherwise run normal batch simulation
    if config.mode == "rl":
        from rl.eval_runner import run_rl_batch

        summary = run_rl_batch(config, args.runs, rl_policy, rl_device, deterministic=True, algo=rl_algo, use_search=rl_use_search)
        print(format_batch_summary(summary, config))
        return

    summary = run_batch(config, args.runs)
    print(format_batch_summary(summary, config))


if __name__ == "__main__":
    main()
