import argparse

import torch

from rl.checkpointing import checkpoint_use_search, load_checkpoint
from rl.eval_runner import evaluate_policy
from simulation import format_batch_summary
from rl.algo import IPPO, MAPPO


def build_arg_parser():
    '''Creates evaluation CLI input parser'''
    parser = argparse.ArgumentParser(description="Evaluate trained IPPO / MAPPO predators.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint/policy to evaluate")
    parser.add_argument("--predators", type=int, default=3, help="Number of predators per run")
    parser.add_argument("--prey", type=int, default=None, help="Number of prey per run. If omitted, evaluate 2, 3, and 4 prey.")
    parser.add_argument("--walls", type=int, default=2, dest="num_walls", help="Number of wall segments to generate and place on the map randomly.")
    parser.add_argument("--wall-size", type=int, default=2, help="Length, in cells, of wall segments")
    parser.add_argument("--runs", type=int, default=50, help="Number of RL runs to execute for each number of prey")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for the runs")
    parser.add_argument("--prey-defend", choices=["stun", "kill"], default=None, dest="prey_defend", help="Enable cooperative prey defense (stun or kill). Omit to disable.")

    return parser


def main():
    '''Main function to evaluate a trained policy'''
    # Read and parse the CLI arguments
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get the policy checkpoint
    policy, ppo_config, checkpoint = load_checkpoint(args.checkpoint, device)
    algo = checkpoint.get("algo", IPPO)
    # Set policy to evaluation mode
    policy.eval()
    use_search = checkpoint_use_search(checkpoint)

    # Define the number of prey to evaluate for
    prey_counts = [args.prey] if args.prey is not None else [2, 3, 4]
    # Evaluate the policy for each number of prey
    results, mean_win_rate = evaluate_policy(policy, device, seed=args.seed, num_runs=args.runs, num_predators=args.predators, prey_counts=prey_counts, walls=args.num_walls, wall_size=args.wall_size, prey_defend=args.prey_defend, algo=algo, use_search=use_search)

    print(
        f"algo={algo} use_search={use_search} mean_win_rate={mean_win_rate:.3f}",
        flush=True,
    )
    for num_prey in prey_counts:
        item = results[num_prey]
        print(format_batch_summary(item["summary"], item["config"]), flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
