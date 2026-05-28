"""Evaluate a trained shared-policy IPPO checkpoint."""

import argparse

import torch

from rl.checkpointing import checkpoint_use_search, load_checkpoint
from rl.eval_runner import evaluate_policy
from simulation import format_batch_summary
from rl.algo import IPPO, MAPPO


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate trained IPPO / MAPPO predators.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--predators", type=int, default=3)
    parser.add_argument("--prey", type=int, default=None, help="If omitted, evaluate 2, 3, and 4 prey.")
    parser.add_argument("--walls", type=int, default=2, dest="num_walls")
    parser.add_argument("--wall-size", type=int, default=2)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prey-defend",
        choices=["stun", "kill"],
        default=None,
        dest="prey_defend",
        help="Enable cooperative prey knockout (stun or kill). Omit to disable.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, _ppo_cfg, payload = load_checkpoint(args.checkpoint, device)
    algo = payload.get("algo", IPPO)
    policy.eval()
    use_search = checkpoint_use_search(payload)

    prey_counts = [args.prey] if args.prey is not None else [2, 3, 4]
    results, mean_win_rate = evaluate_policy(
        policy,
        device,
        seed=args.seed,
        num_runs=args.runs,
        num_predators=args.predators,
        prey_counts=prey_counts,
        walls=args.num_walls,
        wall_size=args.wall_size,
        prey_defend=args.prey_defend,
        algo=algo,
        use_search=use_search,
    )

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
