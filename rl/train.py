"""Shared-policy IPPO training loop."""

import argparse
import csv
import random
from pathlib import Path

import torch

import agent_utils as au
from reward_attribution import predator_team_shaped_reward
from rl.checkpointing import load_checkpoint, save_checkpoint
from rl.eval_runner import curriculum_phase_for_update, evaluate_policy, sample_num_prey
from rl.inference import collect_predator_transitions, make_rl_config
from rl.policy import ActorCritic
from rl.ppo import PPOConfig, RolloutBuffer, ppo_update
from simulation import SimulationState


def collect_rollout(training_rng, policy, device, ppo_cfg, base_config, update, curriculum):
    buffer = RolloutBuffer()
    episode_rewards = []
    episode_lengths = []

    while len(buffer) < ppo_cfg.rollout_steps:
        phase = curriculum_phase_for_update(update, curriculum)
        num_prey = sample_num_prey(training_rng, phase)
        episode_seed = training_rng.randint(0, 2**31 - 1)
        config = make_rl_config(
            width=base_config.width,
            height=base_config.height,
            timesteps=base_config.timesteps,
            vision_radius_predator=base_config.vision_radius_predator,
            vision_radius_prey=base_config.vision_radius_prey,
            num_predators=base_config.num_predators,
            num_prey=num_prey,
            num_walls=base_config.num_walls,
            wall_size=base_config.wall_size,
            prey_defend=base_config.prey_defend,
            seed=episode_seed,
        )
        sim = SimulationState(config, random.Random(episode_seed))
        episode_reward = 0.0
        episode_steps = 0
        visited_cells = {
            (
                sim.env.agent_bodies[agent_id].x,
                sim.env.agent_bodies[agent_id].y,
            )
            for agent_id in sim.predator_agent_ids()
        }

        while sim.outcome == au.OUTCOME_ONGOING and len(buffer) < ppo_cfg.rollout_steps:
            raw_obs = sim.build_step_observations()
            visited_before = set(visited_cells)
            predator_actions, transitions = collect_predator_transitions(
                sim, policy, device, raw_obs, deterministic=False,
            )
            continuing = sim.step_once(
                predator_actions=predator_actions,
                raw_obs=raw_obs,
            )
            for agent_id in sim.predator_agent_ids():
                body = sim.env.agent_bodies[agent_id]
                visited_cells.add((body.x, body.y))
            reward = predator_team_shaped_reward(
                sim,
                raw_obs,
                sim.last_captured,
                visited_before,
            )
            done = not continuing
            episode_reward += reward * len(transitions)
            episode_steps += 1

            for transition in transitions:
                buffer.add(
                    transition["obs"],
                    transition["mask"],
                    transition["action"],
                    transition["log_prob"],
                    transition["value"],
                    reward,
                    done,
                )

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)

    return buffer, episode_rewards, episode_lengths


def append_csv_row(path, fieldnames, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def apply_exploration_config(ppo_cfg, args):
    ppo_cfg.entropy_coef = args.entropy_coef
    ppo_cfg.entropy_floor = args.entropy_floor if args.entropy_floor > 0 else None
    ppo_cfg.entropy_floor_coef = args.entropy_floor_coef


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo_cfg = PPOConfig(
        rollout_steps=args.rollout_steps,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        entropy_floor=args.entropy_floor if args.entropy_floor > 0 else None,
        entropy_floor_coef=args.entropy_floor_coef,
    )
    base_config = make_rl_config(
        num_predators=args.predators,
        num_walls=args.num_walls,
        wall_size=args.wall_size,
        prey_defend=args.prey_defend,
    )

    policy = ActorCritic().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo_cfg.lr)
    start_update = 0
    best_eval = -1.0

    checkpoint_dir = Path(args.checkpoint_dir)
    if args.checkpoint:
        policy, loaded_cfg, payload = load_checkpoint(
            args.checkpoint, device, policy=policy, optimizer=optimizer,
        )
        ppo_cfg = loaded_cfg
        apply_exploration_config(ppo_cfg, args)
        ppo_cfg.rollout_steps = args.rollout_steps
        ppo_cfg.lr = args.lr
        start_update = int(payload.get("update", 0))
        best_eval = float(payload.get("best_eval_score", -1.0))
        for param_group in optimizer.param_groups:
            param_group["lr"] = ppo_cfg.lr

    training_rng = random.Random(args.seed + start_update)

    train_fields = [
        "update",
        "mean_episode_reward",
        "mean_episode_length",
        "policy_loss",
        "value_loss",
        "entropy",
    ]
    eval_fields = [
        "update",
        "mean_win_rate",
        "win_rate_prey_2",
        "win_rate_prey_3",
        "win_rate_prey_4",
    ]

    for update in range(start_update, args.updates):
        policy.train()
        buffer, episode_rewards, episode_lengths = collect_rollout(
            training_rng,
            policy,
            device,
            ppo_cfg,
            base_config,
            update,
            args.curriculum,
        )
        metrics = ppo_update(policy, optimizer, buffer, ppo_cfg, device)
        append_csv_row(
            checkpoint_dir / "train_log.csv",
            train_fields,
            {
                "update": update + 1,
                "mean_episode_reward": sum(episode_rewards) / max(len(episode_rewards), 1),
                "mean_episode_length": sum(episode_lengths) / max(len(episode_lengths), 1),
                "policy_loss": metrics.get("policy_loss", 0.0),
                "value_loss": metrics.get("value_loss", 0.0),
                "entropy": metrics.get("entropy", 0.0),
            },
        )

        save_checkpoint(
            checkpoint_dir / "latest.pt",
            policy,
            optimizer,
            update + 1,
            ppo_cfg,
            extra={"best_eval_score": best_eval},
        )

        if (update + 1) % args.eval_every == 0:
            policy.eval()
            eval_results, mean_win_rate = evaluate_policy(
                policy,
                device,
                seed=args.seed,
                num_runs=args.eval_runs,
                num_predators=args.predators,
                walls=args.num_walls,
                wall_size=args.wall_size,
                prey_defend=args.prey_defend,
            )
            append_csv_row(
                checkpoint_dir / "eval_log.csv",
                eval_fields,
                {
                    "update": update + 1,
                    "mean_win_rate": mean_win_rate,
                    "win_rate_prey_2": eval_results[2]["win_rate"],
                    "win_rate_prey_3": eval_results[3]["win_rate"],
                    "win_rate_prey_4": eval_results[4]["win_rate"],
                },
            )
            print(
                f"update={update + 1} mean_win_rate={mean_win_rate:.3f} "
                f"prey2={eval_results[2]['win_rate']:.3f} "
                f"prey3={eval_results[3]['win_rate']:.3f} "
                f"prey4={eval_results[4]['win_rate']:.3f}",
                flush=True,
            )
            if mean_win_rate >= best_eval:
                best_eval = mean_win_rate
                save_checkpoint(
                    checkpoint_dir / "best_eval.pt",
                    policy,
                    optimizer,
                    update + 1,
                    ppo_cfg,
                    extra={"best_eval_score": best_eval},
                )

        if (update + 1) % args.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"update_{update + 1:04d}.pt",
                policy,
                optimizer,
                update + 1,
                ppo_cfg,
                extra={"best_eval_score": best_eval},
            )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train shared-policy IPPO predators.")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--predators", type=int, default=3)
    parser.add_argument("--num-walls", type=int, default=2, dest="num_walls")
    parser.add_argument("--wall-size", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.02,
        help="Entropy bonus weight (default 0.02; was 0.01).",
    )
    parser.add_argument(
        "--entropy-floor",
        type=float,
        default=0.4,
        help="Penalize policy entropy below this value; set 0 to disable.",
    )
    parser.add_argument(
        "--entropy-floor-coef",
        type=float,
        default=0.05,
        help="Strength of the entropy-floor penalty.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/ippo")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-runs", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Train prey=2 first, then mix 2+3, then 2+3+4.",
    )
    parser.add_argument(
        "--prey-defend",
        choices=["stun", "kill"],
        default=None,
        dest="prey_defend",
        help="Enable cooperative prey knockout (stun or kill). Omit to disable.",
    )
    return parser


def main():
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
