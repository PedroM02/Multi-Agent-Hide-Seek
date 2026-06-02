import argparse
import csv
import random
from pathlib import Path

import torch

import constants as co
from rl.reward_attribution import predator_team_shaped_reward
from rl.algo import IPPO, MAPPO
from rl.checkpointing import create_policy, load_checkpoint, save_checkpoint
from rl.eval_runner import curriculum_phase_for_update, evaluate_policy, sample_num_prey
from rl.inference import collect_predator_transitions, make_rl_config, predator_slot_ids
from rl.team_search import PredatorSearchController
from rl.ppo import MAPPOBuffer, PPOConfig, RolloutBuffer, mappo_update, ppo_update
from simulation import Run


def collect_rollout(training_rng, policy, device, ppo_config, base_config, update, curriculum, algo=IPPO, use_search=False):
    '''Collects transitions for IPPO/MAPPO training using the given policy and stores them in the buffer'''
    
    buffer = MAPPOBuffer() if algo == MAPPO else RolloutBuffer()
    run_rewards = []
    run_lengths = []

    # Execute steps while the buffer is not full
    while len(buffer) < ppo_config.rollout_steps:
        # Sample the number of prey for the current run
        phase = curriculum_phase_for_update(update, curriculum)
        num_prey = sample_num_prey(training_rng, phase)
        run_seed = training_rng.randint(0, 2**31 - 1)
        # Build RL run configuration
        config = make_rl_config(width=base_config.width, height=base_config.height, timesteps=base_config.timesteps, vision_radius_predator=base_config.vision_radius_predator, vision_radius_prey=base_config.vision_radius_prey, num_predators=base_config.num_predators, num_prey=num_prey, num_walls=base_config.num_walls, wall_size=base_config.wall_size, prey_defend=base_config.prey_defend, seed=run_seed)
        # Create run
        run = Run(config, random.Random(run_seed))
        slot_ids = predator_slot_ids(run) if algo == MAPPO else None
        # Create search controller if search is enabled
        search_controller = (PredatorSearchController() if use_search else None)
        # Initialize rewards and steps
        run_reward = 0.0
        run_steps = 0
        visited_cells = {(run.env.agent_bodies[agent_id].x, run.env.agent_bodies[agent_id].y) for agent_id in run.env.alive_predator_ids()}
 

        # Execute steps while the run is ongoing and the buffer is not full
        while run.outcome == co.OUTCOME_ONGOING and len(buffer) < ppo_config.rollout_steps:
            # Build observations for all alive agents
            all_obs = run.build_step_observations()
            visited_before = set(visited_cells)
            # Get predator transitions according to policy and observations
            step_result = collect_predator_transitions(run, policy, device, all_obs, search_controller, deterministic=False, algo=algo, slot_ids=slot_ids)
            if algo == MAPPO:
                predator_actions, transitions, team_value, joint_obs, search_mode = step_result
            else:
                predator_actions, transitions, search_mode = step_result
            # Execute timestep
            continuing = run.step_once(predator_actions=predator_actions, all_obs=all_obs)
            for agent_id in run.env.alive_predator_ids():
                agent_body = run.env.agent_bodies[agent_id]
                visited_cells.add((agent_body.x, agent_body.y))
            # Collect rewards
            reward = predator_team_shaped_reward(run, all_obs, run.last_captured, visited_before, search_mode=search_mode)

            done = not continuing
            run_reward += reward * len(transitions)
            run_steps += 1

            # Add transitions to buffer
            if algo == MAPPO:
                buffer.add_step(joint_obs, team_value, reward, done, transitions)
            else:
                for transition in transitions:
                    buffer.add(transition["obs"], transition["mask"], transition["action"], transition["log_prob"], transition["value"], reward, done)

        run_rewards.append(run_reward)
        run_lengths.append(run_steps)

    return buffer, run_rewards, run_lengths


def append_csv_row(path, fieldnames, row):
    '''Writes data row to CSV file'''
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def apply_exploration_config(ppo_config, args):
    '''Adds entropy terms to PPO config from command line arguments'''
    ppo_config.entropy_coef = args.entropy_coef
    ppo_config.entropy_floor = args.entropy_floor if args.entropy_floor > 0 else None
    ppo_config.entropy_floor_coef = args.entropy_floor_coef


def run_policy_update(algo, policy, optimizer, buffer, ppo_config, device):
    '''Runs the policy update for the given algorithm, return loss metrics'''
    if algo == MAPPO:
        return mappo_update(policy, optimizer, buffer, ppo_config, device)
    return ppo_update(policy, optimizer, buffer, ppo_config, device)


def train(args):
    '''Trains PPO policy using IPPO or MAPPO algorithms'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get algorithm to run
    algo = args.algo
    # Create PPO and RL run configurations
    ppo_config = PPOConfig(rollout_steps=args.rollout_steps, lr=args.lr, entropy_coef=args.entropy_coef, entropy_floor=args.entropy_floor if args.entropy_floor > 0 else None, entropy_floor_coef=args.entropy_floor_coef)
    base_config = make_rl_config(num_predators=args.predators, num_walls=args.num_walls, wall_size=args.wall_size, prey_defend=args.prey_defend)

    # Create policy and optimizer objects
    policy = create_policy(algo, num_predators=args.predators).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo_config.lr)
    start_update = 0
    best_eval = -1.0

    # Load checkpoint to resume training if provided
    checkpoint_dir = Path(args.checkpoint_dir)
    if args.checkpoint:
        policy, loaded_config, payload = load_checkpoint(args.checkpoint, device, policy=policy, optimizer=optimizer, algo=algo, num_predators=args.predators)

        checkpoint_algo = payload.get("algo", IPPO)
        if checkpoint_algo != algo:
            raise ValueError(f"Checkpoint algo={checkpoint_algo!r} does not match --algo {algo!r}")
        
        # Update PPO configuration with loaded configuration
        ppo_config = loaded_config
        apply_exploration_config(ppo_config, args)
        ppo_config.rollout_steps = args.rollout_steps
        ppo_config.lr = args.lr
        start_update = int(payload.get("update", 0))
        best_eval = float(payload.get("best_eval_score", -1.0))
        for param_group in optimizer.param_groups:
            param_group["lr"] = ppo_config.lr

    training_rng = random.Random(args.seed + start_update)
    # Define fields for training and evaluation logs
    train_fields = ["update","mean_run_reward","mean_run_length","policy_loss","value_loss","entropy"]
    eval_fields = ["update", "mean_win_rate", "win_rate_prey_2", "win_rate_prey_3", "win_rate_prey_4"]

    # Iterate over defined number of updates
    for update in range(start_update, args.updates):
        # Set policy to training mode
        policy.train()
        # Collect transitions
        buffer, run_rewards, run_lengths = collect_rollout(training_rng, policy, device, ppo_config, base_config, update, args.curriculum, algo=algo, use_search=args.search)
        # Run policy update
        metrics = run_policy_update(algo, policy, optimizer, buffer, ppo_config, device)
        # Log training metrics
        append_csv_row(checkpoint_dir / "train_log.csv", train_fields,
            {
                "update": update + 1,
                "mean_run_reward": sum(run_rewards) / max(len(run_rewards), 1),
                "mean_run_length": sum(run_lengths) / max(len(run_lengths), 1),
                "policy_loss": metrics.get("policy_loss", 0.0),
                "value_loss": metrics.get("value_loss", 0.0),
                "entropy": metrics.get("entropy", 0.0)})
    
        # Save latest checkpoint
        save_checkpoint(checkpoint_dir / "latest.pt", policy, optimizer, update + 1, ppo_config,
            extra={
                "best_eval_score": best_eval,
                "use_search": bool(args.search)})

        # Evaluate policy at regular defined intervals
        if (update + 1) % args.eval_every == 0:
            policy.eval()
            eval_results, mean_win_rate = evaluate_policy(policy, device, seed=args.seed, num_runs=args.eval_runs, num_predators=args.predators, walls=args.num_walls, wall_size=args.wall_size, prey_defend=args.prey_defend, algo=algo, use_search=args.search)
            # Log evaluation metrics
            append_csv_row(checkpoint_dir / "eval_log.csv", eval_fields,
                {
                    "update": update + 1,
                    "mean_win_rate": mean_win_rate,
                    "win_rate_prey_2": eval_results[2]["win_rate"],
                    "win_rate_prey_3": eval_results[3]["win_rate"],
                    "win_rate_prey_4": eval_results[4]["win_rate"]})
            
            # Output metrics to console
            print(
                f"update={update + 1} mean_win_rate={mean_win_rate:.3f} "
                f"prey2={eval_results[2]['win_rate']:.3f} "
                f"prey3={eval_results[3]['win_rate']:.3f} "
                f"prey4={eval_results[4]['win_rate']:.3f}",
                flush=True,
            )
            # Save best evaluation checkpoint if best performance so far
            if mean_win_rate >= best_eval:
                best_eval = mean_win_rate
                save_checkpoint(checkpoint_dir / "best_eval.pt", policy, optimizer, update + 1, ppo_config,
                    extra={
                        "best_eval_score": best_eval,
                        "use_search": bool(args.search)})

        # Save checkpoint at regular defined intervals
        if (update + 1) % args.save_every == 0:
            save_checkpoint(checkpoint_dir / f"update_{update + 1:04d}.pt", policy, optimizer, update + 1, ppo_config,
                extra={
                    "best_eval_score": best_eval,
                    "use_search": bool(args.search)})


def build_arg_parser():
    '''Builds the argument parser for training CLI inputs'''
    parser = argparse.ArgumentParser(description="Train shared-policy IPPO / MAPPO predators")
    parser.add_argument("--algo", choices=[IPPO, MAPPO], default=IPPO, help="Training algorithm: shared decentralized critic (ippo) or centralized critic (mappo)")
    parser.add_argument("--updates", type=int, default=1000, help="Number of training updates to perform")
    parser.add_argument("--predators", type=int, default=3, help="Number of predators per run")
    parser.add_argument("--num-walls", type=int, default=2, dest="num_walls", help="Number of wall segments to generate and place on the map randomly")
    parser.add_argument("--wall-size", type=int, default=2, help="Length, in cells, of wall segments")
    parser.add_argument("--rollout-steps", type=int, default=4096, help="Number of transitions to collect per policy update")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for the policy optimizer")
    parser.add_argument("--entropy-coef", type=float, default=0.02, help="Entropy bonus weight to encourage exploration")
    parser.add_argument("--entropy-floor", type=float, default=0.4, help="Penalize policy entropy below this value. Set 0 to disable")
    parser.add_argument("--entropy-floor-coef", type=float, default=0.05, help="Strength of the entropy-floor penalty")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save policy checkpoints to")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint to resume training")
    parser.add_argument("--eval-every", type=int, default=25, help="Evaluate policy every N updates")
    parser.add_argument("--eval-runs", type=int, default=20, help="Number of RL runs used to evaluate policy when training")
    parser.add_argument("--save-every", type=int, default=100, help="Save policy checkpoint every N updates")
    parser.add_argument("--curriculum", action="store_true", help="Trains with prey=2 for the first 200 updates, then mix 2+3 prey for another 200 updates, then 2+3+4 prey")
    parser.add_argument("--prey-defend", choices=["stun", "kill"], default=None, dest="prey_defend", help="Enable defense mechanism where prey groups attack up to nr_prey-1 adjacent predators (stun or kill). Omit to disable")
    parser.add_argument("--search", action="store_true", help="Enable predator search when no prey is either visible or communicated. This is not learned behavior")
    return parser


def main():
    '''Main function to execute the training of a policy using IPPO or MAPPO'''
    # Read and parse the CLI arguments
    args = build_arg_parser().parse_args()
    # Set checkpoint directory if not provided
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"checkpoints/{args.algo}"
    train(args)


if __name__ == "__main__":
    main()
