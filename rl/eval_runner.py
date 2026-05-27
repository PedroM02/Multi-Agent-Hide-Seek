"""Shared evaluation helpers for RL episodes."""

import random

import agent_utils as au
from reward_attribution import predator_team_reward
from rl.algo import IPPO, MAPPO
from rl.inference import collect_predator_transitions, make_rl_config, predator_slot_ids
from simulation import BatchSummary, SimulationState, copy_config


def sample_num_prey(rng, curriculum_phase=None):
    if curriculum_phase == 1:
        return 2
    if curriculum_phase == 2:
        return rng.choice([2, 3])
    return rng.choice([2, 3, 4])


def curriculum_phase_for_update(update, curriculum_enabled):
    if not curriculum_enabled:
        return None
    if update < 200:
        return 1
    if update < 400:
        return 2
    return 3


def run_rl_episode(config, rng, policy, device, deterministic=True, algo=IPPO):
    sim = SimulationState(config, rng)
    total_reward = 0.0
    slot_ids = predator_slot_ids(sim) if algo == MAPPO else None

    while sim.outcome == au.OUTCOME_ONGOING:
        raw_obs = sim.build_step_observations()
        step_result = collect_predator_transitions(
            sim,
            policy,
            device,
            raw_obs,
            deterministic=deterministic,
            algo=algo,
            slot_ids=slot_ids,
        )
        if algo == MAPPO:
            predator_actions, _transitions, _team_value, _joint_obs = step_result
        else:
            predator_actions, _transitions = step_result
        sim.step_once(predator_actions=predator_actions, raw_obs=raw_obs)
        total_reward += predator_team_reward(sim.last_captured)

    return sim.outcome, sim.step_index, total_reward


def run_rl_batch(config, num_runs, policy, device, deterministic=True, algo=IPPO):
    accumulator = BatchSummary()
    for run_index in range(num_runs):
        run_config = copy_config(config, seed=config.seed + run_index)
        rng = random.Random(run_config.seed)
        outcome, steps, _reward = run_rl_episode(
            run_config, rng, policy, device, deterministic=deterministic, algo=algo,
        )
        accumulator.runs += 1
        accumulator.total_steps += steps
        if outcome == au.OUTCOME_PREDATORS_WIN:
            accumulator.predator_wins += 1
            accumulator.predator_win_steps += steps
        elif outcome == au.OUTCOME_PREY_WIN:
            accumulator.prey_wins += 1
            accumulator.prey_win_steps += steps
    return accumulator


def evaluate_policy(
    policy,
    device,
    seed=0,
    num_runs=20,
    num_predators=3,
    prey_counts=(2, 3, 4),
    walls=2,
    wall_size=2,
    prey_defend=None,
    algo=IPPO,
):
    results = {}
    for num_prey in prey_counts:
        config = make_rl_config(
            num_predators=num_predators,
            num_prey=num_prey,
            num_walls=walls,
            wall_size=wall_size,
            prey_defend=prey_defend,
            seed=seed,
        )
        summary = run_rl_batch(
            config, num_runs, policy, device, deterministic=True, algo=algo,
        )
        win_rate = summary.predator_wins / max(summary.runs, 1)
        results[num_prey] = {
            "summary": summary,
            "win_rate": win_rate,
            "config": config,
        }
    mean_win_rate = sum(item["win_rate"] for item in results.values()) / len(results)
    return results, mean_win_rate
