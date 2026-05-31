"""Shared evaluation helpers for RL runs."""

import random

import constants as co
from rl.algo import IPPO, MAPPO
from rl.inference import collect_predator_transitions, make_rl_config, predator_slot_ids
from rl.team_search import PredatorSearchController
from simulation import BatchSummary, Run, copy_config


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


def execute_rl_run(
    config,
    rng,
    policy,
    device,
    deterministic=True,
    algo=IPPO,
    use_search=False,
):
    run = Run(config, rng)
    slot_ids = predator_slot_ids(run) if algo == MAPPO else None
    search_controller = PredatorSearchController() if use_search else None

    while run.outcome == co.OUTCOME_ONGOING:
        all_obs = run.build_step_observations()
        step_result = collect_predator_transitions(
            run,
            policy,
            device,
            all_obs,
            search_controller,
            deterministic=deterministic,
            algo=algo,
            slot_ids=slot_ids,
        )
        if algo == MAPPO:
            predator_actions, _transitions, _team_value, _joint_obs, _search_mode = (
                step_result
            )
        else:
            predator_actions, _transitions, _search_mode = step_result
        run.step_once(predator_actions=predator_actions, all_obs=all_obs)

    return run.outcome, run.step_index


def run_rl_batch(
    config,
    num_runs,
    policy,
    device,
    deterministic=True,
    algo=IPPO,
    use_search=False,
):
    accumulator = BatchSummary()
    for run_index in range(num_runs):
        run_config = copy_config(config, seed=config.seed + run_index)
        rng = random.Random(run_config.seed)
        outcome, steps = execute_rl_run(
            run_config,
            rng,
            policy,
            device,
            deterministic=deterministic,
            algo=algo,
            use_search=use_search,
        )
        accumulator.runs += 1
        accumulator.total_steps += steps
        if outcome == co.OUTCOME_PREDATORS_WIN:
            accumulator.predator_wins += 1
            accumulator.predator_win_steps += steps
        elif outcome == co.OUTCOME_PREY_WIN:
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
    use_search=False,
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
            config,
            num_runs,
            policy,
            device,
            deterministic=True,
            algo=algo,
            use_search=use_search,
        )
        win_rate = summary.predator_wins / max(summary.runs, 1)
        results[num_prey] = {
            "summary": summary,
            "win_rate": win_rate,
            "config": config,
        }
    mean_win_rate = sum(item["win_rate"] for item in results.values()) / len(results)
    return results, mean_win_rate
