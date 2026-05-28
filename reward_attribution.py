import agent_utils as au
from distances import manhattan
from rl.team_search import known_prey_positions_from_obs

DEFAULT_STEP_PENALTY = -0.001
RL_STEP_PENALTY = -0.005
DISTANCE_SHAPING_COEF = 0.01
NEW_CELL_SHAPING_COEF = 0.02
DISPERSION_SHAPING_COEF = 0.005


def predator_team_reward(captured_prey_ids, step_penalty=DEFAULT_STEP_PENALTY):
    """Shared team reward broadcast to every alive predator."""
    reward = step_penalty
    if captured_prey_ids:
        reward += 1.0
    return reward


def _known_prey_positions(raw_obs, predator_ids):
    """Team prey targets from visible + comms union (aligned with roles / RL search)."""
    positions = set()
    for agent_id in predator_ids:
        obs = raw_obs.get(agent_id)
        if obs is None:
            continue
        positions.update(known_prey_positions_from_obs(obs))
    return positions


def _predator_positions_from_obs(raw_obs, predator_ids):
    positions = []
    for agent_id in predator_ids:
        obs = raw_obs.get(agent_id)
        if obs is None:
            continue
        positions.append((obs["agent_x"], obs["agent_y"]))
    return positions


def _predator_positions_from_env(env):
    positions = []
    for body in env.alive_bodies():
        if body.team == au.TEAM_PREDATOR:
            positions.append((body.x, body.y))
    return positions


def _team_min_manhattan_to_targets(positions, targets):
    if not positions or not targets:
        return None
    min_dist = None
    for x, y in positions:
        for target_x, target_y in targets:
            distance = manhattan(x, y, target_x, target_y)
            if min_dist is None or distance < min_dist:
                min_dist = distance
    return min_dist


def _mean_pairwise_manhattan(positions):
    if len(positions) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for index, first in enumerate(positions):
        for second in positions[index + 1:]:
            total += manhattan(first[0], first[1], second[0], second[1])
            pairs += 1
    return total / pairs


def predator_team_shaped_reward(
    sim,
    raw_obs_before,
    captured_prey_ids,
    visited_cells_before,
    step_penalty=RL_STEP_PENALTY,
    distance_coef=DISTANCE_SHAPING_COEF,
    new_cell_coef=NEW_CELL_SHAPING_COEF,
    dispersion_coef=DISPERSION_SHAPING_COEF,
    search_mode=False,
):
    """Team reward for RL training: capture bonus, step penalty, optional search shaping."""
    reward = predator_team_reward(captured_prey_ids, step_penalty=step_penalty)

    predator_ids = sim.predator_agent_ids()
    if not predator_ids:
        return reward

    prey_positions = _known_prey_positions(raw_obs_before, predator_ids)
    positions_before = _predator_positions_from_obs(raw_obs_before, predator_ids)
    positions_after = _predator_positions_from_env(sim.env)

    if prey_positions:
        dist_before = _team_min_manhattan_to_targets(
            positions_before, prey_positions,
        )
        dist_after = _team_min_manhattan_to_targets(
            positions_after, prey_positions,
        )
        if (
            dist_before is not None
            and dist_after is not None
            and dist_after < dist_before
        ):
            reward += distance_coef * (dist_before - dist_after)
    elif search_mode:
        new_cells = sum(
            1 for position in positions_after
            if position not in visited_cells_before
        )
        reward += new_cell_coef * new_cells

        dispersion_before = _mean_pairwise_manhattan(positions_before)
        dispersion_after = _mean_pairwise_manhattan(positions_after)
        if dispersion_after > dispersion_before:
            reward += dispersion_coef * (dispersion_after - dispersion_before)

    return reward


def attribute_rewards(env, captured_prey_ids):
    rewards = {agent_id: 0.0 for agent_id in env.agent_bodies}
    if not captured_prey_ids:
        return rewards
    for body in env.agent_bodies.values():
        if body.team == au.TEAM_PREDATOR and body.alive:
            rewards[body.agent_id] += 1.0
    return rewards
