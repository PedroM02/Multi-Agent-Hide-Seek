import constants as co
from utils import manhattan

from rl.team_search import known_prey_positions_from_obs

# Reward and penalty constants
STEP_PENALTY = -0.005  # Penalty for each step taken
DISTANCE_SHAPING_COEF = 0.01  # Reward coefficient for reducing distance to prey
NEW_CELL_SHAPING_COEF = 0.02  # Reward coefficient for exploring new cells
DISPERSION_SHAPING_COEF = 0.005  # Reward coefficient for increasing dispersion towards other predators


def known_prey_positions(all_obs, predator_ids):
    """Visible and communicated prey targets for a predator"""
    positions = set()
    for agent_id in predator_ids:
        obs = all_obs.get(agent_id)
        if obs is None:
            continue
        positions.update(known_prey_positions_from_obs(obs))
    return positions


def predator_positions_from_obs(all_obs, predator_ids):
    """Positions of all alive predators from their observations"""
    positions = []
    for agent_id in predator_ids:
        obs = all_obs.get(agent_id)
        if obs is None:
            continue
        positions.append((obs["agent_x"], obs["agent_y"]))
    return positions


def predator_positions_from_env(env):
    """Positions of all alive predators in the environment"""
    positions = []
    for agent_body in env.alive_bodies():
        if agent_body.team == co.TEAM_PREDATOR:
            positions.append((agent_body.x, agent_body.y))
    return positions


def team_min_manhattan_to_targets(positions, targets):
    """Calculates the overall minimum distance from any agent to any target"""
    if not positions or not targets:
        return None
    min_dist = None
    for x, y in positions:
        for target_x, target_y in targets:
            distance = manhattan(x, y, target_x, target_y)
            if min_dist is None or distance < min_dist:
                min_dist = distance
    return min_dist


def mean_pairwise_manhattan(positions):
    '''Returns the overall mean distance between all pairs of agents'''
    if len(positions) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for index, first in enumerate(positions):
        for second in positions[index + 1:]:
            total += manhattan(first[0], first[1], second[0], second[1])
            pairs += 1
    return total / pairs


def predator_team_shaped_reward(run, all_obs_before, captured_prey_ids, visited_cells_before, step_penalty=STEP_PENALTY, distance_coef=DISTANCE_SHAPING_COEF, new_cell_coef=NEW_CELL_SHAPING_COEF, dispersion_coef=DISPERSION_SHAPING_COEF, search_mode=False):
    """Team reward for RL training: capture bonus, step penalty, optional search shaping."""
    
    # Initialize reward with step penalty
    reward = step_penalty
    # Add capture reward if any prey was captured
    if captured_prey_ids:
        reward += 1.0

    predator_ids = run.env.alive_predator_ids()
    if not predator_ids:
        return reward

    prey_positions = known_prey_positions(all_obs_before, predator_ids)
    positions_before = predator_positions_from_obs(all_obs_before, predator_ids)
    positions_after = predator_positions_from_env(run.env)

    # Attribute reward if distance to prey decreased
    if prey_positions:
        dist_before = team_min_manhattan_to_targets(positions_before, prey_positions)
        dist_after = team_min_manhattan_to_targets(positions_after, prey_positions)
        if (dist_before is not None and dist_after is not None and dist_after < dist_before):
            reward += distance_coef * (dist_before - dist_after)
    
    
    elif search_mode:
        # Attribute reward if new cells were explored
        new_cells = sum(1 for position in positions_after if position not in visited_cells_before)
        reward += new_cell_coef * new_cells

        # Attribute reward if dispersion between predatorsincreased
        dispersion_before = mean_pairwise_manhattan(positions_before)
        dispersion_after = mean_pairwise_manhattan(positions_after)
        if dispersion_after > dispersion_before:
            reward += dispersion_coef * (dispersion_after - dispersion_before)

    return reward
