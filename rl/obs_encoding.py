import numpy as np

import constants as co

# Observation dimensions and constants
OBS_DIM = 102
GRID_CHANNELS = 4
PATCH_SIZE = 5

# Action constants and mappings
ACTIONS = co.MOVE_ACTIONS
ACTION_TO_IDX = {action: index for index, action in enumerate(ACTIONS)}
IDX_TO_ACTION = {index: action for action, index in ACTION_TO_IDX.items()}
NUM_ACTIONS = len(ACTIONS)


def enemy_positions(items):
    '''Returns the non-deduplicated (by ID) positions of all enemies in the observation, both direct and active'''
    positions = set()
    for item in items:
        # If item is a communicated prey, ignore sender ID
        if len(item) == 4:
            sender_id, enemy_x, enemy_y, enemy_id = item
        else:
            enemy_x, enemy_y, enemy_id = item
        positions.add((enemy_x, enemy_y, enemy_id))
    return positions


def ally_positions(raw_obs):
    '''Returns visible allies in set of tuples'''
    return {(ally_x, ally_y, ally_id) for ally_x, ally_y, ally_id in raw_obs.get("visible_allies", ())}


def encode_obs(raw_obs, agent):
    """Return float32 vector of shape (OBS_DIM,) - defaults to OBS_DIM = 102.
       Dimensions of observation are hard-coded (PATCH_SIZE = 5, GRID_CHANNELS = 4) to match project's
       specification and what is expected by policies."""
       
    vision_radius = raw_obs["vision_radius"]
    agent_x = raw_obs["agent_x"]
    agent_y = raw_obs["agent_y"]
    # Build a GRID_CHANNELS number of empty grid patches of size 5x5, given default vision radius of 2
    patch = np.zeros((PATCH_SIZE, PATCH_SIZE, GRID_CHANNELS), dtype=np.float32)

    # Get positions of direct and active enemies and allies
    direct_enemies = enemy_positions(raw_obs.get("visible_enemies", ()))
    active_enemies = enemy_positions(raw_obs.get("active_enemies", ()))
    # Get enemies that came from communications only
    comms_only = active_enemies - direct_enemies
    allies = ally_positions(raw_obs)

    # Populate grid patches with enemies and allies
    # Iterate over all grid cells in the observation patch
    for dy in range(-vision_radius, vision_radius + 1):
        for dx in range(-vision_radius, vision_radius + 1):
            # Get real coordinates of the current grid cell in the world
            world_x = agent_x + dx
            world_y = agent_y + dy
            # Get row and column indices of the current grid cell in the observation patch
            row = dy + vision_radius
            col = dx + vision_radius

            # If the current grid cell is the agent's own cell, set to 1 in the last channel
            if dx == 0 and dy == 0:
                patch[row, col, 3] = 1.0

            # If the current grid cell is a direct enemy, set to 1 in the second channel
            if (world_x, world_y) in {(enemy_x, enemy_y) for enemy_x, enemy_y, enemy_id in direct_enemies}:
                patch[row, col, 1] = 1.0

            # If the current grid cell is a communicated enemy, set to 1 in the third channel
            elif (world_x, world_y) in {(enemy_x, enemy_y) for enemy_x, enemy_y, enemy_id in comms_only}:
                patch[row, col, 2] = 1.0

            # If the current grid cell is an ally, set to 1 in the first channel
            if (world_x, world_y) in {(ally_x, ally_y) for ally_x, ally_y, _ in allies}:
                patch[row, col, 0] = 1.0

    # Flatten grid patches into a 1D array
    grid_flat = patch.reshape(-1)
    # Initialize memory array for last seen enemy
    memory = np.zeros(2, dtype=np.float32)
    last_seen = agent.last_seen_enemy
    # If the agent has an enemy in memory, set it into the memory array, in relative coordinates normalized by vision radius
    if last_seen is not None:
        target_x, target_y = last_seen
        memory[0] = (target_x - agent_x) / max(vision_radius, 1)
        memory[1] = (target_y - agent_y) / max(vision_radius, 1)

    # Return entire observation as a flattened array
    return np.concatenate([grid_flat, memory]).astype(np.float32)


def action_mask(raw_obs):
    """Boolean mask of shape (NUM_ACTIONS,) for legal moves. Legal moves have True in their corresponding
       index, and False otherwise"""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for action in raw_obs.get("legal_actions", ()):
        mask[ACTION_TO_IDX[action]] = True
    return mask


def build_joint_predator_obs(all_obs, predator_slot_ids, agent_lookup):
    """Concatenates local predator observations in fixed slot order for the centralized critic"""
    parts = []
    # Iterate over all predators in fixed slot order
    for agent_id in predator_slot_ids:
        # If the agent does not have observations (such as when the agent is dead), ignore
        if agent_id not in all_obs:
            parts.append(np.zeros(OBS_DIM, dtype=np.float32))
            continue
        # Get agent object
        agent = agent_lookup.get(agent_id)
        parts.append(encode_obs(all_obs[agent_id], agent))
    return np.concatenate(parts).astype(np.float32)
