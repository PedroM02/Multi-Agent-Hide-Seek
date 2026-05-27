"""Egocentric observation encoding for RL predators."""

import numpy as np

import agent_utils as au

OBS_DIM = 102
GRID_CHANNELS = 4
PATCH_SIZE = 5

ACTIONS = au.MOVE_ACTIONS
ACTION_TO_IDX = {action: index for index, action in enumerate(ACTIONS)}
IDX_TO_ACTION = {index: action for action, index in ACTION_TO_IDX.items()}
NUM_ACTIONS = len(ACTIONS)


def _enemy_positions(items):
    positions = set()
    for item in items:
        if len(item) == 4:
            _, enemy_x, enemy_y, enemy_id = item
        else:
            enemy_x, enemy_y, enemy_id = item
        positions.add((enemy_x, enemy_y, enemy_id))
    return positions


def _direct_enemy_positions(raw_obs):
    return _enemy_positions(raw_obs.get("visible_enemies", ()))


def _active_enemy_positions(raw_obs):
    return _enemy_positions(raw_obs.get("active_enemies", ()))


def _ally_positions(raw_obs):
    return {
        (ally_x, ally_y, ally_id)
        for ally_x, ally_y, ally_id in raw_obs.get("visible_allies", ())
    }


def encode_obs(raw_obs, agent):
    """Return float32 vector of shape (OBS_DIM,)."""
    vision_radius = raw_obs["vision_radius"]
    agent_x = raw_obs["agent_x"]
    agent_y = raw_obs["agent_y"]
    patch = np.zeros((PATCH_SIZE, PATCH_SIZE, GRID_CHANNELS), dtype=np.float32)

    direct_enemies = _direct_enemy_positions(raw_obs)
    active_enemies = _active_enemy_positions(raw_obs)
    comms_only = active_enemies - direct_enemies
    allies = _ally_positions(raw_obs)

    for dy in range(-vision_radius, vision_radius + 1):
        for dx in range(-vision_radius, vision_radius + 1):
            world_x = agent_x + dx
            world_y = agent_y + dy
            row = dy + vision_radius
            col = dx + vision_radius

            if dx == 0 and dy == 0:
                patch[row, col, 3] = 1.0

            if (world_x, world_y) in {
                (enemy_x, enemy_y)
                for enemy_x, enemy_y, _ in direct_enemies
            }:
                patch[row, col, 1] = 1.0
            elif (world_x, world_y) in {
                (enemy_x, enemy_y)
                for enemy_x, enemy_y, _ in comms_only
            }:
                patch[row, col, 2] = 1.0

            if (world_x, world_y) in {
                (ally_x, ally_y) for ally_x, ally_y, _ in allies
            }:
                patch[row, col, 0] = 1.0

    grid_flat = patch.reshape(-1)
    memory = np.zeros(2, dtype=np.float32)
    last_seen = agent.last_seen_enemy
    if last_seen is not None:
        target_x, target_y = last_seen
        memory[0] = (target_x - agent_x) / max(vision_radius, 1)
        memory[1] = (target_y - agent_y) / max(vision_radius, 1)

    return np.concatenate([grid_flat, memory]).astype(np.float32)


def action_mask(raw_obs):
    """Boolean mask of shape (NUM_ACTIONS,) for legal moves."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for action in raw_obs.get("legal_actions", ()):
        mask[ACTION_TO_IDX[action]] = True
    return mask


def build_joint_predator_obs(raw_obs, predator_slot_ids, agent_lookup):
    """Concatenate local predator obs in fixed slot order for the centralized critic."""
    parts = []
    for agent_id in predator_slot_ids:
        if agent_id not in raw_obs:
            parts.append(np.zeros(OBS_DIM, dtype=np.float32))
            continue
        agent = agent_lookup.get(agent_id)
        parts.append(encode_obs(raw_obs[agent_id], agent))
    return np.concatenate(parts).astype(np.float32)
