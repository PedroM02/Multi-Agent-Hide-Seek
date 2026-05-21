import agent_utils as au
from environment import Environment


def attribute_rewards(env, captured_prey_ids):
    rewards = {agent_id: 0.0 for agent_id in env.agent_bodies}
    if not captured_prey_ids:
        return rewards
    for body in env.agent_bodies.values():
        if body.team == au.TEAM_PREDATOR and body.alive:
            rewards[body.agent_id] += 1.0
    return rewards
