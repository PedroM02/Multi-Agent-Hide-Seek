import agent_utils as au

DEFAULT_STEP_PENALTY = -0.001


def predator_team_reward(captured_prey_ids, step_penalty=DEFAULT_STEP_PENALTY):
    """Shared team reward broadcast to every alive predator."""
    reward = step_penalty
    if captured_prey_ids:
        reward += 1.0
    return reward


def attribute_rewards(env, captured_prey_ids):
    rewards = {agent_id: 0.0 for agent_id in env.agent_bodies}
    if not captured_prey_ids:
        return rewards
    for body in env.agent_bodies.values():
        if body.team == au.TEAM_PREDATOR and body.alive:
            rewards[body.agent_id] += 1.0
    return rewards
