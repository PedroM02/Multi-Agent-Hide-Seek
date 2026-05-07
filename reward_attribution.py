from __future__ import annotations

from typing import Dict, List

import agent_utils as au
from environment import Environment


def attribute_rewards(env: Environment, captured_prey_ids: List[int]) -> Dict[int, float]:
    rewards: Dict[int, float] = {bid: 0.0 for bid in env.agent_bodies}
    if not captured_prey_ids:
        return rewards
    for b in env.agent_bodies.values():
        if b.team == au.TEAM_PREDATOR and b.alive:
            rewards[b.agent_id] += 1.0
    return rewards
