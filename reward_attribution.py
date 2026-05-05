from __future__ import annotations

from typing import Dict, List

import game_types as gt
from environment import Environment


def attribute_rewards(env: Environment, captured_prey_ids: List[int]) -> Dict[int, float]:
    rewards: Dict[int, float] = {bid: 0.0 for bid in env.bodies}
    if not captured_prey_ids:
        return rewards
    for b in env.bodies.values():
        if b.team == gt.TEAM_PREDATOR and b.alive:
            rewards[b.agent_id] += 1.0
    return rewards
