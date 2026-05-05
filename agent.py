from __future__ import annotations

import random

from decision_making import DecisionMaking
from perception import Perception


class Agent:
    def __init__(
        self,
        agent_id: int,
        team: str,
        perception: Perception,
        decision: DecisionMaking,
    ) -> None:
        self.agent_id = agent_id
        self.team = team
        self.perception = perception
        self.decision = decision
        self.last_seen_enemy: tuple[int, int] | None = None

    def reset_memory(self) -> None:
        self.last_seen_enemy = None

    def decide(self, obs: dict) -> str:
        assert obs["agent_id"] == self.agent_id
        assert obs["team"] == self.team
        vis = Perception.update_last_seen_enemy(
            obs["ego_x"],
            obs["ego_y"],
            self.team,
            obs["visible_enemies"],
        )
        if vis is not None:
            self.last_seen_enemy = vis
        return self.decision.choose_action(obs, self.last_seen_enemy)


def build_agents_for_env(env, rng: random.Random) -> list[Agent]:
    shared_perception = Perception()
    agents: list[Agent] = []
    for body in sorted(env.bodies.values(), key=lambda b: b.agent_id):
        agents.append(
            Agent(
                agent_id=body.agent_id,
                team=body.team,
                perception=shared_perception,
                decision=DecisionMaking(rng),
            )
        )
    return agents
