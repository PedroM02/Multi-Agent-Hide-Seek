from __future__ import annotations
from decision_making import DecisionMaking
from perception import Perception

import random



class Agent:
    def __init__(
        self,
        agent_id: int,
        team: str,
        perception: Perception,
        decision: DecisionMaking,
        rng: random.Random,
    ) -> None:
        self.agent_id = agent_id
        self.team = team
        self.perception = perception
        self.decision = decision
        self.rng = rng
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
            self.rng,
        )
        if vis is not None:
            self.last_seen_enemy = vis
        action, clear_memory = self.decision.choose_action(obs, self.last_seen_enemy)
        if clear_memory:
            self.last_seen_enemy = None
        return action


def build_agents_for_env(env, rng: random.Random) -> list[Agent]:
    shared_perception = Perception()
    agents: list[Agent] = []
    for body in sorted(env.bodies.values(), key=lambda b: b.agent_id):
        # Each agent gets its own RNGs seeded from the parent — avoids
        # coupled "random" choices where both agents draw from the same
        # generator state in the same step. Perception and decision use
        # independent streams so tiebreak draws can't bias action choice.
        decision_seed = rng.randint(0, 2**31)
        perception_seed = rng.randint(0, 2**31)
        agents.append(
            Agent(
                agent_id=body.agent_id,
                team=body.team,
                perception=shared_perception,
                decision=DecisionMaking(random.Random(decision_seed)),
                rng=random.Random(perception_seed),
            )
        )
    return agents