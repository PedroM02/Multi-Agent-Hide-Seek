from __future__ import annotations
from decision_making import DecisionMaking
from perception import Perception

import random

import agent_utils as au



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
        # Role assignment is written by the team role selector each step
        # (see simulation.step_once). The role_target is sticky between
        # selector calls when still valid; the selector itself preserves it.
        self.role: str | None = None
        self.role_target: tuple[int, int] | None = None

    def reset_memory(self) -> None:
        self.last_seen_enemy = None
        self.role = None
        self.role_target = None

    def prepare_observation(
        self,
        obs: dict,
        shared_enemies: tuple[tuple[int, int, int], ...],
    ) -> None:
        """Fuse the raw observation with teammate reports.

        Run once per step before the role selector and before decide. The
        agent uses its perception module to collapse direct sight +
        teammate reports into a single priority-resolved `active_enemies`
        set; both the selector and decision logic consume that set, so
        they can never disagree about who the threat is.
        """
        obs["shared_enemies"] = shared_enemies
        obs["active_enemies"] = self.perception.compute_active_enemies(
            obs["visible_enemies"], shared_enemies,
        )

    def decide(self, obs: dict) -> str:
        assert obs["agent_id"] == self.agent_id
        assert obs["team"] == self.team

        # `active_enemies` was populated by this agent's own
        # prepare_observation earlier in the step (strict priority:
        # direct sight > teammate report > empty). When non-empty, refresh
        # memory from it so the fallback memory is never older than the
        # freshest signal the team produced.
        if self.decision.mode != au.MODE_OPTIMAL:
            active = obs["active_enemies"]
            if active:
                vis = Perception.update_last_seen_enemy(
                    obs["ego_x"], obs["ego_y"], self.team, active, self.rng,
                )
                if vis is not None:
                    self.last_seen_enemy = vis

        action, clear_memory = self.decision.choose_action(obs, self.last_seen_enemy)
        if clear_memory:
            self.last_seen_enemy = None
        return action


def build_agents_for_env(env, rng: random.Random, config) -> list[Agent]:
    shared_perception = Perception()
    agents: list[Agent] = []
    for body in sorted(env.agent_bodies.values(), key=lambda b: b.agent_id):
        # Each agent gets its own RNGs seeded from the parent — avoids
        # coupled "random" choices where both agents draw from the same
        # generator state in the same step. Perception and decision use
        # independent streams so tiebreak draws can't bias action choice.
        decision_seed = rng.randint(0, 2**31)
        perception_seed = rng.randint(0, 2**31)
        decision_mode = (
            config.mode if body.team == au.TEAM_PREDATOR else au.MODE_CHASE
        )
        agents.append(
            Agent(
                agent_id=body.agent_id,
                team=body.team,
                perception=shared_perception,
                decision=DecisionMaking(
                    random.Random(decision_seed), mode=decision_mode,
                ),
                rng=random.Random(perception_seed),
            )
        )
    return agents