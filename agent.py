from decision_making import DecisionMaking
from perception import Perception

import random

import agent_utils as au


class Agent:
    def __init__(self, agent_id, team, perception, decision, rng):
        self.agent_id = agent_id
        self.team = team
        self.perception = perception
        self.decision = decision
        self.rng = rng
        self.last_seen_enemy = None
        # Written each step by derive_role in `--mode roles` (for GUI).
        self.role = None
        self.role_target = None

    def reset_memory(self):
        self.last_seen_enemy = None
        self.role = None
        self.role_target = None

    def prepare_observation(self, obs, shared_enemies, shared_allies=()):
        """Fuse the raw observation with teammate reports.

        Run once per step before decide. The agent uses its perception
        module to collapse direct sight + teammate reports into a single
        priority-resolved `active_enemies` set.
        """
        obs["shared_enemies"] = shared_enemies
        obs["shared_allies"] = shared_allies
        obs["active_enemies"] = self.perception.compute_active_enemies(
            obs["visible_enemies"], shared_enemies,
        )

    def decide(self, obs):
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
                visible_position = self.perception.update_last_seen_enemy(
                    obs["agent_x"], obs["agent_y"], active, self.rng,
                )
                if visible_position is not None:
                    self.last_seen_enemy = visible_position

        obs_for_decision = obs
        if self.decision.mode == au.MODE_ROLES:
            self.role, self.role_target = self.decision.derive_role(obs)
            obs_for_decision = dict(obs)
            obs_for_decision["role"] = self.role
            obs_for_decision["role_target"] = self.role_target
        else:
            self.role = None
            self.role_target = None

        action, clear_memory = self.decision.choose_action(
            obs_for_decision, self.last_seen_enemy,
        )
        if clear_memory:
            self.last_seen_enemy = None
        return action


def build_agents_for_env(env, rng, config):
    agents = []
    for body in sorted(env.agent_bodies.values(), key=lambda body: body.agent_id):
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
                perception=Perception(),
                decision=DecisionMaking(
                    random.Random(decision_seed),
                    mode=decision_mode,
                    searcher_enabled=(
                        config.roles_searcher
                        if body.team == au.TEAM_PREDATOR
                        else False
                    ),
                ),
                rng=random.Random(perception_seed),
            )
        )
    return agents
