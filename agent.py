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
        # Persisted cardinal direction while ROLE_SEARCHER (`--searcher`).
        self.search_heading = None
        # Ally ids visible last step while searching (re-head on new entries).
        self.search_seen_ally_ids = set()
        # Written each step by derive_role in `--mode roles` (for GUI).
        self.role = None
        self.role_target = None

    def reset_memory(self):
        self.last_seen_enemy = None
        self.search_heading = None
        self.search_seen_ally_ids = set()
        self.role = None
        self.role_target = None

    def update_memory_from_obs(self, obs):
        """Refresh `last_seen_enemy` from fused `active_enemies`."""
        if self.decision.mode == au.MODE_OPTIMAL:
            return
        active = obs.get("active_enemies", ())
        if not active:
            return
        visible_position = self.perception.update_last_seen_enemy(
            obs["agent_x"], obs["agent_y"], active, self.rng,
        )
        if visible_position is not None:
            self.last_seen_enemy = visible_position

    def clear_stale_memory_if_at_cell(self, obs):
        """Drop memory when standing on a stale last-seen cell with no prey known."""
        if obs.get("active_enemies", ()):
            return
        last_seen = self.last_seen_enemy
        if last_seen is None:
            return
        if (obs["agent_x"], obs["agent_y"]) == last_seen:
            self.last_seen_enemy = None

    def clear_memory_at_positions(self, positions):
        if self.last_seen_enemy is not None and self.last_seen_enemy in positions:
            self.last_seen_enemy = None

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

        self.update_memory_from_obs(obs)

        obs_for_decision = obs
        if self.decision.mode == au.MODE_ROLES:
            prev_role = self.role
            self.role, self.role_target = self.decision.derive_role(obs)
            if self.decision.searcher_enabled:
                if self.role == au.ROLE_SEARCHER:
                    visible_ally_ids = {
                        ally_id
                        for _, _, ally_id in obs.get("visible_allies", ())
                    }
                    if prev_role != au.ROLE_SEARCHER:
                        self.search_heading = (
                            self.decision.init_search_heading(obs)
                        )
                    elif visible_ally_ids - self.search_seen_ally_ids:
                        self.search_heading = (
                            self.decision.init_search_heading(obs)
                        )
                    self.search_seen_ally_ids = visible_ally_ids
                else:
                    self.search_heading = None
                    self.search_seen_ally_ids = set()
            obs_for_decision = dict(obs)
            obs_for_decision["role"] = self.role
            obs_for_decision["role_target"] = self.role_target
            obs_for_decision["search_heading"] = self.search_heading
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
