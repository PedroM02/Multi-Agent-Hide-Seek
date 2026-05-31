from decision_making import DecisionMaking
from perception import Perception
import random
import constants as co


class Agent:
    '''Agent class that holds internal state of the agent'''

    def __init__(self, agent_id, team, perception, decision, rng):
        self.agent_id = agent_id
        self.team = team
        self.perception = perception
        self.decision = decision
        self.rng = rng
        self.last_seen_enemy = None
        self.search_heading = None
        self.search_seen_ally_ids = set()
        self.role = None
        self.role_target = None

    def reset_memory(self):
        '''Resets the memory of the agent to empty defaults'''
        self.last_seen_enemy = None
        self.search_heading = None
        self.search_seen_ally_ids = set()
        self.role = None
        self.role_target = None

    def update_memory_from_obs(self, obs):
        """Refresh last_seen_enemy from active_enemies, which include visible enemies and communicated enemies"""
        
        # If decision mode is optimal, agents have full view of the environment and do not need to update their memory
        if self.decision.mode == co.MODE_OPTIMAL:
            return
        if not obs.get("active_enemies", ()):
            return
        # Get closest enemy and update last seen enemy memory
        closest_enemy_position = self.perception.update_last_seen_enemy(obs, self.rng)
        if closest_enemy_position is not None:
            self.last_seen_enemy = closest_enemy_position

    def clear_stale_memory_if_at_cell(self, obs):
        """Clear memory when standing on last seen prey cell with no prey there nor any prey known (both visible and communicated)"""

        # If we still have active enemies, then last seen enemy memory is still valid and it is the position of the closest enemy
        if obs.get("active_enemies", ()):
            return
        # If no memory of enemy is stored, then there is no memory to clear
        if self.last_seen_enemy is None:
            return
        # If the agent is standing on the last seen enemy cell, then clear the memory
        if (obs["agent_x"], obs["agent_y"]) == self.last_seen_enemy:
            self.last_seen_enemy = None

    def clear_memory_at_positions(self, positions):
        '''Clear memory when standing on a cell that contained a prey which was consumed in the current step. RL only'''
        if self.last_seen_enemy is not None and self.last_seen_enemy in positions:
            self.last_seen_enemy = None

    def perceive(self, obs):
        """Enrich observation for this step"""
        self.perception.perceive(obs)

    def decide(self, obs):
        """Returns action intention considering roles, if enabled"""

        # Update memory
        self.update_memory_from_obs(obs)

        # Update role if enabled
        if self.decision.mode == co.MODE_ROLES:
            prev_role = self.role
            # Derive new role and role target position
            self.role, self.role_target = self.decision.derive_role(obs)
            # If search is enabled and it's the agent's role, get the agent's search direction
            # Heading is chosen when entering search and again when there is a new visible ally, to ensure dispersion
            if self.decision.searcher_enabled:
                if self.role == co.ROLE_SEARCHER:
                    # Get visible allies from observation
                    visible_ally_ids = {ally_id for ally_x, ally_y, ally_id in obs.get("visible_allies", ())}
                    # If it's the first time searching, calculate search direction
                    if prev_role != co.ROLE_SEARCHER:
                        self.search_heading = (self.decision.init_search_heading(obs))
                    # If the visible allies contains a new ally, recalculate search direction
                    elif visible_ally_ids - self.search_seen_ally_ids:
                        self.search_heading = (self.decision.init_search_heading(obs))
                    self.search_seen_ally_ids = visible_ally_ids
                else:
                    self.search_heading = None
                    self.search_seen_ally_ids = set()
            
            # Add information to observation for decision making
            obs["role"] = self.role
            obs["role_target"] = self.role_target
            obs["search_heading"] = self.search_heading
        else:
            self.role = None
            self.role_target = None

        # Get action intention from decision making and clear stale memory
        action = self.decision.choose_action(obs, self.last_seen_enemy)
        self.clear_stale_memory_if_at_cell(obs)
        return action


def build_agents_for_env(env, rng, config):
    '''Builds agents' logic modules to associate with bodies created in the environment'''

    agents = []
    # Iterate over agent bodies in order of agent ID
    for agent_id in sorted(env.agent_bodies):
        # Retrieve agent body
        agent_body = env.agent_bodies[agent_id]
        # Get individual seeds for perception and decision
        decision_seed = rng.randint(0, 10**10)
        perception_seed = rng.randint(0, 10**10)
        # Define predators' decision mode.Prey always flee, so they do not consider any decision modes
        decision_mode = (config.mode if agent_body.team == co.TEAM_PREDATOR else co.MODE_CHASE)
        # Create agent and add to list of agents
        agents.append(Agent(agent_id=agent_id,
                            team=agent_body.team,
                            perception=Perception(),
                            decision=DecisionMaking(rng=random.Random(decision_seed),
                                                    mode=decision_mode,
                                                    searcher_enabled=(config.roles_searcher if agent_body.team == co.TEAM_PREDATOR else False)),
                            rng=random.Random(perception_seed)))
    
    return agents
