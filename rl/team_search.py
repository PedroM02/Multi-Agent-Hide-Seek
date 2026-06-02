import random

from decision_making import DecisionMaking


def known_prey_positions_from_obs(obs):
    """Returns deduplicated positions of known enemies"""
    if obs is None:
        return set()
    positions = set()
    seen_ids = set()
    for enemy_x, enemy_y, enemy_id in obs.get("known_enemies", ()):
        if enemy_id in seen_ids:
            continue
        seen_ids.add(enemy_id)
        positions.add((enemy_x, enemy_y))
    return positions


class PredatorSearchController:
    """Returns True when predator is in search and False otherwise.
       Starts search when no prey is in sight or communicated. Leaves search when prey is known again.
       Also returns whether each predator just entered search"""

    def __init__(self, rng=None):
        self.search_logic = DecisionMaking(rng or random.Random(0))
        self.in_search = {}

    def reset(self):
        self.in_search.clear()

    def update(self, all_obs, predator_ids):
        """Update each predator. Returns (agent_in_search, agent_just_entered_search)."""
        agent_in_search = {}
        agent_just_entered = {}

        # For each predator, put them in search if they don't know prey, or take them out of search otherwise
        for agent_id in predator_ids:
            obs = all_obs.get(agent_id)
            knows_prey = bool(known_prey_positions_from_obs(obs))
            was_search = self.in_search.get(agent_id, False)
            now_search = not knows_prey

            self.in_search[agent_id] = now_search
            agent_in_search[agent_id] = now_search
            agent_just_entered[agent_id] = now_search and not was_search

        return agent_in_search, agent_just_entered


def update_predator_search_headings(agents_by_id, all_obs, predator_ids, search_logic, agent_in_search, agent_just_entered):
    """Applies search via roles-mode direction persistence per searching predator."""
    
    # For each predator, check if they are in search and update direction if needed
    for agent_id in predator_ids:
        agent = agents_by_id[agent_id]
        # If agent is not in search, clear search direction and then skip
        if not agent_in_search.get(agent_id, False):
            agent.search_heading = None
            agent.search_seen_ally_ids = set()
            continue

        obs = all_obs[agent_id]
        visible_ally_ids = {ally_id for ally_x, ally_y, ally_id in obs.get("visible_allies", ())}
        # If agent entered search, give it direction
        if (agent_just_entered.get(agent_id, False) or agent.search_heading is None):
            agent.search_heading = search_logic.init_search_heading(obs)
        # If new ally appeared, recalculate direction to ensure dispersion
        elif visible_ally_ids - agent.search_seen_ally_ids:
            agent.search_heading = search_logic.init_search_heading(obs)
        agent.search_seen_ally_ids = visible_ally_ids


def choose_search_action(search_logic, agent, raw_obs):
    """Picks a legal move using the persisted direction on the agent."""
    obs_for_search = dict(raw_obs)
    obs_for_search["search_heading"] = agent.search_heading
    legal = list(raw_obs["legal_actions"])
    return search_logic.search(obs_for_search, legal)


def any_agent_in_search(agent_in_search):
    '''Checks if any predator is currently searching'''
    return any(agent_in_search.values())
