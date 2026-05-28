"""Per-predator search fallback for RL (option B: heuristic only, not trained)."""

import random

from decision_making import _iter_enemy_sightings
from decision_making import DecisionMaking


def known_prey_positions_from_obs(obs):
    """Set of (x, y) for prey known via visible + shared comms (roles union)."""
    if obs is None:
        return set()
    positions = set()
    seen_ids = set()
    for enemy_x, enemy_y, enemy_id in _iter_enemy_sightings(
        obs.get("visible_enemies", ()),
        obs.get("shared_enemies", ()),
    ):
        if enemy_id in seen_ids:
            continue
        seen_ids.add(enemy_id)
        positions.add((enemy_x, enemy_y))
    return positions


class PredatorSearchController:
    """Per-agent search when no prey in sight/comms union; exit when prey is known again."""

    def __init__(self, rng=None):
        self.search_logic = DecisionMaking(rng or random.Random(0))
        self.in_search = {}

    def reset(self):
        self.in_search.clear()

    def update(self, raw_obs, predator_ids):
        """Update each predator. Returns (agent_in_search, agent_just_entered_search)."""
        agent_in_search = {}
        agent_just_entered = {}

        for agent_id in predator_ids:
            obs = raw_obs.get(agent_id)
            knows_prey = bool(known_prey_positions_from_obs(obs))
            was_search = self.in_search.get(agent_id, False)
            now_search = not knows_prey

            self.in_search[agent_id] = now_search
            agent_in_search[agent_id] = now_search
            agent_just_entered[agent_id] = now_search and not was_search

        return agent_in_search, agent_just_entered


def update_predator_search_headings(
    agents_by_id,
    raw_obs,
    predator_ids,
    search_logic,
    agent_in_search,
    agent_just_entered,
):
    """Mirror roles-mode heading persistence per searching predator."""
    for agent_id in predator_ids:
        agent = agents_by_id[agent_id]
        if not agent_in_search.get(agent_id, False):
            agent.search_heading = None
            agent.search_seen_ally_ids = set()
            continue

        obs = raw_obs[agent_id]
        visible_ally_ids = {
            ally_id for _, _, ally_id in obs.get("visible_allies", ())
        }
        if (
            agent_just_entered.get(agent_id, False)
            or agent.search_heading is None
        ):
            agent.search_heading = search_logic.init_search_heading(obs)
        elif visible_ally_ids - agent.search_seen_ally_ids:
            agent.search_heading = search_logic.init_search_heading(obs)
        agent.search_seen_ally_ids = visible_ally_ids


def choose_search_action(search_logic, agent, raw_obs):
    """Pick a legal search move using the persisted heading on the agent."""
    obs_for_search = dict(raw_obs)
    obs_for_search["search_heading"] = agent.search_heading
    legal = list(raw_obs["legal_actions"])
    return search_logic._search(obs_for_search, legal)


def any_agent_in_search(agent_in_search):
    return any(agent_in_search.values())
