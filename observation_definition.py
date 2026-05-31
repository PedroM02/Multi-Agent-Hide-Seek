import constants as co
from utils import chebyshev


def build_observation(env, agent_id, vision_radius):
    '''Builds and returns basic observation dictionary for a given agent'''

    # Retrieve agent body and its legal actions
    agent_body = env.agent_bodies[agent_id]
    legal_actions = env.legal_actions(agent_id)
    enemies = []
    allies = []

    # Iterate over all other agents and add them to the observation if they are within the vision radius, either enemies or allies
    for other in env.agent_bodies.values():
        # If the other agent is the agent itself or is not alive, skip
        if other.agent_id == agent_id or not other.alive:
            continue
        # If the other agent is not within the vision radius, skip
        if chebyshev(agent_body.x, agent_body.y, other.x, other.y) > vision_radius:
            continue
        # If the other agent is on the same team, add it to the allies list, otherwise add it to the enemies list
        if other.team == agent_body.team:
            allies.append((other.x, other.y, other.agent_id))
        else:
            enemies.append((other.x, other.y, other.agent_id))
    enemies.sort(key=lambda t: (t[2], t[0], t[1]))
    allies.sort(key=lambda t: (t[2], t[0], t[1]))

    return {
        "agent_id": agent_id,
        "team": agent_body.team,
        "agent_x": agent_body.x,
        "agent_y": agent_body.y,
        "vision_radius": vision_radius,
        "visible_enemies": tuple(enemies),
        "visible_allies": tuple(allies),
        "legal_actions": legal_actions,
    }


def comms_enabled_for_team(config, team):
    """Returns True when communications are enabled for the given team"""
    mode = config.comms
    if mode is None:
        return False
    if mode == "both":
        return True
    if mode == "predators":
        return team == co.TEAM_PREDATOR
    if mode == "prey":
        return team == co.TEAM_PREY
    return False


def dedupe_enemy_messages(messages):
    """Deduplicates enemy messages by sender ID and enemy ID. Keeps first occurrence of each message."""

    # Initialize set to store seen keys (sender ID and enemy ID)
    seen_keys = set()
    # Initialize list to store deduplicated messages
    deduped = []
    # Iterate over all messages
    for message in messages:
        # Extract sender ID, enemy position, and enemy ID from message
        sender_id, enemy_x, enemy_y, enemy_id = message
        key = (sender_id, enemy_id)
        # If message has already been seen, skip
        if key in seen_keys:
            continue
        # Add key to seen keys and add message to deduplicated list
        seen_keys.add(key)
        deduped.append((sender_id, enemy_x, enemy_y, enemy_id))
    deduped.sort(key=lambda t: (t[0], t[3], t[1], t[2]))
    return tuple(deduped)


def exchange_team_messages(all_obs, config):
    """Returns dictionary of agent IDs with their received enemy reports.
       Every agent broadcasts the enemies it directly sees this step to all
       teammates inside its own vision radius. Each message is tagged with the sender's ID and
       contains the enemy's position and ID. The messages are deduplicated by sender ID and enemy ID,
       keeping the first occurrence of each message."""
    # Initialize store for messages received by each agent
    shared_enemies = {receiver_id: [] for receiver_id in all_obs}
    # Iterate over all observations to send them to visible allies
    for sender_obs in all_obs.values():
        # If communications are not enabled for the sender's team, skip
        if not comms_enabled_for_team(config, sender_obs["team"]):
            continue
        # Extract enemies directly seen by the sender and tag them with the sender's ID
        enemies = [(sender_obs["agent_id"], enemy_x, enemy_y, enemy_id) for enemy_x, enemy_y, enemy_id in sender_obs["visible_enemies"]]
        # If no enemies were seen by the sender, skip
        if not enemies:
            continue
        # Iterate over all visible allies to send them the messages
        for receiver_x, receiver_y, receiver_id in sender_obs["visible_allies"]:
            # If the ally is not in the shared enemies store, skip
            if receiver_id not in shared_enemies:
                continue
            shared_enemies[receiver_id].extend(enemies)
    # Return deduplicates messages for each agent
    return {
        receiver_id: dedupe_enemy_messages(messages)
        for receiver_id, messages in shared_enemies.items()
    }


def attach_team_comms(all_obs, config):
    """Attaches teammate communicated enemies to each observation"""
    # If communications are enabled, retrieve messages received by each agent
    if config.comms is not None:
        shared_enemies = exchange_team_messages(all_obs, config)
    # If communications are not enabled, each agent gets empty comms
    else:
        shared_enemies = {agent_id: tuple() for agent_id in all_obs}

    # Attached shared enemies to each observation if its team has them enabled
    for agent_id, obs in all_obs.items():
        if comms_enabled_for_team(config, obs["team"]):
            obs["shared_enemies"] = shared_enemies[agent_id]
        else:
            obs["shared_enemies"] = tuple()
