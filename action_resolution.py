import agent_utils as au


def calculate_target_cell(env, agent_id, move_intentions):
    '''Returns the target cell for an agent given its intended action and the intentions of other agents.

       The agent cannot move into a cell if it's out of bounds or on a wall, or if it's occupied by another agent who stayed there
       from the previous step. A prey can never move into a predator's cell but the reverse is possible and will result in capture.
       Conflict between same-team agents who want to move into the same cell is resolved in resolve_actions().'''

    # Retrieve agent body and its intended action
    agent_body = env.agent_bodies[agent_id]
    action = move_intentions[agent_id]
    # If agent is not alive, it stays in place
    if not agent_body.alive:
        return (agent_body.x, agent_body.y)

    # Calculate target coordinates given intended action and its coordinates' deltas
    delta_x, delta_y = au.ACTION_DELTA[action]
    target_x, target_y = agent_body.x + delta_x, agent_body.y + delta_y
    # If target coordinates are out of bounds, agent stays in place
    if not (0 <= target_x < env.width and 0 <= target_y < env.height):
        return (agent_body.x, agent_body.y)
    # If target coordinates are on a wall, agent stays in place
    if env.is_wall(target_x, target_y):
        return (agent_body.x, agent_body.y)

    # Check if target cell is occupied by another agent by iterating over all agent bodies and seeing their positions
    for other_agent_body in env.agent_bodies.values():
        # If the other agent is not alive or is the agent itself, skip
        if not other_agent_body.alive or other_agent_body.agent_id == agent_id:
            continue
        # If the other agent is not in the target cell, skip
        if other_agent_body.x != target_x or other_agent_body.y != target_y:
            continue
        # If the other agent belongs to the same team, check its move intention
        if other_agent_body.team == agent_body.team:
            other_agent_action = move_intentions.get(other_agent_body.agent_id, au.STAY)
            other_agent_delta_x, other_agent_delta_y = au.ACTION_DELTA[other_agent_action]
            # If the other agent is occupying target cell because it is staying, then the agent cannot move into target cell
            if other_agent_delta_x == 0 and other_agent_delta_y == 0:
                return (agent_body.x, agent_body.y)
        else:
            # If agent is a prey and the other agent is a predator, then the agent cannot move into target cell
            # However, the reverse can happen and will result in capture
            if agent_body.team == au.TEAM_PREY:
                return (agent_body.x, agent_body.y)

    return (target_x, target_y)


def resolve_actions(env, intentions):
    '''Resolves the actions of all agents in the environment in one step given their intentions.
       Conflict between same-team agents who want to move into the same cell is resolved with agent with lowest ID having priority.
       If the target cell is occupied by a same-team agent who is staying, then the agent cannot move into the target cell and is
       forced to stay in its current cell. This is cascaded to other same-team agents who are also forced to stay in their current cells.'''

    # Retrieve all alive agents' IDs
    alive_ids = [agent_id for agent_id, agent_body in env.agent_bodies.items() if agent_body.alive]
    # Build dict with move intentions for all alive agents ensuring missing intentions are set to staying
    move_intentions = {agent_id: intentions.get(agent_id, au.STAY) for agent_id in alive_ids}
    # Build dict with target cells for all alive agents given their intended actions
    targets = {agent_id: calculate_target_cell(env, agent_id, move_intentions) for agent_id in move_intentions}

    # Initialize dict for next positions of all alive agents. This will be updated
    next_pos = {agent_id: (env.agent_bodies[agent_id].x, env.agent_bodies[agent_id].y) for agent_id in alive_ids}
    # Iterate over each team and resolve conflicts between same-team agents whose target cells are the same
    teams = (au.TEAM_PREDATOR, au.TEAM_PREY)

    for team in teams:
        # Retrieve agent IDs in current team
        team_ids = [agent_id for agent_id in move_intentions if env.agent_bodies[agent_id].team == team]
        # Retrieve current positions of agents in team
        current_pos = {agent_id: (env.agent_bodies[agent_id].x, env.agent_bodies[agent_id].y) for agent_id in team_ids}

        # Group agents by target cell in dict of form {target_cell: [agent_id1, agent_id2, ...]}
        agents_by_cell = {}
        for agent_id in team_ids:
            target_cell = targets[agent_id]
            if target_cell not in agents_by_cell:
                agents_by_cell[target_cell] = []
            agents_by_cell[target_cell].append(agent_id)

        # Initialize set of agents who are forced to stay in their current cells
        forced_stay = set()
        # Initialize dict for candidate next position cells for each agent. These are reassessed in the cascading flow
        candidate_next_pos = {}
        # Iterate over each target cell and its possibly conflicting agents
        for target_cell, conflicting_agents in agents_by_cell.items():
            # If there is only one agent targeting the cell, there is no conflict and agent can move to target
            if len(conflicting_agents) == 1:
                candidate_next_pos[conflicting_agents[0]] = target_cell
                continue
            # If there is more than one agent targeting the same cell, the one with lowest ID is the winner
            winner = min(conflicting_agents)
            candidate_next_pos[winner] = target_cell
            # Losing agents are forced to stay in their current cells
            for agent_id in conflicting_agents:
                if agent_id != winner:
                    forced_stay.add(agent_id)

        # Iterate over agents in team and check if their candidate next position is occupied by an
        # agent who was forced to stay. If so, agents are also forced to stay. Loop until no more
        # agents are forced to stay.
        cells_occupied_by_stayers = {current_pos[agent_id] for agent_id in forced_stay}
        changed = True
        while changed:
            changed = False
            for agent_id in team_ids:
                if agent_id in forced_stay:
                    continue
                # If agent wanted to move to a cell occupied by an agent who was forced to stay, it too
                # is forced to stay and its cell becomes one occupied by an agent who was forced to stay.
                if candidate_next_pos.get(agent_id) in cells_occupied_by_stayers:
                    forced_stay.add(agent_id)
                    cells_occupied_by_stayers.add(current_pos[agent_id])
                    changed = True

        # Update next positions of agents in team based on whether they were forced to stay or not
        for agent_id in team_ids:
            if agent_id in forced_stay:
                next_pos[agent_id] = current_pos[agent_id]
            else:
                next_pos[agent_id] = candidate_next_pos[agent_id]

    # Update positions of all alive agents in environment
    for agent_id, (x, y) in next_pos.items():
        env.set_position(agent_id, x, y)
