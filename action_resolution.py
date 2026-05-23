import agent_utils as au


def _target_cell(env, agent_id, action, move_intentions):
    body = env.agent_bodies[agent_id]
    if not body.alive:
        return (body.x, body.y)
    delta_x, delta_y = au.ACTION_DELTA[action]
    target_x, target_y = body.x + delta_x, body.y + delta_y
    if not (0 <= target_x < env.width and 0 <= target_y < env.height):
        return (body.x, body.y)
    if env.is_wall(target_x, target_y):
        return (body.x, body.y)

    # Agent occupancy of the target cell. Two rules to encode:
    #   1. Same-team agents never share a cell. If a teammate is staying
    #      here this step (literal STAY — there are no other zero-delta
    #      actions in the current action set), deny the entry now. If
    #      the teammate is moving away we leave the decision to the
    #      same-team `forced_stay` propagation later in resolve_actions,
    #      which already cascades blocks if the teammate ends up stuck.
    #   2. The only legal cross-team co-location is a predator stepping
    #      onto a prey cell to capture. Prey moving onto a predator is
    #      always rejected.
    for other_body in env.agent_bodies.values():
        if not other_body.alive or other_body.agent_id == agent_id:
            continue
        if other_body.x != target_x or other_body.y != target_y:
            continue
        if other_body.team == body.team:
            occupant_action = move_intentions.get(other_body.agent_id, au.STAY)
            occupant_delta_x, occupant_delta_y = au.ACTION_DELTA[occupant_action]
            if occupant_delta_x == 0 and occupant_delta_y == 0:
                return (body.x, body.y)
        else:
            if body.team == au.TEAM_PREY:
                return (body.x, body.y)

    return (target_x, target_y)


def resolve_actions(env, intentions):
    alive_ids = [
        agent_id for agent_id, body in env.agent_bodies.items() if body.alive
    ]

    move_intentions = {
        agent_id: intentions.get(agent_id, au.STAY) for agent_id in alive_ids
    }

    targets = {}
    for agent_id in move_intentions:
        targets[agent_id] = _target_cell(
            env, agent_id, move_intentions[agent_id], move_intentions,
        )

    final_pos = {
        agent_id: (env.agent_bodies[agent_id].x, env.agent_bodies[agent_id].y)
        for agent_id in alive_ids
    }
    teams = (au.TEAM_PREDATOR, au.TEAM_PREY)

    for team in teams:
        team_move_ids = [
            agent_id
            for agent_id in move_intentions
            if env.agent_bodies[agent_id].team == team
        ]
        old_pos = {
            agent_id: (env.agent_bodies[agent_id].x, env.agent_bodies[agent_id].y)
            for agent_id in team_move_ids
        }

        # First pass: same-team agents cannot claim the same destination.
        # Ties go to the lowest agent id (deterministic convention).
        by_cell = {}
        for agent_id in team_move_ids:
            by_cell.setdefault(targets[agent_id], []).append(agent_id)

        forced_stay = set()
        candidate_target = {}
        for cell, claimants in by_cell.items():
            if len(claimants) == 1:
                candidate_target[claimants[0]] = cell
                continue
            winner = min(claimants)
            candidate_target[winner] = cell
            for agent_id in claimants:
                if agent_id != winner:
                    forced_stay.add(agent_id)

        # Second pass: propagate fallback blocking.
        # If A is forced to stay at cell X, any teammate trying to move to X
        # must also stay; this can cascade.
        occupied_by_stayers = {old_pos[agent_id] for agent_id in forced_stay}
        changed = True
        while changed:
            changed = False
            for agent_id in team_move_ids:
                if agent_id in forced_stay:
                    continue
                if candidate_target.get(agent_id) in occupied_by_stayers:
                    forced_stay.add(agent_id)
                    occupied_by_stayers.add(old_pos[agent_id])
                    changed = True

        for agent_id in team_move_ids:
            if agent_id in forced_stay:
                final_pos[agent_id] = old_pos[agent_id]
            else:
                final_pos[agent_id] = candidate_target[agent_id]

    for agent_id, (x, y) in final_pos.items():
        env.set_position(agent_id, x, y)
