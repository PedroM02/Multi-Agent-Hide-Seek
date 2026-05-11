from __future__ import annotations

import random
from typing import Dict, List, Tuple

import agent_utils as au
from environment import Environment


def _target_cell(
    env: Environment,
    agent_id: int,
    action: str,
    move_intentions: Dict[int, str],
) -> Tuple[int, int]:
    body = env.agent_bodies[agent_id]
    if not body.alive:
        return (body.x, body.y)
    dx, dy = au.ACTION_DELTA[action]
    tx, ty = body.x + dx, body.y + dy
    if not (0 <= tx < env.width and 0 <= ty < env.height):
        return (body.x, body.y)
    if env.is_wall(tx, ty):
        return (body.x, body.y)

    # Agent occupancy of the target cell. Two rules to encode:
    #   1. Same-team agents never share a cell. If a teammate is staying
    #      here this step (literal STAY, or PICKUP / DROP — neither of
    #      which appears in `move_intentions` and so defaults to STAY),
    #      we deny the entry now. If the teammate is moving away we
    #      leave the decision to the same-team `forced_stay` propagation
    #      later in resolve_actions, which already cascades blocks if
    #      the teammate ends up stuck.
    #   2. The only legal cross-team co-location is a predator stepping
    #      onto a prey cell to capture. Prey moving onto a predator is
    #      always rejected.
    for ab in env.agent_bodies.values():
        if not ab.alive or ab.agent_id == agent_id:
            continue
        if ab.x != tx or ab.y != ty:
            continue
        if ab.team == body.team:
            ab_act = move_intentions.get(ab.agent_id, au.STAY)
            ddx, ddy = au.ACTION_DELTA[ab_act]
            if ddx == 0 and ddy == 0:
                return (body.x, body.y)
            # Teammate is moving away — do not block on the agent alone.
            # Fall through to the obstacle/simultaneity check below: if
            # there is an unheld obstacle co-located with the leaving
            # teammate (e.g. they dropped last step and are now fleeing),
            # `cell_stays_occupied` will be False and the obstacle will
            # re-assert as a wall, correctly denying the entry. With no
            # obstacle the entrant moves into the vacated cell, and any
            # cascading block from a failed teammate move is picked up
            # later by the same-team `forced_stay` propagation in
            # resolve_actions.
        else:
            if body.team == au.TEAM_PREY:
                return (body.x, body.y)
            # Predator entering an enemy cell: allowed (capture). The
            # obstacle/simultaneity check below still applies — if the
            # prey is leaving the cell the same step, the cell reverts
            # to obstacle-as-wall and entry is denied there.

    for obstacle in env.obstacles.values():
        if obstacle.held_by is None and obstacle.x == tx and obstacle.y == ty:
            # In owner-passable lock mode, members of the team that owns the
            # obstacle can walk through it; in the default symmetric mode
            # obstacles always block movement.
            if env.lock_mode == "owner-passable" and obstacle.locked_team == body.team:
                continue
            # The cell is treated as enterable (capture-by-stepping-in
            # works) only when at least one live agent at the cell is
            # *staying* this step. An occupant whose move intention has a
            # non-zero delta is leaving — by the time this step resolves
            # the obstacle would be alone on the cell and back to acting
            # as a wall, so we deny the move now rather than letting the
            # mover land on an enemy-locked obstacle.
            # Pickup/drop actions don't appear in `move_intentions` (they
            # were resolved earlier in resolve_actions and don't change
            # the agent's cell), so `.get(..., au.STAY)` correctly treats
            # those occupants as staying.
            cell_stays_occupied = False
            for ab in env.agent_bodies.values():
                if not (ab.alive and ab.x == tx and ab.y == ty):
                    continue
                ab_act = move_intentions.get(ab.agent_id, au.STAY)
                ddx, ddy = au.ACTION_DELTA[ab_act]
                if ddx == 0 and ddy == 0:
                    cell_stays_occupied = True
                    break
            if cell_stays_occupied:
                continue
            return (body.x, body.y)
    return (tx, ty)


def resolve_actions(
    env: Environment,
    intentions: Dict[int, str],
    rng: random.Random,
) -> dict:
    alive_ids = [bid for bid, b in env.agent_bodies.items() if b.alive]

    move_intentions: Dict[int, str] = {}
    for aid in alive_ids:
        act = intentions.get(aid, au.STAY)
        if act == au.PICKUP:
            env.pickup_obstacle(aid)
        elif act == au.DROP:
            env.drop_obstacle(aid)
        else:
            move_intentions[aid] = act  


    targets: Dict[int, Tuple[int, int]] = {}
    for aid in move_intentions:
        targets[aid] = _target_cell(env, aid, move_intentions[aid], move_intentions)

    # Agents that perform PICKUP/DROP (or STAY) do not participate in
    # move-collision resolution; they remain at their current cells.
    final_pos: Dict[int, Tuple[int, int]] = {
        aid: (env.agent_bodies[aid].x, env.agent_bodies[aid].y) for aid in alive_ids
    }
    teams = (au.TEAM_PREDATOR, au.TEAM_PREY)

    for team in teams:
        team_move_ids = [
            aid for aid in move_intentions if env.agent_bodies[aid].team == team
        ]
        old_pos: Dict[int, Tuple[int, int]] = {
            aid: (env.agent_bodies[aid].x, env.agent_bodies[aid].y) for aid in team_move_ids
        }

        # First pass: same-team agents cannot claim the same destination.
        by_cell: Dict[Tuple[int, int], List[int]] = {}
        for aid in team_move_ids:
            by_cell.setdefault(targets[aid], []).append(aid)

        forced_stay: set[int] = set()
        candidate_target: Dict[int, Tuple[int, int]] = {}
        for cell, claimants in by_cell.items():
            if len(claimants) == 1:
                candidate_target[claimants[0]] = cell
                continue
            winner = rng.choice(claimants)
            candidate_target[winner] = cell
            for aid in claimants:
                if aid != winner:
                    forced_stay.add(aid)

        # Second pass: propagate fallback blocking.
        # If A is forced to stay at cell X, any teammate trying to move to X
        # must also stay; this can cascade.
        occupied_by_stayers = {old_pos[aid] for aid in forced_stay}
        changed = True
        while changed:
            changed = False
            for aid in team_move_ids:
                if aid in forced_stay:
                    continue
                if candidate_target.get(aid) in occupied_by_stayers:
                    forced_stay.add(aid)
                    occupied_by_stayers.add(old_pos[aid])
                    changed = True

        for aid in team_move_ids:
            if aid in forced_stay:
                final_pos[aid] = old_pos[aid]
            else:
                final_pos[aid] = candidate_target[aid]

    for aid, (x, y) in final_pos.items():
        env.set_position(aid, x, y)

    return {"intended": dict(intentions), "final_positions": final_pos}
