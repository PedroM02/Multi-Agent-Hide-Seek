from __future__ import annotations

import random
from typing import Dict, List, Tuple

import agent_utils as au
from environment import Environment


def _target_cell(env: Environment, agent_id: int, action: str) -> Tuple[int, int]:
    body = env.agent_bodies[agent_id]
    if not body.alive:
        return (body.x, body.y)
    dx, dy = au.ACTION_DELTA[action]
    tx, ty = body.x + dx, body.y + dy
    if not (0 <= tx < env.width and 0 <= ty < env.height):
        return (body.x, body.y)
    if env.is_wall(tx, ty):
        return (body.x, body.y)
    return (tx, ty)


def resolve_actions(
    env: Environment,
    intentions: Dict[int, str],
    rng: random.Random,
) -> dict:
    alive_ids = [bid for bid, b in env.agent_bodies.items() if b.alive]
    targets: Dict[int, Tuple[int, int]] = {}
    for aid in alive_ids:
        act = intentions.get(aid, au.STAY)
        targets[aid] = _target_cell(env, aid, act)

    final_pos: Dict[int, Tuple[int, int]] = {}
    teams = (au.TEAM_PREDATOR, au.TEAM_PREY)

    for team in teams:
        team_ids = [aid for aid in alive_ids if env.agent_bodies[aid].team == team]
        old_pos: Dict[int, Tuple[int, int]] = {
            aid: (env.agent_bodies[aid].x, env.agent_bodies[aid].y) for aid in team_ids
        }

        # First pass: same-team agents cannot claim the same destination.
        by_cell: Dict[Tuple[int, int], List[int]] = {}
        for aid in team_ids:
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
            for aid in team_ids:
                if aid in forced_stay:
                    continue
                if candidate_target.get(aid) in occupied_by_stayers:
                    forced_stay.add(aid)
                    occupied_by_stayers.add(old_pos[aid])
                    changed = True

        for aid in team_ids:
            if aid in forced_stay:
                final_pos[aid] = old_pos[aid]
            else:
                final_pos[aid] = candidate_target[aid]

    for aid, (x, y) in final_pos.items():
        env.set_position(aid, x, y)

    return {"intended": dict(intentions), "final_positions": final_pos}
