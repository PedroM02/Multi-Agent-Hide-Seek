from __future__ import annotations

import random
from typing import Dict, List, Tuple

import agent_utils as au
from environment import Environment


def _target_cell(env: Environment, agent_id: int, action: str) -> Tuple[int, int]:
    body = env.bodies[agent_id]
    if not body.alive:
        return (body.x, body.y)
    dx, dy = au.ACTION_DELTA[action]
    tx, ty = body.x + dx, body.y + dy
    if not (0 <= tx < env.width and 0 <= ty < env.height):
        return (body.x, body.y)
    if env.is_wall(tx, ty):
        return (body.x, body.y)
    for obstacle in env.obstacles.values():
        if obstacle.held_by is None and obstacle.x == tx and obstacle.y == ty:
            return (body.x, body.y)
    return (tx, ty)


def resolve_actions(
    env: Environment,
    intentions: Dict[int, str],
    rng: random.Random,
) -> dict:
    alive_ids = [bid for bid, b in env.bodies.items() if b.alive]

    move_intentions: Dict[int, str] = {}
    for aid in alive_ids:
        act = intentions.get(aid, gt.STAY)
        if act == gt.PICKUP:
            env.pickup_obstacle(aid)
        elif act == gt.DROP:
            env.drop_obstacle(aid)
        else:
            move_intentions[aid] = act  


    targets: Dict[int, Tuple[int, int]] = {}
    for aid in move_intentions:
        targets[aid] = _target_cell(env, aid, move_intentions[aid])

    by_cell: Dict[Tuple[int, int], List[int]] = {}
    for aid, cell in targets.items():
        by_cell.setdefault(cell, []).append(aid)

    final_pos: Dict[int, Tuple[int, int]] = {}
    for cell, claimants in by_cell.items():
        if len(claimants) == 1:
            final_pos[claimants[0]] = cell
            continue
        winner = rng.choice(claimants)
        for aid in claimants:
            if aid == winner:
                final_pos[aid] = cell
            else:
                b = env.bodies[aid]
                final_pos[aid] = (b.x, b.y)

    for aid, (x, y) in final_pos.items():
        env.set_position(aid, x, y)

    return {"intended": dict(intentions), "final_positions": final_pos}
