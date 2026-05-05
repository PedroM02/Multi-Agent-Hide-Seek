from __future__ import annotations

import random
from typing import Dict, List, Tuple

import game_types as gt
from environment import Environment


def _target_cell(env: Environment, agent_id: int, action: str) -> Tuple[int, int]:
    body = env.bodies[agent_id]
    if not body.alive:
        return (body.x, body.y)
    dx, dy = gt.ACTION_DELTA[action]
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
    alive_ids = [bid for bid, b in env.bodies.items() if b.alive]
    targets: Dict[int, Tuple[int, int]] = {}
    for aid in alive_ids:
        act = intentions.get(aid, gt.STAY)
        targets[aid] = _target_cell(env, aid, act)

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
