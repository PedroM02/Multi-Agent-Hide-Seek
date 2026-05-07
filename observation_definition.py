from __future__ import annotations

from environment import Environment
from distances import chebyshev


def build_observation(
    env: Environment,
    agent_id: int,
    vision_radius: int,
) -> dict:
    body = env.agent_bodies[agent_id]
    legal = env.legal_actions(agent_id)
    enemies: list[tuple[int, int, int]] = []
    if not body.alive:
        return {
            "agent_id": agent_id,
            "team": body.team,
            "ego_x": body.x,
            "ego_y": body.y,
            "vision_radius": vision_radius,
            "visible_enemies": tuple(),
            "legal_actions": legal,
        }
    my_team = body.team
    for other in env.agent_bodies.values():
        if other.agent_id == agent_id or not other.alive:
            continue
        if other.team == my_team:
            continue
        if chebyshev(body.x, body.y, other.x, other.y) <= vision_radius:
            enemies.append((other.x, other.y, other.agent_id))
    enemies.sort(key=lambda t: (t[2], t[0], t[1]))
    return {
        "agent_id": agent_id,
        "team": body.team,
        "ego_x": body.x,
        "ego_y": body.y,
        "vision_radius": vision_radius,
        "visible_enemies": tuple(enemies),
        "legal_actions": legal,
    }
