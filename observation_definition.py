from __future__ import annotations

from environment import Environment
from distances import chebyshev


def build_observation(
    env: Environment,
    agent_id: int,
    vision_radius: int,
) -> dict:
    body = env.bodies[agent_id]
    legal = env.legal_actions_for(agent_id)
    enemies: list[tuple[int, int, int]] = []
    if not body.alive:
        return {
            "agent_id": agent_id,
            "team": body.team,
            "ego_x": body.x,
            "ego_y": body.y,
            "vision_radius": vision_radius,
            "visible_enemies": tuple(),
            "visible_obstacles": tuple(),
            "held_obstacle": None,
            "legal_actions": legal,
        }
    my_team = body.team
    for other in env.bodies.values():
        if other.agent_id == agent_id or not other.alive:
            continue
        if other.team == my_team:
            continue
        if chebyshev(body.x, body.y, other.x, other.y) <= vision_radius:
            enemies.append((other.x, other.y, other.agent_id))
    enemies.sort(key=lambda t: (t[2], t[0], t[1]))

    visible_obstacles = tuple(
        (it.x, it.y, it.obstacle_id)
        for it in env.obstacles.values()
        if it.held_by is None
        and _chebyshev_dist(body.x, body.y, it.x, it.y) <= vision_radius
    )

    held = next((it.obstacle_id for it in env.obstacles.values() if it.held_by == agent_id), None)

    return {
        "agent_id": agent_id,
        "team": body.team,
        "ego_x": body.x,
        "ego_y": body.y,
        "vision_radius": vision_radius,
        "visible_enemies": tuple(enemies),
        "visible_obstacles": visible_obstacles,
        "held_obstacle": held,
        "legal_actions": legal,
    }
