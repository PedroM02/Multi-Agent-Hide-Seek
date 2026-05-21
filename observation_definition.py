from environment import Environment
from distances import chebyshev


def build_observation(env, agent_id, vision_radius):
    body = env.agent_bodies[agent_id]
    legal_actions = env.legal_actions(agent_id)
    enemies = []
    allies = []
    if not body.alive:
        return {
            "agent_id": agent_id,
            "team": body.team,
            "agent_x": body.x,
            "agent_y": body.y,
            "vision_radius": vision_radius,
            "visible_enemies": tuple(),
            "visible_allies": tuple(),
            "legal_actions": legal_actions,
        }
    my_team = body.team
    for other in env.agent_bodies.values():
        if other.agent_id == agent_id or not other.alive:
            continue
        if chebyshev(body.x, body.y, other.x, other.y) > vision_radius:
            continue
        if other.team == my_team:
            allies.append((other.x, other.y, other.agent_id))
        else:
            enemies.append((other.x, other.y, other.agent_id))
    enemies.sort(key=lambda t: (t[2], t[0], t[1]))
    allies.sort(key=lambda t: (t[2], t[0], t[1]))

    return {
        "agent_id": agent_id,
        "team": body.team,
        "agent_x": body.x,
        "agent_y": body.y,
        "vision_radius": vision_radius,
        "visible_enemies": tuple(enemies),
        "visible_allies": tuple(allies),
        "legal_actions": legal_actions,
    }
