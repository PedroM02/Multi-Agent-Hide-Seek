from __future__ import annotations

import agent_utils as au
from environment import Environment
from distances import chebyshev


def _is_blocked_for_team(env: Environment, x: int, y: int, team: str) -> bool:
    """Return True if cell (x, y) is unenterable for an agent of `team`.

    Off-grid and walls are always blocked. An unheld obstacle blocks
    everyone in symmetric lock mode; in owner-passable mode, an obstacle
    locked to the agent's own team is passable. An obstacle that shares
    its cell with a live agent is treated as non-blocking (capture must
    still be possible); the cell only behaves like a wall once the
    occupant moves off it. Mirrors `action_resolution._target_cell`.
    """
    if not env.is_in_bounds(x, y):
        return True
    if env.is_wall(x, y):
        return True
    for obstacle in env.obstacles.values():
        if obstacle.held_by is None and obstacle.x == x and obstacle.y == y:
            if env.lock_mode == "owner-passable" and obstacle.locked_team == team:
                continue
            if any(
                ab.alive and ab.x == x and ab.y == y
                for ab in env.agent_bodies.values()
            ):
                continue
            return True
    return False


def build_observation(
    env: Environment,
    agent_id: int,
    vision_radius: int,
) -> dict:
    body = env.agent_bodies[agent_id]
    legal = env.legal_actions(agent_id)
    enemies: list[tuple[int, int, int]] = []
    allies: list[tuple[int, int, int]] = []
    if not body.alive:
        return {
            "agent_id": agent_id,
            "team": body.team,
            "ego_x": body.x,
            "ego_y": body.y,
            "vision_radius": vision_radius,
            "visible_enemies": tuple(),
            "visible_allies": tuple(),
            "visible_obstacles": tuple(),
            "held_obstacle": None,
            "legal_actions": legal,
            "on_obstacle_cell": False,
            "cardinal_blocked": (True, True, True, True),
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

    # Each visible obstacle tuple is (x, y, obstacle_id, locked_team) where
    # locked_team is None for unclaimed obstacles, or the owning team string.
    visible_obstacles = tuple(
        (it.x, it.y, it.obstacle_id, it.locked_team)
        for it in env.obstacles.values()
        if it.held_by is None
        and chebyshev(body.x, body.y, it.x, it.y) <= vision_radius
    )

    held = next((it.obstacle_id for it in env.obstacles.values() if it.held_by == agent_id), None)

    # Is the agent currently standing on an unheld obstacle? An agent that
    # has just dropped one is co-located with it. In symmetric lock mode the
    # cell becomes unenterable by anyone (including enemies), which is what
    # makes the prey "hole-up" tactic work.
    on_obstacle_cell = any(
        it.held_by is None and it.x == body.x and it.y == body.y
        for it in env.obstacles.values()
    )

    # cardinal_blocked is a 4-tuple aligned with (UP, DOWN, LEFT, RIGHT)
    # indicating whether each cardinal neighbour cell is unenterable for
    # an agent of this team given the current lock_mode.
    cardinal_blocked = tuple(
        _is_blocked_for_team(
            env,
            body.x + au.ACTION_DELTA[a][0],
            body.y + au.ACTION_DELTA[a][1],
            body.team,
        )
        for a in (au.UP, au.DOWN, au.LEFT, au.RIGHT)
    )

    return {
        "agent_id": agent_id,
        "team": body.team,
        "ego_x": body.x,
        "ego_y": body.y,
        "vision_radius": vision_radius,
        "visible_enemies": tuple(enemies),
        "visible_allies": tuple(allies),
        "visible_obstacles": visible_obstacles,
        "held_obstacle": held,
        "legal_actions": legal,
        "on_obstacle_cell": on_obstacle_cell,
        "cardinal_blocked": cardinal_blocked,
    }
