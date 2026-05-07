from __future__ import annotations

from typing import Iterator, Optional, Sequence, Set, Tuple

import agent_utils as au


class AgentBody:
    def __init__(self, agent_id: int, team: str, x: int, y: int, alive: bool = True) -> None:
        self.agent_id = agent_id
        self.team = team
        self.x = x
        self.y = y
        self.alive = alive

class Obstacle:
    def __init__(self, obstacle_id, x, y):
        self.obstacle_id = obstacle_id
        self.x = x
        self.y = y
        self.held_by: int | None = None

class Environment:
    """Playable rectangle [0, width) x [0, height); optional obstacle cells. Out-of-bounds moves are blocked elsewhere."""

    def __init__(
        self,
        width: int,
        height: int,
        walls: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        if width < 1 or height < 1:
            raise ValueError("width and height must be at least 1 (playable cell counts).")
        self.width = width
        self.height = height
        self.wall_cells: Set[Tuple[int, int]] = set()
        if walls:
            for wx, wy in walls:
                if self._in_bounds(wx, wy):
                    self.wall_cells.add((wx, wy))
        self.bodies: dict[int, AgentBody] = {}
        self.obstacles: dict[int, Obstacle] = {}

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_wall(self, x: int, y: int) -> bool:
        return (x, y) in self.wall_cells

    def free_cells(self) -> list[Tuple[int, int]]:
        out: list[Tuple[int, int]] = []
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) not in self.wall_cells:
                    out.append((x, y))
        return out

    def clear_agents(self) -> None:
        self.bodies.clear()

    def place_agent(self, body: AgentBody) -> None:
        if not self._in_bounds(body.x, body.y):
            raise ValueError("Agent out of bounds")
        if self.is_wall(body.x, body.y):
            raise ValueError("Cannot place agent on wall")
        self.bodies[body.agent_id] = body

    def reset_agent_positions_random(
        self,
        num_predators: int,
        num_prey: int,
        rng,
        next_id_start: int = 0,
    ) -> list[AgentBody]:
        self.clear_agents()
        free = self.free_cells()
        if len(free) < num_predators + num_prey:
            raise ValueError("Not enough free cells for all agents.")
        rng.shuffle(free)
        bodies: list[AgentBody] = []
        idx = 0
        aid = next_id_start
        for _ in range(num_predators):
            x, y = free[idx]
            idx += 1
            b = AgentBody(aid, au.TEAM_PREDATOR, x, y)
            self.place_agent(b)
            bodies.append(b)
            aid += 1
        for _ in range(num_prey):
            x, y = free[idx]
            idx += 1
            b = AgentBody(aid, au.TEAM_PREY, x, y)
            self.place_agent(b)
            bodies.append(b)
            aid += 1
        return bodies
    
    def place_obstacle_random(self, n: int, rng) -> None:
        self.obstacles.clear()
        occupied = {(b.x, b.y) for b in self.bodies.values()}
        free = [c for c in self.free_cells() if c not in occupied]
        rng.shuffle(free)
        for i, (x, y) in enumerate(free[:n]):
            self.obstacles[i] = Obstacle(obstacle_id=i, x=x, y=y)

            
    def pickup_obstacle(self, agent_id: int) -> None:
        body = self.bodies[agent_id]
        already_holding = any(it.held_by == agent_id for it in self.obstacles.values())
        if already_holding:
            return
        for obstacle in self.obstacles.values():
            if obstacle.held_by is None:
                continue
            if abs(obstacle.x - body.x) <= 1 and abs(obstacle.y - body.y) <= 1 and (obstacle.x != body.x or obstacle.y != body.y):
                obstacle.held_by = agent_id
                return
    
    def drop_obstacle(self, agent_id: int) -> None:
        body = self.bodies[agent_id]
        for obstacle in self.obstacles.values():
            if obstacle.held_by == agent_id:
                self.obstacle.held_by = None
                obstacle.x = body.x
                obstacle.y = body.y
                return

    def alive_bodies(self) -> Iterator[AgentBody]:
        for b in self.bodies.values():
            if b.alive:
                yield b

    def legal_actions_for(self, agent_id: int) -> tuple[str, ...]:
        body = self.bodies[agent_id]
        if not body.alive:
            return (au.STAY,)
        out: list[str] = []
        for act in au.MOVE_ACTIONS:
            dx, dy = au.ACTION_DELTA[act]
            tx, ty = body.x + dx, body.y + dy
            if not self._in_bounds(tx, ty) or self.is_wall(tx, ty):
                if act == au.STAY:
                    out.append(au.STAY)
                continue
            out.append(act)
        # pickup if an unclaimed obstacle is here
        if any(
            it.held_by is None 
            and abs(it.x - body.x) <= 1 and (it.y - body.y) <= 1
            and (it.x != body.x or it.y != body.y)
            for it in self.obstacles.values()):
            out.append(gt.PICKUP)
        # drop if holding something
        if any(it.held_by == body.agent_id for it in self.obstacles.values()):
            out.append(gt.DROP)     
        return tuple(out)

    def set_position(self, agent_id: int, x: int, y: int) -> None:
        b = self.bodies[agent_id]
        b.x, b.y = x, y

    def apply_captures(self) -> list[int]:
        by_cell: dict[Tuple[int, int], list[AgentBody]] = {}
        for b in self.alive_bodies():
            by_cell.setdefault((b.x, b.y), []).append(b)
        captured: list[int] = []
        for _cell, group in by_cell.items():
            preds = [b for b in group if b.team == au.TEAM_PREDATOR]
            preys = [b for b in group if b.team == au.TEAM_PREY]
            if preds and preys:
                for p in preys:
                    p.alive = False
                    captured.append(p.agent_id)
        return captured

    def any_prey_alive(self) -> bool:
        return any(b.team == au.TEAM_PREY and b.alive for b in self.bodies.values())

    def any_predator_alive(self) -> bool:
        return any(b.team == au.TEAM_PREDATOR and b.alive for b in self.bodies.values())
