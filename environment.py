from __future__ import annotations

from typing import Iterator, Optional, Sequence, Set, Tuple

import game_types as gt


class AgentBody:
    def __init__(self, agent_id: int, team: str, x: int, y: int, alive: bool = True) -> None:
        self.agent_id = agent_id
        self.team = team
        self.x = x
        self.y = y
        self.alive = alive


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
            b = AgentBody(aid, gt.TEAM_PREDATOR, x, y)
            self.place_agent(b)
            bodies.append(b)
            aid += 1
        for _ in range(num_prey):
            x, y = free[idx]
            idx += 1
            b = AgentBody(aid, gt.TEAM_PREY, x, y)
            self.place_agent(b)
            bodies.append(b)
            aid += 1
        return bodies

    def alive_bodies(self) -> Iterator[AgentBody]:
        for b in self.bodies.values():
            if b.alive:
                yield b

    def legal_actions_for(self, agent_id: int) -> tuple[str, ...]:
        body = self.bodies[agent_id]
        if not body.alive:
            return (gt.STAY,)
        out: list[str] = []
        for act in gt.ALL_MOVE_ACTIONS:
            dx, dy = gt.ACTION_DELTA[act]
            tx, ty = body.x + dx, body.y + dy
            if not self._in_bounds(tx, ty) or self.is_wall(tx, ty):
                if act == gt.STAY:
                    out.append(gt.STAY)
                continue
            out.append(act)
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
            preds = [b for b in group if b.team == gt.TEAM_PREDATOR]
            preys = [b for b in group if b.team == gt.TEAM_PREY]
            if preds and preys:
                for p in preys:
                    p.alive = False
                    captured.append(p.agent_id)
        return captured

    def any_prey_alive(self) -> bool:
        return any(b.team == gt.TEAM_PREY and b.alive for b in self.bodies.values())

    def any_predator_alive(self) -> bool:
        return any(b.team == gt.TEAM_PREDATOR and b.alive for b in self.bodies.values())
