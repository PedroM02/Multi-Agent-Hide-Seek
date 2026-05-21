from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, Optional, Set, Tuple

import agent_utils as au

Coord = Tuple[int, int]


def manhattan(x1: int, y1: int, x2: int, y2: int) -> int:
    return abs(x1 - x2) + abs(y1 - y2)


def chebyshev(x1: int, y1: int, x2: int, y2: int) -> int:
    return max(abs(x1 - x2), abs(y1 - y2))


def _grid_neighbors(x: int, y: int) -> Iterable[Coord]:
    for action in (au.UP, au.DOWN, au.LEFT, au.RIGHT):
        dx, dy = au.ACTION_DELTA[action]
        yield x + dx, y + dy


def _in_grid_bounds(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


def bfs_distance(
    start: Coord,
    goal: Coord,
    width: int,
    height: int,
    wall_cells: Set[Coord],
) -> Optional[int]:
    """Shortest 4-neighbor path length from `start` to `goal`, or None if unreachable."""
    if start == goal:
        return 0
    gx, gy = goal
    if not _in_grid_bounds(gx, gy, width, height) or (gx, gy) in wall_cells:
        return None

    q: deque[Coord] = deque([start])
    dist: Dict[Coord, int] = {start: 0}
    while q:
        x, y = q.popleft()
        d = dist[(x, y)]
        for nx, ny in _grid_neighbors(x, y):
            if not _in_grid_bounds(nx, ny, width, height):
                continue
            if (nx, ny) in wall_cells:
                continue
            if (nx, ny) in dist:
                continue
            nd = d + 1
            if (nx, ny) == goal:
                return nd
            dist[(nx, ny)] = nd
            q.append((nx, ny))
    return None
