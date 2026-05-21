import agent_utils as au


def manhattan(first_x, first_y, second_x, second_y):
    return abs(first_x - second_x) + abs(first_y - second_y)


def chebyshev(first_x, first_y, second_x, second_y):
    return max(abs(first_x - second_x), abs(first_y - second_y))


def _grid_neighbors(x, y):
    for action in (au.UP, au.DOWN, au.LEFT, au.RIGHT):
        delta_x, delta_y = au.ACTION_DELTA[action]
        yield x + delta_x, y + delta_y


def _in_grid_bounds(x, y, width, height):
    return 0 <= x < width and 0 <= y < height


def bfs_distance(start, goal, width, height, wall_cells):
    """Shortest 4-neighbor path length from `start` to `goal`, or None if unreachable."""
    if start == goal:
        return 0
    goal_x, goal_y = goal
    if not _in_grid_bounds(goal_x, goal_y, width, height) or (goal_x, goal_y) in wall_cells:
        return None

    queue = [start]
    distances = {start: 0}
    while queue:
        x, y = queue.pop(0)
        distance = distances[(x, y)]
        for neighbor_x, neighbor_y in _grid_neighbors(x, y):
            if not _in_grid_bounds(neighbor_x, neighbor_y, width, height):
                continue
            if (neighbor_x, neighbor_y) in wall_cells:
                continue
            if (neighbor_x, neighbor_y) in distances:
                continue
            new_distance = distance + 1
            if (neighbor_x, neighbor_y) == goal:
                return new_distance
            distances[(neighbor_x, neighbor_y)] = new_distance
            queue.append((neighbor_x, neighbor_y))
    return None
