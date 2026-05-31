import constants as co

# Distances
def manhattan(first_x, first_y, second_x, second_y):
    '''Calculates the Manhattan distance between two points'''
    return abs(first_x - second_x) + abs(first_y - second_y)


def chebyshev(first_x, first_y, second_x, second_y):
    '''Calculates the Chebyshev distance between two points'''
    return max(abs(first_x - second_x), abs(first_y - second_y))

def bfs_search(start, goal, width, height, wall_cells):
    """Shortest path search from start to goal. Returns the distance and the immediately previous cell of the goal along the path.
       Returns None if the goal is unreachable."""

    goal_x, goal_y = goal

    # If start and goal are the same, the distance is 0
    if start == goal:
        return 0, {start: None}
     # If goal is out of bounds or on a wall, no distance is defined
    if not in_grid_bounds(goal_x, goal_y, width, height) or (goal_x, goal_y) in wall_cells:
        return None

    # Initialize previous cells, distance, furthest cells (cells on the outermost edge of the search), and visited cells
    previous_cells = {start: None}
    distance = 0
    furthest_cells = [start]
    visited_cells = {start}

    # Iterate over increasingly further cells from the start until goal is reached
    while furthest_cells:
        distance += 1
        next_furthest_cells = []
        # For every cell on the outermost edge of the search, check all its neighbors and whether any of them is the goal
        for x, y in furthest_cells:
            for neighbor_x, neighbor_y in grid_neighbors(x, y):
                # If neighbor is the goal, return the distance and the immediately previous cell
                if (neighbor_x, neighbor_y) == goal:
                    previous_cells[goal] = (x, y)
                    return distance, previous_cells
                # Skip if neighbor is out of bounds, on wall, or already visited
                if not in_grid_bounds(neighbor_x, neighbor_y, width, height):
                    continue
                if (neighbor_x, neighbor_y) in wall_cells:
                    continue
                if (neighbor_x, neighbor_y) in visited_cells:
                    continue
                # Add neighbor to visited cells and furthest cells, and register previous cell for the neighbor
                visited_cells.add((neighbor_x, neighbor_y))
                next_furthest_cells.append((neighbor_x, neighbor_y))
                previous_cells[(neighbor_x, neighbor_y)] = (x, y)
        # Move onto the next outermost edge of the search
        furthest_cells = next_furthest_cells
    # If goal is not reachable, return None
    return None


def bfs_best_distance_greedy(start, goal, width, height, wall_cells, legal_actions, rng):
    """Fallback when the first step on the shortest path from start to goal is not a legal action.
       Returns the action that minimizes the immediate distance to the goal."""
    start_x, start_y = start
    best_actions = []
    best_distance = None

    # Evaluate the shortest path from where each legal action lands to the goal
    for action in legal_actions:
        neighbor_x, neighbor_y = apply_action(start_x, start_y, action)
        # Compute shortest path from the neighbor to the goal
        search = bfs_search((neighbor_x, neighbor_y), goal, width, height, wall_cells,)
        # If the path is not found, skip this action
        if search is None:
            continue
        else:
            distance, previous_cells = search
        # If this action leads to a shorter path, update the best action to take and its distance
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_actions = [action]
        # If this action leads to the same distance as the best action so far, add to possible actions
        elif distance == best_distance:
            best_actions.append(action)
    
    # If there are more than one possible best actions, return a random one
    if best_actions:
        return rng.choice(best_actions)
    # If there are no possible best actions, return a random legal action
    return rng.choice(legal_actions)


def bfs_first_step(start, goal, width, height, wall_cells, legal_actions, rng):
    """Returns a legal action that follows one step along a shortest path from start to goal.
       If no legal action is possible, returns the action that minimizes the immediate distance."""
    # Failsafe: if there are no legal actions, return STAY
    if not legal_actions:
        return co.STAY
    # Failsafe: if start and goal are the same, return STAY if STAY is legal, otherwise return a random legal action
    if start == goal:
        return co.STAY if co.STAY in legal_actions else rng.choice(legal_actions)

    # Search for the shortest path from start to goal
    search = bfs_search(start, goal, width, height, wall_cells)
    # If the path is not found, return the action that minimizes the immediate distance
    if search is None:
        return bfs_best_distance_greedy(start, goal, width, height, wall_cells, legal_actions, rng)

    # Get the distance and the previous cells' mapping along the path
    distance, previous_cells = search

    # Iterate from goal backwards to start by following the shortest path, using the previous cells' mapping
    current = goal
    while previous_cells[current] is not None and previous_cells[current] != start:
        current = previous_cells[current]
    
    # Retrieve first step and start coordinates
    first_step_x, first_step_y = current
    start_x, start_y = start
    # Check which legal action leads to first step from start
    for action in legal_actions:
        if apply_action(start_x, start_y, action) == (first_step_x, first_step_y):
            return action
    # If no legal action leads to first step, return the action that minimizes the immediate distance
    return bfs_best_distance_greedy(start, goal, width, height, wall_cells, legal_actions, rng)


# Grid info
def grid_neighbors(x, y):
    '''Returns the four adjacent cells to the given cell'''
    neighbors = []
    for action in (co.UP, co.DOWN, co.LEFT, co.RIGHT):
        neighbors.append(apply_action(x, y, action))
    return tuple(neighbors)


def in_grid_bounds(x, y, width, height):
    '''Checks if the given cell is within the grid bounds'''
    return 0 <= x < width and 0 <= y < height


# Action
def apply_action(x, y, action):
    '''Applies the given action to the given cell and returns the resulting cell'''
    delta_x, delta_y = co.ACTION_DELTA[action]
    return x + delta_x, y + delta_y





