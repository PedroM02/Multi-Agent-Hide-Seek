"""Simulation constants and position deltas from actions."""


# Teams
TEAM_PREDATOR = "predator"
TEAM_PREY = "prey"

# Actions
UP = "up"
DOWN = "down"
LEFT = "left"
RIGHT = "right"
STAY = "stay"
PICKUP = "pickup"
DROP = "drop"

MOVE_ACTIONS = (UP, DOWN, LEFT, RIGHT, STAY)
ALL_ACTIONS = MOVE_ACTIONS + (PICKUP, DROP)


ACTION_DELTA = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0),
    STAY: (0, 0),
    PICKUP: (0, 0),
    DROP: (0, 0),
}

# Episode outcomes
OUTCOME_ONGOING = "Ongoing"
OUTCOME_PREDATORS_WIN = "Predators Win"
OUTCOME_PREY_WIN = "Prey Win"
