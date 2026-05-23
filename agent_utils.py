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

MOVE_ACTIONS = (UP, DOWN, LEFT, RIGHT, STAY)


ACTION_DELTA = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0),
    STAY: (0, 0),
}

# Episode outcomes
OUTCOME_ONGOING = "Ongoing"
OUTCOME_PREDATORS_WIN = "Predators Win"
OUTCOME_PREY_WIN = "Prey Win"


# Predator decision modes
MODE_RANDOM = "random"
MODE_CHASE = "chase"
MODE_ROLES = "roles"
MODE_OPTIMAL = "optimal"
MODE_PACK = "pack"

# Agent roles. Active only when mode is "roles"
ROLE_CHASER = "chaser"
ROLE_FLEE = "flee"
ROLE_FLANKER = "flanker"

ROLE_LETTER = {
    ROLE_CHASER: "C",
    ROLE_FLEE: "F",
    ROLE_FLANKER: "K",
}
