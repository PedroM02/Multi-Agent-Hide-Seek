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


# Agent roles. Currently only the two defaults (predators chase, prey
# flee) are wired into the team role selector — see
# `decision_making.select_team_roles`. Additional roles (hunting and
# protection variants) will be added back as the selector grows.
ROLE_CHASER = "chaser"
ROLE_FLEE = "flee"

ROLE_LETTER = {
    ROLE_CHASER: "C",
    ROLE_FLEE: "F",
}
