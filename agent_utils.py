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


# Agent roles. Roles drive obstacle strategy in decision_making.choose_action.
# Predator roles
ROLE_CHASER = "chaser"
ROLE_FLANKER = "flanker"
ROLE_NET = "net"
# Prey roles
ROLE_FLEE = "flee"
ROLE_BREADCRUMB = "breadcrumb"
ROLE_SHIELDER = "shielder"
ROLE_BUNKER = "bunker"
ROLE_FUNNELER = "funneler"

ROLE_LETTER = {
    ROLE_CHASER: "C",
    ROLE_FLANKER: "F",
    ROLE_NET: "N",
    ROLE_FLEE: "-",
    ROLE_BREADCRUMB: "R",
    ROLE_SHIELDER: "S",
    ROLE_BUNKER: "K",
    ROLE_FUNNELER: "U",
}
