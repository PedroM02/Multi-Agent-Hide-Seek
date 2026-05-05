"""Plain string constants and action deltas (no enums/dataclasses)."""

from __future__ import annotations

# Teams
TEAM_PREDATOR = "predator"
TEAM_PREY = "prey"

# Actions
UP = "up"
DOWN = "down"
LEFT = "left"
RIGHT = "right"
STAY = "stay"

ALL_MOVE_ACTIONS = (UP, DOWN, LEFT, RIGHT, STAY)

ACTION_DELTA: dict[str, tuple[int, int]] = {
    UP: (0, -1),
    DOWN: (0, 1),
    LEFT: (-1, 0),
    RIGHT: (1, 0),
    STAY: (0, 0),
}

# Episode outcomes (strings)
OUTCOME_ONGOING = "ongoing"
OUTCOME_PREDATORS_WIN = "predators_win"
OUTCOME_PREY_WIN_TIMEOUT = "prey_win_timeout"
