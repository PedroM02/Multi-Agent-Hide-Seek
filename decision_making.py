from __future__ import annotations

import random

import game_types as gt
from perception import Perception


def _manhattan(ax: int, ay: int, bx: int, by: int) -> int:
    return abs(ax - bx) + abs(ay - by)


def _apply_action(x: int, y: int, action: str) -> tuple[int, int]:
    dx, dy = gt.ACTION_DELTA[action]
    return x + dx, y + dy


class DecisionMaking:
    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def choose_action(
        self,
        obs: dict,
        last_seen_enemy: tuple[int, int] | None,
    ) -> tuple[str, bool]:
        """Returns (chosen_action, should_clear_memory).

        should_clear_memory is True when the agent reached the last-seen
        position without finding the enemy there — so the caller can drop
        the stale memory instead of looping forever.
        """
        legal = list(obs["legal_actions"])
        if not legal:
            return gt.STAY, False

        visible = obs["visible_enemies"]
        if visible:
            target = Perception.update_last_seen_enemy(
                obs["ego_x"],
                obs["ego_y"],
                obs["team"],
                visible,
            )
            stale = False
        elif last_seen_enemy is not None:
            target = last_seen_enemy
            stale = True
        else:
            return self._rng.choice(legal), False

        tx, ty = target
        ego_x, ego_y = obs["ego_x"], obs["ego_y"]

        # If we've arrived at the last-seen position and the enemy isn't
        # here, clear memory so we fall back to exploration.
        if stale and ego_x == tx and ego_y == ty:
            return self._rng.choice(legal), True

        if obs["team"] == gt.TEAM_PREDATOR:

            def score(act: str) -> tuple[int, int]:
                nx, ny = _apply_action(ego_x, ego_y, act)
                # Random tiebreaker prevents deterministic corner loops.
                return (_manhattan(nx, ny, tx, ty), self._rng.randint(0, 1000))

            return min(legal, key=score), False

        def score_prey(act: str) -> tuple[int, int]:
            nx, ny = _apply_action(ego_x, ego_y, act)
            return (-_manhattan(nx, ny, tx, ty), self._rng.randint(0, 1000))

        return min(legal, key=score_prey), False