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
    ) -> str:
        legal = list(obs["legal_actions"])
        if not legal:
            return gt.STAY

        visible = obs["visible_enemies"]
        if visible:
            target = Perception.update_last_seen_enemy(
                obs["ego_x"],
                obs["ego_y"],
                obs["team"],
                visible,
            )
        elif last_seen_enemy is not None:
            target = last_seen_enemy
        else:
            return self._rng.choice(legal)

        tx, ty = target
        ego_x, ego_y = obs["ego_x"], obs["ego_y"]

        if obs["team"] == gt.TEAM_PREDATOR:

            def score(act: str) -> tuple[int, str]:
                nx, ny = _apply_action(ego_x, ego_y, act)
                return (_manhattan(nx, ny, tx, ty), act)

            return min(legal, key=score)

        def score_prey(act: str) -> tuple[int, str]:
            nx, ny = _apply_action(ego_x, ego_y, act)
            return (-_manhattan(nx, ny, tx, ty), act)

        return min(legal, key=score_prey)
