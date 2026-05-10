from __future__ import annotations

import random

import agent_utils as au
from distances import manhattan


def _apply_action(x: int, y: int, action: str) -> tuple[int, int]:
    dx, dy = au.ACTION_DELTA[action]
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
            return au.STAY, False

        if au.PICKUP in legal:
            return au.PICKUP, False

        # `active_enemies` is the set the agent is currently tracking this
        # step, chosen by priority in Agent.decide: direct sightings if any,
        # otherwise teammate reports, otherwise empty (memory-only fallback).
        active = obs["active_enemies"]
        if active:
            # Agent.decide already refreshed last_seen_enemy to the chosen
            # tracked enemy this step, so reuse it instead of re-running
            # perception (which would also burn another RNG draw).
            target = last_seen_enemy
            stale_target = False
        elif last_seen_enemy is not None:
            target = last_seen_enemy
            stale_target = True
        else:
            if au.DROP in legal:
                return au.DROP, False
            return self._rng.choice(legal), False

        assert target is not None

        tx, ty = target
        ego_x, ego_y = obs["ego_x"], obs["ego_y"]

        # If we've arrived at the last-seen position and the enemy isn't
        # here, clear memory so we fall back to exploration.
        if stale_target and ego_x == tx and ego_y == ty:
            return self._rng.choice(legal), True

        if obs["team"] == au.TEAM_PREDATOR:

            def score(act: str) -> tuple[int, int]:
                nx, ny = _apply_action(ego_x, ego_y, act)
                # Random tiebreaker prevents deterministic corner loops.
                return (manhattan(nx, ny, tx, ty), self._rng.randint(0, 1000))

            return min(legal, key=score), False

        # Prey primarily flees the closest tracked predator (target), but
        # when possible it also avoids moves that reduce distance to any
        # other predator in the active set (direct sight or teammate
        # report, whichever supplied the active set this step).
        current_other_dists: dict[int, int] = {}
        for ex, ey, eid in active:
            if ex == tx and ey == ty:
                continue
            current_other_dists[eid] = manhattan(ego_x, ego_y, ex, ey)

        safe_actions: list[str] = []
        for act in legal:
            nx, ny = _apply_action(ego_x, ego_y, act)
            gets_closer_to_other = False
            for ex, ey, eid in active:
                if eid not in current_other_dists:
                    continue
                if manhattan(nx, ny, ex, ey) < current_other_dists[eid]:
                    gets_closer_to_other = True
                    break
            if not gets_closer_to_other:
                safe_actions.append(act)

        candidate_actions = safe_actions if safe_actions else legal

        def score_prey(act: str) -> tuple[int, int]:
            nx, ny = _apply_action(ego_x, ego_y, act)
            return (-manhattan(nx, ny, tx, ty), self._rng.randint(0, 1000))

        return min(candidate_actions, key=score_prey), False