from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import agent_utils as au
from distances import manhattan
from environment import Environment


# ---------------------------------------------------------------------------
# Module-level helpers.
# ---------------------------------------------------------------------------


def _apply_action(x: int, y: int, action: str) -> Tuple[int, int]:
    dx, dy = au.ACTION_DELTA[action]
    return x + dx, y + dy


# ---------------------------------------------------------------------------
# Team-level role selector. Currently trivial; kept as the entry point
# for future hunting / protection strategies.
# ---------------------------------------------------------------------------


def select_team_roles(
    team: str,
    team_ids: List[int],
    obs_by_id: Dict[int, dict],
    agents_by_id: Dict[int, Any],
    env: Environment,
) -> Dict[int, Tuple[str, Optional[Tuple[int, int]]]]:
    """Assign (role, role_target) to every alive agent in `team`.

    Currently trivial: predators all become CHASER, prey all become FLEE.
    The function is kept as the single hook for future role logic — when
    a richer role taxonomy is reintroduced it slots in here and
    `DecisionMaking.choose_action` learns to dispatch on it.
    """
    if not team_ids:
        return {}
    if team == au.TEAM_PREDATOR:
        return {aid: (au.ROLE_CHASER, None) for aid in team_ids}
    return {aid: (au.ROLE_FLEE, None) for aid in team_ids}


# ---------------------------------------------------------------------------
# Per-agent action selection.
# ---------------------------------------------------------------------------


class DecisionMaking:
    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def choose_action(
        self,
        obs: dict,
        last_seen_enemy: Optional[Tuple[int, int]],
    ) -> Tuple[str, bool]:
        """Return (chosen_action, should_clear_memory).

        should_clear_memory is True when the agent reached its last-seen
        position without finding the enemy there — the caller drops the
        stale memory instead of looping forever.
        """
        legal = list(obs["legal_actions"])
        if not legal:
            return au.STAY, False

        if obs["team"] == au.TEAM_PREDATOR:
            return self._chase(obs, last_seen_enemy, legal)
        return self._flee(obs, last_seen_enemy, legal)

    def _chase(
        self,
        obs: dict,
        last_seen_enemy: Optional[Tuple[int, int]],
        legal: List[str],
    ) -> Tuple[str, bool]:
        active = obs["active_enemies"]
        ego_x, ego_y = obs["ego_x"], obs["ego_y"]

        if active:
            target = last_seen_enemy
            stale = False
        elif last_seen_enemy is not None:
            target = last_seen_enemy
            stale = True
        else:
            return self._rng.choice(legal), False

        assert target is not None
        tx, ty = target
        if stale and ego_x == tx and ego_y == ty:
            return self._rng.choice(legal), True
        return self._navigate_to(ego_x, ego_y, tx, ty, legal), False

    def _flee(
        self,
        obs: dict,
        last_seen_enemy: Optional[Tuple[int, int]],
        legal: List[str],
    ) -> Tuple[str, bool]:
        active = obs["active_enemies"]
        ego_x, ego_y = obs["ego_x"], obs["ego_y"]

        if active:
            target = last_seen_enemy
            stale = False
        elif last_seen_enemy is not None:
            target = last_seen_enemy
            stale = True
        else:
            return self._rng.choice(legal), False

        assert target is not None
        tx, ty = target
        if stale and ego_x == tx and ego_y == ty:
            return self._rng.choice(legal), True

        # Avoid moves that close distance to any *other* predator in the
        # active set; only the primary target governs flee direction.
        current_other_dists: Dict[int, int] = {}
        for ex, ey, eid in active:
            if ex == tx and ey == ty:
                continue
            current_other_dists[eid] = manhattan(ego_x, ego_y, ex, ey)

        safe_actions: List[str] = []
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

        def score_prey(act: str) -> Tuple[int, int]:
            nx, ny = _apply_action(ego_x, ego_y, act)
            return (-manhattan(nx, ny, tx, ty), self._rng.randint(0, 1000))

        return min(candidate_actions, key=score_prey), False

    def _navigate_to(
        self,
        ego_x: int,
        ego_y: int,
        tx: int,
        ty: int,
        legal: List[str],
    ) -> str:
        def score(act: str) -> Tuple[int, int]:
            nx, ny = _apply_action(ego_x, ego_y, act)
            return (abs(nx - tx) + abs(ny - ty), self._rng.randint(0, 1000))

        return min(legal, key=score)
