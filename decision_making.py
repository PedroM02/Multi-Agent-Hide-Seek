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

        # Current distances to all *non-primary* threats. The "safe"
        # check below only fires on these — only the primary target
        # governs the actual flee direction.
        current_other_dists: Dict[int, int] = {}
        for ex, ey, eid in active:
            if ex == tx and ey == ty:
                continue
            current_other_dists[eid] = manhattan(ego_x, ego_y, ex, ey)

        # Split STAY out of the legal set before computing the
        # "safe" subset. STAY trivially never closes distance to anyone
        # (zero delta), so treating it as just another safe action lets
        # the prey freeze whenever every real cardinal looks unsafe —
        # the primary predator then walks in for free. We treat STAY
        # explicitly below instead.
        cardinals = [a for a in legal if a != au.STAY]

        # Suicide guard: drop any cardinal whose target cell is
        # currently occupied by a known active enemy. `legal_actions`
        # only checks bounds and walls, so a prey adjacent to a
        # predator can otherwise have a move like DOWN→(predator cell)
        # in its candidate set; the `safe_cardinals` filter happily
        # accepts it (moves further from all *other* predators) and
        # the scorer then prefers it to STAY on Manhattan-to-primary,
        # producing a step-into-capture. The agent already sees the
        # active enemies' positions in `obs`, so this guard uses no
        # new information channel.
        active_cells = {(ex, ey) for ex, ey, _ in active}
        cardinals = [
            act for act in cardinals
            if _apply_action(ego_x, ego_y, act) not in active_cells
        ]

        safe_cardinals: List[str] = []
        for act in cardinals:
            nx, ny = _apply_action(ego_x, ego_y, act)
            gets_closer_to_other = False
            for ex, ey, eid in active:
                if eid not in current_other_dists:
                    continue
                if manhattan(nx, ny, ex, ey) < current_other_dists[eid]:
                    gets_closer_to_other = True
                    break
            if not gets_closer_to_other:
                safe_cardinals.append(act)

        if safe_cardinals:
            # At least one real cardinal doesn't close on any
            # secondary threat. STAY is trivially safe; include it so
            # the scoring can still pick it when no cardinal beats it
            # (e.g. cornered prey with a single diagonal predator).
            candidate_actions = safe_cardinals + (
                [au.STAY] if au.STAY in legal else []
            )
        elif cardinals:
            # Every cardinal closes on at least one secondary threat.
            # Rather than freezing on STAY and letting the primary
            # (closest) predator walk in, accept the secondary closure
            # and pick the cardinal that most increases distance to
            # the primary.
            candidate_actions = cardinals
        else:
            # No legal cardinals at all (surrounded by walls). Whatever
            # is legal — typically just STAY.
            candidate_actions = legal

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
