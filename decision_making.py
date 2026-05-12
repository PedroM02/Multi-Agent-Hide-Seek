from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import agent_utils as au
from distances import manhattan
from environment import Environment


# ---------------------------------------------------------------------------
# Module-level geometry helpers shared by the selector and decision logic.
# ---------------------------------------------------------------------------


def _apply_action(x: int, y: int, action: str) -> Tuple[int, int]:
    dx, dy = au.ACTION_DELTA[action]
    return x + dx, y + dy


def _is_walkable_for_team(env: Environment, x: int, y: int, team: str) -> bool:
    """Cell is enterable next step for an agent of `team` ignoring agent occupancy.

    Off-grid and walls are always blocked. Unheld obstacles block both teams
    in the symmetric lock mode; in owner-passable they block only the enemy.
    """
    if not env.is_in_bounds(x, y) or env.is_wall(x, y):
        return False
    for obstacle in env.obstacles.values():
        if obstacle.held_by is None and obstacle.x == x and obstacle.y == y:
            if env.lock_mode == "owner-passable" and obstacle.locked_team == team:
                continue
            return False
    return True


def _cardinal_cells_around(
    cell: Tuple[int, int], env: Environment, team: str
) -> List[Tuple[int, int]]:
    """Return the (up to four) cardinal neighbours of `cell` walkable by `team`."""
    px, py = cell
    out: List[Tuple[int, int]] = []
    for action in (au.UP, au.DOWN, au.LEFT, au.RIGHT):
        dx, dy = au.ACTION_DELTA[action]
        nx, ny = px + dx, py + dy
        if _is_walkable_for_team(env, nx, ny, team):
            out.append((nx, ny))
    return out


def _flank_cell(
    prey_cell: Tuple[int, int],
    centroid: Tuple[float, float],
    env: Environment,
    team: str,
) -> Optional[Tuple[int, int]]:
    """Pick the cardinal of `prey_cell` furthest from the predator centroid.

    The intuition is the "far side" of the prey relative to the pack —
    a carrier that takes that cell forces the prey to flee back through
    its teammates. Tiebreak is deterministic on (x, y).
    """
    candidates = _cardinal_cells_around(prey_cell, env, team)
    if not candidates:
        return None
    cx, cy = centroid
    return max(
        candidates,
        key=lambda c: (abs(c[0] - cx) + abs(c[1] - cy), c[0], c[1]),
    )


def _predict_predator_next_cell(
    pred_cell: Tuple[int, int],
    prey_cell: Tuple[int, int],
    env: Environment,
    team: str,
) -> Optional[Tuple[int, int]]:
    """Greedy 1-step prediction of where a chase-only predator would step.

    Used by the prey's FUNNELER role to pick a drop target on the
    predator's predicted path. Mirrors the CHASER's own min-Manhattan
    cardinal step, with deterministic (x, y) tiebreak.
    """
    px, py = pred_cell
    qx, qy = prey_cell
    candidates: List[Tuple[int, int]] = []
    for action in (au.UP, au.DOWN, au.LEFT, au.RIGHT):
        dx, dy = au.ACTION_DELTA[action]
        nx, ny = px + dx, py + dy
        if _is_walkable_for_team(env, nx, ny, team):
            candidates.append((nx, ny))
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda c: (abs(c[0] - qx) + abs(c[1] - qy), c[0], c[1]),
    )


# ---------------------------------------------------------------------------
# Team-level role selector. Called once per team, per step, by simulation.
# ---------------------------------------------------------------------------


def select_team_roles(
    team: str,
    team_ids: List[int],
    obs_by_id: Dict[int, dict],
    agents_by_id: Dict[int, Any],
    env: Environment,
) -> Dict[int, Tuple[str, Optional[Tuple[int, int]]]]:
    """Assign (role, role_target) to every alive agent in `team`.

    The selector reads each agent's comms-augmented observation (so threat
    awareness is consistent with the per-agent decision step) and the
    agent's previous role_target for stickiness. Returns a mapping that
    callers should write back onto each Agent before agent.decide runs.
    """
    if not team_ids:
        return {}
    if team == au.TEAM_PREDATOR:
        return _select_predator_roles(team_ids, obs_by_id, agents_by_id, env)
    return _select_prey_roles(team_ids, obs_by_id, agents_by_id, env)


def _select_predator_roles(
    team_ids: List[int],
    obs_by_id: Dict[int, dict],
    agents_by_id: Dict[int, Any],
    env: Environment,
) -> Dict[int, Tuple[str, Optional[Tuple[int, int]]]]:
    # Aggregate the team's view of prey (active_enemies = direct + shared).
    prey_seen: Dict[int, Tuple[int, int]] = {}
    for aid in team_ids:
        for ex, ey, eid in obs_by_id[aid].get("active_enemies", ()):
            prey_seen[eid] = (ex, ey)

    if not prey_seen:
        return {aid: (au.ROLE_CHASER, None) for aid in team_ids}

    n = len(team_ids)
    centroid = (
        sum(obs_by_id[aid]["ego_x"] for aid in team_ids) / n,
        sum(obs_by_id[aid]["ego_y"] for aid in team_ids) / n,
    )

    target_eid = min(
        prey_seen.keys(),
        key=lambda e: (
            abs(prey_seen[e][0] - centroid[0])
            + abs(prey_seen[e][1] - centroid[1]),
            e,
        ),
    )
    target_prey = prey_seen[target_eid]

    carriers = sorted(
        aid for aid in team_ids if obs_by_id[aid]["held_obstacle"] is not None
    )
    non_carriers = [aid for aid in team_ids if aid not in carriers]

    roles: Dict[int, Tuple[str, Optional[Tuple[int, int]]]] = {
        aid: (au.ROLE_CHASER, None) for aid in non_carriers
    }

    if not carriers:
        return roles

    if len(carriers) == 1:
        # Solo carrier: S1 outflank. Compute a flank cell on the prey's far
        # side relative to the centroid and stick to it as long as it's a
        # valid cardinal of the current target prey.
        aid = carriers[0]
        agent = agents_by_id[aid]
        prev = agent.role_target
        prev_valid = (
            agent.role == au.ROLE_FLANKER
            and prev is not None
            and abs(prev[0] - target_prey[0]) + abs(prev[1] - target_prey[1]) == 1
            and _is_walkable_for_team(env, prev[0], prev[1], au.TEAM_PREDATOR)
        )
        if prev_valid:
            roles[aid] = (au.ROLE_FLANKER, prev)
        else:
            fc = _flank_cell(target_prey, centroid, env, au.TEAM_PREDATOR)
            roles[aid] = (au.ROLE_FLANKER, fc) if fc is not None else (au.ROLE_CHASER, None)
        return roles

    # Two or more carriers: S4 net formation. Greedily assign each carrier
    # to a distinct cardinal of the prey, honouring stickiness when the
    # previous target is still a valid cardinal slot.
    cells = _cardinal_cells_around(target_prey, env, au.TEAM_PREDATOR)
    if not cells:
        for aid in carriers:
            roles[aid] = (au.ROLE_CHASER, None)
        return roles

    available = list(cells)
    unassigned: List[int] = []
    for aid in carriers:
        prev = agents_by_id[aid].role_target
        if prev in available:
            roles[aid] = (au.ROLE_NET, prev)
            available.remove(prev)
        else:
            unassigned.append(aid)

    for aid in sorted(unassigned):
        if not available:
            roles[aid] = (au.ROLE_CHASER, None)
            continue
        ax, ay = obs_by_id[aid]["ego_x"], obs_by_id[aid]["ego_y"]
        cell = min(
            available,
            key=lambda c: (abs(c[0] - ax) + abs(c[1] - ay), c[0], c[1]),
        )
        roles[aid] = (au.ROLE_NET, cell)
        available.remove(cell)

    return roles


def _select_prey_roles(
    team_ids: List[int],
    obs_by_id: Dict[int, dict],
    agents_by_id: Dict[int, Any],
    env: Environment,
) -> Dict[int, Tuple[str, Optional[Tuple[int, int]]]]:
    """Prey role policy.

    An obstacle dropped at the prey's own cell only blocks the cell after
    the prey leaves (an enemy can step in while the prey is still
    standing on it). So dropping is only worth the obstacle when the
    predator is roughly one step away from where the new wall will sit —
    far enough that we get to leave before the enemy arrives, close
    enough that the wall actually intersects the chase. Roughly:

    - not holding  -> FLEE
    - holding, no threat        -> BUNKER (carry, do not spend obstacle)
    - holding, threat at M <= 1 -> FLEE  (drop would be wasted, captured next step)
    - holding, threat at M == 2 -> SHIELDER (drop, wall lands as predator arrives)
    - holding, threat at M == 3 -> BREADCRUMB (drop, build a chase-extending trail)
    - holding, threat at M >= 4 -> BUNKER (still too far, save the obstacle)
    """
    roles: Dict[int, Tuple[str, Optional[Tuple[int, int]]]] = {}
    for aid in team_ids:
        ob = obs_by_id[aid]
        holding = ob["held_obstacle"] is not None
        active = ob.get("active_enemies", ())
        ego = (ob["ego_x"], ob["ego_y"])

        if not holding:
            roles[aid] = (au.ROLE_FLEE, None)
            continue

        if not active:
            roles[aid] = (au.ROLE_BUNKER, None)
            continue

        min_dist = min(abs(ex - ego[0]) + abs(ey - ego[1]) for ex, ey, _eid in active)

        if min_dist <= 1:
            roles[aid] = (au.ROLE_FLEE, None)
            continue
        if min_dist == 2:
            roles[aid] = (au.ROLE_SHIELDER, None)
            continue
        if min_dist == 3:
            roles[aid] = (au.ROLE_BREADCRUMB, None)
            continue
        roles[aid] = (au.ROLE_BUNKER, None)

    return roles


# ---------------------------------------------------------------------------
# Per-agent action selection. Dispatches on the role written by the selector.
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

        active = obs["active_enemies"]
        ego_x, ego_y = obs["ego_x"], obs["ego_y"]
        team = obs["team"]
        role = obs.get(
            "role",
            au.ROLE_CHASER if team == au.TEAM_PREDATOR else au.ROLE_FLEE,
        )
        role_target = obs.get("role_target", None)
        enable_roles = obs.get("enable_roles", False)

        # The full obstacle pipeline (PICKUP here, DROP / FLANKER / NET /
        # SHIELDER / BREADCRUMB downstream) is gated on enable_roles.
        # When roles are off we strip PICKUP / DROP out of `legal` up
        # front: the downstream chase / flee branches consult `legal`
        # both for explicit choices and for their random-walk fallback,
        # and PICKUP / DROP have a zero movement delta so they would
        # otherwise tie with STAY in `_navigate_to`. Stripping here is
        # the single switch that guarantees "no obstacle interaction"
        # under the default flag set.
        if not enable_roles:
            legal = [a for a in legal if a not in (au.PICKUP, au.DROP)]
            if not legal:
                return au.STAY, False

        # Capture-imminent pickup guard (roles-on only — when roles are
        # off PICKUP was already stripped above). With 4-cardinal
        # movement, an enemy at Manhattan <= 1 can step into ego next
        # turn. Picking up forfeits movement, so we skip PICKUP and let
        # the chase / flee / role logic handle the urgent step. The
        # guard only inspects active_enemies (direct + teammate
        # reports); stale memory does not block pickup.
        if au.PICKUP in legal:
            threat_within_one = any(
                abs(ex - ego_x) + abs(ey - ego_y) <= 1
                for ex, ey, _eid in active
            )
            if not threat_within_one:
                return au.PICKUP, False

        if team == au.TEAM_PREDATOR:
            return self._predator_act(obs, role, role_target, last_seen_enemy, legal)
        return self._prey_act(obs, role, role_target, last_seen_enemy, legal)

    # ----- Predator branches --------------------------------------------------

    def _predator_act(
        self,
        obs: dict,
        role: str,
        role_target: Optional[Tuple[int, int]],
        last_seen_enemy: Optional[Tuple[int, int]],
        legal: List[str],
    ) -> Tuple[str, bool]:
        ego_x, ego_y = obs["ego_x"], obs["ego_y"]
        holding = obs["held_obstacle"] is not None

        if role in (au.ROLE_FLANKER, au.ROLE_NET) and role_target is not None and holding:
            tx, ty = role_target
            if ego_x == tx and ego_y == ty:
                if au.DROP in legal:
                    return au.DROP, False
                # No-op: holding but cannot drop — fall through to nav (rare).
            return self._navigate_to(ego_x, ego_y, tx, ty, legal), False

        return self._chase(obs, last_seen_enemy, legal)

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
            if au.DROP in legal:
                return au.DROP, False
            return self._rng.choice(legal), False

        assert target is not None
        tx, ty = target
        if stale and ego_x == tx and ego_y == ty:
            return self._rng.choice(legal), True
        return self._navigate_to(ego_x, ego_y, tx, ty, legal), False

    # ----- Prey branches ------------------------------------------------------

    def _prey_act(
        self,
        obs: dict,
        role: str,
        role_target: Optional[Tuple[int, int]],
        last_seen_enemy: Optional[Tuple[int, int]],
        legal: List[str],
    ) -> Tuple[str, bool]:
        holding = obs["held_obstacle"] is not None

        # Holding-with-threat-at-the-right-range roles drop on the prey's
        # current cell. The dropped obstacle blocks the cell only after the
        # prey moves off it (see action_resolution._target_cell), so the
        # behaviour is "drop now, flee next step" — the flee fallback at
        # the bottom of the function handles the step right after the drop
        # automatically because the prey is no longer holding.
        if role in (au.ROLE_SHIELDER, au.ROLE_BREADCRUMB) and holding:
            if au.DROP in legal:
                return au.DROP, False
            return self._flee(obs, last_seen_enemy, legal)

        # BUNKER no longer means "drop on self and stay"; the new
        # movement rule lets a predator step onto a co-located cell. The
        # role now means "I'm holding an obstacle but there is no threat
        # near enough to make a drop pay off". Keep carrying.
        # FLEE and FUNNELER fall through to the same flee logic — the
        # latter exists only as a legacy label and currently is not
        # assigned by the selector.
        return self._flee(obs, last_seen_enemy, legal)

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
            if au.DROP in legal:
                return au.DROP, False
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

    # ----- Shared helper ------------------------------------------------------

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
