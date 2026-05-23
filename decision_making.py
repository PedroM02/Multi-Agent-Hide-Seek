import random

import agent_utils as au
from distances import _grid_neighbors, _in_grid_bounds, bfs_distance, chebyshev, manhattan


# ---------------------------------------------------------------------------
# Module-level helpers.
# ---------------------------------------------------------------------------


def _apply_action(x, y, action):
    delta_x, delta_y = au.ACTION_DELTA[action]
    return x + delta_x, y + delta_y


def _iter_enemy_sightings(*sources):
    """Yield ``(enemy_x, enemy_y, enemy_id)`` from direct or comms reports."""
    for source in sources:
        for item in source:
            if len(item) == 4:
                sender_id, enemy_x, enemy_y, enemy_id = item
            else:
                enemy_x, enemy_y, enemy_id = item
            yield enemy_x, enemy_y, enemy_id


# ---------------------------------------------------------------------------
# Grid path steps for Level 6 optimal modes (uses distances.bfs_distance).
# ---------------------------------------------------------------------------


def _bfs_best_distance_greedy(
    start,
    goal,
    width,
    height,
    wall_cells,
    legal_actions,
    rng,
):
    """Fallback when goal is unreachable: minimize BFS distance after one step."""
    start_x, start_y = start
    best = []
    best_distance = None
    for action in legal_actions:
        delta_x, delta_y = au.ACTION_DELTA[action]
        neighbor_x, neighbor_y = start_x + delta_x, start_y + delta_y
        distance = bfs_distance(
            (neighbor_x, neighbor_y), goal, width, height, wall_cells,
        )
        if distance is None:
            continue
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best = [action]
        elif distance == best_distance:
            best.append(action)
    if best:
        return min(best, key=lambda a: rng.randint(0, 1000))
    return rng.choice(legal_actions)


def bfs_first_step(
    start,
    goal,
    width,
    height,
    wall_cells,
    legal_actions,
    rng,
):
    """Pick a legal action that follows one step along a shortest BFS path."""
    if not legal_actions:
        return au.STAY
    if start == goal:
        return au.STAY if au.STAY in legal_actions else rng.choice(legal_actions)

    start_x, start_y = start
    queue = [start]
    parent = {start: None}

    while queue:
        x, y = queue.pop(0)
        if (x, y) == goal:
            break
        for neighbor_x, neighbor_y in _grid_neighbors(x, y):
            if not _in_grid_bounds(neighbor_x, neighbor_y, width, height):
                continue
            if (neighbor_x, neighbor_y) in wall_cells:
                continue
            if (neighbor_x, neighbor_y) in parent:
                continue
            parent[(neighbor_x, neighbor_y)] = (x, y)
            queue.append((neighbor_x, neighbor_y))
    else:
        return _bfs_best_distance_greedy(
            start, goal, width, height, wall_cells, legal_actions, rng,
        )

    current = goal
    while parent[current] is not None and parent[current] != start:
        current = parent[current]
    first_step_x, first_step_y = current
    preferred = []
    for action in legal_actions:
        delta_x, delta_y = au.ACTION_DELTA[action]
        if start_x + delta_x == first_step_x and start_y + delta_y == first_step_y:
            preferred.append(action)
    if preferred:
        return min(preferred, key=lambda a: rng.randint(0, 1000))
    return _bfs_best_distance_greedy(
        start, goal, width, height, wall_cells, legal_actions, rng,
    )


def select_pack_prey_id(
    predator_positions,
    oracle_prey,
    width,
    height,
    wall_cells,
):
    """Prey id minimizing sum of BFS distances from all predators."""
    if not oracle_prey or not predator_positions:
        return None

    unreachable_penalty = width * height * len(predator_positions)
    best_id = None
    best_score = None

    for prey_x, prey_y, prey_id in oracle_prey:
        total = 0
        for predator_position in predator_positions:
            distance = bfs_distance(
                predator_position, (prey_x, prey_y), width, height, wall_cells,
            )
            total += distance if distance is not None else unreachable_penalty
        if best_score is None or total < best_score or (
            total == best_score and (best_id is None or prey_id < best_id)
        ):
            best_score = total
            best_id = prey_id
    return best_id


# ---------------------------------------------------------------------------
# Per-agent action selection.
# ---------------------------------------------------------------------------


class DecisionMaking:
    def __init__(self, rng, mode=au.MODE_CHASE):
        self.rng = rng
        self.mode = mode

    def choose_action(self, obs, last_seen_enemy):
        """Return (chosen_action, should_clear_memory).

        should_clear_memory is True when the agent reached its last-seen
        position without finding the enemy there — the caller drops the
        stale memory instead of looping forever.
        """
        legal = list(obs["legal_actions"])
        if not legal:
            return au.STAY, False

        if obs["team"] == au.TEAM_PREDATOR:
            if self.mode == au.MODE_RANDOM:
                return self._random_move(legal), False
            if self.mode == au.MODE_OPTIMAL:
                return self._optimal(obs, legal), False
            if self.mode == au.MODE_PACK:
                return self._pack(obs, last_seen_enemy, legal)
            if self.mode == au.MODE_ROLES:
                return self._roles(obs, last_seen_enemy, legal)
            return self._chase(obs, last_seen_enemy, legal)
        return self._flee(obs, last_seen_enemy, legal)

    def _random_move(self, legal):
        """Level 1: uniform random over legal actions (ignores perception)."""
        return self.rng.choice(legal)

    def _optimal(self, obs, legal):
        """Level 6: BFS toward the shared pack target injected by simulation."""
        target = obs.get("pack_target")
        if target is None:
            return self._random_move(legal)
        target_x, target_y = target
        return self._optimal_bfs_step(obs, target_x, target_y, legal)

    def _optimal_bfs_step(self, obs, target_x, target_y, legal):
        return bfs_first_step(
            (obs["agent_x"], obs["agent_y"]),
            (target_x, target_y),
            obs["grid_width"],
            obs["grid_height"],
            obs["wall_cells"],
            legal,
            self.rng,
        )

    def _chase(self, obs, last_seen_enemy, legal):
        active = obs["active_enemies"]
        agent_x, agent_y = obs["agent_x"], obs["agent_y"]

        if active:
            target = last_seen_enemy
            stale = False
        elif last_seen_enemy is not None:
            target = last_seen_enemy
            stale = True
        else:
            return self.rng.choice(legal), False

        assert target is not None
        target_x, target_y = target
        if stale and agent_x == target_x and agent_y == target_y:
            return self.rng.choice(legal), True
        return self._navigate_to(agent_x, agent_y, target_x, target_y, legal), False

    def derive_role(self, obs):
        """Local role derivation for `--mode roles` (no simulation assignment)."""
        if obs["team"] == au.TEAM_PREY:
            return au.ROLE_FLEE, None
        return self._derive_predator_role(obs)

    def _derive_predator_role(self, obs):
        if not obs.get("visible_allies", ()):
            return au.ROLE_CHASER, None

        peers = self._known_peers(obs, include_shared_allies=False)
        prey_candidates = self._pack_prey_candidates(obs)
        if not prey_candidates:
            return au.ROLE_CHASER, None

        peer_positions = [(x, y) for _, x, y in peers]
        focus_prey = self._pick_pack_prey(peer_positions, prey_candidates)
        if focus_prey is None:
            return au.ROLE_CHASER, None

        chaser_id = self._pick_chaser_id(peers, focus_prey)
        my_id = obs["agent_id"]
        if my_id == chaser_id:
            return au.ROLE_CHASER, focus_prey

        if not self._knows_prey_at(obs, focus_prey):
            return au.ROLE_CHASER, None

        flank_target = self._my_flank_target(obs, peers, chaser_id, focus_prey)
        if flank_target is None:
            return au.ROLE_CHASER, None
        return au.ROLE_FLANKER, flank_target

    def _roles(self, obs, last_seen_enemy, legal):
        role = obs.get("role")
        role_target = obs.get("role_target")
        agent_x, agent_y = obs["agent_x"], obs["agent_y"]

        if role == au.ROLE_FLANKER:
            return self._flank(obs, legal)
        if role == au.ROLE_CHASER and role_target is not None:
            target_x, target_y = role_target
            return self._navigate_to(
                agent_x, agent_y, target_x, target_y, legal,
            ), False
        return self._chase(obs, last_seen_enemy, legal)

    def _pick_chaser_id(self, peers, focus_prey):
        focus_x, focus_y = focus_prey
        best_id = None
        best_dist = None
        for peer_id, px, py in peers:
            distance = manhattan(px, py, focus_x, focus_y)
            if best_dist is None or distance < best_dist or (
                distance == best_dist and peer_id < best_id
            ):
                best_dist = distance
                best_id = peer_id
        return best_id

    def _knows_prey_at(self, obs, prey_pos):
        px, py = prey_pos
        for enemy_x, enemy_y, enemy_id in _iter_enemy_sightings(
            obs.get("visible_enemies", ()),
            obs.get("shared_enemies", ()),
        ):
            if (enemy_x, enemy_y) == (px, py):
                return True
        return False

    def _ally_can_see_prey_at(self, obs, ally_x, ally_y, prey_pos):
        px, py = prey_pos
        return chebyshev(ally_x, ally_y, px, py) <= obs["vision_radius"]

    def _peer_reported_prey_at(self, obs, peer_id, prey_pos):
        px, py = prey_pos
        for item in obs.get("shared_enemies", ()):
            if len(item) != 4:
                continue
            sender_id, enemy_x, enemy_y, _ = item
            if sender_id == peer_id and (enemy_x, enemy_y) == (px, py):
                return True
        return False

    def _peer_can_see_focus_prey(self, obs, peer_id, ally_x, ally_y, focus_prey):
        return (
            self._ally_can_see_prey_at(obs, ally_x, ally_y, focus_prey)
            or self._peer_reported_prey_at(obs, peer_id, focus_prey)
        )

    def _flanker_candidate_ids(self, obs, peers, chaser_id, focus_prey):
        """Non-chaser peers eligible to flank (can see focus prey).

        Self: prey at focus in visible or attributed comms reports. Other
        visible allies: Chebyshev range to focus prey, or they reported it.
        Slot assignment stays id-based.
        """
        my_id = obs["agent_id"]
        candidate_ids = []
        for peer_id, ally_x, ally_y in peers:
            if peer_id == chaser_id:
                continue
            if peer_id == my_id:
                if self._knows_prey_at(obs, focus_prey):
                    candidate_ids.append(peer_id)
            elif self._peer_can_see_focus_prey(
                obs, peer_id, ally_x, ally_y, focus_prey,
            ):
                candidate_ids.append(peer_id)
        return sorted(candidate_ids)

    def _unit_step(self, dx, dy):
        """Sign per axis for a non-zero direction (Manhattan step)."""
        step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
        step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
        return step_x, step_y

    def _flank_options(self, chaser_pos, focus_prey):
        """Two cells one step perpendicular to the chaser→prey axis."""
        chaser_x, chaser_y = chaser_pos
        focus_x, focus_y = focus_prey
        dx = focus_x - chaser_x
        dy = focus_y - chaser_y
        if dx == 0 and dy == 0:
            return (focus_x + 1, focus_y), (focus_x - 1, focus_y)
        perp_x, perp_y = self._unit_step(-dy, dx)
        return (
            (focus_x + perp_x, focus_y + perp_y),
            (focus_x - perp_x, focus_y - perp_y),
        )

    def _my_flank_target(self, obs, peers, chaser_id, focus_prey):
        chaser_pos = next(
            (x, y) for peer_id, x, y in peers if peer_id == chaser_id
        )
        option1, option2 = self._flank_options(chaser_pos, focus_prey)
        flanker_ids = self._flanker_candidate_ids(
            obs, peers, chaser_id, focus_prey,
        )
        if obs["agent_id"] not in flanker_ids:
            return None

        used_slots = []
        assignments = {}
        for flanker_id in flanker_ids:
            flanker_pos = next(
                (x, y) for peer_id, x, y in peers if peer_id == flanker_id
            )
            distance1 = manhattan(
                flanker_pos[0], flanker_pos[1], option1[0], option1[1],
            )
            distance2 = manhattan(
                flanker_pos[0], flanker_pos[1], option2[0], option2[1],
            )
            if option1 not in used_slots and option2 not in used_slots:
                chosen = option1 if distance1 <= distance2 else option2
            elif option1 in used_slots:
                chosen = option2
            elif option2 in used_slots:
                chosen = option1
            else:
                chosen = option1 if distance1 <= distance2 else option2
            used_slots.append(chosen)
            assignments[flanker_id] = chosen
        return assignments.get(obs["agent_id"])

    def _pack(self, obs, last_seen_enemy, legal):
        """Level 4: local pack reasoning — no simulation-assigned targets.

        A pack exists when this predator sees at least one ally, or has
        received ally positions via two-pass pack comms. Prey candidates
        are the union of own sightings and teammate reports
        (``shared_enemies``), not ``active_enemies`` alone. Peer positions
        include self, ``visible_allies``, and ``shared_allies`` (relay).
        """
        visible_allies = obs.get("visible_allies", ())
        shared_allies = obs.get("shared_allies", ())
        if not visible_allies and not shared_allies:
            return self._chase(obs, last_seen_enemy, legal)

        agent_x, agent_y = obs["agent_x"], obs["agent_y"]
        peer_positions = self._known_peer_positions(
            obs, include_shared_allies=True,
        )

        prey_candidates = self._pack_prey_candidates(obs)
        target = self._pick_pack_prey(peer_positions, prey_candidates)
        if target is not None:
            target_x, target_y = target
            return self._navigate_to(
                agent_x, agent_y, target_x, target_y, legal,
            ), False

        return self._chase(obs, last_seen_enemy, legal)

    def _known_peers(self, obs, include_shared_allies=False):
        """(peer_id, x, y) for self and known allies, sorted by id."""
        peers = [(obs["agent_id"], obs["agent_x"], obs["agent_y"])]
        seen_ids = {obs["agent_id"]}
        sources = [obs.get("visible_allies", ())]
        if include_shared_allies:
            sources.append(obs.get("shared_allies", ()))
        for source in sources:
            for ally_x, ally_y, ally_id in source:
                if ally_id in seen_ids:
                    continue
                seen_ids.add(ally_id)
                peers.append((ally_id, ally_x, ally_y))
        peers.sort(key=lambda peer: peer[0])
        return peers

    def _known_peer_positions(self, obs, include_shared_allies=False):
        return [
            (x, y) for _, x, y in self._known_peers(
                obs, include_shared_allies=include_shared_allies,
            )
        ]

    def _pack_prey_candidates(self, obs):
        """Own direct prey plus prey reported by visible allies via comms."""
        seen_ids = set()
        candidates = []
        for enemy_x, enemy_y, enemy_id in _iter_enemy_sightings(
            obs.get("visible_enemies", ()),
            obs.get("shared_enemies", ()),
        ):
            if enemy_id in seen_ids:
                continue
            seen_ids.add(enemy_id)
            candidates.append((enemy_x, enemy_y, enemy_id))
        candidates.sort(key=lambda t: (t[2], t[0], t[1]))
        return candidates

    def _pick_pack_prey(self, peer_positions, prey_candidates):
        """Prey minimizing sum of Manhattan distances from pack peers."""
        if not prey_candidates:
            return None

        best_target = None
        best_score = None
        best_id = None
        for prey_x, prey_y, prey_id in prey_candidates:
            total = sum(
                manhattan(px, py, prey_x, prey_y)
                for px, py in peer_positions
            )
            if best_score is None or total < best_score or (
                total == best_score and prey_id < best_id
            ):
                best_score = total
                best_target = (prey_x, prey_y)
                best_id = prey_id
        return best_target

    def _flank(self, obs, legal):
        role_target = obs.get("role_target")
        agent_x, agent_y = obs["agent_x"], obs["agent_y"]

        if role_target is None:
            return self._chase(obs, None, legal)

        target_x, target_y = role_target
        # if we reached the flank position, stay put and wait
        if agent_x == target_x and agent_y == target_y:
            return au.STAY, False

        return self._navigate_to(agent_x, agent_y, target_x, target_y, legal), False

    def _flee(self, obs, last_seen_enemy, legal):
        active = obs["active_enemies"]
        visible_allies = obs.get("visible_allies", ())
        agent_x, agent_y = obs["agent_x"], obs["agent_y"]

        if active:
            target = last_seen_enemy
            stale = False
        elif last_seen_enemy is not None:
            target = last_seen_enemy
            stale = True
        else:
            # No active threat and no memory. Wander, but bias toward
            # visible allies so prey preemptively pair up before any
            # threat appears (this is what makes the cooperative
            # knockout mechanic reachable in practice).
            return self._wander_with_cohesion(
                agent_x, agent_y, legal, visible_allies
            ), False

        assert target is not None
        target_x, target_y = target
        if stale and agent_x == target_x and agent_y == target_y:
            # Reached the stale memory cell without finding the enemy;
            # drop the memory and wander (still cohesion-biased).
            return self._wander_with_cohesion(
                agent_x, agent_y, legal, visible_allies
            ), True

        # Current distances to all *non-primary* threats. The "safe"
        # check below only fires on these — only the primary target
        # governs the actual flee direction.
        current_other_distances = {}
        for enemy_x, enemy_y, enemy_id in active:
            if enemy_x == target_x and enemy_y == target_y:
                continue
            current_other_distances[enemy_id] = manhattan(
                agent_x, agent_y, enemy_x, enemy_y,
            )

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
        active_cells = {
            (enemy_x, enemy_y) for enemy_x, enemy_y, _ in active
        }
        cardinals = [
            action for action in cardinals
            if _apply_action(agent_x, agent_y, action) not in active_cells
        ]

        # No ally-stack guard here. A move that lands on a teammate's
        # current cell ("stacking") is *not* pruned: when the
        # teammate is moving away this step, the action resolver
        # cleanly cascades and the prey follows into the vacated
        # cell — that's the "cascade-follow" path which is sometimes
        # the only safe escape (e.g. cornered prey with the only
        # safe cardinal pointing through a fleeing ally). The
        # cohesion term in `score_prey` below uses |d_a - 1| (not
        # d_a - 1), so d_a = 0 (stack) ties d_a = 2 (one cell away)
        # at score 1 instead of being uniquely rewarded — the
        # scorer no longer prefers stacking purely for its own
        # sake. When the teammate actually stays put the resolver
        # blocks the stack and forces STAY for this step (one
        # wasted step), which is strictly cheaper than the
        # alternative (refusing to stack and walking into a
        # predator instead).

        safe_cardinals = []
        for action in cardinals:
            neighbor_x, neighbor_y = _apply_action(agent_x, agent_y, action)
            gets_closer_to_other = False
            for enemy_x, enemy_y, enemy_id in active:
                if enemy_id not in current_other_distances:
                    continue
                if manhattan(neighbor_x, neighbor_y, enemy_x, enemy_y) < current_other_distances[enemy_id]:
                    gets_closer_to_other = True
                    break
            if not gets_closer_to_other:
                safe_cardinals.append(action)

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
            # No legal cardinals remain after the suicide guard
            # (every cardinal would step onto an active enemy, or
            # there were no cardinals to begin with). Default to
            # STAY — `legal` may still contain pruned cardinals, and
            # we don't want the scorer to pick one of them after we
            # just decided they're bad.
            candidate_actions = [au.STAY] if au.STAY in legal else list(legal)

        def score_prey(action):
            neighbor_x, neighbor_y = _apply_action(agent_x, agent_y, action)
            primary_distance = manhattan(neighbor_x, neighbor_y, target_x, target_y)
            # Cohesion term: prefer moves that keep the prey within
            # Chebyshev-1 of at least one visible teammate. d_a = 1
            # is the sweet spot for the cooperative-knockout mechanic
            # (two adjacent prey can stun a predator that is
            # Chebyshev-1 of both). Using |d_a - 1| (not d_a - 1)
            # means d_a = 0 (stacking onto an ally cell) is no longer
            # the unique optimum — it ties d_a = 2 at score 1. The
            # scorer therefore doesn't prefer stack moves for their
            # own sake; when one is picked it's because the primary-
            # distance key (the first lex key) already strictly
            # favoured it, in which case the action resolver will
            # cascade-follow the ally if it moves, or force a STAY
            # if it doesn't — both acceptable. With no visible ally
            # the term is neutral, preserving solo-prey behaviour.
            if visible_allies:
                ally_distance = min(
                    chebyshev(neighbor_x, neighbor_y, ally_x, ally_y)
                    for (ally_x, ally_y, _) in visible_allies
                )
                ally_term = abs(ally_distance - 1)
            else:
                ally_term = 0
            return (-primary_distance, ally_term, self.rng.randint(0, 1000))

        return min(candidate_actions, key=score_prey), False

    def _wander_with_cohesion(
        self,
        agent_x,
        agent_y,
        legal,
        visible_allies,
    ):
        """Cohesion-biased wander.

        With no visible ally this is just a uniform random pick over
        `legal` — same as the prior behaviour. With at least one
        visible ally we apply the same stack guard as the flee path
        and pick the move minimising Chebyshev distance to the
        nearest ally (target d_a = 1), with random jitter as the
        tiebreaker. STAY is always considered, so a prey that is
        already adjacent to an ally happily holds the formation.
        """
        if not visible_allies:
            return self.rng.choice(legal)

        cardinals = [a for a in legal if a != au.STAY]
        ally_cells = {
            (ally_x, ally_y) for ally_x, ally_y, _ in visible_allies
        }
        cardinals = [
            action for action in cardinals
            if _apply_action(agent_x, agent_y, action) not in ally_cells
        ]
        candidates = cardinals + (
            [au.STAY] if au.STAY in legal else []
        )
        if not candidates:
            candidates = list(legal)

        def score(action):
            neighbor_x, neighbor_y = _apply_action(agent_x, agent_y, action)
            ally_distance = min(
                chebyshev(neighbor_x, neighbor_y, ally_x, ally_y)
                for (ally_x, ally_y, _) in visible_allies
            )
            return (abs(ally_distance - 1), self.rng.randint(0, 1000))

        return min(candidates, key=score)

    def _navigate_to(self, agent_x, agent_y, target_x, target_y, legal):
        def score(action):
            neighbor_x, neighbor_y = _apply_action(agent_x, agent_y, action)
            return (
                abs(neighbor_x - target_x) + abs(neighbor_y - target_y),
                self.rng.randint(0, 1000),
            )

        return min(legal, key=score)
