from __future__ import annotations

import random

from distances import manhattan


class Perception:
    """Uses observation only (no full-env reads)."""

    @staticmethod
    def compute_active_enemies(
        visible_enemies: tuple[tuple[int, int, int], ...],
        shared_enemies: tuple[tuple[int, int, int], ...],
    ) -> tuple[tuple[int, int, int], ...]:
        """Priority-resolved set of enemies the agent is currently tracking.

        Direct sightings dominate (an enemy I see now is fresher than any
        teammate report); when I see nothing, teammate reports take over;
        when neither has anything, the active set is empty and the caller
        can still fall back to its own `last_seen_enemy` memory.

        Pure function over the two enemy lists — no agent state is read.
        """
        if visible_enemies:
            return visible_enemies
        return shared_enemies

    @staticmethod
    def update_last_seen_enemy(
        ego_x: int,
        ego_y: int,
        _team: str,
        visible_enemies: tuple[tuple[int, int, int], ...],
        rng: random.Random,
    ) -> tuple[int, int] | None:
        if not visible_enemies:
            return None
        best: tuple[int, int] | None = None
        best_key: tuple[float, float] | None = None
        for ex, ey, _eid in visible_enemies:
            # Random tiebreak: when several visible enemies are equidistant,
            # pick one uniformly instead of biasing by agent id.
            key = (float(manhattan(ego_x, ego_y, ex, ey)), rng.random())
            if best_key is None or key < best_key:
                best_key = key
                best = (ex, ey)
        assert best is not None
        return best
