from __future__ import annotations


def _manhattan(ax: int, ay: int, bx: int, by: int) -> int:
    return abs(ax - bx) + abs(ay - by)


class Perception:
    """Uses observation only (no full-env reads)."""

    @staticmethod
    def update_last_seen_enemy(
        ego_x: int,
        ego_y: int,
        _team: str,
        visible_enemies: tuple[tuple[int, int, int], ...],
    ) -> tuple[int, int] | None:
        if not visible_enemies:
            return None
        best: tuple[int, int] | None = None
        best_key: tuple[int, int] | None = None
        for ex, ey, eid in visible_enemies:
            key = (_manhattan(ego_x, ego_y, ex, ey), eid)
            if best_key is None or key < best_key:
                best_key = key
                best = (ex, ey)
        assert best is not None
        return best
