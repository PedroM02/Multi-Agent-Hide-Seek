import random

from distances import manhattan


class Perception:
    """Uses observation only (no full-env reads)."""

    def compute_active_enemies(self, visible_enemies, shared_enemies):
        """Priority-resolved set of enemies the agent is currently tracking.

        Direct sightings dominate (an enemy I see now is fresher than any
        teammate report); when I see nothing, teammate reports take over;
        when neither has anything, the active set is empty and the caller
        can still fall back to its own `last_seen_enemy` memory.

        Pure function over the two enemy lists — no agent state is read.
        """
        if visible_enemies:
            return visible_enemies
        positions = []
        
        for item in shared_enemies:
            if len(item) == 4:
                sender_id, enemy_x, enemy_y, enemy_id = item
            else:
                enemy_x, enemy_y, enemy_id = item
            positions.append((enemy_x, enemy_y, enemy_id))
        return tuple(positions)

    def update_last_seen_enemy(
        self,
        agent_x,
        agent_y,
        visible_enemies,
        rng,
    ):
        if not visible_enemies:
            return None
        best = None
        best_key = None
        for enemy_x, enemy_y, enemy_id in visible_enemies:
            # Random tiebreak: when several visible enemies are equidistant,
            # pick one uniformly instead of biasing by agent id.
            key = (float(manhattan(agent_x, agent_y, enemy_x, enemy_y)), rng.random())
            if best_key is None or key < best_key:
                best_key = key
                best = (enemy_x, enemy_y)
        assert best is not None
        return best
