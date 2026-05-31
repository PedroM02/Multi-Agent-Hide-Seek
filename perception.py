from utils import manhattan


class Perception:
    """Uses observation to process information and enrich the observation with the processed information."""

    def compute_enemies(self, visible_enemies, shared_enemies):
        """Returns (active_enemies, known_enemies) from direct sights and communication reports.

        active_enemies gives priority to direct sightings when present, otherwise it's communicated enemies.
        known_enemies is the union of both sources, with sender_id removed and with deduplication.
        """
        # Compute known enemies (union of direct sights and communication reports, deduplicated)
        known_set = set()
        for source in (visible_enemies, shared_enemies):
            for item in source:
                # If enemy was communicated by another agent
                if len(item) == 4:
                    sender_id, enemy_x, enemy_y, enemy_id = item
                # If enemy was directly seen by the agent
                else:
                    enemy_x, enemy_y, enemy_id = item
                known_set.add((enemy_x, enemy_y, enemy_id))
        known_enemies = tuple(known_set)

        # If there are direct sightings, active enemies are those
        if visible_enemies:
            active_enemies = visible_enemies
        # Otherwise, known enemies are the communicated enemies and those are the only we know and can act on
        else:
            active_enemies = known_enemies

        return active_enemies, known_enemies

    def perceive(self, obs):
        """Enriches an observation with comms and enemy knowledge.

        active_enemies gives priority to direct sightings when present, otherwise it's communicated enemies.
        known_enemies is the union of both sources, with sender_id removed and with deduplication.
        """
        visible_enemies = obs.get("visible_enemies", ())
        shared_enemies = obs.get("shared_enemies", ())
        active_enemies, known_enemies = self.compute_enemies(visible_enemies, shared_enemies)
        obs["active_enemies"] = active_enemies
        obs["known_enemies"] = known_enemies

    def update_last_seen_enemy(self, obs, rng):
        """Updates the last seen enemy memory to the closest active enemy, which can be a direct sight or a communicated enemy.
           If there are several equidistant enemies, a random one is chosen."""

        # If there are no active enemies, we do not update
        active_enemies = obs.get("active_enemies", ())
        if not active_enemies:
            return None

        # Get agent position
        agent_x, agent_y = obs["agent_x"], obs["agent_y"]

        # Compute the distance to each active enemy, storing alongside the enemy position
        distances = []
        for enemy_x, enemy_y, enemy_id in active_enemies:

            distance = (manhattan(agent_x, agent_y, enemy_x, enemy_y), enemy_x, enemy_y)
            distances.append(distance)

        # Get minimum distance
        min_distance = min(distance for distance, enemy_x, enemy_y in distances)
        # Get enemy positions that sit at the minimum distance (may be more than one)
        closest_enemies = [(enemy_x, enemy_y) for distance, enemy_x, enemy_y in distances if distance == min_distance]

        # Return a random enemy from the closest enemies
        return rng.choice(closest_enemies)
