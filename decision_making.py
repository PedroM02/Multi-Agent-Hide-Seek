import random
import constants as co
from utils import apply_action, bfs_first_step, chebyshev, manhattan


class DecisionMaking:
    '''Decision making module that chooses an action intention for the agent based on observation and reasoning mode.
       Roles mode can have searcher role or not'''
    def __init__(self, rng, mode=co.MODE_CHASE, searcher_enabled=False):
        self.rng = rng
        self.mode = mode
        self.searcher_enabled = searcher_enabled


############################################ General action functions #######################################################


    def choose_action(self, obs, last_seen_enemy):
        """Return the chosen action intention for the current step."""
        
        # If no legal actions are available, stay in place
        legal = list(obs["legal_actions"])
        if not legal:
            return co.STAY

        # If the agent is a predator, choose an action intention based on the mode
        if obs["team"] == co.TEAM_PREDATOR:
            if self.mode == co.MODE_RL:
                return co.STAY
            if self.mode == co.MODE_RANDOM:
                return self.rng.choice(legal)
            if self.mode == co.MODE_ROLES:
                return self.roles(obs, last_seen_enemy, legal)
            if self.mode == co.MODE_OPTIMAL:
                return self.optimal(obs, legal)
            # If mode is neither of the above nor RL, use chase mode
            return self.chase(obs, last_seen_enemy, legal)
        # If the agent is a prey, use flee mode
        return self.flee(obs, last_seen_enemy, legal)

    def move_to(self, agent_x, agent_y, target_x, target_y, legal):
        """Returns action intention that most reduces distance to the target position"""
        # Initialize best distance and best actions
        best_distance = None
        best_actions = []
        # Iterate over all legal actions and compute the distance to the target
        for action in legal:
            # Compute the next position after taking the action
            next_x, next_y = apply_action(agent_x, agent_y, action)
            # Compute the Manhattan distance to the target
            distance = manhattan(next_x, next_y, target_x, target_y)
            # If this action leads to a shorter distance than the best so far, update the best action and distance
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_actions = [action]
            # If this action leads to the same distance as the best so far, add to possible actions
            elif distance == best_distance:
                best_actions.append(action)
        return self.rng.choice(best_actions)

############################################ Chase mode #######################################################

    def chase(self, obs, last_seen_enemy, legal):
        """Returns action intention that moves toward the last seen enemy, which could be the current 
           closest active enemy or a previously seen enemy from memory"""
        active = obs["active_enemies"]
        agent_x, agent_y = obs["agent_x"], obs["agent_y"]

        # If there is an active enemy, last_seen_enemy is the current closest active enemy
        if active:
            target = last_seen_enemy
            stale = False
        # If there is no active enemy, last_seen_enemy is a previously seen enemy from memory and we chase that
        elif last_seen_enemy is not None:
            target = last_seen_enemy
            stale = True
        # If there is nothing to chase, return a random legal action
        else:
            return self.rng.choice(legal)

        # If the target's information is stale and the agent has reached the target, return a random legal action
        target_x, target_y = target
        if stale and agent_x == target_x and agent_y == target_y:
            return self.rng.choice(legal)
        # Return action that moves towards the target
        return self.move_to(agent_x, agent_y, target_x, target_y, legal)

############################################ Roles mode #######################################################

    def derive_role(self, obs):
        """Role derivation when roles mode is enabled. Prey have no specific roles. Predators
           can be chaser, flanker, or searcher. Chaser if they are closest to a selected pack
           prey. Flanker if they are not the closest but still within communication range. Searcher if
           they know no prey.
           
           Returns the role and target position for the role (relevant for flanker)"""

        # Prey don't have specific roles
        if obs["team"] == co.TEAM_PREY:
            return co.ROLE_FLEE, None

        # Retrieve known prey candidates
        prey_candidates = self.pack_prey_candidates(obs)
        # If searcher is enabled and there are no prey candidates, return searcher role
        if self.searcher_enabled and not prey_candidates:
            return co.ROLE_SEARCHER, None
        # If there are no visible allies around, return chaser role
        if not obs.get("visible_allies", ()):
            return co.ROLE_CHASER, None
        # Retrieve known peers and their positions, including the agent
        peers = self.known_peers(obs)
        peer_positions = [(x, y) for peer_id, x, y in peers]
        # Pick the pack prey that minimizes the sum of Manhattan distances from pack peers
        focus_prey = self.pick_pack_prey(peer_positions, prey_candidates)
        # If no pack prey is found, return chaser role
        if focus_prey is None:
            return co.ROLE_CHASER, None

        # Identify the closest peer to the pack prey
        chaser_id = self.pick_chaser_id(peers, focus_prey)
        # If the agent is the closest predator to prey, agent is the chaser
        if obs["agent_id"] == chaser_id:
            return co.ROLE_CHASER, focus_prey

        # If the agent does not know of the pack prey, return chaser role. Only agents that are close enough to
        # the prey can know of it and thus are eligible to be a flanker (below)
        if not self.knows_prey_at(obs, focus_prey):
            return co.ROLE_CHASER, None

        # Identify if agent possesses conditions to be flanker and, if so, return its role along with flank target cell
        # flank_target() handles both eligibility and cell assignment
        flank_target = self.flank_target(obs, peers, chaser_id, focus_prey)
        if flank_target is None:
            return co.ROLE_CHASER, None
        return co.ROLE_FLANKER, flank_target

    def roles(self, obs, last_seen_enemy, legal):
        '''Returns the action intention for the agent based on its role'''

        # Get role and possibly target from observation
        role = obs.get("role")
        role_target = obs.get("role_target")
        agent_x, agent_y = obs["agent_x"], obs["agent_y"]

        # Return appropriate action intention based on role
        if role == co.ROLE_FLANKER:
            return self.flank(obs, legal)
        if role == co.ROLE_SEARCHER:
            return self.search(obs, legal)
        # If agent is a chaser within a flanking formation, move to focus prey (role_target)
        if role == co.ROLE_CHASER and role_target is not None:
            target_x, target_y = role_target
            return self.move_to(agent_x, agent_y, target_x, target_y, legal)
        # If agent is just a chaser, chase the last seen enemy
        return self.chase(obs, last_seen_enemy, legal)

    ################################### Chase and Flank #######################################################

    def pick_chaser_id(self, peers, focus_prey):
        '''Returns the closest predator in the pack to the selected prey'''
        # Initialize best ID and distance
        focus_x, focus_y = focus_prey
        best_id = None
        best_distance = None
        # Iterate over all peers and compute the Manhattan distance to the focus prey
        for peer_id, peer_x, peer_y in peers:
            # Compute the Manhattan distance to the focus prey
            distance = manhattan(peer_x, peer_y, focus_x, focus_y)
            # If this peer is closer than the best so far, update the best ID and distance
            # If this peer has the same distance as the best so far, keep the peer with smallest ID
            if best_distance is None or distance < best_distance or (distance == best_distance and peer_id < best_id):
                best_distance = distance
                best_id = peer_id
        return best_id

    def knows_prey_at(self, obs, prey_pos):
        '''Returns True if the agent knows the prey at the given position either directly or through communication'''
        prey_x, prey_y = prey_pos
        # Iterate over all known enemies and check if any of them is at the given position
        for enemy_x, enemy_y, enemy_id in obs.get("known_enemies", ()):
            if (enemy_x, enemy_y) == (prey_x, prey_y):
                return True
        return False

    def ally_can_see_prey_at(self, obs, ally_x, ally_y, prey_pos):
        '''Returns True if an ally is within vision range of the prey'''
        prey_x, prey_y = prey_pos
        return chebyshev(ally_x, ally_y, prey_x, prey_y) <= obs["vision_radius"]

    def peer_reported_prey_at(self, obs, peer_id, prey_pos):
        '''Returns True if a peer has reported the prey at the given position'''
        prey_x, prey_y = prey_pos
        # Iterate over communicated enemies and check if any of them is at the relevant position
        # and was reported by the given peer
        for item in obs.get("shared_enemies", ()):
            sender_id, enemy_x, enemy_y, enemy_id = item
            if sender_id == peer_id and (enemy_x, enemy_y) == (prey_x, prey_y):
                return True
        return False

    def flanker_candidate_ids(self, obs, peers, chaser_id, focus_prey):
        """Returns non-chaser peers which are eligible to flank.

        Can be the agent itself if prey at focus is in visible enemies or comms reports. 
        Can be any of the visible allies if they are within vision range to focus prey, or they reported it"""

        candidate_ids = []

        # Iterate over all peers and check if they check eligibility conditions to flank
        for peer_id, ally_x, ally_y in peers:
            # If peer is already chaser, skip
            if peer_id == chaser_id:
                continue
            # If current peer is the agent, check if it knows of the prey
            if peer_id == obs["agent_id"]:
                if self.knows_prey_at(obs, focus_prey):
                    candidate_ids.append(peer_id)
            # If the peer is a visible ally, check if they can see the prey or report it
            elif self.ally_can_see_prey_at(obs, ally_x, ally_y, focus_prey) or self.peer_reported_prey_at(obs, peer_id, focus_prey):
                candidate_ids.append(peer_id)
        return sorted(candidate_ids)


    def flank_options(self, chaser_pos, focus_prey):
        """Returns the two adjacent cells to the prey that are perpendicular to the chaser's direction"""
        chaser_x, chaser_y = chaser_pos
        focus_x, focus_y = focus_prey
        # Calculate the positional difference between the chaser and prey (their axis, chaser's direction vector)
        dx = focus_x - chaser_x
        dy = focus_y - chaser_y
        # Failsafe: if chaser is on top of prey, return two adjacent cells
        if dx == 0 and dy == 0:
            return (focus_x + 1, focus_y), (focus_x - 1, focus_y)

        # Perpendicular to (dx, dy) direction on the grid means rotating to (-dy, dx)
        rot_x = -dy
        rot_y = dx
        # Perpendicular one-step coordinate increments
        # (if prey and chaser on the same x or y coordinate, the other coordinate needs no change,
        # just go under/over or left/right of prey)
        perp_x = 0 if rot_x == 0 else (1 if rot_x > 0 else -1)
        perp_y = 0 if rot_y == 0 else (1 if rot_y > 0 else -1)
        # Return the two prey adjacent cells that are one cell away, along the perpendicular direction
        return ((focus_x + perp_x, focus_y + perp_y), (focus_x - perp_x, focus_y - perp_y))

    def flank_target(self, obs, peers, chaser_id, focus_prey):
        '''Returns the target cell for the agent to flank the prey, closest from options if not taken'''

        chaser_pos = next((x, y) for peer_id, x, y in peers if peer_id == chaser_id)
        # Get the two flank options (perpendicular adjacent cells to the prey)
        option1, option2 = self.flank_options(chaser_pos, focus_prey)
        # Check possible flankers and if agent is eligible to flank, return no target if not
        flanker_ids = self.flanker_candidate_ids(obs, peers, chaser_id, focus_prey)
        if obs["agent_id"] not in flanker_ids:
            return None

        # Assign the two flank options to the flankers, closest from options if not taken
        used_slots = []
        assignments = {}
        # Iterate over flanker candidates, get their position and compute closest flank option
        for flanker_id in flanker_ids:
            flanker_pos = next((x, y) for peer_id, x, y in peers if peer_id == flanker_id)
            distance1 = manhattan(flanker_pos[0], flanker_pos[1], option1[0], option1[1])
            distance2 = manhattan(flanker_pos[0], flanker_pos[1], option2[0], option2[1])
            # If both options are not taken, choose the closest
            if option1 not in used_slots and option2 not in used_slots:
                chosen = option1 if distance1 <= distance2 else option2
            # If any option is taken, choose the other
            elif option1 in used_slots:
                chosen = option2
            elif option2 in used_slots:
                chosen = option1
            # Failsafe: for the scenario where both options are taken but there are more than two flankers
            else:
                chosen = option1 if distance1 <= distance2 else option2
            # Assign the chosen option to the flanker and add to used slots
            used_slots.append(chosen)
            assignments[flanker_id] = chosen
        # Return agent's assigned target
        return assignments.get(obs["agent_id"])

    def known_peers(self, obs):
        """Return (peer_id, x, y) for the agent itself and visible allies, sorted by id"""
        
        # Initialize peers list with the agent itself and the IDs seen for deduplication
        peers = [(obs["agent_id"], obs["agent_x"], obs["agent_y"])]
        seen_ids = {obs["agent_id"]}
        # Iterate over visible allies and add to peers if not already seen
        for ally_x, ally_y, ally_id in obs.get("visible_allies", ()):
            if ally_id in seen_ids:
                continue
            seen_ids.add(ally_id)
            peers.append((ally_id, ally_x, ally_y))
        # Sort peers by ID (sorts with first element of tuple first)
        peers.sort()
        return peers

    def pack_prey_candidates(self, obs):
        """Known prey, one entry per ID: direct sightings have priority over comm reports"""

        seen_ids = set()
        candidates = []
        # Iterate over visible prey and add to candidates if not already seen
        for enemy_x, enemy_y, enemy_id in obs.get("visible_enemies", ()):
            # Store visible prey IDs and positions
            seen_ids.add(enemy_id)
            candidates.append((enemy_x, enemy_y, enemy_id))
        # Iterate over prey communicated to agent
        for prey in obs.get("shared_enemies", ()):
            sender_id, enemy_x, enemy_y, enemy_id = prey
            # If there is direct information about prey, skip comm report
            if enemy_id in seen_ids:
                continue
            seen_ids.add(enemy_id)
            candidates.append((enemy_x, enemy_y, enemy_id))
        return candidates

    def pick_pack_prey(self, peer_positions, prey_candidates):
        """Returns prey that minimizes the sum of Manhattan distances from pack peers"""
        
        # Failsafe: if there is no prey candidates, return no prey
        if not prey_candidates:
            return None
        # Initialize best target, distance, and ID
        best_target = None
        best_score = None
        best_id = None
        # Iterate over all prey candidates and compute the sum of Manhattan distances to pack peers
        for prey_x, prey_y, prey_id in prey_candidates:
            total = sum(manhattan(peer_x, peer_y, prey_x, prey_y) for peer_x, peer_y in peer_positions)
            # If this prey has a smaller total distance than the best so far, update the best target, distance, and ID
            # If this prey has the same total distance, keep the prey with smallest ID
            if best_score is None or total < best_score or (total == best_score and prey_id < best_id):
                best_score = total
                best_target = (prey_x, prey_y)
                best_id = prey_id
        return best_target

    def flank(self, obs, legal):
        '''Returns the action intention for the agent to flank the prey'''
        
        # Get role target and agent position
        role_target = obs.get("role_target")
        agent_x, agent_y = obs["agent_x"], obs["agent_y"]
        # If no role target, just chase the last seen enemy
        if role_target is None:
            return self.chase(obs, None, legal)
        # If we reached the flank position, stay and wait
        target_x, target_y = role_target
        if agent_x == target_x and agent_y == target_y:
            return co.STAY
        # Otherwise, move to the flank target position
        return self.move_to(agent_x, agent_y, target_x, target_y, legal)


    ################################### Search ################################################################

    def init_search_heading(self, obs):
        """Pick a heading direction to persist while searching. Agent will keep moving in this
           direction until a new visible ally appears or prey is known.
           Uses the sum of vectors away from each visible ally to get best direction. 
           With no visible allies, picks a deterministic direction"""

        agent_x, agent_y = obs["agent_x"], obs["agent_y"]
        visible_allies = obs.get("visible_allies", ())
        # If there are visible allies, iterate over them and compute the sum of vectors away from each visible ally
        if visible_allies:
            x_away = 0
            y_away = 0
            for ally_x, ally_y, ally_id in visible_allies:
                x_away += agent_x - ally_x
                y_away += agent_y - ally_y
            # If the sum of vectors is zero (such as when surrounded by allies in all directions), pick a deterministic direction
            if x_away == 0 and y_away == 0:
                x_away = 1 if obs["agent_id"] % 2 == 0 else -1
                y_away = 1 if obs["agent_id"] % 3 == 0 else -1
            # Get best matching direction from the sum of vectors
            return self.heading_from_vector(x_away, y_away, obs["agent_id"])
        # If there are no visible allies, pick a deterministic direction
        directions = (co.UP, co.DOWN, co.LEFT, co.RIGHT)
        return directions[obs["agent_id"] % len(directions)]

    def heading_from_vector(self, vector_x, vector_y, agent_id):
        """Computes the best-matching action from a direction vector"""

        candidates = []
        # Iterate over all possible actions and compute an alignment score
        for action in (co.UP, co.DOWN, co.LEFT, co.RIGHT):
            delta_x, delta_y = co.ACTION_DELTA[action]
            # Failsafe: if the action is STAY, skip as it's not a valid direction
            if delta_x == 0 and delta_y == 0:
                continue
            # Compute alignment score. If <= 0, then action is either perpendicular or against the direction vector and we discard it
            alignment = delta_x * vector_x + delta_y * vector_y
            if alignment <= 0:
                continue
            candidates.append((alignment, action))
        # If no action is good enough, pick a deterministic action
        if not candidates:
            cardinals = (co.UP, co.DOWN, co.LEFT, co.RIGHT)
            return cardinals[agent_id % len(cardinals)]
        # Sort candidates by alignment score and then by action
        candidates.sort(key=lambda item: (-item[0], item[1]))
        # Get best aligned action
        best_alignment = candidates[0][0]
        # Check if there are multiple actions with the same best alignment score
        tied = [action for alignment, action in candidates if alignment == best_alignment]
        # If there are multiple actions with the same best alignment score, pick one in alphabetical order
        return min(tied)

    def search(self, obs, legal):
        """Returns the action intention for the agent to search for prey.
           Means returning the action that is best aligned with the search direction,
           especially when the search direction's best action is not legal"""
        
        # If no heading exists, pick a random legal move
        heading = obs.get("search_heading")
        if heading is None:
            directions = [action for action in legal if action != co.STAY]
            return self.rng.choice(directions) if directions else co.STAY

        # If the heading is legal, return it
        if heading in legal:
            return heading
        # Otherwise, compute the second best aligned action (the first one is the heading itself)
        return self.best_aligned_action(legal, heading)

    def best_aligned_action(self, legal, heading):
        '''Computes the best aligned legal action from a heading, to be used when
           the heading's best action is not legal'''
        
        # Get the heading vector coordinates
        heading_x, heading_y = co.ACTION_DELTA[heading]
        best = None
        best_score = None
        # Iterate over all legal actions and compute the alignment score with the heading vector
        for action in legal:
            # Failsafe: if the action is STAY, skip as it's not a valid direction
            if action == co.STAY:
                continue
            # Get the candidate action's direction vector
            delta_x, delta_y = co.ACTION_DELTA[action]
            # Compute the alignment score
            score = delta_x * heading_x + delta_y * heading_y
            # If this action has a better alignment score than the best so far, update the best action and score
            if best_score is None or score > best_score:
                best_score = score
                best = action
        # If a best action was found, return it
        if best is not None:
            return best
        # If no best action was found, stay if possible, otherwise pick a random legal action
        return co.STAY if co.STAY in legal else self.rng.choice(legal)

############################################ Optimal mode #######################################################

    def optimal(self, obs, legal):
        """Returns action intention that moves toward pack prey in the shortest path possible"""
        
        # Get pack target from observation
        target = obs.get("pack_target")
        # Failsafe: if no pack target is found, return a random legal action
        if target is None:
            return self.rng.choice(legal)
        # Return optimal action intention towards pack target
        target_x, target_y = target
        return bfs_first_step((obs["agent_x"], obs["agent_y"]), (target_x, target_y), obs["grid_width"], obs["grid_height"], obs["wall_cells"], legal, self.rng)


############################################ Flee mode (prey) #######################################################


    def flee(self, obs, last_seen_enemy, legal):
        '''Returns the action intention for the agent to flee from predators.
           Chooses to avoid fleeing into cells currently occupied by active enemies. If
           no active enemies are present, it will wander around the map with a bias towards staying
           close to visible allies. When possible, actions that keep prey close to visible allies are preferred.
           This helps in stunning predators'''
        
        # Get the active enemies and visible allies
        active = obs["active_enemies"]
        visible_allies = obs.get("visible_allies", ())
        agent_x, agent_y = obs["agent_x"], obs["agent_y"]

        # If there is an active enemy, last_seen_enemy is the current closest active enemy
        if active:
            target = last_seen_enemy
            stale = False
        # If there is no active enemy, last_seen_enemy is a previously seen enemy from memory and the agent flees from them
        elif last_seen_enemy is not None:
            target = last_seen_enemy
            stale = True
        # If there is no active enemy nor memory to flee from, wander around the map with a bias towards staying
        # close to visible allies
        else:
            return self.wander_with_cohesion(agent_x, agent_y, legal, visible_allies)

        # If prey end up on the last seen enemy's position, it means predator is no longer there and prey can wander map
        target_x, target_y = target
        if stale and agent_x == target_x and agent_y == target_y:
            return self.wander_with_cohesion(agent_x, agent_y, legal, visible_allies)

        # Compute current distances to all non-primary enemies
        current_other_distances = {}
        # Iterate over all active enemies, skip primary enemy, and compute distance to each enemy
        for enemy_x, enemy_y, enemy_id in active:
            if enemy_x == target_x and enemy_y == target_y:
                continue
            current_other_distances[enemy_id] = manhattan(agent_x, agent_y, enemy_x, enemy_y)

        # Get all legal actions except STAY
        directions = [action for action in legal if action != co.STAY]
        # Get all cells currently occupied by active enemies and filter directions that would lead to them
        enemy_cells = {(enemy_x, enemy_y) for enemy_x, enemy_y, enemy_id in active}
        directions = [action for action in directions if apply_action(agent_x, agent_y, action) not in enemy_cells]

        # Check if actions move towards any secondary enemy and avoid those
        safe_directions = []
        for action in directions:
            # Get the next position after taking the action
            next_x, next_y = apply_action(agent_x, agent_y, action)
            gets_closer_to_other = False
            # Iterate over all active secondary enemies and check if the action moves towards them
            for enemy_x, enemy_y, enemy_id in active:
                if enemy_id not in current_other_distances:
                    continue
                # Check if the action moves towards the enemy (next distance is less than current distance)
                if manhattan(next_x, next_y, enemy_x, enemy_y) < current_other_distances[enemy_id]:
                    gets_closer_to_other = True
                    break
            # If the action does not move towards any secondary enemy, add it to the safe directions
            if not gets_closer_to_other:
                safe_directions.append(action)

        # If there are safe directions, add STAY if it's legal
        if safe_directions:
            candidate_actions = safe_directions + ([co.STAY] if co.STAY in legal else [])
        # If there are no safe directions, keep legal directions. If every action closes in on a secondary
        # enemy, rather than simply staying, pick the direction that moves away from the primary enemy
        elif directions:
            candidate_actions = directions
        # If there are no safe directions nor legal directions, stay if possible
        else:
            candidate_actions = [co.STAY] if co.STAY in legal else list(legal)

        # Definition of the score for the candidate actions
        def score_prey(action):
            '''Return the score for candidate actions. Contains two terms: the distance to the 
               primary enemy and the cohesion term. Prey will prefer actions that move away from 
               primary enemy, and will choose actions that stay close to visible allies when
               primary distance is tied'''
            # Get the next position after taking the action
            next_x, next_y = apply_action(agent_x, agent_y, action)
            # Calculate distance to primary enemy
            primary_distance = manhattan(next_x, next_y, target_x, target_y)
            # Calculate cohesion term, where prey prefer actions that keep them within radius 1 of a visible ally
            if visible_allies:
                # Get minimum Chebyshev distance to any visible ally
                ally_distance = min(chebyshev(next_x, next_y, ally_x, ally_y) for (ally_x, ally_y, ally_id) in visible_allies)
                # Compute cohesion term
                ally_term = abs(ally_distance - 1)
            else:
                ally_term = 0
            # Return the score, where higher distance is preferred, and lower ally term is preferred
            return (-primary_distance, ally_term, self.rng.randint(0, 1000))

        # Return the action with the minimum score (moves furthest from primary enemy, and closest to allies)
        return min(candidate_actions, key=score_prey)

    def wander_with_cohesion(self, agent_x, agent_y, legal, visible_allies):
        """Returns the action intention for the agent to wander around the map with a bias towards staying
           close to visible allies"""

        # If there are no visible allies, return a random legal action
        if not visible_allies:
            return self.rng.choice(legal)

        # Get all legal actions except stay and filter directions that would lead to allies
        directions = [action for action in legal if action != co.STAY]
        ally_positions = {(ally_x, ally_y) for ally_x, ally_y, ally_id in visible_allies}
        directions = [action for action in directions if apply_action(agent_x, agent_y, action) not in ally_positions]
        # Bring back stay if it's legal
        candidates = directions + ([co.STAY] if co.STAY in legal else [])
        # If no actions do not lead to allies, return all legal actions
        if not candidates:
            candidates = list(legal)

        def score(action):
            '''Returns the cohesion score for a candidate action.
               The score is the minimum Chebyshev distance to any visible ally. The lower the score,
               the better the action is'''

            # Get next position after taking the action and the minimum Chebyshev distance to any visible ally from that position
            next_x, next_y = apply_action(agent_x, agent_y, action)
            ally_distance = min(chebyshev(next_x, next_y, ally_x, ally_y) for (ally_x, ally_y, ally_id) in visible_allies)
            # Return the score, where lower distance is preferred
            return (abs(ally_distance - 1), self.rng.randint(0, 1000))
        # Return the action with the minimum score (moves closest to allies)
        return min(candidates, key=score)


