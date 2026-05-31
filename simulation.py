import random

import constants as co
from action_resolution import resolve_actions
from agent import build_agents_for_env
from utils import chebyshev, bfs_search
from environment import Environment
from observation_definition import attach_team_comms, build_observation


def generate_walls(width, height, num_walls, wall_size, rng):
    """Generates the specified number of walls of a wall_size number of segments. Walls are
       placed randomly on the map but never fill an entire row or column.

       Returns the list of wall cell coordinates."""

    occupied = set()
    walls = []

    # Generate the specified number of walls
    for _ in range(num_walls):
        # Try up to 50 random placements before giving up on this wall.
        for attempt in range(50):
            # Choose a random direction for the wall
            horizontal = rng.choice([True, False])
            
            if horizontal:
                # Get the maximum x coordinate for the wall taking into account wall size
                max_x = width - wall_size
                # If the wall would be too large, skip this attempt
                if max_x < 0:
                    continue
                # Choose a random origin for the wall
                origin_x = rng.randint(0, max_x)
                origin_y = rng.randint(0, height - 1)
                # Generate the cells for the wall from left to right
                cells = [(origin_x + i, origin_y) for i in range(wall_size)]
                # If generated wall blocks the full width of the map, skip this attempt
                if len(cells) >= width:
                    continue
            else:
                # Get the maximum y coordinate for the wall taking into account wall size
                max_y = height - wall_size
                # If the wall would be too large, skip this attempt
                if max_y < 0:
                    continue
                # Choose a random origin for the wall
                origin_x = rng.randint(0, width - 1)
                origin_y = rng.randint(0, max_y)
                # Generate the cells for the wall from top to bottom
                cells = [(origin_x, origin_y + i) for i in range(wall_size)]
                # If generated wall blocks the full height of the map, skip this attempt
                if len(cells) >= height:
                    continue

            # Skip attempt if any of the generated cell walls are already occupied
            if any(cell in occupied for cell in cells):
                continue

            # Add the generated cells to the occupied cells and walls
            for cell in cells:
                occupied.add(cell)
            walls.extend(cells)
            # If attempt was successful, break out of attempts loop and move on to next wall
            break

    return walls


def alive_oracle_prey(env):
    """Returns positions of all prey in environment, regardless of vision radius. To be used for optimal predators"""
    prey = []
    # Iterate over all agent bodies and store positions and ID of those belong to alive prey
    for agent_body in env.agent_bodies.values():
        if agent_body.alive and agent_body.team == co.TEAM_PREY:
            prey.append((agent_body.x, agent_body.y, agent_body.agent_id))
    prey.sort(key=lambda t: (t[2], t[0], t[1]))
    return tuple(prey)

def select_pack_prey_id(predator_positions, oracle_prey, width, height, wall_cells):
    """Returns prey ID of prey that minimizes the sum of the shortest paths distances to all predators"""
    
    # Failsafe: if there is no prey positions given by oracle or no predator positions, no prey is selected
    if not oracle_prey or not predator_positions:
        return None

    # Penalty for a prey that is unreachable by a predator. Calculated as the number of cells needed to 
    # walk the whole map times the number of predators, so that if one predator cannot reach the prey, its
    # distance score becomes worse than any prey that is not unreachable by all predators
    unreachable_penalty = width * height * len(predator_positions)
    best_id = None
    best_distance = None

    # Iterate over all prey given by oracle
    for prey_x, prey_y, prey_id in oracle_prey:
        # Initialize total distance to 0
        total = 0
        # Iterate over all predators
        for predator_position in predator_positions:
            # Compute shortest path from predator to prey
            search = bfs_search(predator_position, (prey_x, prey_y), width, height, wall_cells)
            # If the path is not found, apply unreachable penalty
            if search is None:
                total += unreachable_penalty
            # If the path is found, add the distance to the total distance
            else:
                distance, previous_cells = search
                total += distance
        # If this prey has a smaller total distance than the best so far, update the best prey
        # If this prey has the same total distance, keep the prey with smallest ID
        if best_distance is None or total < best_distance or (total == best_distance and (best_id is None or prey_id < best_id)):
            best_distance = total
            best_id = prey_id
    return best_id





class SimulationConfig:
    def __init__(self):
        self.width = 10
        self.height = 8
        self.timesteps = 200
        self.vision_radius_predator = 2
        self.vision_radius_prey = 2
        self.num_predators = 1
        self.num_prey = 1
        self.seed = 0
        self.num_walls = 0
        self.wall_size = 2
        self.comms = None
        self.prey_defend = None
        self.stun_duration = 3
        self.mode = co.MODE_CHASE
        self.roles_searcher = False

# Config copy to override and increment seed at each run
def copy_config(base, **overrides):
    new_config = SimulationConfig()
    new_config.width = base.width
    new_config.height = base.height
    new_config.timesteps = base.timesteps
    new_config.vision_radius_predator = base.vision_radius_predator
    new_config.vision_radius_prey = base.vision_radius_prey
    new_config.num_predators = base.num_predators
    new_config.num_prey = base.num_prey
    new_config.seed = base.seed
    new_config.num_walls = base.num_walls
    new_config.wall_size = base.wall_size
    new_config.comms = base.comms
    new_config.prey_defend = base.prey_defend
    new_config.stun_duration = base.stun_duration
    new_config.mode = base.mode
    new_config.roles_searcher = base.roles_searcher
    for key, value in overrides.items():
        setattr(new_config, key, value)
    return new_config


class Run:
    '''Class representing a single run of the simulation, with multiple timesteps. Initializes the 
       environment and agents, and runs each timestep until the run is over.'''
    def __init__(self, config, rng):
        self.config = config
        self.rng = rng

        # Generate walls if specified in config
        walls = None
        if config.num_walls > 0:
            walls = generate_walls(config.width, config.height, config.num_walls, config.wall_size, rng)

        # Initialize environment with walls
        self.env = Environment(config.width, config.height, walls)
        # Assign positions to agents
        self.env.set_agent_positions(self.config.num_predators, self.config.num_prey, self.rng)
        # Build agents for the environment
        self.agents = build_agents_for_env(self.env, self.rng, self.config)
        for agent in self.agents:
            agent.reset_memory()

        self.step_index = 0
        self.outcome = co.OUTCOME_ONGOING
        self.optimal_focus_prey_id = None
        self.last_captured = []

    def build_step_observations(self):
        """Builds observations for all alive agents in the timestep.
           Starts by building raw basic observation, enriches it with comms, agent perceives and processes
           the information to create final observation. If optimal mode is enabled, full map information and
           shared prey target is injected into the observation."""
        # Build and store observations for all alive agents by agent ID
        all_obs = {}
        for agent in self.agents:
            agent_body = self.env.agent_bodies[agent.agent_id]
            if not agent_body.alive:
                continue
            all_obs[agent.agent_id] = build_observation(self.env, agent.agent_id, self.config.vision_radius_predator if agent_body.team == co.TEAM_PREDATOR else self.config.vision_radius_prey)

        # Attach team communications to observations
        attach_team_comms(all_obs, self.config)
        # Allow the agents to process and enrich the observations
        for agent in self.agents:
            obs = all_obs.get(agent.agent_id)
            if obs is not None:
                agent.perceive(obs)
        # Inject oracle observations if optimal mode is enabled
        self.inject_oracle_obs(all_obs)
        return all_obs

    def inject_oracle_obs(self, all_obs):
        """Injection of full map information and shared prey target into the observation when in optimal mode"""
        # Return nothing if not in optimal mode
        if self.config.mode != co.MODE_OPTIMAL:
            return

        # Retrieve all alive prey, walls and grid dimensions
        all_alive_prey = alive_oracle_prey(self.env)
        walls = set(self.env.wall_cells)
        grid_width, grid_height = self.env.width, self.env.height

        # Retrieve positions of all alive predators
        predator_positions = []
        for agent in self.agents:
            agent_body = self.env.agent_bodies[agent.agent_id]
            if agent_body.alive and agent_body.team == co.TEAM_PREDATOR:
                predator_positions.append((agent_body.x, agent_body.y))

        # Select the prey that minimizes the sum of the shortest paths distances to all predators
        pack_target = None
        alive_prey_ids = {prey_id for prey_x, prey_y, prey_id in all_alive_prey}
        # Failsafe: if the optimal focus prey ID is not alive, set it to None
        if self.optimal_focus_prey_id not in alive_prey_ids:
            self.optimal_focus_prey_id = None
        # If there is no optimal focus prey, select one
        if all_alive_prey and self.optimal_focus_prey_id is None:
            self.optimal_focus_prey_id = select_pack_prey_id(predator_positions, all_alive_prey, grid_width, grid_height, walls)
        # If an optimal focus prey has been chosen, look up its position from its ID and store it
        if self.optimal_focus_prey_id is not None:
            for prey_x, prey_y, prey_id in all_alive_prey:
                if prey_id == self.optimal_focus_prey_id:
                    pack_target = (prey_x, prey_y)
                    break

        # Inject full map information and shared prey target into the observation for all predators
        for obs in all_obs.values():
            if obs["team"] != co.TEAM_PREDATOR:
                continue
            obs["wall_cells"] = walls
            obs["grid_width"] = grid_width
            obs["grid_height"] = grid_height
            if pack_target is not None:
                obs["pack_target"] = pack_target

    def resolve_knockouts(self, intentions, mode):
        """Resolves knockouts from prey groups to predators. Identifies adjacent prey groups and applies
           stun to nr_prey-1 surrounding alive, non-stunned predators per group. Prey are forced to STAY when
           stunning."""

        # Retrieve all alive prey by ID
        alive_prey_by_id = {agent_body.agent_id: agent_body for agent_body in self.env.agent_bodies.values() if agent_body.alive and agent_body.team == co.TEAM_PREY}
        # If there is only one prey, there is no possible knockout
        if len(alive_prey_by_id) < 2:
            return

        # Retrieve all alive predators by ID
        alive_pred_by_id = {agent_body.agent_id: agent_body for agent_body in self.env.agent_bodies.values() if agent_body.alive and agent_body.team == co.TEAM_PREDATOR}
        # Failsafe: if there are no alive predators, there is no possible knockout
        if not alive_pred_by_id:
            return

        # Assess adjacent prey groups
        prey_ids = list(alive_prey_by_id)
        adjacent_prey = {prey_id: [] for prey_id in prey_ids}
        # Iterate over all prey and assess their adjacency (including diagonal neighbors)
        for i, prey_id_1 in enumerate(prey_ids):
            # Get first prey
            prey_body_1 = alive_prey_by_id[prey_id_1]
            # Iterate over all other prey
            for prey_id_2 in prey_ids[i + 1:]:
                # Get second prey
                prey_body_2 = alive_prey_by_id[prey_id_2]
                # Check if they are adjacent and add them to their adjacent prey list
                if chebyshev(prey_body_1.x, prey_body_1.y, prey_body_2.x, prey_body_2.y) <= 1:
                    adjacent_prey[prey_id_1].append(prey_id_2)
                    adjacent_prey[prey_id_2].append(prey_id_1)

        # Identify groups of adjacent prey given recursive assessment of adjacent prey
        visited = set()
        groups = []
        # Iterate over all prey and identify groups of adjacent prey
        for prey_id in prey_ids:
            # If the prey has already been visited, skip it
            if prey_id in visited:
                continue
            # Initialize stack with only current prey and group
            stack = [prey_id]
            group = []
            # Iterate over all adjacent prey and their respective adjacent prey, "recursively", and add them to the group until no more adjacent prey are found
            while stack:
                current = stack.pop()
                # If we have already seen this prey's neighbors, skip it
                if current in visited:
                    continue
                # Add the prey to the visited set and to the group
                visited.add(current)
                group.append(current)
                # Check all adjacent prey and add them to the stack if they have not been visited so their neighbors can be checked
                for neighbor_id in adjacent_prey[current]:
                    if neighbor_id not in visited:
                        stack.append(neighbor_id)
            group.sort()
            # Append group of neighboring prey
            groups.append(group)
        groups.sort(key=lambda group: group[0])



        defeated_this_step = set()
        predator_ids_sorted = sorted(alive_pred_by_id)

        # Iterate over all groups of adjacent prey and select predators to defeat
        for group in groups:
            # If group is a single prey, skip
            group_size = len(group)
            if group_size < 2:
                continue
            # Maximum number of predators that can be knocked out is nr_prey-1
            max_knockouts = group_size - 1

            # Initialize list of candidates to defeat
            candidates = []
            # Iterate over all predators and check if they are a candidate to defeat
            for predator_id in predator_ids_sorted:
                # If the predator has already been defeated this step, skip
                if predator_id in defeated_this_step:
                    continue
                # Get predator body
                predator_body = alive_pred_by_id[predator_id]
                # If the predator is stunned, skip
                if predator_body.stun_remaining > 0:
                    continue
                count = 0
                # Iterate over all prey in the group and check if there are at least 2 prey that are adjacent to the predator
                for prey_id in group:
                    prey_body = alive_prey_by_id[prey_id]
                    if chebyshev(predator_body.x, predator_body.y, prey_body.x, prey_body.y,) <= 1:
                        count += 1
                        if count >= 2:
                            break
                if count >= 2:
                    candidates.append(predator_id)

            # If there are no predators around the group of prey, skip
            if not candidates:
                continue
            # Select the maximum number of predators to defeat
            chosen = candidates[:max_knockouts]
            for predator_id in chosen:
                # Add the predator to the list of knocked out predators this step
                defeated_this_step.add(predator_id)
                predator_body = alive_pred_by_id[predator_id]
                # Update predator status based on prey defense mode
                if mode == "stun":
                    predator_body.stun_remaining = self.config.stun_duration
                else:
                    predator_body.alive = False
                # Force the predator to STAY for this step
                intentions[predator_id] = co.STAY
                # Force all prey in the group to STAY for this step
                for prey_id in group:
                    prey_body = alive_prey_by_id[prey_id]
                    if chebyshev(predator_body.x, predator_body.y, prey_body.x, prey_body.y) <= 1:
                        intentions[prey_id] = co.STAY

    def step_once(self, predator_actions=None, all_obs=None):
        """Runs a single timestep of the simulation."""
        # If the outcome of the previous step is not ongoing (some team won), stop the run
        if self.outcome != co.OUTCOME_ONGOING:
            return False
        # Build observations for all alive agents
        if all_obs is None:
            all_obs = self.build_step_observations()
        # Retrieve agents by ID
        agents_by_id = {agent.agent_id: agent for agent in self.agents}
        # Retrieve agent action intentions. From decision making if not in RL mode, injected from policy otherwise
        intentions = {}
        for agent in self.agents:
            agent_body = self.env.agent_bodies[agent.agent_id]
            if not agent_body.alive:
                continue
            obs = all_obs[agent.agent_id]
            # If predator actions exist, we are in RL mode and get actions from policy
            if (agent_body.team == co.TEAM_PREDATOR and predator_actions is not None):
                intentions[agent.agent_id] = predator_actions[agent.agent_id]
                predator_agent = agents_by_id[agent.agent_id]
                predator_agent.update_memory_from_obs(obs)
                predator_agent.clear_stale_memory_if_at_cell(obs)
            # Get action intention from decision making
            else:
                intentions[agent.agent_id] = agent.decide(obs)

        # If prey defend is set to stun, override the action of any predator with a stun remaining into STAY
        if self.config.prey_defend == "stun":
            for agent_body in self.env.agent_bodies.values():
                if (agent_body.team == co.TEAM_PREDATOR and agent_body.alive and agent_body.stun_remaining > 0):
                    intentions[agent_body.agent_id] = co.STAY
        # If the prey can defend themselves, resolve the knockouts (either stun or kill)
        if self.config.prey_defend is not None:
            self.resolve_knockouts(intentions, self.config.prey_defend)
        
        # Resolve actions for all alive agents
        resolve_actions(self.env, intentions)
        # Apply captures for all alive agents
        captured = self.env.apply_captures()
        # Store the last captured prey
        self.last_captured = captured
        # If there were captures and predator_actions was provided (RL mode), clear memory of predators for the captured prey's positions
        # This is to avoid continuous chasing memory for RL agents
        if captured and predator_actions is not None:
            capture_positions = {(self.env.agent_bodies[prey_id].x, self.env.agent_bodies[prey_id].y) for prey_id in captured}
            for agent in self.agents:
                agent_body = self.env.agent_bodies[agent.agent_id]
                if agent_body.team == co.TEAM_PREDATOR and agent_body.alive:
                    agent.clear_memory_at_positions(capture_positions)

        # If prey defend is set to stun, decrease the stun timer of all stunned predators
        if self.config.prey_defend == "stun":
            for agent_body in self.env.agent_bodies.values():
                if agent_body.team == co.TEAM_PREDATOR and agent_body.stun_remaining > 0:
                    agent_body.stun_remaining -= 1

        self.step_index += 1

        # Make run termination checks
        if not self.env.any_prey_alive():
            self.outcome = co.OUTCOME_PREDATORS_WIN
            return False
        if not self.env.any_predator_alive():
            self.outcome = co.OUTCOME_PREY_WIN
            return False
        if self.step_index >= self.config.timesteps:
            if self.env.any_prey_alive():
                self.outcome = co.OUTCOME_PREY_WIN
            else:
                self.outcome = co.OUTCOME_PREDATORS_WIN
            return False
        return True


class BatchSummary:
    def __init__(self):
        self.predator_wins = 0
        self.prey_wins = 0
        self.total_steps = 0
        self.predator_win_steps = 0
        self.prey_win_steps = 0
        self.runs = 0


def run_batch(config, num_runs):
    '''Runs a batch of runs and returns the summary of results across all runs.'''
    # Initialize summary
    accumulator = BatchSummary()
    # Execute each run and store results
    for i in range(num_runs):
        # Create a new run config with a different seed
        run_config = copy_config(config, seed=config.seed + i)
        rng = random.Random(run_config.seed)
        # Create a new run with the config and random number generator
        run = Run(run_config, rng)
        # Execute all steps in the run until it is over
        while run.step_once():
            pass
        # Store accumulated results in the summary
        accumulator.runs += 1
        accumulator.total_steps += run.step_index
        if run.outcome == co.OUTCOME_PREDATORS_WIN:
            accumulator.predator_wins += 1
            accumulator.predator_win_steps += run.step_index
        elif run.outcome == co.OUTCOME_PREY_WIN:
            accumulator.prey_wins += 1
            accumulator.prey_win_steps += run.step_index
    return accumulator


def format_batch_summary(summary, config):
    """Formats the run batch summary into a summary table for output."""

    def fmt_mean(total, count):
        '''Avoids division by zero and formats the mean to 2 decimal places'''
        if count <= 0:
            return ""
        return f"{total / count:.2f}"
    
    # Define headers and header widths
    header_cells = [
        ("Number of Predators", 19),
        ("Number of Prey", 14),
        ("Predator Wins", 13),
        ("Prey Wins", 9),
        ("Mean Run Timesteps", 18),
        ("Mean Run Timesteps in Predator-won", 34),
        ("Mean Run Timesteps in Prey-won", 30)]
    
    # Define content
    data_cells = [
        str(config.num_predators),
        str(config.num_prey),
        str(summary.predator_wins),
        str(summary.prey_wins),
        fmt_mean(summary.total_steps, summary.runs),
        fmt_mean(summary.predator_win_steps, summary.predator_wins),
        fmt_mean(summary.prey_win_steps, summary.prey_wins)]

    # Define whole header
    header = "| " + " | ".join(label for label, width in header_cells) + " |"
    # Define --- separator between header and content
    sep = "| " + " | ".join("-" * width for label, width in header_cells) + " |"
    # Define content row, keeping header widths for alignment
    row = "| " + " | ".join(f"{value:>{width}}" for value, (label, width) in zip(data_cells, header_cells)) + " |"
    # Return table
    return "\n".join([header, sep, row])
