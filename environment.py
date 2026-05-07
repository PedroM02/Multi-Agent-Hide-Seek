import agent_utils as au


class AgentBody:
    """Defines an agent's body/physical presence in the environment."""
    def __init__(self, agent_id, team, x, y, alive=True):
        self.agent_id = agent_id
        self.team = team
        self.x = x
        self.y = y
        self.alive = alive


class Environment:
    """Rectangle of dimensions (width, height) with optional obstacle cells."""

    def __init__(self, width, height, walls):

        # Initialize basic environment with walls
        if width < 1 or height < 1:
            raise ValueError("Width and Height must be larger than 0")
        self.width = width
        self.height = height
        self.wall_cells = set()
        self.agent_bodies = {}

        # Add walls to the environment
        if walls:
            for wall_x, wall_y in walls:
                # Check if the wall is in bounds
                if self.is_in_bounds(wall_x, wall_y):
                    self.wall_cells.add((wall_x, wall_y))


    def is_in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_wall(self, x, y):
        return (x, y) in self.wall_cells

    def get_free_cells(self):
        free_cells = []
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) not in self.wall_cells:
                    free_cells.append((x, y))
        return free_cells
        

    def place_agent(self, body):
        # Check agent position is in bounds
        if not self.is_in_bounds(body.x, body.y):
            raise ValueError("Agent position is out of bounds")
        # Check agent position is not on a wall
        if self.is_wall(body.x, body.y):
            raise ValueError("Agent position is on a wall")

        self.agent_bodies[body.agent_id] = body

    def set_agent_positions(self, num_predators, num_prey, rng, first_id=0):

        # Reset agent bodies and get available cells to place agents
        self.agent_bodies.clear()
        free_cells = self.get_free_cells()

        if len(free_cells) < num_predators + num_prey:
            raise ValueError("Not enough free cells for all agents.")

        # Shuffle free cell list so they are not in a specific order
        rng.shuffle(free_cells)

        # Place predators and prey in the environment
        bodies = []
        idx = 0 # Common index for free cells so prey are placed in cells where predators were not placed
        agent_id = first_id
        for _ in range(num_predators):
            x, y = free_cells[idx]
            idx += 1
            agent_body = AgentBody(agent_id, au.TEAM_PREDATOR, x, y)
            self.place_agent(agent_body)
            bodies.append(agent_body)
            agent_id += 1
        for _ in range(num_prey):
            x, y = free_cells[idx]
            idx += 1
            agent_body = AgentBody(agent_id, au.TEAM_PREY, x, y)
            self.place_agent(agent_body)
            bodies.append(agent_body)    
            agent_id += 1
        return bodies

    def alive_bodies(self):
        for agent_body in self.agent_bodies.values():
            if agent_body.alive:
                yield agent_body

    def legal_actions(self, agent_id):
        agent_body = self.agent_bodies[agent_id]
        output_actions = []
        # If agent is not alive, it must not do anything
        if not agent_body.alive:
            return (au.STAY,)
        # Check all possible actions and assess legality
        for action in au.MOVE_ACTIONS:
            delta_x, delta_y = au.ACTION_DELTA[action]
            next_x, next_y = agent_body.x + delta_x, agent_body.y + delta_y

            if not self.is_in_bounds(next_x, next_y) or self.is_wall(next_x, next_y):
                # Failsafe to guarantee agent has always a legal action if somehow ends up in an unexpected situation
                if action == au.STAY:
                    output_actions.append(au.STAY)
                continue
            output_actions.append(action)
        return tuple(output_actions)

    def set_position(self, agent_id, x, y):
        agent_body = self.agent_bodies[agent_id]
        agent_body.x, agent_body.y = x, y

    def apply_captures(self):
        cells_with_agents = {}
        # Check alive agents' positions and group them by cell
        for agent_body in self.alive_bodies():
            agent_cell = (agent_body.x, agent_body.y)
            cells_with_agents[agent_cell] = cells_with_agents.get(agent_cell, []) + [agent_body]
        
        # Check if there are predators and prey in the same cell and capture the prey
        captured = []
        for cell, bodies_in_cell in cells_with_agents.items():
            preds = [agent_body for agent_body in bodies_in_cell if agent_body.team == au.TEAM_PREDATOR]
            preys = [agent_body for agent_body in bodies_in_cell if agent_body.team == au.TEAM_PREY]
            if preds and preys:
                for prey in preys:
                    prey.alive = False
                    captured.append(prey.agent_id)
        return captured

    def any_prey_alive(self):
        return any(agent_body.team == au.TEAM_PREY and agent_body.alive for agent_body in self.agent_bodies.values())

    def any_predator_alive(self):
        return any(agent_body.team == au.TEAM_PREDATOR and agent_body.alive for agent_body in self.agent_bodies.values())
