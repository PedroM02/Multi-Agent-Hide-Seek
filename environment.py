import constants as co
from utils import apply_action, in_grid_bounds


class AgentBody:
    """Defines an agent's body/physical presence in the environment. Includes agent's Id, team, position,
       whether it is alive and the number of turns it has been stunned for"""
    def __init__(self, agent_id, team, x, y, alive=True):
        self.agent_id = agent_id
        self.team = team
        self.x = x
        self.y = y
        self.alive = alive
        self.stun_remaining = 0


class Environment:
    """Rectangle of dimensions (width, height) with optional wall cells."""

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
                if in_grid_bounds(wall_x, wall_y, self.width, self.height):
                    self.wall_cells.add((wall_x, wall_y))


    def is_wall(self, x, y):
        '''Returns True if the cell at (x,y) is a wall'''
        return (x, y) in self.wall_cells

    def get_free_cells(self):
        '''Returns a list of all cells in map that are not walls'''
        free_cells = []
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) not in self.wall_cells:
                    free_cells.append((x, y))
        return free_cells


    def place_agent(self, agent_body):
        '''Places an agent in the environment if its assigned position is valid'''
        # Check agent position is in bounds
        if not in_grid_bounds(agent_body.x, agent_body.y, self.width, self.height):
            raise ValueError("Agent position is out of bounds")
        # Check agent position is not on a wall
        if self.is_wall(agent_body.x, agent_body.y):
            raise ValueError("Agent position is on a wall")

        self.agent_bodies[agent_body.agent_id] = agent_body

    def set_agent_positions(self, num_predators, num_prey, rng, first_id=0):
        '''Assigns available map positions to agent bodies and places them in the environment. First predators, then prey'''

        # Reset agent bodies and get available cells to place agents
        self.agent_bodies.clear()
        free_cells = self.get_free_cells()

        if len(free_cells) < num_predators + num_prey:
            raise ValueError("Not enough free cells for all agents.")

        # Shuffle free cell list so they are not in a specific order
        rng.shuffle(free_cells)

        # Place predators and prey in the environment with common index for free cells so prey are placed in cells where predators were not placed
        idx = 0
        agent_id = first_id
        for _ in range(num_predators):
            x, y = free_cells[idx]
            idx += 1
            agent_body = AgentBody(agent_id, co.TEAM_PREDATOR, x, y)
            self.place_agent(agent_body)
            agent_id += 1
        for _ in range(num_prey):
            x, y = free_cells[idx]
            idx += 1
            agent_body = AgentBody(agent_id, co.TEAM_PREY, x, y)
            self.place_agent(agent_body)
            agent_id += 1

    def alive_bodies(self):
        '''Returns agent bodies that are alive'''
        return [agent_body for agent_body in self.agent_bodies.values() if agent_body.alive]

    def alive_predator_ids(self):
        '''Returns IDs os all alive predators'''
        return [body.agent_id for body in self.alive_bodies() if body.team == co.TEAM_PREDATOR]

    def legal_actions(self, agent_id):
        '''Returns a tuple of all legal actions for a given agent'''
        agent_body = self.agent_bodies[agent_id]
        output_actions = []
        # If agent is not alive, it must not do anything
        if not agent_body.alive:
            return (co.STAY,)

        # Check all possible movement actions and assess legality
        for action in co.MOVE_ACTIONS:
            next_x, next_y = apply_action(agent_body.x, agent_body.y, action)

            if not in_grid_bounds(next_x, next_y, self.width, self.height) or self.is_wall(next_x, next_y):
                # Failsafe: guarantee agent has always a legal action if somehow ends up in an unexpected situation
                if action == co.STAY:
                    output_actions.append(co.STAY)
                continue
            output_actions.append(action)

        return tuple(output_actions)

    def set_position(self, agent_id, x, y):
        '''Sets the positional attributes of an agent body'''
        agent_body = self.agent_bodies[agent_id]
        agent_body.x, agent_body.y = x, y

    def apply_captures(self):
        cells_with_agents = {}
        # Check alive agents' positions and group them by cell
        for agent_body in self.alive_bodies():
            agent_cell = (agent_body.x, agent_body.y)
            cells_with_agents[agent_cell] = cells_with_agents.get(agent_cell, []) + [agent_body]
        
        # Check if there are predators and prey in the same cell and capture the prey.
        # Stunned predators (stun_remaining > 0) are skipped here since stunned predators do not capture
        captured = []
        for cell, bodies_in_cell in cells_with_agents.items():
            preds = [agent_body for agent_body in bodies_in_cell if agent_body.team == co.TEAM_PREDATOR and agent_body.stun_remaining == 0]
            preys = [agent_body for agent_body in bodies_in_cell if agent_body.team == co.TEAM_PREY]
            if preds and preys:
                for prey in preys:
                    prey.alive = False
                    captured.append(prey.agent_id)
        return captured

    def any_prey_alive(self):
        return any(agent_body.team == co.TEAM_PREY and agent_body.alive for agent_body in self.agent_bodies.values())

    def any_predator_alive(self):
        return any(agent_body.team == co.TEAM_PREDATOR and agent_body.alive for agent_body in self.agent_bodies.values())
