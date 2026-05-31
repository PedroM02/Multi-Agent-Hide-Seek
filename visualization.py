import random

import pygame

import constants as co
from simulation import (BatchSummary, Run, format_batch_summary)

# Define constants for the visualization, such as dimensions and colors
CELL = 36
MARGIN_TOP = 92
PAD_X = 8
HUD_COLOR = (20, 20, 24)
GRID_WALL = (35, 35, 40)
GRID_EMPTY = (210, 210, 215)
GRID_PRED = (110, 55, 180)
GRID_PREY = (55, 160, 75)
ROLE_LETTER_COLOR = (245, 245, 245)


def draw_legend(surface, font, x, y):
    """Draws team color squares and labels onto window legend area"""
    square_width, square_height = 14, 14
    foreground = (240, 240, 245)
    # Draw predator color square and label and glue it to window
    pygame.draw.rect(surface, GRID_PRED, (x, y + 2, square_width, square_height))
    surface.blit(font.render("Predator", True, foreground), (x + square_width + 6, y))
    # Draw prey color square and label and glue it to window
    prey_legend_x = x + square_width + 6 + font.size("Predator")[0] + 16
    pygame.draw.rect(surface, GRID_PREY, (prey_legend_x, y + 2, square_width, square_height))
    surface.blit(font.render("Prey", True, foreground), (prey_legend_x + square_width + 6, y))



def draw_grid(surface, env, agents, font, status, vision_radius_predator, vision_radius_prey):
    '''Draws the entire GUI grid and all its elements'''

    surface.fill(HUD_COLOR)
    grid_w = env.width * CELL
    grid_h = env.height * CELL
    grid_surface = pygame.Surface((grid_w, grid_h))
    grid_surface.fill(GRID_EMPTY)

    # Draw cells and walls on the grid surface
    for x in range(env.width):
        for y in range(env.height):
            # Draw cell rectangle on the grid surface
            rect = pygame.Rect(x * CELL, y * CELL, CELL, CELL)
            if env.is_wall(x, y):
                # Draw wall rectangle on the grid surface
                pygame.draw.rect(grid_surface, GRID_WALL, rect)
            # Draw cell border on the grid surface
            pygame.draw.rect(grid_surface, (0, 0, 0), rect, 1)

    # Get agents by ID
    agent_by_id = {agent.agent_id: agent for agent in agents}
    # Iterate over agents and draw them on grid
    for agent_body in env.agent_bodies.values():
        if not agent_body.alive:
            continue
        # Get agent rectangle and color
        rect = pygame.Rect(agent_body.x * CELL + 2, agent_body.y * CELL + 2, CELL - 4, CELL - 4)
        color = GRID_PRED if agent_body.team == co.TEAM_PREDATOR else GRID_PREY
        # Stunned predators are drawn dimmed
        stunned = agent_body.team == co.TEAM_PREDATOR and agent_body.stun_remaining > 0
        if stunned:
            color = tuple(channel // 2 for channel in color)
        # Draw agent rectangle on the grid surface
        pygame.draw.rect(grid_surface, color, rect, border_radius=4)
        # Draw roles letters on agent rectangle
        agent = agent_by_id.get(agent_body.agent_id)
        if agent is not None and agent.role is not None:
            letter = co.ROLE_LETTER.get(agent.role)
            if letter is not None:
                letter_color = (tuple(channel // 2 for channel in ROLE_LETTER_COLOR) if stunned else ROLE_LETTER_COLOR)
                label = font.render(letter, True, letter_color)
                label_width, label_height = label.get_size()
                center_x = agent_body.x * CELL + CELL // 2
                center_y = agent_body.y * CELL + CELL // 2
                grid_surface.blit(label, (center_x - label_width // 2, center_y - label_height // 2))

    # Add grid surface to main surface, draw legend and add extra hint text
    grid_x = max(0, (surface.get_width() - grid_w) // 2)
    surface.blit(grid_surface, (grid_x, MARGIN_TOP))

    surface.blit(font.render(status, True, (240, 240, 245)), (PAD_X, 8))
    draw_legend(surface, font, PAD_X, 34)
    hint_1 = font.render(
        f"Vision: Chebyshev predator r={vision_radius_predator} and prey r={vision_radius_prey}",
        True,
        (170, 170, 180),
    )
    hint_2 = font.render(
        "Next timestep: space/right   Auto Run: a   Reset Run: r   Next Run: n   Quit: esc",
        True,
        (170, 170, 180),
    )
    surface.blit(hint_1, (PAD_X, 58))
    surface.blit(hint_2, (PAD_X, 74))


def run_visualization(config, num_runs=1, rl_policy=None, rl_device=None, rl_algo=None, rl_use_search=False):
    '''Runs the visualization and the run-wise/step-wise simulation'''

    pygame.init()
    total_runs = max(1, num_runs)
    run_index = 0
    run_seed = config.seed
    rng = random.Random(run_seed)
    # Create a new run with the config and random number generator
    run = Run(config, rng)
    # Get font and sample status text
    font = pygame.font.Font(None, 18)
    status_sample = (f"Run {total_runs}/{total_runs}  Timestep 0/{config.timesteps}  Outcome={co.OUTCOME_ONGOING}  |  Seed={config.seed + total_runs - 1}")
    hint_text_1 = (f"Vision: Chebyshev Predator Radius={config.vision_radius_predator} and Prey Radius={config.vision_radius_prey}")
    hint_text_2 = "Next timestep: space/right   Auto Run: a   Reset Run: r   Next Run: n   Quit: esc"
    # Window sizing and config
    min_header_w = max(font.size(status_sample)[0], font.size(hint_text_1)[0], font.size(hint_text_2)[0]) + PAD_X * 2
    win_w = max(config.width * CELL, min_header_w)
    win_h = config.height * CELL + MARGIN_TOP
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("PredPrey Sim")
    # Create auto-run utils
    clock = pygame.time.Clock()
    auto = False
    step_ms = 180
    pygame.time.set_timer(pygame.USEREVENT, 0)

    # Initialize results summary and logged runs
    summary = BatchSummary()
    logged_runs = set()
    # Create RL search controller if enabled
    search_controller = None
    if rl_policy is not None and rl_use_search:
        from rl.team_search import PredatorSearchController

        search_controller = PredatorSearchController()

    def simulation_step():
        '''Runs a single step of the simulation, either RL or normal'''
        if rl_policy is not None:
            from rl.algo import IPPO
            from rl.inference import select_predator_actions

            predator_actions, all_obs = select_predator_actions(run, rl_policy, rl_device, search_controller, deterministic=True, algo=rl_algo or IPPO)
            run.step_once(predator_actions=predator_actions,all_obs=all_obs)
        else:
            run.step_once()

    def log_current_run_if_done():
        """Record results of the current run into the batch summary once it finishes."""
        # If the run is still ongoing, do not log results
        if run.outcome == co.OUTCOME_ONGOING:
            return
        # If the run has already been logged, do not log again
        if run_index in logged_runs:
            return
        # Add the run index to the logged runs set
        logged_runs.add(run_index)
        # Update the summary with the run results
        summary.runs += 1
        summary.total_steps += run.step_index
        if run.outcome == co.OUTCOME_PREDATORS_WIN:
            summary.predator_wins += 1
            summary.predator_win_steps += run.step_index
        elif run.outcome == co.OUTCOME_PREY_WIN:
            summary.prey_wins += 1
            summary.prey_win_steps += run.step_index
        print(f"run={run_index + 1}/{total_runs}  seed={run_seed}  outcome={run.outcome}  steps={run.step_index}", flush=True)

    def load_run(idx):
        '''Loads a new run of the given index. Run is reset if idx brings run_seed to the current seed. Nonlocal variables update variables outside function scope'''
        nonlocal run_index, run_seed, rng, run, search_controller
        run_index = idx
        # Get run seed from config. If resetting run, run_seed stays the same as previous run
        run_seed = config.seed + run_index
        rng = random.Random(run_seed)
        # Create a new run and resets search controller memory if enabled
        run = Run(config, rng)
        if search_controller is not None:
            search_controller.reset()

    def try_advance_run():
        '''Loads the next run if it exists'''
        if run_index + 1 >= total_runs:
            return False
        log_current_run_if_done()
        load_run(run_index + 1)
        return True

    def status_text():
        '''Returns the current run's status text'''
        return (
            f"Run {run_index + 1}/{total_runs}  "
            f"Timestep {run.step_index}/{run.config.timesteps}  "
            f"Current Outcome={run.outcome}  |  Seed={run_seed}"
        )

    running = True
    while running:
        for event in pygame.event.get():
            # Handle quit events
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                # If user presses space or right arrow, move to the next timestep if the run is ongoing
                elif event.key in (pygame.K_SPACE, pygame.K_RIGHT):
                    if run.outcome == co.OUTCOME_ONGOING:
                        simulation_step()
                        # If run has ended, log reuslts
                        if run.outcome != co.OUTCOME_ONGOING:
                            log_current_run_if_done()
                # If the user presses "a", toggle auto-run (turn on or off, depending on current state)
                elif event.key == pygame.K_a:
                    auto = not auto
                    pygame.time.set_timer(pygame.USEREVENT, step_ms if auto else 0)
                # If the user presses "r", reset the current run
                elif event.key == pygame.K_r:
                    load_run(run_index)
                    auto = False
                    pygame.time.set_timer(pygame.USEREVENT, 0)
                # If the user presses "n", advance to the next run if it exists
                elif event.key == pygame.K_n:
                    if run.outcome != co.OUTCOME_ONGOING:
                        if try_advance_run():
                            auto = False
                            pygame.time.set_timer(pygame.USEREVENT, 0)
            # Auto-run
            elif event.type == pygame.USEREVENT and auto:
                if run.outcome == co.OUTCOME_ONGOING:
                    simulation_step()
                    if run.outcome != co.OUTCOME_ONGOING:
                        log_current_run_if_done()
                elif not try_advance_run():
                    auto = False
                    pygame.time.set_timer(pygame.USEREVENT, 0)

        # Redraw the grid for next run, update the display and tick the clock
        draw_grid(screen, run.env, run.agents, font, status_text(), config.vision_radius_predator, config.vision_radius_prey)
        pygame.display.flip()
        clock.tick(60)

    log_current_run_if_done()
    pygame.quit()

    # If the total number of actual runs completed is less than the expected, print abort message with summary
    fully_completed = summary.runs >= total_runs
    if not fully_completed:
        print(f"simulation aborted ({summary.runs}/{total_runs} runs completed)", flush=True)
    # Print regular summary
    if summary.runs > 0:
        print(format_batch_summary(summary, config), flush=True)
