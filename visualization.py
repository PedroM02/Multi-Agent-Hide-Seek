from __future__ import annotations

import random

import pygame

import agent_utils as au
from environment import Environment
from simulation import SimulationConfig, SimulationState


CELL = 36
MARGIN_TOP = 92
PAD_X = 8
HUD_COLOR = (20, 20, 24)
GRID_WALL = (35, 35, 40)
GRID_EMPTY = (210, 210, 215)
GRID_PRED = (110, 55, 180)
GRID_PREY = (55, 160, 75)
GRID_OBSTACLE = (220, 180, 40)


def _blit_legend(surface: pygame.Surface, font: pygame.font.Font, x: int, y: int) -> int:
    """Draw color swatches + team labels; returns x after last label (for layout)."""
    sw, sh = 14, 14
    fg = (240, 240, 245)
    pygame.draw.rect(surface, GRID_PRED, (x, y + 2, sw, sh))
    surface.blit(font.render("Predator", True, fg), (x + sw + 6, y))
    x2 = x + sw + 6 + font.size("Predator")[0] + 16
    pygame.draw.rect(surface, GRID_PREY, (x2, y + 2, sw, sh))
    surface.blit(font.render("Prey", True, fg), (x2 + sw + 6, y))
    return x2 + sw + 6 + font.size("Prey")[0]


def _draw_grid(
    surface: pygame.Surface,
    env: Environment,
    font: pygame.font.Font,
    status: str,
    vision_radius_predator: int,
    vision_radius_prey: int
) -> None:
    surface.fill(HUD_COLOR)
    grid_w = env.width * CELL
    grid_h = env.height * CELL
    grid_surf = pygame.Surface((grid_w, grid_h))
    grid_surf.fill(GRID_EMPTY)
    for x in range(env.width):
        for y in range(env.height):
            rect = pygame.Rect(x * CELL, y * CELL, CELL, CELL)
            if env.is_wall(x, y):
                pygame.draw.rect(grid_surf, GRID_WALL, rect)
            pygame.draw.rect(grid_surf, (0, 0, 0), rect, 1)
    for b in env.bodies.values():
        if not b.alive:
            continue
        rect = pygame.Rect(b.x * CELL + 2, b.y * CELL + 2, CELL - 4, CELL - 4)
        col = GRID_PRED if b.team == au.TEAM_PREDATOR else GRID_PREY
        pygame.draw.rect(grid_surf, col, rect, border_radius=4)
    for item in env.obstacles.values():
        if item.held_by is not None:
            continue
        cx = item.x * CELL + CELL // 2
        cy = item.y * CELL + CELL // 2
        pygame.draw.circle(grid_surf, GRID_OBSTACLE, (cx, cy), CELL // 5)   
    grid_x = max(0, (surface.get_width() - grid_w) // 2)
    surface.blit(grid_surf, (grid_x, MARGIN_TOP))

    surface.blit(font.render(status, True, (240, 240, 245)), (PAD_X, 8))
    _blit_legend(surface, font, PAD_X, 34)
    hint_1 = font.render(
        f"Vision: Chebyshev predaror r={vision_radius_predator} and prey r={vision_radius_prey} (square side {2 * vision_radius_predator + 1})  |  "
        "Chase: visible enemy if any, else last seen",
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


def run_visualization(config: SimulationConfig, num_runs: int = 1) -> None:
    pygame.init()
    total_runs = max(1, num_runs)
    run_index = 0
    run_seed = config.seed
    rng = random.Random(run_seed)
    sim = SimulationState(config, rng)
    font = pygame.font.Font(None, 18)
    status_sample = (
        f"Run {total_runs}/{total_runs}  Timestep 0/{config.timesteps}  "
        f"Outcome={au.OUTCOME_ONGOING}  |  Seed={config.seed + total_runs - 1}"
    )
    hint_text_1 = (
        f"Vision: Chebyshev Predaror Radius={config.vision_radius_predator} and Prey Radius={config.vision_radius_prey} (square side {2 * config.vision_radius_predator + 1})  |  "
        "Chase: visible enemy if any, else last seen"
    )
    hint_text_2 = "Next timestep: space/right   Auto Run: a   Reset Run: r   Next Run: n   Quit: esc"
    min_header_w = max(font.size(status_sample)[0], font.size(hint_text_1)[0], font.size(hint_text_2)[0]) + PAD_X * 2
    win_w = max(config.width * CELL, min_header_w)
    win_h = config.height * CELL + MARGIN_TOP
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("PredPrey Sim")
    clock = pygame.time.Clock()
    auto = False
    step_ms = 180
    pygame.time.set_timer(pygame.USEREVENT, 0)

    def load_run(idx: int) -> None:
        nonlocal run_index, run_seed, rng, sim
        run_index = idx
        run_seed = config.seed + run_index
        rng = random.Random(run_seed)
        sim = SimulationState(config, rng)

    def try_advance_run() -> bool:
        if run_index + 1 >= total_runs:
            return False
        load_run(run_index + 1)
        return True

    def status_text() -> str:
        return f"Run {run_index + 1}/{total_runs}  {sim.status_line()}  |  Seed={run_seed}"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_SPACE, pygame.K_RIGHT):
                    if sim.outcome == au.OUTCOME_ONGOING:
                        sim.step_once()
                elif event.key == pygame.K_a:
                    auto = not auto
                    pygame.time.set_timer(pygame.USEREVENT, step_ms if auto else 0)
                elif event.key == pygame.K_r:
                    load_run(run_index)
                    auto = False
                    pygame.time.set_timer(pygame.USEREVENT, 0)
                elif event.key == pygame.K_n:
                    if sim.outcome != au.OUTCOME_ONGOING:
                        if try_advance_run():
                            auto = False
                            pygame.time.set_timer(pygame.USEREVENT, 0)
            elif event.type == pygame.USEREVENT and auto:
                if sim.outcome == au.OUTCOME_ONGOING:
                    sim.step_once()
                elif not try_advance_run():
                    auto = False
                    pygame.time.set_timer(pygame.USEREVENT, 0)

        _draw_grid(screen, sim.env, font, status_text(), config.vision_radius_predator, config.vision_radius_prey)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()