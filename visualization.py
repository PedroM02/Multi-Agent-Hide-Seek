from __future__ import annotations

import random

import pygame

import agent_utils as au
from environment import Environment
from simulation import BatchSummary, SimulationConfig, SimulationState


CELL = 36
MARGIN_TOP = 92
PAD_X = 8
HUD_COLOR = (20, 20, 24)
GRID_WALL = (35, 35, 40)
GRID_EMPTY = (210, 210, 215)
GRID_PRED = (110, 55, 180)
GRID_PREY = (55, 160, 75)
ROLE_LETTER_COLOR = (245, 245, 245)


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
    agents,
    font: pygame.font.Font,
    status: str,
    vision_radius_predator: int,
    vision_radius_prey: int,
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

    agent_by_id = {a.agent_id: a for a in agents}

    for b in env.agent_bodies.values():
        if not b.alive:
            continue
        rect = pygame.Rect(b.x * CELL + 2, b.y * CELL + 2, CELL - 4, CELL - 4)
        col = GRID_PRED if b.team == au.TEAM_PREDATOR else GRID_PREY
        pygame.draw.rect(grid_surf, col, rect, border_radius=4)

        a = agent_by_id.get(b.agent_id)
        if a is not None:
            letter = au.ROLE_LETTER.get(a.role, "?")
            label = font.render(letter, True, ROLE_LETTER_COLOR)
            lw, lh = label.get_size()
            cx = b.x * CELL + CELL // 2
            cy = b.y * CELL + CELL // 2
            grid_surf.blit(label, (cx - lw // 2, cy - lh // 2))

    grid_x = max(0, (surface.get_width() - grid_w) // 2)
    surface.blit(grid_surf, (grid_x, MARGIN_TOP))

    surface.blit(font.render(status, True, (240, 240, 245)), (PAD_X, 8))
    _blit_legend(surface, font, PAD_X, 34)
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
        f"Vision: Chebyshev Predator Radius={config.vision_radius_predator} and Prey Radius={config.vision_radius_prey}"
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

    summary = BatchSummary()
    logged_runs: set[int] = set()

    def log_current_run_if_done() -> None:
        """Record the current run into the batch summary once it finishes.

        Called both when advancing to the next run and on quit. The
        logged_runs guard means a reset of an already-counted run does
        not double-count or change the recorded outcome.
        """
        if sim.outcome == au.OUTCOME_ONGOING:
            return
        if run_index in logged_runs:
            return
        logged_runs.add(run_index)
        summary.runs += 1
        summary.total_steps += sim.step_index
        if sim.outcome == au.OUTCOME_PREDATORS_WIN:
            summary.predator_wins += 1
        elif sim.outcome == au.OUTCOME_PREY_WIN:
            summary.prey_timeout_wins += 1
        print(
            f"run={run_index + 1}/{total_runs}  seed={run_seed}  "
            f"outcome={sim.outcome}  steps={sim.step_index}",
            flush=True,
        )

    def load_run(idx: int) -> None:
        nonlocal run_index, run_seed, rng, sim
        run_index = idx
        run_seed = config.seed + run_index
        rng = random.Random(run_seed)
        sim = SimulationState(config, rng)

    def try_advance_run() -> bool:
        if run_index + 1 >= total_runs:
            return False
        log_current_run_if_done()
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
                        if sim.outcome != au.OUTCOME_ONGOING:
                            log_current_run_if_done()
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
                    if sim.outcome != au.OUTCOME_ONGOING:
                        log_current_run_if_done()
                elif not try_advance_run():
                    auto = False
                    pygame.time.set_timer(pygame.USEREVENT, 0)

        _draw_grid(screen, sim.env, sim.agents, font, status_text(), config.vision_radius_predator, config.vision_radius_prey)
        pygame.display.flip()
        clock.tick(60)

    log_current_run_if_done()
    pygame.quit()

    # A "natural" end is one where every seeded run actually finished.
    # Anything else (closing the window mid-run, leaving early after some
    # but not all runs) is an abort and gets reported as such so the
    # printed summary doesn't silently misrepresent the experiment.
    completed_cleanly = summary.runs >= total_runs
    if not completed_cleanly:
        print(
            f"simulation aborted ({summary.runs}/{total_runs} runs completed)",
            flush=True,
        )

    if summary.runs > 0:
        mean_steps = summary.total_steps / summary.runs
        print(
            f"runs={summary.runs}  predator_wins={summary.predator_wins}  "
            f"prey_timeout_wins={summary.prey_timeout_wins}",
            flush=True,
        )
        print(f"mean_steps={mean_steps:.2f}", flush=True)