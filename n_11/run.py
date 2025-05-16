import sys

import pygame

from n_11.const import BLACK, GRAY_COL, LIGHT_BLUE_COL, GREEN_COL, BLUE_COL, RED_COL, WHITE
from n_11.const import L1, L2, M1, M2, T_MAX, FPS
from n_11.const import SCREEN_WIDTH, SCREEN_HEIGHT, CENTER_X, CENTER_Y
from n_11.pendulum import init_state, init_random_state, run_simulation_logic


def main():
    # inicializace stavu pro první spuštění
    y0_state = init_state()

    # spuštění simulace
    time_points, x1_sim, y1_sim, x2_sim, y2_sim = run_simulation_logic(y0_state)

    pygame.init()

    scale_factor = min(SCREEN_WIDTH, SCREEN_HEIGHT) / (2.5 * (L1 + L2))
    bob_radius1 = int(M1 ** (1 / 3) * 15)
    bob_radius2 = int(M2 ** (1 / 3) * 15)
    max_trace_len = 500

    # obrazovka a hodiny
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("dvojité kyvadlo - pygame")
    clock = pygame.time.Clock()
    main_font = pygame.font.Font(None, 36)

    # proměnné pro animaci
    current_frame_index = 0
    trace_path = []
    running = True

    while running:
        # zachycení pygame událostí
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # konec
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # konec
                    running = False
                if event.key == pygame.K_r:
                    # restart simulace
                    # nový stav
                    y0_state_new = init_random_state()
                    # proveď simulaci (ta se následně jen vyrenderuje)
                    time_points, x1_sim, y1_sim, x2_sim, y2_sim = run_simulation_logic(y0_state_new)
                    current_frame_index = 0
                    trace_path = []

        #
        # vykreslení aktuálního snímku
        #
        if current_frame_index < len(time_points):
            # aktuální fyzikální souřadnice
            phys_x1 = x1_sim[current_frame_index]
            phys_y1 = y1_sim[current_frame_index]
            phys_x2 = x2_sim[current_frame_index]
            phys_y2 = y2_sim[current_frame_index]

            # převod na souřadnice obrazovky
            screen_origin_x = CENTER_X
            screen_origin_y = CENTER_Y

            scr_x1 = int(screen_origin_x + phys_x1 * scale_factor)
            scr_y1 = int(screen_origin_y - phys_y1 * scale_factor)
            scr_x2 = int(screen_origin_x + phys_x2 * scale_factor)
            scr_y2 = int(screen_origin_y - phys_y2 * scale_factor)

            # stopa
            trace_path.append((scr_x2, scr_y2))
            if len(trace_path) > max_trace_len:
                # umazávání starých bodů
                trace_path.pop(0)

            # vykreslování
            screen.fill(BLACK)

            if len(trace_path) > 1:
                # vykreslení stopy kyvadla
                pygame.draw.lines(screen, GRAY_COL, False, trace_path, 1)

            # střed kyvadla
            pygame.draw.circle(screen, WHITE, (screen_origin_x, screen_origin_y), 5)
            # l1
            pygame.draw.line(screen, LIGHT_BLUE_COL, (screen_origin_x, screen_origin_y), (scr_x1, scr_y1), 3)
            # první kulíčka
            pygame.draw.circle(screen, BLUE_COL, (scr_x1, scr_y1), bob_radius1)
            # l2
            pygame.draw.line(screen, GREEN_COL, (scr_x1, scr_y1), (scr_x2, scr_y2), 3)
            # druhá kulíčka
            pygame.draw.circle(screen, RED_COL, (scr_x2, scr_y2), bob_radius2)

            # texty
            time_val = time_points[current_frame_index]
            time_surf = main_font.render(f"čas: {time_val:.1f}s / {T_MAX:.1f}s", True, WHITE)
            screen.blit(time_surf, (10, 10))
            fps_surf = main_font.render(f"fps: {int(clock.get_fps())}", True, WHITE)
            screen.blit(fps_surf, (10, 40))

            help_text_items = ["r - restart", "esc - konec"]
            for i, line in enumerate(help_text_items):
                help_line_surf = main_font.render(line, True, WHITE)
                screen.blit(help_line_surf, (10, SCREEN_HEIGHT - (len(help_text_items) - i) * 30 - 10))

            current_frame_index += 1
        else:
            # konec předpočítané simulace
            end_msg = main_font.render("simulace dokončena. 'r' pro restart.", True, WHITE)
            msg_rect = end_msg.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 150))
            screen.blit(end_msg, msg_rect)

        # konec framu
        #
        # flipneme buffer
        pygame.display.flip()
        # update clock simulace
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
