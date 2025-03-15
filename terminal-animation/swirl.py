import numpy as np
from utils import Ansi, aprint, Colors, animation_loop


def prepare_terminal():
    aprint(Ansi.CLEAR_SCREEN)
    aprint(Ansi.HIDE_CURSOR)


def reset_terminal():
    """clean up terminal back to orignal state"""
    aprint(Ansi.CLEAR_SCREEN)
    aprint(Ansi.SHOW_CURSOR)
    aprint(Ansi.CURSOR_TOP_LEFT)


GRADIENT = np.array(list(" ,;~=>§#%"))
FG_COLOR = Colors.GREEN_I.foreground
BG_COLOR = Colors.BLACK.background


def render(cols: int, rows: int, frame: int) -> str:
    frame_output = Ansi.CURSOR_TOP_LEFT

    for y in range(rows):
        x = np.arange(cols)
        norm_x = (x - cols / 2) / (cols / 2)
        norm_y = (y - rows / 2) / (rows / 2)
        frame_factor = np.sin(frame * 0.02) ** 2
        radius = np.sqrt(norm_x**2 + norm_y**2 + frame_factor)
        angle = np.atan2(norm_y, norm_x)
        value = np.sin(15 * radius - angle + frame_factor)
        index = ((value + 1) / 2 * (len(GRADIENT) - 1)).astype(np.uint)
        frame_output += "".join(GRADIENT[index])
        frame_output += "\n"

    return Ansi.BOLD + BG_COLOR + FG_COLOR + frame_output + Ansi.COLOR_RESET


if __name__ == "__main__":
    try:
        prepare_terminal()
        animation_loop(
            render_func=render,
            fps=30,
            duration_in_seconds=10,
            show_fps=True,
        )
    except KeyboardInterrupt:
        pass  # allow clean exit with ctrl+c
    finally:
        reset_terminal()
