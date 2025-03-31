import shutil
import sys
import time
from dataclasses import dataclass


@dataclass
class Ansi:
    CLEAR_SCREEN = "\033[2J"  # Clears the entire screen
    HIDE_CURSOR = "\033[?25l"  # Hides the cursor
    SHOW_CURSOR = "\033[?25h"  # Shows the cursor
    CURSOR_TOP_LEFT = "\033[H"  # Moves cursor to the top-left corner
    LINE_CLEAR_TO_EOL = "\033[0K"  # Clears from cursor to end of line
    LINE_CLEAR_TO_BOL = "\033[1K"  # Clears from cursor to beginning of line
    LINE_CLEAR = "\033[2K"  # Clears the entire line
    MOVE_UP = "\033[A"  # Moves cursor up one line
    MOVE_DOWN = "\033[B"  # Moves cursor down one line
    MOVE_RIGHT = "\033[C"  # Moves cursor right one character
    MOVE_LEFT = "\033[D"  # Moves cursor left one character
    SAVE_CURSOR = "\033[s"  # Saves current cursor position
    RESTORE_CURSOR = "\033[u"  # Restores saved cursor position
    BOLD = "\033[1m"  # Makes text bold
    DIM = "\033[2m"  # Makes text dimmer
    UNDERLINE = "\033[4m"  # Underlines text
    BLINK = "\033[5m"  # Makes text blink (may not work in all terminals)
    REVERSE = "\033[7m"  # Inverts foreground and background colors
    HIDDEN = "\033[8m"  # Hides text (useful for passwords)
    COLOR_RESET = "\033[0m"  # Resets all text formatting


@dataclass
class SpecialChars:
    UPPER_HALF_BLOCK = "▀"
    LOWER_HALF_BLOCK = "▄"
    FULL_BLOCK = "█"
    LIGHT_SHADE = "░"
    MEDIUM_SHADE = "▒"
    DARK_SHADE = "▓"
    VERTICAL_BAR = "│"
    HORIZONTAL_BAR = "─"
    CROSS = "┼"
    TOP_LEFT_CORNER = "┌"
    TOP_RIGHT_CORNER = "┐"
    BOTTOM_LEFT_CORNER = "└"
    BOTTOM_RIGHT_CORNER = "┘"
    LEFT_T = "├"
    RIGHT_T = "┤"
    TOP_T = "┬"
    BOTTOM_T = "┴"
    DOUBLE_VERTICAL_BAR = "║"
    DOUBLE_HORIZONTAL_BAR = "═"
    DOUBLE_TOP_LEFT_CORNER = "╔"
    DOUBLE_TOP_RIGHT_CORNER = "╗"
    DOUBLE_BOTTOM_LEFT_CORNER = "╚"
    DOUBLE_BOTTOM_RIGHT_CORNER = "╝"
    BULLET = "•"
    MIDDLE_DOT = "·"
    ARROW_UP = "↑"
    ARROW_DOWN = "↓"
    ARROW_LEFT = "←"
    ARROW_RIGHT = "→"
    DOUBLE_ARROW_LEFT = "«"
    DOUBLE_ARROW_RIGHT = "»"
    MUSIC_NOTE = "♪"
    SUN = "☀"
    MOON = "☾"
    PI = "π"
    COPYRIGHT = "©"
    REGISTERED = "®"
    TRADEMARK = "™"


@dataclass
class Color:
    code: int

    @property
    def fg(self):
        return f"\033[38;5;{self.code}m"

    @property
    def bg(self):
        return f"\033[48;5;{self.code}m"


@dataclass
class Colors:
    # STANDARD COLORS
    BLACK = Color(0)
    RED = Color(1)
    GREEN = Color(2)
    YELLOW = Color(3)
    BLUE = Color(4)
    PURPLE = Color(5)
    LIGHT_BLUE = Color(6)
    LIGHT_GRAY = Color(7)
    # HIGH INTENSITY COLORS
    DARK_GRAY = Color(8)
    RED_I = Color(9)
    GREEN_I = Color(10)
    YELLOW_I = Color(11)
    BLUE_I = Color(12)
    PURPLE_I = Color(13)
    LIGHT_BLUE_I = Color(14)
    WHITE = Color(15)
    # GRAY VALUES
    GRAY_00 = Color(232)
    GRAY_01 = Color(233)
    GRAY_02 = Color(234)
    GRAY_03 = Color(235)
    GRAY_04 = Color(236)
    GRAY_05 = Color(237)
    GRAY_06 = Color(238)
    GRAY_07 = Color(239)
    GRAY_08 = Color(240)
    GRAY_09 = Color(241)
    GRAY_10 = Color(242)
    GRAY_11 = Color(243)
    GRAY_12 = Color(244)
    GRAY_13 = Color(245)
    GRAY_14 = Color(246)
    GRAY_15 = Color(247)
    GRAY_16 = Color(248)
    GRAY_17 = Color(249)
    GRAY_18 = Color(250)
    GRAY_19 = Color(251)
    GRAY_20 = Color(252)
    GRAY_21 = Color(253)
    GRAY_22 = Color(254)
    GRAY_23 = Color(255)


GRAYSCALE = [
    Colors.BLACK,
    Colors.GRAY_00,
    Colors.GRAY_01,
    Colors.GRAY_02,
    Colors.GRAY_03,
    Colors.GRAY_04,
    Colors.GRAY_05,
    Colors.GRAY_06,
    Colors.GRAY_07,
    Colors.GRAY_08,
    Colors.GRAY_09,
    Colors.GRAY_10,
    Colors.GRAY_11,
    Colors.GRAY_12,
    Colors.GRAY_13,
    Colors.GRAY_14,
    Colors.GRAY_15,
    Colors.GRAY_16,
    Colors.GRAY_17,
    Colors.GRAY_18,
    Colors.GRAY_19,
    Colors.GRAY_20,
    Colors.GRAY_21,
    Colors.GRAY_22,
    Colors.GRAY_23,
    Colors.WHITE,
]


def aprint(s: str):
    """advanced print: write and flush"""
    sys.stdout.write(s)
    sys.stdout.flush()


def true_color_str(s: str, r: int, g: int, b: int, bg_r: int = None, bg_g: int = None, bg_b: int = None):
    fg_color = f"\x1b[38;2;{r};{g};{b}m"
    if bg_r is not None and bg_g is not None and bg_b is not None:
        bg_color = f"\x1b[48;2;{bg_r};{bg_g};{bg_b}m"
        return f"{bg_color}{fg_color}{s}\x1b[0m"
    return f"{fg_color}{s}\x1b[0m"


def get_terminal_size() -> tuple[int, int]:
    """get the current terminal size in a cross-platform way"""
    columns, rows = shutil.get_terminal_size()
    # TODO: subtract 1 from rows to avoid flickering in some terminals
    rows -= 1
    return columns, rows


def animation_loop(
    render_func,
    fps: int,
    duration_in_seconds: float,
    show_fps: bool,
):
    """starts an animation loop
    render_func(cols: int, rows: int, frame: int) -> str"""
    num_frames = int(fps * duration_in_seconds)
    wait_between_frames = 1.0 / fps
    for frame in range(num_frames):
        cols, rows = get_terminal_size()
        if show_fps:
            rows -= 1

        frame_start = time.time()
        frame_output = render_func(cols=cols, rows=rows, frame=frame)
        frame_time = time.time() - frame_start

        if frame_time < wait_between_frames:
            time.sleep(wait_between_frames - frame_time)
        if show_fps:
            actual_fps = 1.0 / (time.time() - frame_start)
            fps_info = Colors.RED_I.fg + f"\nFPS: {actual_fps:.2f}" + Ansi.COLOR_RESET
        else:
            fps_info = ""

        aprint(frame_output + fps_info)
