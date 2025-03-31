from utils import Ansi, aprint, Colors, animation_loop
import random


def prepare_terminal():
    aprint(Ansi.CLEAR_SCREEN)
    aprint(Ansi.HIDE_CURSOR)


def reset_terminal():
    """clean up terminal back to orignal state"""
    aprint(Ansi.CLEAR_SCREEN)
    aprint(Ansi.SHOW_CURSOR)
    aprint(Ansi.CURSOR_TOP_LEFT)


def load_ascii_art(filename: str) -> str:
    with open(filename, "r") as f:
        return f.read()


cow = load_ascii_art(filename="../ascii-by-hand/cow.txt")
char_choices = []
NUM_CHOICES = 8
for char in cow:
    if char == "\n":
        char_choices.append(["\n"])
        continue
    choices = [char]
    for _ in range(random.randint(NUM_CHOICES - 2, NUM_CHOICES)):
        choice = random.randint(33, 126)
        while chr(choice) == char:
            choice = random.randint(33, 126)
        choices.append(chr(choice))
    char_choices.append(choices)


def render(cols: int, rows: int, frame: int) -> str:
    frame_output = [Ansi.CURSOR_TOP_LEFT]

    for i, char_choice in enumerate(char_choices):
        frame_output.append(random.choice(char_choice))
        if frame % 24 == 0:
            if len(char_choices[i]) > 1 and random.randint(0, 1) == 0:
                _ = char_choices[i].pop()

    return (
        Ansi.BOLD + Colors.GREEN_I.fg + "".join(frame_output) + Ansi.COLOR_RESET
    )


if __name__ == "__main__":
    try:
        prepare_terminal()
        animation_loop(
            render_func=render,
            fps=30,
            duration_in_seconds=18,
            show_fps=True,
        )
    except KeyboardInterrupt:
        pass  # allow clean exit with ctrl+c
    finally:
        reset_terminal()
