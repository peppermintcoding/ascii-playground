from utils import Ansi, aprint, Colors, animation_loop, SpecialChars
import random


octopus = """
              .---.         ,,   
   ,,        /     \       ;,,'  
  ;, ;      (  °  ° )      ; ;   
    ;,';,,,  \  ~- /      ,; ;   
  ,,  ;,,,,;;,`   '-,;'''',,,'   
   ;,,,;    ~~'  '';,,''',,;''''   
   (;,, ,,,,   ,;  ,,,'';;,,;''';
                      '''       
"""


def prepare_terminal():
    aprint(Ansi.CLEAR_SCREEN)
    aprint(Ansi.HIDE_CURSOR)


def reset_terminal():
    aprint(Ansi.CLEAR_SCREEN)
    aprint(Ansi.SHOW_CURSOR)
    aprint(Ansi.CURSOR_TOP_LEFT)


def _create_empty_cavas(cols: int, rows: int) -> list[str]:
    canvas = []
    for _ in range(rows):
        for _ in range(cols):
            canvas.append(" ")
        canvas.append("\n")
    return canvas


def _add_ground(canvas: list[str], cols: int, rows: int) -> list[str]:
    for x in range(cols):
        canvas[(rows - 1) * (cols + 1) + x] = SpecialChars.LIGHT_SHADE
    return canvas


class Seagrass:
    def __init__(
        self, x: int, height: int, left: str = None, right: str = None, c: Colors = None
    ):
        self.x = x
        self.height = height
        self.left = "(" if left is None else left
        self.right = ")" if right is None else right
        self.c = Colors.GREEN_I.foreground if c is None else c

    def draw(self, canvas: list[str], cols: int, rows: int, frame: int) -> list[str]:
        if frame % 15 == 0:
            self.left, self.right = self.right, self.left

        for i in range(self.height):
            canvas[(rows - 2 - i) * (cols + 1) + self.x] = (
                self.c + self.left + Ansi.COLOR_RESET
                if i % 2 == 0
                else self.c + self.right + Ansi.COLOR_RESET
            )
        return canvas


class Fish:
    def __init__(self, x: int, y: int, direction: int, c: Colors):
        self.x = x
        self.y = y
        self.direction = direction
        self.c = c

    def draw(self, canvas: list[str], cols: int, rows: int, frame: int) -> list[str]:
        if self.direction == 1:
            canvas[self.y * (cols + 1) + self.x] = ">" + Ansi.COLOR_RESET
            canvas[self.y * (cols + 1) + self.x - self.direction] = "<"
            canvas[self.y * (cols + 1) + self.x - (self.direction * 2)] = self.c + ">"
        else:
            canvas[self.y * (cols + 1) + self.x] = self.c + "<"
            canvas[self.y * (cols + 1) + self.x - self.direction] = ">"
            canvas[self.y * (cols + 1) + self.x - (self.direction * 2)] = (
                "<" + Ansi.COLOR_RESET
            )
        if frame % 3 == 0:
            if self.x + self.direction >= cols - 3 or self.x + self.direction <= 2:
                self.direction *= -1
            self.x += self.direction
        return canvas


class Bubble:
    def __init__(self, x: int, y: int = None):
        self.x = x
        self.y = y

    def draw(self, canvas: list[str], cols: int, rows: int, frame: int) -> list[str]:
        if self.y is None:
            self.y = rows - (2 + random.randint(0, 6))
        canvas[self.y * (cols + 1) + self.x] = "°"
        if frame % 7 == 0:
            self.y -= 1
            self.x += random.randint(-1, 1)
            if self.y <= 1:
                self.y = rows - 2
        return canvas


def _add_text_block(
    canvas: list[str], text_block: str, cols: int, x: int, y: int
) -> list[str]:
    ix = x
    for char in text_block:
        if char == "\n":
            y += 1
            ix = x
            continue
        canvas[y * (cols + 1) + ix] = char
        ix += 1
    return canvas


def render(cols: int, rows: int, frame: int) -> str:
    canvas = _create_empty_cavas(cols=cols, rows=rows)
    canvas = _add_ground(canvas=canvas, cols=cols, rows=rows)

    for grass in grasses:
        grass.draw(canvas=canvas, cols=cols, rows=rows, frame=frame)

    for fish in school:
        fish.draw(canvas=canvas, cols=cols, rows=rows, frame=frame)

    for bubble in bubbles:
        bubble.draw(canvas=canvas, cols=cols, rows=rows, frame=frame)

    canvas = _add_text_block(canvas=canvas, text_block=octopus, cols=cols, x=42, y=13)

    return Ansi.CURSOR_TOP_LEFT + "".join(canvas)


if __name__ == "__main__":
    try:
        prepare_terminal()
        grasses = [
            Seagrass(x=94, height=5),
            Seagrass(x=90, height=7),
            Seagrass(x=87, height=4),
            Seagrass(x=5, height=5),
            Seagrass(x=12, height=4),
            Seagrass(x=18, height=2),
            Seagrass(x=19, height=1),
            Seagrass(x=23, height=8),
        ]

        school = [
            Fish(x=2, y=0, direction=1, c=Colors.RED.foreground),
            Fish(x=12, y=2, direction=1, c=Colors.PURPLE_I.foreground),
            Fish(x=64, y=3, direction=-1, c=Colors.YELLOW.foreground),
            Fish(x=24, y=5, direction=-1, c=Colors.WHITE.foreground),
            Fish(x=29, y=8, direction=1, c=Colors.LIGHT_BLUE_I.foreground),
            Fish(x=19, y=10, direction=-1, c=Colors.GREEN_I.foreground),
        ]

        bubbles = [
            Bubble(x=7),
            Bubble(x=9),
            Bubble(x=84),
            Bubble(x=89),
        ]

        animation_loop(
            render_func=render,
            fps=30,
            duration_in_seconds=60,
            show_fps=True,
        )
    except KeyboardInterrupt:
        pass  # allow clean exit with ctrl+c
    finally:
        reset_terminal()
