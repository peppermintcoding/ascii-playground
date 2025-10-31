from utils import SpecialChars, aprint, Colors, Ansi


class PixelArtImage:
    def __init__(
        self,
        width: int,
        height: int,
        pixels: list[int],
        fg_color: str = Colors.WHITE.fg,
        bg_color: str = Colors.BLACK.bg,
    ):
        assert height % 2 == 0
        self.width = width
        self.height = height
        self.pixels = pixels
        self.fg_color = fg_color
        self.bg_color = bg_color
    

    def render(self):
        text = []
        for y in range(0, self.height, 2):
            line = []
            for x in range(self.width):
                fg_value = self.pixels[x + y * self.width]
                bg_value = self.pixels[x + (y + 1) * self.width]
                if fg_value == 1 and bg_value == 1:
                    pixel = SpecialChars.FULL_BLOCK
                elif fg_value == 1 and bg_value == 0:
                    pixel = SpecialChars.UPPER_HALF_BLOCK
                elif fg_value == 0 and bg_value == 1:
                    pixel = SpecialChars.LOWER_HALF_BLOCK
                else:
                    pixel = " "
                line.append(self.fg_color + self.bg_color + pixel)
            line.append(Ansi.COLOR_RESET + "\n")
            text.append("".join(line))
        return "".join(text)

    def print(self):
        aprint(self.render())


if __name__ == "__main__":
    pixels = [
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 0, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]
    smiley = PixelArtImage(width=9, height=8, pixels=pixels)
    smiley.print()
