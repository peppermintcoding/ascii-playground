from utils import SpecialChars, aprint, Colors, Ansi


class PixelArtImage:
    def __init__(
        self,
        width: int,
        height: int,
        pixels: list[int],
        foreground_color: str = Colors.WHITE.foreground,
        background_color: str = Colors.BLACK.background,
    ):
        assert height % 2 == 0
        self.width = width
        self.height = height
        self.pixels = pixels
        self.foreground_color = foreground_color
        self.background_color = background_color

    def print(self):
        for y in range(0, self.height, 2):
            for x in range(self.width):
                foreground_value = self.pixels[x + y * self.width]
                background_value = self.pixels[x + (y + 1) * self.width]
                if foreground_value == 1 and background_value == 1:
                    pixel = SpecialChars.FULL_BLOCK
                elif foreground_value == 1 and background_value == 0:
                    pixel = SpecialChars.UPPER_HALF_BLOCK
                elif foreground_value == 0 and background_value == 1:
                    pixel = SpecialChars.LOWER_HALF_BLOCK
                else:
                    pixel = " "
                aprint(self.foreground_color + self.background_color + pixel)
            aprint(Ansi.COLOR_RESET + "\n")


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
