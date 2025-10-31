import time

import pandas as pd
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
import shutil
import numpy as np

from pixel_art import PixelArtImage

ZERO = [
    0, 0, 0, 0, 0, 0,
    0, 0, 1, 1, 0, 0,
    0, 1, 0, 0, 1, 0,
    0, 1, 0, 0, 1, 0,
    0, 1, 0, 0, 1, 0,
    0, 1, 0, 0, 1, 0,
    0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0,
]
ONE = [
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 1, 1, 0,
    0, 0, 1, 0, 1, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0,
]
TWO = [
    0, 0, 0, 0, 0, 0,
    0, 0, 1, 1, 0, 0,
    0, 1, 0, 0, 1, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 1, 0, 0,
    0, 0, 1, 0, 0, 0,
    0, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0,
]
THREE = [
    0, 0, 0, 0, 0, 0,
    0, 0, 1, 1, 0, 0,
    0, 1, 0, 0, 1, 0,
    0, 0, 0, 1, 1, 0,
    0, 0, 0, 1, 1, 0,
    0, 1, 0, 0, 1, 0,
    0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0,
]
FOUR = [
    0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0,
    0, 1, 0, 1, 0, 0,
    0, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0,
]
FIVE = [
    0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 0,
    0, 1, 0, 0, 0, 0,
    0, 1, 1, 1, 0, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 1, 0,
    0, 1, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0,
]
SIX = [
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 1, 0,
    0, 0, 1, 0, 0, 0,
    0, 1, 0, 1, 0, 0,
    0, 1, 1, 0, 1, 0,
    0, 1, 0, 0, 1, 0,
    0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0,
]
SEVEN = [
    0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 1, 0, 0,
    0, 0, 1, 0, 0, 0,
    0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
]
EIGHT = [
    0, 0, 0, 0, 0, 0,
    0, 0, 1, 1, 0, 0,
    0, 1, 0, 0, 1, 0,
    0, 0, 1, 1, 0, 0,
    0, 0, 1, 1, 0, 0,
    0, 1, 0, 0, 1, 0,
    0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0,
]
NINE = [
    0, 0, 0, 0, 0, 0,
    0, 0, 1, 1, 1, 0,
    0, 1, 0, 0, 1, 0,
    0, 1, 0, 0, 1, 0,
    0, 0, 1, 1, 1, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0,
]
DOT = [
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0,
    0, 0, 1, 1, 0, 0,
]




if __name__ == "__main__":
    console = Console()
    big_digits = {
        "0": ZERO,
        "1": ONE,
        "2": TWO,
        "3": THREE,
        "4": FOUR,
        "5": FIVE,
        "6": SIX,
        "7": SEVEN,
        "8": EIGHT,
        "9": NINE,
        ".": DOT,
    }

    print("Enter you date of birth in the format: YYYY-MM-DD HH:MM:SS.")
    print("The hours, minutes and seconds can be omitted.")
    birth = pd.Timestamp(input(">> "))
    now = pd.Timestamp.today()
    age = (now - birth).total_seconds() / (365.25 * 24 * 60 * 60)
    width, height = shutil.get_terminal_size()

    try:
        with Live(auto_refresh=True, screen=True, console=console) as live:
            while True:
                digit_text = f"{age:.12f}"
                big_digit_text = [np.array(big_digits[digit]).reshape(8, 6) for digit in digit_text]
                big_digit_text = np.hstack(big_digit_text)
                pixel_art = PixelArtImage(width=6*len(digit_text), height=8, pixels=big_digit_text.flatten().tolist())
                live.update(
                    Panel(
                        Align.center(Text.from_ansi(pixel_art.render()), vertical="middle"),
                        width=width,
                        height=height,
                        expand=True,
                        title="Time flies..",
                    )
                )
                time.sleep(0.1)
                now = pd.Timestamp.today()
                age = (now - birth).total_seconds() / (365.25 * 24 * 60 * 60)
    except KeyboardInterrupt:
        pass
