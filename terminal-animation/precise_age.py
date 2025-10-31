#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rich"
# ]
# ///
import shutil
import time
from datetime import datetime

from pixel_art import PixelArtImage
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

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


def parse_date(date: str) -> datetime.timestamp:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(date, fmt)
        except ValueError:
            continue
    raise ValueError("Invalid date format. Use YYYY-MM-DD [HH:MM[:SS]].")


def seconds_to_years(seconds: float) -> float:
    return seconds / (365.25 * 24 * 60 * 60)


def reshape(x: list, rows: int, cols: int) -> list:
    return [x[i * cols:(i + 1) * cols] for i in range(rows)]


def hstack(matrices: list):
    combined = []
    for row_group in zip(*matrices):
        row = []
        for subrow in row_group:
            row.extend(subrow)
        combined.append(row)
    return combined


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

    print("Enter your date of birth in the format: YYYY-MM-DD HH:MM:SS.")
    print("The hours, minutes and seconds can be omitted.")
    birth = parse_date(input(">> "))
    now = datetime.now()
    age = seconds_to_years((now - birth).total_seconds())
    width, height = shutil.get_terminal_size()

    try:
        with Live(auto_refresh=True, screen=True, console=console) as live:
            while True:
                digit_text = f"{age:.12f}"
                big_digit_text = [reshape(big_digits[d], 8, 6) for d in digit_text]
                big_digit_text = hstack(big_digit_text)
                pixel_art = PixelArtImage(width=6 * len(digit_text), height=8, pixels=[p for row in big_digit_text for p in row])
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
                now = datetime.now()
                age = seconds_to_years((now - birth).total_seconds())
    except KeyboardInterrupt:
        pass
