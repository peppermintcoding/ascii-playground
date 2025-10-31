import time

import pandas as pd
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
import shutil

if __name__ == "__main__":
    console = Console()

    print("Enter you date of birth in the format: YYYY-MM-DD HH:MM:SS.")
    print("The hours, minutes and seconds can be omitted.")
    birth = pd.Timestamp(input(">> "))
    now = pd.Timestamp.today()
    age = (now - birth).total_seconds() / (365.25 * 24 * 60 * 60)
    width, height = shutil.get_terminal_size()

    try:
        with Live(auto_refresh=True, screen=True, console=console) as live:
            while True:
                live.update(
                    Panel(
                        Align.center(f"{age:.12f}", vertical="middle"),
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