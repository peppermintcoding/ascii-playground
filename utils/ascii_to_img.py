"""
rendering ascii as a simple image so it easily sharable. works with most input
sizes but might need some fiddling with sizing of the font and whitespace to make
it look good.
"""

import argparse

from PIL import Image, ImageDraw, ImageFont

X_MARGIN = 32
Y_MARGIN = 32
FONT_SIZE = 16


def save_ascii_as_png(
    ascii_str_path: str,
    output_filename: str,
    bg_color: tuple[int] = (24, 25, 41),
    font_color: tuple[int] = (255, 255, 255),
):
    with open(ascii_str_path, "r") as file:
        ascii_str = file.readlines()

    ascii_xy = (max([len(line) for line in ascii_str]), len(ascii_str))
    print(f"size of ascii image: {ascii_xy[0]} x {ascii_xy[1]}")

    img = Image.new(
        mode="RGB",
        size=(ascii_xy[0] * int(FONT_SIZE / 1.6), ascii_xy[1] * int(FONT_SIZE * 1.4)),
        color=bg_color,
    )
    I1 = ImageDraw.Draw(img)

    try:
        monospace_font = ImageFont.truetype("consola.ttf", FONT_SIZE)
    except IOError:
        monospace_font = ImageFont.truetype("LiberationMono-Regular.ttf", FONT_SIZE)
    
    for y_idx, line in enumerate(ascii_str):
        I1.text(
            xy=(X_MARGIN, y_idx * int(FONT_SIZE * 1.2) + Y_MARGIN),
            text=line,
            font=monospace_font,
            fill=font_color,
        )

    img.save(output_filename)
    print(f"resulting image: {img.width} x {img.height}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cli for ascii to img converter")
    parser.add_argument("ascii_str_filename", type=str, help="path to the input txt")
    parser.add_argument("output_filename", type=str, help="output file name")
    args = parser.parse_args()

    save_ascii_as_png(
        ascii_str_path=args.ascii_str_filename,
        output_filename=args.output_filename,
    )
