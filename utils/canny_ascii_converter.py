"""
the idea is to take any image and convert it to ascii art but not only depending
on the brightness of every pixel, but on the edges of the images. in this case
we can use the canny edge detection algorithm to get the edges of any subject in
an image and represent these with | / _ and so on.
"""

import argparse

import cv2
import numpy as np


def convert_img_to_ascii_with_canny(
    image_path: str, width: int = 80, fill_background: bool = True
):
    image = cv2.imread(filename=image_path, flags=cv2.IMREAD_GRAYSCALE)

    aspect_ratio = image.shape[0] / image.shape[1]
    # Multiply by 0.5 to account for terminal line spacing
    height = int(width * aspect_ratio * 0.5)
    image = cv2.resize(image, (width, height))
    edges = cv2.Canny(image, 100, 200)

    ascii_shapes = {
        "horizontal": "_",
        "vertical": "|",
        "diagonal_right": "/",
        "diagonal_left": "\\",
    }
    # change this to play around with the background gradients
    ascii_shades = ["&", "#", '"', ".", ".", " ", " ", " "]

    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    def pixel_to_ascii(x, y):
        if edges[y, x] == 0:
            return ascii_shades[int(image[y, x] / 32)] if fill_background else " "
        angle = np.degrees(np.arctan2(gradient_y[y, x], gradient_x[y, x]))
        if abs(angle) < 22.5 or abs(angle) > 157.5:
            return ascii_shapes["vertical"]
        if 22.5 <= abs(angle) < 67.5:
            return ascii_shapes["diagonal_right"]
        if 67.5 <= abs(angle) < 112.5:
            return ascii_shapes["horizontal"]
        return ascii_shapes["diagonal_left"]

    return "\n".join(
        "".join(pixel_to_ascii(x, y) for x in range(width)) for y in range(height)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="cli for ascii image converter via edge detection"
    )
    parser.add_argument("image_path", type=str, help="path to the input image")
    parser.add_argument("-o", type=str, help="output file name")
    parser.add_argument("-w", type=int, default=80, help="Width of the ascii art")
    parser.add_argument("--no-fill", dest="fill_background", action="store_false")
    parser.set_defaults(fill_background=True)
    args = parser.parse_args()

    ascii_image = convert_img_to_ascii_with_canny(
        image_path=args.image_path,
        width=args.w,
        fill_background=args.fill_background,
    )

    if args.o is None:
        print(ascii_image)
    else:
        with open(args.o, "w") as file:
            file.write(ascii_image)
