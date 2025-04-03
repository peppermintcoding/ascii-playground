import argparse

import numpy as np
from PIL import Image


def floyd_steinberg_dither(image: Image):
    grayscale = image.convert("L")
    img_array = np.array(grayscale, dtype=np.float32) / 255.0
    height, width = img_array.shape
    binary_img = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            old_pixel = img_array[y, x]
            new_pixel = 1 if old_pixel > 0.5 else 0
            binary_img[y, x] = new_pixel
            error = old_pixel - new_pixel

            if x + 1 < width:
                img_array[y, x + 1] += error * 7 / 16
            if x - 1 >= 0 and y + 1 < height:
                img_array[y + 1, x - 1] += error * 3 / 16
            if y + 1 < height:
                img_array[y + 1, x] += error * 5 / 16
            if x + 1 < width and y + 1 < height:
                img_array[y + 1, x + 1] += error * 1 / 16

    return binary_img


def thresholding(image: Image, threshold: float):
    grayscale = image.convert("L")
    img_array = np.array(grayscale, dtype=np.float32) / 255.0
    height, width = img_array.shape
    binary_img = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            if img_array[y, x] > threshold:
                binary_img[y, x] = 1
            else:
                binary_img[y, x] = 0
    return binary_img


def render_braille(img: np.ndarray):
    unicode_braille_start = 0x2800
    height, width = img.shape
    assert width % 2 == 0
    assert height % 4 == 0

    for y in range(0, height, 4):
        for x in range(0, width, 2):
            offset = 0
            offset += int(img[y + 0, x + 0] * 2**0)
            offset += int(img[y + 1, x + 0] * 2**1)
            offset += int(img[y + 2, x + 0] * 2**2)
            offset += int(img[y + 0, x + 1] * 2**3)
            offset += int(img[y + 1, x + 1] * 2**4)
            offset += int(img[y + 2, x + 1] * 2**5)
            offset += int(img[y + 3, x + 0] * 2**6)
            offset += int(img[y + 3, x + 1] * 2**7)
            char = chr(unicode_braille_start + offset)
            print(char, end="")
        print()


def main(image_path: str, compression_factor: int, method: str = "floyd"):
    image = Image.open(image_path)
    new_width = image.width // compression_factor
    new_width = new_width - (new_width % 2)
    new_height = image.height // compression_factor
    new_height = new_height - (new_height % 4)
    image = image.resize((new_width, new_height))
    if method == "floyd":
        binary_img = floyd_steinberg_dither(image=image)
    elif method == "threshold":
        binary_img = thresholding(image=image, threshold=0.5)
    else:
        raise ValueError(f"invalid method: {method}")
    render_braille(img=binary_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="braille dithered image",
        description="uses floyd steinberg dithering on image and prints with braille characters",
    )
    parser.add_argument("-i", type=str, help="path to image", default="img/selena.jpeg")
    parser.add_argument("-c", type=int, help="compression factor", default=4)
    parser.add_argument("-m", type=str, help="pixelating method, 'floyd' or 'threshold'", default="floyd")
    args = parser.parse_args()
    main(image_path=args.i, compression_factor=args.c, method=args.m)
