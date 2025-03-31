import argparse
import random

from tinygrad.nn.datasets import mnist

from utils import GRAYSCALE, Ansi, SpecialChars, true_color_str


def show_mnist_grid(rows: int, cols: int, true_color: bool):
    X_train, Y_train, _, _ = mnist()

    for _ in range(rows):
        indices = [random.randint(0, X_train.shape[0]) for _ in range(cols)]

        print(SpecialChars.DOUBLE_HORIZONTAL_BAR, end="")
        for index in indices:
            print(
                f"{13 * SpecialChars.DOUBLE_HORIZONTAL_BAR}-{Y_train[index].numpy()}-{13 * SpecialChars.DOUBLE_HORIZONTAL_BAR}",
                end="",
            )
        print()

        for y in range(0, X_train.shape[2], 2):
            print(SpecialChars.DOUBLE_VERTICAL_BAR, end="")
            for index in indices:
                img = X_train[index].squeeze().numpy()
                for x in range(img.shape[1]):
                    fg_value = img[y, x]
                    bg_value = img[y + 1, x]
                    if true_color:
                        print(
                            true_color_str(
                                s=SpecialChars.UPPER_HALF_BLOCK,
                                r=fg_value,
                                g=fg_value,
                                b=fg_value,
                                bg_r=bg_value,
                                bg_g=bg_value,
                                bg_b=bg_value,
                            ),
                            end="",
                        )
                    else:
                        fg_color = GRAYSCALE[
                            int(fg_value / 255 * (len(GRAYSCALE) - 1))
                        ].fg
                        bg_color = GRAYSCALE[
                            int(bg_value / 255 * (len(GRAYSCALE) - 1))
                        ].bg
                        print(
                            fg_color
                            + bg_color
                            + SpecialChars.UPPER_HALF_BLOCK
                            + Ansi.COLOR_RESET,
                            end="",
                        )
                print(SpecialChars.DOUBLE_VERTICAL_BAR, end="")
            print()
    print((cols * 29 + 1) * SpecialChars.DOUBLE_HORIZONTAL_BAR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="MNIST Showroom", description="shows mnist examples in a grid"
    )
    parser.add_argument("-r", type=int, help="number rows used", default=3)
    parser.add_argument("-c", type=int, help="number of columns used", default=4)
    parser.add_argument("-tc", type=bool, help="use true color", default=False)
    args = parser.parse_args()
    show_mnist_grid(rows=args.r, cols=args.c, true_color=args.tc)
