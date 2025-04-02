from tinygrad import Tensor, TinyJit, nn
from tinygrad.nn.datasets import mnist
import numpy as np

from utils import GRAYSCALE, Ansi, SpecialChars


class Model:
    def __init__(self):
        self.l1 = nn.Linear(in_features=28 * 28, out_features=32 * 32)
        self.l2 = nn.Linear(in_features=32 * 32, out_features=16 * 16)
        self.l3 = nn.Linear(16 * 16, 10)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x.flatten(1)).relu()
        x = self.l2(x).relu()
        return self.l3(x)


X_train, Y_train, X_test, Y_test = mnist()
model = Model()


optim = nn.optim.Adam(nn.state.get_parameters(model))
batch_size = 128


def step():
    Tensor.training = True  # makes dropout work
    samples = Tensor.randint(batch_size, high=X_train.shape[0])
    X, Y = X_train[samples], Y_train[samples]
    optim.zero_grad()
    loss = model(X).sparse_categorical_crossentropy(Y).backward()
    optim.step()
    return loss, X, Y


jit_step = TinyJit(step)


def train(steps: int):
    for step in range(steps):
        loss, X, Y = jit_step()
        if step % 100 == 0:
            Tensor.training = False
            acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
            print(f"step {step:4d}, loss {loss.item():.2f}, acc {acc * 100.0:.2f}%")

            visualize_mnist_digit(X=X, Y=Y, idx=0)
            x_model = model.l1(X.squeeze().flatten(1))
            visualize_layer(dim=32, weights=x_model, title="layer 01")
            x_model = model.l2(x=x_model)
            visualize_layer(dim=16, weights=x_model, title="layer 02")


def visualize_layer(dim: int, weights: np.ndarray, title: str):
    print(
        f"{11 * SpecialChars.DOUBLE_HORIZONTAL_BAR}-{title}-{11 * SpecialChars.DOUBLE_HORIZONTAL_BAR}{Ansi.COLOR_RESET}",
    )
    for y in range(0, dim, 2):
        img = weights[0].squeeze().numpy()
        for x in range(dim):
            min_value = abs(min(img))
            fg_value = img[x + y * dim] + min_value
            bg_value = img[x + (y+1) * dim] + min_value
            max_value = max(img) + min_value
            fg_color = GRAYSCALE[int(fg_value / max_value * (len(GRAYSCALE) - 1))].fg
            bg_color = GRAYSCALE[int(bg_value / max_value * (len(GRAYSCALE) - 1))].bg
            print(
                fg_color + bg_color + SpecialChars.UPPER_HALF_BLOCK + Ansi.COLOR_RESET,
                end="",
            )
        print()


def visualize_mnist_digit(X, Y, idx: int):
    print(
        f"{12 * SpecialChars.DOUBLE_HORIZONTAL_BAR}-{Y[idx].numpy()}-{13 * SpecialChars.DOUBLE_HORIZONTAL_BAR}"
    )
    for y in range(0, X_train.shape[2], 2):
        img = X[idx].squeeze().numpy()
        for x in range(img.shape[1]):
            fg_value = img[y, x]
            bg_value = img[y + 1, x]
            fg_color = GRAYSCALE[int(fg_value / 255 * (len(GRAYSCALE) - 1))].fg
            bg_color = GRAYSCALE[int(bg_value / 255 * (len(GRAYSCALE) - 1))].bg
            print(
                fg_color + bg_color + SpecialChars.UPPER_HALF_BLOCK + Ansi.COLOR_RESET,
                end="",
            )
        print()

train(steps=1_000)
