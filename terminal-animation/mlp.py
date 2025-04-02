import numpy as np
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from tinygrad import Tensor, TinyJit, nn
from tinygrad.nn.datasets import mnist

from utils import GRAYSCALE, Ansi, Colors, SpecialChars


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
console = Console()


def step():
    Tensor.training = True
    samples = Tensor.randint(batch_size, high=X_train.shape[0])
    X, Y = X_train[samples], Y_train[samples]
    optim.zero_grad()
    loss = model(X).sparse_categorical_crossentropy(Y).backward()
    optim.step()
    return loss


jit_step = TinyJit(step)


def visualize(step: int, max_steps: int, acc: float):
    samples = Tensor.randint(batch_size, high=X_test.shape[0])
    X, Y = X_test[samples], Y_test[samples]
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
    )
    layout["main"].split_column(Layout(name="example_0"), Layout(name="example_1"), Layout(name="example_2"))

    for idx in range(3):
        layout["main"][f"example_{idx}"].split_row(
            Layout(name="digit"),
            Layout(name="layer01"),
            Layout(name="layer02"),
            Layout(name="last_layer"),
        )

        mnist_digit = visualize_mnist_digit(X=X, idx=idx)
        x_model = model.l1(X.squeeze().flatten(1)).relu()
        layer01 = visualize_layer(dim=32, weights=x_model, idx=idx)
        x_model = model.l2(x=x_model).relu()
        layer02 = visualize_layer(dim=16, weights=x_model, idx=idx)
        x_model = model.l3(x=x_model)
        last_layer = visualize_last_layer(dim=10, weights=x_model, labels=Y, idx=idx)

        left_text = "[b]MNIST[/b] MLP training"
        right_text = f"Step: {step} / {max_steps} | Accuracy: {acc:.2%}"

        header_content = Columns([left_text, Align.right(right_text)], expand=True)

        layout["header"].update(Panel(header_content, style="white on black"))

        layout["main"][f"example_{idx}"]["digit"].update(Panel(Text.from_ansi(mnist_digit), title="MNIST digit"))
        layout["main"][f"example_{idx}"]["layer01"].update(Panel(Text.from_ansi(layer01), title="layer 01: 32x32"))
        layout["main"][f"example_{idx}"]["layer02"].update(Panel(Text.from_ansi(layer02), title="layer 02: 16x16"))
        layout["main"][f"example_{idx}"]["last_layer"].update(Panel(Text.from_ansi(last_layer), title="last layer"))

    return layout


def train(steps: int):
    with Live(auto_refresh=True, screen=True, console=console) as live:
        for step in range(steps):
            _ = jit_step()
            if step % 5 == 0:
                Tensor.training = False
                acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
                live.update(visualize(step=step, max_steps=steps, acc=acc))


def visualize_layer(dim: int, weights: np.ndarray, idx: int) -> str:
    fmt_str = []
    img = weights[idx].squeeze().numpy()
    min_value = abs(min(img))
    max_value = max(img) + min_value
    for y in range(0, dim, 2):
        for x in range(dim):
            fg_value = img[x + y * dim] + min_value
            bg_value = img[x + (y + 1) * dim] + min_value
            fg_color = GRAYSCALE[int(fg_value / max_value * (len(GRAYSCALE) - 1))].fg
            bg_color = GRAYSCALE[int(bg_value / max_value * (len(GRAYSCALE) - 1))].bg
            fmt_str.append(fg_color + bg_color + SpecialChars.UPPER_HALF_BLOCK + Ansi.COLOR_RESET)
        fmt_str.append("\n")
    return "".join(fmt_str)


def visualize_last_layer(dim: int, weights: np.ndarray, labels, idx: int) -> str:
    fmt_str = []
    img = weights[idx].squeeze().numpy()
    min_value = abs(min(img))
    max_value = max(img) + min_value
    for x in range(dim):
        fg_value = img[x] + min_value
        fg_color = GRAYSCALE[int(fg_value / max_value * (len(GRAYSCALE) - 1))].fg
        fmt_str.append(fg_color + SpecialChars.FULL_BLOCK + Ansi.COLOR_RESET)
    fmt_str.append("\n")
    color = Colors.RED.fg if weights[idx].argmax().item() != labels[idx].numpy().item() else Colors.GREEN.fg
    fmt_str.append(f"pred: {color}{weights[idx].argmax().item()}{Ansi.COLOR_RESET}")
    return "".join(fmt_str)


def visualize_mnist_digit(X, idx: int) -> str:
    fmt_str = []
    img = X[idx].squeeze().numpy()
    for y in range(0, X_train.shape[2], 2):
        for x in range(img.shape[1]):
            fg_value = img[y, x]
            bg_value = img[y + 1, x]
            fg_color = GRAYSCALE[int(fg_value / 255 * (len(GRAYSCALE) - 1))].fg
            bg_color = GRAYSCALE[int(bg_value / 255 * (len(GRAYSCALE) - 1))].bg
            fmt_str.append(fg_color + bg_color + SpecialChars.UPPER_HALF_BLOCK + Ansi.COLOR_RESET)
        fmt_str.append("\n")
    return "".join(fmt_str)


train(steps=1_000)
