import numpy as np
from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from tinygrad import Tensor, TinyJit, nn
from utils import Ansi, SpecialChars, GRAYSCALE, Colors


class SimpleCNN:
    def __init__(self):
        # Input: (batch, 1, 28, 28)
        self.c1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2, bias=False)
        # Output: (batch, 4, 28, 28)
        # Stride defaults to 1. padding=2 keeps size 28x28 with kernel 5x5.
        self.c2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1, bias=False)
        # Output: (batch, 8, 28, 28)
        # padding=1 keeps size 28x28 with kernel 3x3.
        # For this example (no pooling): 8 channels * 28 height * 28 width
        self.l1 = nn.Linear(in_features=8 * 28 * 28, out_features=10)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.c1(x.float()).relu()
        x = self.c2(x).relu()
        return self.l1(x.flatten(1))


@TinyJit
def step():
    Tensor.training = True
    samples = Tensor.randint(batch_size, high=X_train.shape[0])
    X = X_train[samples]
    Y = Y_train[samples]

    optim.zero_grad()
    out = model(X)
    loss = out.sparse_categorical_crossentropy(Y).backward()
    optim.step()
    return loss.item()


def visualize_mnist_digit(X: Tensor, idx: int) -> str:
    """Visualizes a single MNIST digit image (28x28)."""
    fmt_str = []
    img = X[idx].squeeze().numpy()
    H, W = img.shape
    for y in range(0, H, 2):  # Step by 2 for half-block character
        for x in range(W):
            fg_value = img[y, x]
            bg_value = img[y + 1, x]
            fg_color = GRAYSCALE[int(fg_value / 255 * (len(GRAYSCALE) - 1))].fg
            bg_color = GRAYSCALE[int(bg_value / 255 * (len(GRAYSCALE) - 1))].bg
            fmt_str.append(fg_color + bg_color + SpecialChars.UPPER_HALF_BLOCK + Ansi.COLOR_RESET)
        fmt_str.append("\n")
    return "".join(fmt_str)


def visualize_feature_maps(feature_map_tensor: Tensor, idx: int, channel: int = 0) -> str:
    """Visualizes feature maps from a convolutional layer for a specific example."""
    # Input shape: (batch, channels, height, width)
    maps = feature_map_tensor[idx].numpy()
    _, H, W = maps.shape

    channel_map = maps[channel]
    # Normalize this specific channel for better contrast
    min_val = channel_map.min()
    max_val = channel_map.max()
    range_val = max_val - min_val
    if range_val < 1e-6:
        range_val = 1.0  # Avoid division by zero

    normalized_channel = (channel_map - min_val) / range_val

    fmt_str = []
    for y in range(0, H, 2):
        line_str = []
        for x in range(W):
            fg_val = normalized_channel[y, x]
            bg_val = normalized_channel[y + 1, x]

            fg_idx = int(np.clip(fg_val * (len(GRAYSCALE) - 1), 0, len(GRAYSCALE) - 1))
            bg_idx = int(np.clip(bg_val * (len(GRAYSCALE) - 1), 0, len(GRAYSCALE) - 1))
            fg_color = GRAYSCALE[fg_idx].fg
            bg_color = GRAYSCALE[bg_idx].bg

            line_str.append(fg_color + bg_color + SpecialChars.UPPER_HALF_BLOCK + Ansi.COLOR_RESET)
        fmt_str.append("".join(line_str))

    header = f"Channel {channel}:\n"
    return header + "\n".join(fmt_str)


def visualize_last_layer(logits: Tensor, labels: Tensor, idx: int, bar_length: int = 18) -> str:
    """Visualizes the final layer output (logits) as a bar chart."""
    fmt_str = []
    # Logits shape: (batch, 10), Labels shape: (batch,)
    img_logits = logits[idx].numpy()
    label = labels[idx].numpy().item()

    probs = Tensor(img_logits).softmax().numpy()
    max_prob = probs.max()

    for digit in range(10):
        value = probs[digit] / max_prob
        bars = int(value * bar_length)
        label_color = Colors.YELLOW.fg if digit != label else Colors.GREEN.fg  # Highlight true label
        fmt_str.append(
            f"{label_color}{digit}{Ansi.COLOR_RESET}: ["
            + SpecialChars.DOUBLE_HORIZONTAL_BAR * bars
            + " " * (bar_length - bars)
            + f"] {probs[digit]:.2f}\n"
        )
    fmt_str.append("\n")
    pred_digit = img_logits.argmax()
    color = Colors.RED.fg if pred_digit != label else Colors.GREEN.fg
    fmt_str.append(f"True: {label} | Pred: {color}{pred_digit}{Ansi.COLOR_RESET}")
    return "".join(fmt_str)


def visualize(step_num: int, max_steps: int, acc: float, num_examples: int = 3):
    Tensor.training = False
    samples = Tensor.randint(batch_size, high=X_test.shape[0])
    X, Y = X_test[samples], Y_test[samples]
    X_reshaped = X.reshape(-1, 1, 28, 28).float()

    h1 = model.c1(X_reshaped).relu()
    h2 = model.c2(h1).relu()
    h_flat = h2.flatten(1)
    logits = model.l1(h_flat)

    layout = Layout()
    layout.split(Layout(name="header", size=3), Layout(name="content", ratio=1))
    layout["content"].split_column(Layout(name="main", ratio=1))

    main_splits = [Layout(name=f"example_{i}") for i in range(num_examples)]
    layout["main"].split_column(*main_splits)

    h1_shape = h1.shape[1:]  # C, H, W
    h2_shape = h2.shape[1:]  # C, H, W

    for idx in range(num_examples):
        layout["main"][f"example_{idx}"].split_row(
            Layout(name="digit"),
            Layout(name="layer01"),
            Layout(name="layer02"),
            Layout(name="last_layer"),
        )

        mnist_digit_vis = visualize_mnist_digit(X_reshaped, idx=idx)
        layer01_vis = visualize_feature_maps(h1, idx=idx)
        layer02_vis = visualize_feature_maps(h2, idx=idx)
        last_layer_vis = visualize_last_layer(logits, Y, idx=idx)

        left_text = "[b]MNIST CNN Training[/b]"
        right_text = f"Step: {step_num} / {max_steps} | Accuracy: {acc:.2%}"
        header_content = Columns([left_text, Align.right(right_text)], expand=True)
        layout["header"].update(Panel(header_content, style="white on black"))

        layout["main"][f"example_{idx}"]["digit"].update(
            Panel(
                Align.center(Text.from_ansi(mnist_digit_vis), vertical="middle"),
                title="Input Digit",
            )
        )
        layout["main"][f"example_{idx}"]["layer01"].update(
            Panel(
                Align.center(Text.from_ansi(layer01_vis), vertical="middle"),
                title=f"Conv1 Out ({h1_shape[0]}x{h1_shape[1]}x{h1_shape[2]})",
            )
        )
        layout["main"][f"example_{idx}"]["layer02"].update(
            Panel(
                Align.center(Text.from_ansi(layer02_vis), vertical="middle"),
                title=f"Conv2 Out ({h2_shape[0]}x{h2_shape[1]}x{h2_shape[2]})",
            )
        )
        layout["main"][f"example_{idx}"]["last_layer"].update(
            Panel(
                Align.center(Text.from_ansi(last_layer_vis), vertical="middle"),
                title="Prediction",
            )
        )

    return layout


def train(steps: int):
    with Live(auto_refresh=False, screen=True, console=console, vertical_overflow="visible") as live:
        live.update(visualize(step_num=0, max_steps=steps, acc=np.nan), refresh=True)
        _ = input()
        for i in range(steps):
            _ = step()
            if i % 20 == 0 or i == steps - 1:
                Tensor.training = False
                test_samples = 128
                rand_idx = Tensor.randint(test_samples, high=X_test.shape[0])
                X_test_sample = X_test[rand_idx].reshape(-1, 1, 28, 28).float()
                Y_test_sample = Y_test[rand_idx]
                acc = (model(X_test_sample).argmax(axis=1) == Y_test_sample).mean().item()

                live.update(visualize(step_num=i, max_steps=steps, acc=acc), refresh=True)


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = nn.datasets.mnist()
    model = SimpleCNN()
    optim = nn.optim.Adam(nn.state.get_parameters(model), lr=1e-4)
    batch_size = 64
    console = Console()
    try:
        train(steps=2000)
    except KeyboardInterrupt:
        pass
