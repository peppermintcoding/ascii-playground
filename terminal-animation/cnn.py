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
from utils import Ansi, SpecialChars, GRAYSCALE, Colors


class SimpleCNN:
    def __init__(self):
        # Simple CNN: Conv -> ReLU -> Conv -> ReLU -> Flatten -> Linear
        # Input: (batch, 1, 28, 28)
        self.c1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2, bias=False)
        # Output: (batch, 4, 28, 28)
        # Stride defaults to 1. padding=2 keeps size 28x28 with kernel 5x5.
        self.c2 = nn.Conv2d(
            in_channels=4, out_channels=8, kernel_size=3, padding=1, bias=False
        )  # Output: (batch, 8, 28, 28)
        # padding=1 keeps size 28x28 with kernel 3x3.

        # For this example (no pooling): 8 channels * 28 height * 28 width
        fc_in_features = 8 * 28 * 28
        self.l1 = nn.Linear(in_features=fc_in_features, out_features=10)

    def __call__(self, x: Tensor) -> Tensor:
        x = x.float()
        x = self.c1(x).relu()
        x = self.c2(x).relu()
        x = x.flatten(1)
        x = self.l1(x)
        return x


X_train, Y_train, X_test, Y_test = mnist()
model = SimpleCNN()
optim = nn.optim.Adam(nn.state.get_parameters(model), lr=1e-4)
batch_size = 64
console = Console()


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


def visualize_mnist_digit(X, idx: int) -> str:
    """Visualizes a single MNIST digit image."""
    fmt_str = []
    img = X[idx].squeeze().numpy()  # Shape (28, 28)
    H, W = img.shape
    for y in range(0, H, 2):  # Step by 2 for half-block character
        for x in range(W):
            fg_value = img[y, x]
            # Handle the case for the last row if height is odd
            bg_value = img[y + 1, x] if (y + 1) < H else img[y, x]  # Repeat value for padding

            fg_color = GRAYSCALE[int(fg_value / 255 * (len(GRAYSCALE) - 1))].fg
            bg_color = GRAYSCALE[int(bg_value / 255 * (len(GRAYSCALE) - 1))].bg
            fmt_str.append(fg_color + bg_color + SpecialChars.UPPER_HALF_BLOCK + Ansi.COLOR_RESET)
        fmt_str.append("\n")
    return "".join(fmt_str)


def visualize_feature_maps(feature_map_tensor: Tensor, idx: int) -> str:
    """Visualizes feature maps from a convolutional layer for a specific example."""
    # Input shape: (batch, channels, height, width)
    maps = feature_map_tensor[idx].numpy()  # Shape: (channels, height, width)
    C, H, W = maps.shape

    # render just one channel
    CHANNEL = 0
    channel_map = maps[CHANNEL]

    # Normalize this specific channel for better contrast
    min_val = channel_map.min()
    max_val = channel_map.max()
    range_val = max_val - min_val
    if range_val < 1e-6:
        range_val = 1.0  # Avoid division by zero

    normalized_channel = (channel_map - min_val) / range_val

    channel_render = []
    for y in range(0, H, 2):
        line_str = []
        for x in range(W):
            fg_val = normalized_channel[y, x]
            # Handle odd height, repeat value for padding
            bg_val = normalized_channel[y + 1, x] if (y + 1) < H else normalized_channel[y, x]

            fg_idx = int(np.clip(fg_val * (len(GRAYSCALE) - 1), 0, len(GRAYSCALE) - 1))
            bg_idx = int(np.clip(bg_val * (len(GRAYSCALE) - 1), 0, len(GRAYSCALE) - 1))

            fg_color = GRAYSCALE[fg_idx].fg
            bg_color = GRAYSCALE[bg_idx].bg

            line_str.append(fg_color + bg_color + SpecialChars.UPPER_HALF_BLOCK + Ansi.COLOR_RESET)
        channel_render.append("".join(line_str))

    header = f"Channel {CHANNEL}:\n"
    return header + "\n".join(channel_render)


def visualize_last_layer(logits: Tensor, labels: Tensor, idx: int, bar_length: int = 18) -> str:
    """Visualizes the final layer output (logits) as a bar chart."""
    fmt_str = []
    # Logits shape: (batch, 10), Labels shape: (batch,)
    img_logits = logits[idx].numpy()  # Shape: (10,)
    label = labels[idx].numpy().item()  # Get scalar label

    probs = Tensor(img_logits).softmax().numpy()
    max_prob = probs.max()

    for digit in range(10):
        value = probs[digit] / max_prob if max_prob > 1e-6 else 0  # Scale 0-1 relative to max
        bars = int(np.clip(value, 0, 1) * bar_length)
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
    Tensor.training = False  # Set model to evaluation mode for visualization
    # Get a sample batch from the test set
    samples = Tensor.randint(batch_size, high=X_test.shape[0])
    X, Y = X_test[samples], Y_test[samples]
    # Reshape X for the CNN model
    X_reshaped = X.reshape(-1, 1, 28, 28).float()  # Ensure float32

    h1 = model.c1(X_reshaped).relu()  # Output of first Conv + ReLU
    h2 = model.c2(h1).relu()  # Output of second Conv + ReLU
    h_flat = h2.flatten(1)  # Flatten output of last conv/pool
    logits = model.l1(h_flat)  # Final output logits

    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="content", ratio=1),
    )
    layout["content"].split_column(
        Layout(name="main", ratio=1),
    )

    main_splits = [Layout(name=f"example_{i}") for i in range(num_examples)]
    layout["main"].split_column(*main_splits)

    # Get shapes for titles
    h1_shape = h1.shape[1:]  # C, H, W
    h2_shape = h2.shape[1:]  # C, H, W

    for idx in range(num_examples):
        layout["main"][f"example_{idx}"].split_row(
            Layout(name="digit"),
            Layout(name="layer01"),
            Layout(name="layer02"),
            Layout(name="last_layer"),
        )

        # Use the reshaped X for visualization
        mnist_digit_vis = visualize_mnist_digit(X_reshaped, idx=idx)
        # Pass intermediate activations to the feature map visualizer
        layer01_vis = visualize_feature_maps(h1, idx=idx)
        layer02_vis = visualize_feature_maps(h2, idx=idx)
        # Pass final logits and true labels to the last layer visualizer
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
        for i in range(steps):
            _ = step()  # Use the jit_step function directly
            if i % 20 == 0 or i == steps - 1:
                Tensor.training = False  # Set model to evaluation mode for accuracy calculation
                test_samples = 128
                rand_idx = Tensor.randint(test_samples, high=X_test.shape[0])
                X_test_sample = X_test[rand_idx].reshape(-1, 1, 28, 28).float()  # Ensure float32
                Y_test_sample = Y_test[rand_idx]
                acc = (model(X_test_sample).argmax(axis=1) == Y_test_sample).mean().item()

                live.update(visualize(step_num=i, max_steps=steps, acc=acc), refresh=True)


if __name__ == "__main__":
    try:
        train(steps=2000)
    except KeyboardInterrupt:
        pass
