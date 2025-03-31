import argparse
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import GRAYSCALE, Ansi, SpecialChars


def visualize_embedding(dim: int, embedding: torch.tensor, title: str):
    print(
        f"{11 * SpecialChars.DOUBLE_HORIZONTAL_BAR}-{title}-{11 * SpecialChars.DOUBLE_HORIZONTAL_BAR}{Ansi.COLOR_RESET}",
    )
    for y in range(0, dim, 2):
        img = embedding[0].squeeze().numpy()
        for x in range(dim):
            min_value = abs(min(img))
            fg_value = img[x + y * dim] + min_value
            bg_value = img[x + (y + 1) * dim] + min_value
            max_value = max(img) + min_value
            fg_color = GRAYSCALE[int(fg_value / max_value * (len(GRAYSCALE) - 1))].fg
            bg_color = GRAYSCALE[int(bg_value / max_value * (len(GRAYSCALE) - 1))].bg
            print(
                fg_color + bg_color + SpecialChars.UPPER_HALF_BLOCK + Ansi.COLOR_RESET,
                end="",
            )
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="llm embedding showroom", description="shows embedding space of llm"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="huggingface model id",
        default="HuggingFaceTB/SmolLM2-135M",
    )
    parser.add_argument("-w", type=str, help="words to embed", default="dog")
    parser.add_argument("-c", type=int, help="number of cols per row", default=4)
    args = parser.parse_args()

    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    tokens = tokenizer.encode(args.w, return_tensors="pt").to(device)
    print(f"{args.w} -> {tokens}")
    rows = math.ceil(tokens.shape[1] // args.c)
    embedding_layer = model.get_input_embeddings()

    dim = 24
    for row in range(rows):
        row_tokens = tokens[0].cpu().tolist()[row * args.c : (row + 1) * args.c]
        print(SpecialChars.DOUBLE_HORIZONTAL_BAR, end="")
        for token in row_tokens:
            word_token = tokenizer.decode(token)
            print(
                f"{3 * SpecialChars.DOUBLE_HORIZONTAL_BAR}-{word_token}-{(20 - len(word_token)) * SpecialChars.DOUBLE_HORIZONTAL_BAR}",
                end="",
            )
        print()

        for y in range(0, dim, 2):
            print(SpecialChars.DOUBLE_VERTICAL_BAR, end="")
            for token in row_tokens:
                with torch.no_grad():
                    img = embedding_layer(torch.tensor(token)).cpu().numpy()
                for x in range(dim):
                    min_value = abs(min(img))
                    fg_value = img[x + y * dim] + min_value
                    bg_value = img[x + (y + 1) * dim] + min_value
                    max_value = max(img) + min_value
                    fg_color = GRAYSCALE[
                        int(fg_value / max_value * (len(GRAYSCALE) - 1))
                    ].fg
                    bg_color = GRAYSCALE[
                        int(bg_value / max_value * (len(GRAYSCALE) - 1))
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
    print((args.c * 25 + 1) * SpecialChars.DOUBLE_HORIZONTAL_BAR)
