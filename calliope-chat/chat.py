"""
a little cli interface for my poetry language model as a reference for other stuff
with stub functions for the actual model
"""

line_divider = (
    ".~^~._.~^~._.~^~._.~^~._.~^~._.~^~._.~^~._.~^~._.~^~._.~^~._.~^~._.~^~._.~^~._-~"
)
thin_line_divider = (
    "- -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - --- -- - -- -"
)
header = rf"""
{line_divider}
 .~----~.                                                              .~----~.
(  ~^^~  )             ______       __ __ _                           (  ~^^~  ) 
 )      (             / ____/____  / // /(_)___  ____  ___             )      (
(  (())  )           / /    / __ \/ // // / __ \/ __ \/ _ \           (  (())  ) 
 |  ||  |            | |___/ /_/ / // // / /_/ / /_/ / ___/            |  ||  |
 |  ||  |            \____/\__._/_//_//_/\____/ .___/\___/             |  ||  |
 |  ||  |                                    /_/                       |  ||  |
 |  ||  |                   -- your personal muse --                   |  ||  |
(________)                                                            (________)
{line_divider}
"""

DEVICE = "cpu"

def setup():
    return None, None


def generate(prompt: str) -> str:
    return prompt + "\n" + "sphinx of black quartz,\njudge my vow"


def get_multiline_input(start: str):
    lines = []
    while True:
        line = input(start)
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    print(header)
    print(f"loading calliope on {DEVICE}..")
    model, tokenizer = setup()
    print("loaded stub model + tokenizer c:")
    print(thin_line_divider)

    print("prompt calliope, start a poem's song")
    print("an empty line tells calliope to follow along")
    print("write 'exit' to leave, if you won't stay long")

    prompt = get_multiline_input(start=">> ")
    while prompt != "exit":
        if len(prompt) != 0:
            print(line_divider)
            print()
            print(generate(prompt=prompt))
            print()
            print(line_divider)
        prompt = get_multiline_input(start=">> ")

    print("bye wanderer, may your path be ever light")
