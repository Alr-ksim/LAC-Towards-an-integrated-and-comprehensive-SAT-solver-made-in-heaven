import argparse

from cdcl import cdcl
from utils import read_cnf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, default="examples/bmc-1.cnf"
    )

    return parser.parse_args()

def main(args):
    # Create problem.
    with open(args.input, "r") as f:
        sentence, num_vars = read_cnf(f)

    # Create CDCL solver and solve it!
    res = cdcl(sentence, num_vars)

    if res is None:
        print("✘ No solution found")
    else:
        print(f"✔ Successfully found a solution: {res}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
