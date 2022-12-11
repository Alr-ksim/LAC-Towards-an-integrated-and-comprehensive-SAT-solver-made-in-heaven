import argparse

from cdcl import cdcl
from utils import read_cnf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, default="examples/bmc-2.cnf"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="results/output.txt"
    )

    return parser.parse_args()

def main(args):
    # Create problem.
    with open(args.input, "r") as f:
        sentence, num_vars = read_cnf(f)

    # Create CDCL solver and solve it!
    res = cdcl(sentence, num_vars)

    save_path = open(args.output, "w", encoding="utf-8")

    if res is None:
        save_path.write("✘ No solution found")
    else:
        save_path.write(f"✔ Successfully found a solution: {res}")

    save_path.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)
