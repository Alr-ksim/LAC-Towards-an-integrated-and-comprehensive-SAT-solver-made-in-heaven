import argparse
from cdcl_ import cdcl
from utils import read_cnf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, default="examples/bmc-1.cnf"
    )
    parser.add_argument(
        "-d", "--decide_method", type=str, default="LRB", choices=['VSIDS', 'CHB', 'LRB']
    )

    return parser.parse_args()


def main(args):
    # Create problem.
    with open(args.input, "r") as f:
        sentence, num_vars = read_cnf(f)

    # Create CDCL solver and solve it!
    solver = cdcl(sentence, num_vars, args.input, args.decide_method)
    res = solver.solve()

    if res is None:
        print("✘ No solution found")
    else:
        print(f"✔ Successfully found a solution: {res}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
