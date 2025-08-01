import argparse

parser = argparse.ArgumentParser(description="Control output display")
parser.add_argument("--display", "-d", action="store_true", help="Display output")

args = parser.parse_args()

print(args.display)