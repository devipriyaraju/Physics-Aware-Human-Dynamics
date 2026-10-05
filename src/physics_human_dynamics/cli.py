import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Physics-aware human dynamics project"
    )
    parser.add_argument(
        "--show-layout",
        action="store_true",
        help="Print the expected data layout",
    )
    args = parser.parse_args()
    if args.show_layout:
        print(Path("data/raw/README.md").read_text(encoding="utf-8"))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
