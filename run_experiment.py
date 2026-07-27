import sys

from main import main


if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        sys.argv.insert(1, "--config")

    main()
