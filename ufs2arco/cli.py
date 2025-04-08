"""Thanks to ChatGPT"""
import argparse
from ufs2arco.driver import Driver

def main():
    parser = argparse.ArgumentParser(
        description="Run the ufs2arco workflow with a given YAML recipe.",
    )
    parser.add_argument(
        "yaml_file",
        type=str,
        help="Path to the YAML recipe file.",
    )
    args = parser.parse_args()

    Driver(args.yaml_file).run()

if __name__ == "__main__":
    main()
