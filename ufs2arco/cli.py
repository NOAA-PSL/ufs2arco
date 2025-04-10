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

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Pass this flag to overwrite an existing zarr store in the specified location",
    )
    args = parser.parse_args()

    Driver(args.yaml_file).run(overwrite=args.overwrite)

if __name__ == "__main__":
    main()
