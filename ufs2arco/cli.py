import argparse
import yaml
from ufs2arco.driver import Driver
from ufs2arco.multidriver import MultiDriver

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

    with open(args.yaml_file, "r") as f:
        config = yaml.safe_load(f)

    if "multisource" in config.keys():
        MultiDriver(args.yaml_file).run(overwrite=args.overwrite)
    else:
        Driver(args.yaml_file).run(overwrite=args.overwrite)

if __name__ == "__main__":
    main()
