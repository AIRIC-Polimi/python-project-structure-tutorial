import configparser
from os import path


def parse_config(argparse_parser, path_keys=[], print_config=False):
    args = argparse_parser.parse_args().__dict__

    if args["config_file"]:
        if not path.exists(args["config_file"]):
            print(f"[WARNING] Provided config file {args['config_file']} not found.")
        else:
            config_parser = configparser.ConfigParser()
            config_parser.read(args["config_file"])

            for key, value in config_parser.items("DEFAULT"):
                if key not in args or args[key] is None:
                    args[key] = value

                    if key in path_keys:
                        args[key] = path.abspath(path.join(path.dirname(__file__), "../..", args[key]))

    # if "dataset_path" not in args or args["dataset_path"] is None:
    #     print("[ERROR] Must provide the path to the dataset with the --dataset-path option")
    #     exit(1)

    if print_config:
        max_key_len = max([len(k) for k in args])
        print("--- CONFIGURATION ---" + "-" * max_key_len)
        for k, v in args.items():
            print(f"{k.replace('_', ' '):<{max_key_len+1}}: {v}")
        print("---------------------" + "-" * max_key_len)

    return args
