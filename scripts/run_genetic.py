import argparse
import json
import time
from os import makedirs, path

from my_project.algorithms import GeneticAlgorithm
from scripts.utils.parse_config import parse_config

parser = argparse.ArgumentParser(
    prog="run_genetic.py",
    description="""Implementation of a Genetic Algorithm which aims to produce the user specified target string.
This implementation calculates each candidate's fitness based on the alphabetical distance between the candidate the target.
A candidate is selected as a parent with probabilities proportional to the candidate's fitness.
Reproduction is implemented as a single-point crossover between pairs of parents.
Mutation is done by randomly assigning new characters with uniform probability.""",
)

parser.add_argument("target_string", type=str, help="The target string to be guessed.")
parser.add_argument("-p", "--population", type=int, help="The population size.")
parser.add_argument("-m", "--mutation-rate", type=float, help="The mutation rate.")
parser.add_argument("-i", "--iterations", type=int, help="The number of iterations to run the algorithm for.")
parser.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Whether to increase the log level for a more verbose output.",
)
parser.add_argument("--seed", type=int, help="The random seed, if needed.")
parser.add_argument(
    "--config-file",
    type=str,
    help="The path to a config file to read from. Configs from the file can be overridden by commandline options.",
)
parser.add_argument(
    "--experiments-folder", type=str, help="The path to a folder where to store experiments results (optional)."
)


def run():
    config = parse_config(parser, path_keys=["experiments_folder"], print_config=True)

    if config["experiments_folder"] is not None:
        run_folder = path.join(config["experiments_folder"], str(int(time.time())))
        makedirs(run_folder, exist_ok=True)
        with open(path.join(run_folder, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    GeneticAlgorithm(
        target_string=config["target_string"],
        population_size=int(config["population"]),
        mutation_rate=float(config["mutation_rate"]),
        random_seed=config["seed"],
        log_level="DEBUG" if config["verbose"] else "INFO",
        log_file=path.join(run_folder, "logs") if run_folder is not None else None,
    ).run(iterations=int(config["iterations"]))


if __name__ == "__main__":
    run()
