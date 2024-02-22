import argparse

from my_project.algorithms import GeneticAlgorithm


def parse_config(print_config=False):
    parser = argparse.ArgumentParser(
        prog="run_genetic.py",
        description="""Implementation of a Genetic Algorithm which aims to produce the user specified target string.
This implementation calculates each candidate's fitness based on the alphabetical distance between the candidate the target.
A candidate is selected as a parent with probabilities proportional to the candidate's fitness.
Reproduction is implemented as a single-point crossover between pairs of parents.
Mutation is done by randomly assigning new characters with uniform probability.""",
    )

    parser.add_argument("target_string", type=str, help="The target string to be guessed.")
    parser.add_argument("-p", "--population", type=int, default=100, help="The population size.")
    parser.add_argument("-m", "--mutation-rate", type=float, default=0.02, help="The mutation rate.")
    parser.add_argument(
        "-i", "--iterations", type=int, default=1000, help="The number of iterations to run the algorithm for."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Whether to increase the log level for a more verbose output.",
    )
    parser.add_argument("--seed", type=int, help="The random seed, if needed.")

    args = parser.parse_args()

    if print_config:
        max_key_len = max([len(k) for k in args.__dict__])
        print("--- CONFIGURATION ---" + "-" * max_key_len)
        for k, v in args.__dict__.items():
            print(f"{k.replace('_', ' '):<{max_key_len+1}}: {v}")
        print("---------------------" + "-" * max_key_len)

    return args


def run():
    config = parse_config(print_config=True)

    result = GeneticAlgorithm(
        target_string=config.target_string,
        population_size=config.population,
        mutation_rate=config.mutation_rate,
        random_seed=config.seed,
        log_level="DEBUG" if config.verbose else "INFO",
    ).run(iterations=config.iterations)


if __name__ == "__main__":
    run()
