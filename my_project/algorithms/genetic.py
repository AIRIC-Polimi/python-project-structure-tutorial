import string
import numpy as np
import time
from tqdm import tqdm
from ..logger import get_logger

class GeneticAlgorithm():
    """An implementation of a Genetic Algorithm which will try to produce the user
    specified target string.

    Parameters:
    -----------
    target_string: string
        The string which the GA should try to produce.
    population_size: int
        The number of individuals (possible solutions) in the population.
    mutation_rate: float
        The rate (or probability) of which the alleles (chars in this case) should be
        randomly changed.
    random_seed: int
        The seed to use to initialize the random number generators. Defaults to the
        current timestamp.
    log_level: str
        The logging level. If not given, uses the defaults from `aim_tutorial/logger.py`.
    """
    def __init__(self, target_string, population_size, mutation_rate, random_seed=None, log_level=None):
        self.target = target_string
        self.target_tokens = np.array([ord(l) for l in self.target])
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tokens = np.array([ord(l) for l in [" "] + list(string.ascii_letters)])
        self.max_loss = len(target_string) * (max([t for t in self.tokens]) - min([t for t in self.tokens]))

        if random_seed is None:
            random_seed = int(time.time())
        np.random.seed = random_seed
        self.rng = np.random.default_rng(seed=random_seed)
        self.logger = get_logger(log_level)

    def _initialize(self):
        """ Initialize population with random strings """
        self.population = []
        for _ in range(self.population_size):
            # Select random tokens as new individual
            individual = self.rng.choice(self.tokens, size=len(self.target))
            self.population.append(individual)
        self.population = np.array(self.population, dtype='int8')

    def _calculate_fitness(self):
        """ Calculates the fitness of each individual in the population """
        token_distances = np.abs(self.population - self.target_tokens)
        population_fitness = 1 / (token_distances.sum(axis=1) + 1)
        return population_fitness

    def _mutate(self, individual):
        """ Randomly change the individual's characters with probability
        self.mutation_rate """
        mask = self.rng.random(len(individual)) < self.mutation_rate
        # individual[mask] = rng.choice(self.tokens, size=mask.sum())
        random_indices = np.random.randint(0, len(self.tokens), size=mask.sum())
        individual[mask] = self.tokens[random_indices]
        return individual

    def _crossover(self, parent1, parent2):
        """ Create children from parents by crossover """
        # Select random crossover point
        cross_i = np.random.randint(0, len(parent1))
        child1 = np.concatenate([parent1[:cross_i], parent2[cross_i:]])
        child2 = np.concatenate([parent2[:cross_i], parent1[cross_i:]])
        return child1, child2

    def run(self, iterations):
        # Initialize new population
        self._initialize()

        new_population = np.zeros(self.population.shape, dtype='int8')
        for epoch in tqdm(range(iterations)):
            population_fitness = self._calculate_fitness()

            fittest_individual = self.population[population_fitness.argmax()]
            highest_fitness = population_fitness.max()

            # If we have found individual which matches the target => Done
            if (fittest_individual == self.target_tokens).all():
                break

            # Set the probability that the individual should be selected as a parent
            # proportionate to the individual's fitness.
            parent_probabilities = population_fitness / population_fitness.sum()

            # Determine the next generation
            parents = self.rng.choice(self.population, size=self.population_size, p=parent_probabilities, replace=True)
            for i in np.arange(0, self.population_size, 2):
                # Select two parents randomly according to probabilities
                parent1, parent2 = parents[i, :], parents[i+1, :]
                # Perform crossover to produce offspring
                child1, child2 = self._crossover(parent1, parent2)
                # Save mutated offspring for next generation
                new_population[i, :] = self._mutate(child1)
                new_population[i+1, :] = self._mutate(child2)

            # 1-elitism
            new_population[0, :] = fittest_individual

            self.logger.debug("[%d Closest Candidate: '%s', Fitness: %.8f]" % (epoch, ''.join([chr(t) for t in fittest_individual]), highest_fitness))
            self.population = np.copy(new_population)

        self.logger.info("[%d Answer: '%s']" % (epoch, ''.join([chr(t) for t in fittest_individual])))
        return ''.join([chr(t) for t in fittest_individual]), epoch