from random import choice, sample, random
from operator import attrgetter
import numpy as np


# CLASS SNAKE
class Snake:
    def __init__(self, representation=None, size=None, replacement=True):
        if representation is None:
            if replacement:
                self.representation = [choice(np.arange(-1, 1, step=0.01)) for _ in range(size)]
            elif not replacement:
                self.representation = [sample(np.arange(-1, 1, step=0.01), size)]
        else:
            self.representation = representation

        self.fitness, self.score = self.get_fitness()
        print(f'Fitness: {self.fitness} || Score: {self.score}')

    def get_fitness(self):
        raise Exception("You need to monkey patch the fitness path.")

    def get_neighbours(self, func, **kwargs):
        raise Exception("You need to monkey patch the neighbourhood function.")

    def index(self, value):
        return self.representation.index(value)

    def __len__(self):
        return len(self.representation)

    def __getitem__(self, position):
        return self.representation[position]

    def __setitem__(self, position, value):
        self.representation[position] = value

    def __repr__(self):
        return f'Fitness: {self.fitness} || Score: {self.score}'


# CLASS POPULATION
class Population:
    def __init__(self, size, optim, **kwargs):
        self.snakes = []
        self.size = size
        self.optim = optim
        for s in range(size):
            self.snakes.append(
                Snake(
                    size=kwargs["sol_size"],
                    replacement=kwargs["replacement"],
                )
            )

    def evolve(self, gens, select, crossover, mutate, co_p, mu_p, elitism, n_elit=5):
        maxFit = 0
        maxGen = 0

        for gen in range(gens):
            print(f'\n ------ Beginning of Generation {gen + 1} ------')
            # List to store new population
            new_pop = []

            sorted_snakes = []
            best_snakes = []

            if self.optim == "max":
                # Top snakes in terms of fitness
                sorted_snakes = sorted(self.snakes, key=attrgetter("fitness"), reverse=True)  # descending way
                best_snakes = sorted_snakes[:n_elit]

            elif self.optim == "min":
                sorted_snakes = self.snakes.sort(key=attrgetter("fitness"), reverse=False)
                best_snakes = sorted_snakes[:n_elit]

            # If elitism, append these snakes to the new generation
            if elitism:
                new_pop += best_snakes
                print(f'Best {n_elit} individuals of previous generation:')
                for s in best_snakes:
                    print(f'--> {s}')

            # Do this until new population is full
            while len(new_pop) < self.size:
                size_tournament = 10
                parent1, parent2 = select(self.optim, sorted_snakes, size_tournament), \
                                   select(self.optim, sorted_snakes, size_tournament)

                # Crossover
                if random() < co_p:
                    offspring1, offspring2 = crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2

                # Mutation
                if random() < mu_p:
                    offspring1 = mutate(offspring1)
                if random() < mu_p:
                    offspring2 = mutate(offspring2)

                new_pop.append(Snake(representation=offspring1))

                if len(new_pop) < self.size:
                    new_pop.append(Snake(representation=offspring2))

            self.snakes = new_pop

            if self.optim == "max":
                maximum = max(self, key=attrgetter("fitness"))
                if maximum.fitness > maxFit:
                    maxFit = maximum.fitness
                    maxGen = gen + 1
                    best_weights = maximum.representation
                print(f'Best Individual: {maximum} ---> In Generation {maxGen}')

            elif self.optim == "min":
                maximum = min(self, key=attrgetter("fitness"))
                if maximum.fitness < maxFit:
                    maxFit = maximum.fitness
                    maxGen = gen + 1
                    best_weights = maximum.representation
                print(f'Best Individual: {maximum} ---> In Generation {maxGen}')

        print(best_weights)

    def __len__(self):
        return len(self.snakes)

    def __getitem__(self, position):
        return self.snakes[position]

    def __repr__(self):
        return f"Population(size={len(self.snakes)}, individual_size={len(self.snakes[0])})"
