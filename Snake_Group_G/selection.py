from random import uniform, sample
from operator import attrgetter
from random import choice


# --------------- FITNESS PROPORTIONATE IMPLEMENTATION ----------------
def fps(population):
    """
    Fitness proportionate selection implementation.
    Args: population (Population): The population we want to select from.
    Returns: Individual: selected individual.
    """

    if population.optim == "max":

        # Sum total fitness
        total_fitness = sum([i.fitness for i in population])

        # Get a 'position' on the wheel
        spin = uniform(0, total_fitness)
        position = 0

        # Find individual in the position of the spin
        for individual in population:
            position += individual.fitness
            if position > spin:
                return individual

    elif population.optim == "min":
        raise NotImplementedError

    else:
        raise Exception("No optimization specified (min or max).")


# --------------- TOURNAMENT IMPLEMENTATION ----------------
def tournament(optim, sorted_snakes, size=10):
    """
    Tournament selection implementation.
    Args: optim (string): if it is a maximization or minimization problem
          sorted_snakes (list): list with all the snakes in a descending way
          size (int): Size of the tournament.
    Returns: Individual: the best individual in the tournament.
    """

    # Select individuals based on tournament size
    tournament_list = [choice(sorted_snakes) for i in range(size)]
    # Check if the problem is max or min
    if optim == 'max':
        return max(tournament_list, key=attrgetter("fitness"))
    elif optim == 'min':
        return min(tournament_list, key=attrgetter("fitness"))
    else:
        raise Exception("No optimization specified (min or max).")


# --------------- TOURNAMENT 2.0 IMPLEMENTATION ----------------
def tournament_new(optim, sorted_snakes, size=5):
    """
    Variation of the tournament selection implementation, where we extract the size of the tournament from a list with
        just the best snakes from the previous generation (if population size = 100, choose the snakes to go to the
        tournament from the best 25 (25%)

    Args: optim (string): if it is a maximization or minimization problem
          sorted_snakes (list): list with all the snakes in a descending way
          size (int): Size of the tournament.

    Returns: Individual: the best individual in the tournament.
    """

    # Select individuals based on tournament size
    tournament_list = sample(sorted_snakes[:25], size)  # with replacement
    # tournament = sample(population.snakes, size)

    # Check if the problem is max or min
    if optim == 'max':
        return max(tournament_list, key=attrgetter("fitness"))
    elif optim == 'min':
        return min(tournament_list, key=attrgetter("fitness"))
    else:
        raise Exception("No optimization specified (min or max).")

