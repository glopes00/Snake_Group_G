from random import sample, randint


# ---------------- SWAP MUTATION ----------------
def swap_mutation(individual):
    """
    Swap mutation for a GA individual
    Args:
        individual (Individual): A GA individual from classes.py
    Returns:
        Individual: Mutated Individual
    """
    # Get two mutation points
    mut_points = sample(range(len(individual)), 2)
    # Swap them
    individual[mut_points[0]], individual[mut_points[1]] = individual[mut_points[1]], individual[mut_points[0]]

    return individual


# ---------------- INVERSION MUTATION ----------------
def inversion_mutation(individual):
    """
    Inversion mutation for a GA individual
    Args:
        individual (Individual): A GA individual from classes.py
    Returns:
        Individual: Mutated Individual
    """
    # Position of the start and end of substring
    mut_points = sample(range(len(individual)), 2)

    # This method assumes that the second point is after (on the right of) the first one --> sort the list
    mut_points.sort()

    # Invert for the mutation --> goes from the first to the last element (goes backwards)
    individual[mut_points[0]:mut_points[1]] = individual[mut_points[0]:mut_points[1]][::-1]

    return individual
