from random import uniform, sample, randint


# ---------------- SINGLE POINT CROSSOVER ----------------
def single_point_co(p1, p2):
    """
    Implementation of single point crossover.
    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.
    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    co_point = randint(1, len(p1) - 2)

    offspring1 = p1[:co_point] + p2[co_point:]
    offspring2 = p2[:co_point] + p1[co_point:]

    return offspring1, offspring2


# ---------------- CYCLE CROSSOVER ----------------
def cycle_co(p1, p2):
    """
    Implementation of cycle crossover.
    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.
    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """

    # Offspring placeholders - None values make it easy to debug for errors
    offspring1 = [None] * len(p1)
    offspring2 = [None] * len(p2)
    # While there are still None values in offspring, get the first index of
    # None and start a "cycle" according to the cycle crossover method
    while None in offspring1:
        index = offspring1.index(None)
        val1 = p1[index]
        val2 = p2[index]

        while val1 != val2:
            offspring1[index] = p1[index]
            offspring2[index] = p2[index]
            val2 = p2[index]
            index = p1.index(val2)

        for element in offspring1:
            if element is None:
                index = offspring1.index(None)
                if offspring1[index] is None:
                    offspring1[index] = p2[index]
                    offspring2[index] = p1[index]

    return offspring1, offspring2


# ---------------- PARTIALLY MATCHED/MAPPED CROSSOVER ----------------
def pmx_co(p1, p2):
    """
    Implementation of partially matched/mapped crossover.
    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.
    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """

    co_points = sample(range(len(p1)), 2)
    co_points.sort()

    # dictionary creation using the segment elements from both parents
    # the dictionary will be working two ways
    keys = p1[co_points[0]:co_points[1]] + p2[co_points[0]:co_points[1]]
    values = p2[co_points[0]:co_points[1]] + p1[co_points[0]:co_points[1]]

    # segment dictionary
    segment = {keys[i]: values[i] for i in range(len(keys))}

    # empty offsprings
    o1 = [None] * len(p1)
    o2 = [None] * len(p2)

    # where pmx happens
    def pmx(o, p):
        for i, element in enumerate(p):
            # if element not in the segment, copy
            if element not in segment:
                o[i] = p[i]
            # if element in the segment, take the value of the key from
            # segment/dictionary
            else:
                o[i] = segment.get(element)  # get(key) --> returns the value
        return o

    # repeat the procedure for each offspring
    o1 = pmx(o1, p1)
    o2 = pmx(o2, p2)

    return o1, o2


# ---------------- ARITHMETIC CROSSOVER ----------------
def arithmetic_co(p1, p2):
    """
    Implementation of arithmetic crossover.
    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.
    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """

    # Offspring placeholders - None values make it easy to debug for errors
    offspring1 = [None] * len(p1)
    offspring2 = [None] * len(p1)

    # Set a value for alpha between 0 and 1
    alpha = uniform(0, 1)

    # Take weighted sum of two parents, invert alpha for second offspring
    for i in range(len(p1)):
        offspring1[i] = round(p1[i] * alpha + (1 - alpha) * p2[i], 2)
        offspring2[i] = round(p2[i] * alpha + (1 - alpha) * p1[i], 2)

    return offspring1, offspring2


# ---------------- ARITHMETIC & UNIFORM CROSSOVER ----------------
def crossover_mix(p1, p2):
    """
    Implementation of a combination between arithmetic and uniform crossover.

    Arithmetic: one of the offspring is calculated with arithmetic crossover, just like above
    Uniform: Each gene (bit) is selected randomly from one of the corresponding genes of the parent chromosomes.
             Use tossing of a coin as an example technique.
    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.
    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """

    # Offspring placeholders - None values make it easy to debug for errors
    offspring1 = [None] * len(p1)
    offspring2 = [None] * len(p1)

    # Set a value for alpha between 0 and 1
    alpha = uniform(0, 1)

    # Take weighted sum of two parents, invert alpha for second offspring
    for i in range(len(p1)):
        offspring1[i] = round(p1[i] * alpha + (1 - alpha) * p2[i], 2)

        if alpha < 0.5:
            offspring2[i] = p1[i]
        else:
            offspring2[i] = p2[i]

    return offspring1, offspring2
