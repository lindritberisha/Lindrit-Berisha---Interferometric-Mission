import numpy as np


def evaluate_individual(x: np.ndarray, udp) -> float:
    """
    Evaluates an individual using the UDP fitness function.

    udp.fitness() returns a list containing one fitness value.

    Args:
        x: NumPy array containing the individual's 6N variables.
        udp: OrbitalGolombArraySimplified evaluator.

    Returns:
        Fitness value as a float.
    """
    f = udp.fitness(x)
    return float(f[0])
