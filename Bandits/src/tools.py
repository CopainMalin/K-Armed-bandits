import numpy as np
from os import getcwd, listdir

from Bandits.src.Arm import Arm

def create_k_armed_bandit(k: int) -> np.array:
    """Créer les k bras du bandit en reprenant la méthode des "10-armed Testbed" présentée par Sutton et Barto.
        "The true value q*(a) of each of the actions was selected according to a normal distribution with mean zero and unit
        variance, and then the actual rewards were selected according to a mean q*(a) unit variance
        normal distribution."
    Args:
        k (int): Le nombre de bras du bandit à créer.

    Returns:
        np.array, np.array: la liste des bras générés ainsi que la liste des gains espérés associés.
    """
    q_stars = np.random.normal(loc=0, scale=1, size=k)
    return np.array([Arm(mean) for mean in q_stars]), q_stars
