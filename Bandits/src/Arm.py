import numpy as np
from collections.abc import Sequence


class Arm:
    """Classe relative aux bras du bandit."""

    def __init__(self, mean: float):
        """Génère un des bras du bandit. Un bras génèrera des nombres issues d'une distribution gaussiènne de moyenne mean et de variance 1.

        Args:
            mean (float): La moyenne de la distribution associée au bras.
        """
        self.mean = mean

    def generate_rewards(self, n: int = None):
        """Génère n gains basés sur la distribution associée au bras.

        Args:
            n (int): Le nombre de gain à générer.

        Returns:
            (float ou np.array): La liste des gain générés ou le gain généré seul.
        """
        return np.random.normal(loc=self.mean, scale=1, size=n)
