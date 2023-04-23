from collections.abc import Sequence
from Bandits.src.Arm import Arm
import numpy as np


class Agent:
    """
    Abstract. Classe pour laquelle tous les agents vont hériter.
    """

    def __init__(self, nb_arms: int):
        self.estimated_q_values = np.zeros(nb_arms)
        self.action_taken = np.zeros(nb_arms)
        self.rewards = list()

    def get_best_choice(self):
        pass

    def agent_step(self, arms: Sequence[Arm]):
        pass

    def n_steps(self, arms: Sequence[Arm], n: int = 1000):
        pass


class GreedyAgent(Agent):
    """Classe relative au premier type d'agent implémenté, l'agent "Greedy" qui ne choisit que l'exploitation."""

    def __init__(self, nb_arms: int):
        """Initialise l'agent.

        Args:
            nb_arms (int): Le nombre de bras du bandit.
        """
        super(GreedyAgent, self).__init__(nb_arms)

    def get_best_choice(self) -> int:
        """Choisi le meilleur choix possible pour l'agent en se basant sur les q_values estimées.

        Returns:
            int: Le bras à choisir.
        """
        best_choices = np.where(
            self.estimated_q_values == np.max(self.estimated_q_values)
        )[
            0
        ]  # Get the best choices ie, the choices with the higher associated expected reward
        chosen_arm = np.random.choice(
            best_choices
        )  # Pick one arm randomly between the bests
        return chosen_arm

    def agent_step(self, arms: Sequence[Arm]):
        """Réalise une action et atualise les q_values en fonction de l'action choisie.

        Args:
            arms (Sequence[Arm]): La séquence des bras du bandit.
        """
        chosen_arm = self.get_best_choice()
        self.action_taken[chosen_arm] += 1
        # update beliefs
        reward = arms[chosen_arm].generate_rewards()
        self.estimated_q_values[chosen_arm] += (
            reward - self.estimated_q_values[chosen_arm]
        ) / self.action_taken[chosen_arm]
        self.rewards.append(reward)

    def n_steps(self, arms: Sequence[Arm], n: int = 1000):
        """Réalise n action et actualise les q_values à chaque action.

        Args:
            arms (Sequence[Arm]): La séquence des bras du bandit.
            n (int, optional): Le nombre d'actions à réaliser. Defaults to 1000.
        """
        for _ in range(n):
            self.agent_step(arms)

class EpsilonGreedyAgent(Agent):
    """Classe relative au second type d'agent implémenté, l'agent "epsilon-greedy" qui choisit l'exploration avec une probabilité epsilon, ou l'exploration avec une probabilité 1-epsilon.
    """
    def __init__(self, nb_arms: int, epsilon: float = .1) -> None:
        """Initialise l'agent.

        Args:
            nb_arms (int): Le nombre de bras du bandit.
            epsilon (float, optional): La probabilité d'explorer à chaque action. Defaults to .1.
        """
        super(EpsilonGreedyAgent, self).__init__(nb_arms)
        self.epsilon = epsilon
        
    def get_best_choice(self) -> int:
        """Choisi le meilleur choix possible pour l'agent en se basant sur les q_values estimées ou choisit un bras aléatoire avec une probabilité epsilon.

        Returns:
            int: Le bras à choisir.
        """
        if np.random.uniform(low = 0, high = 1) < self.epsilon:
            chosen_arm = np.random.randint(len(self.estimated_q_values))
        else:
            best_choices = np.where(self.estimated_q_values == np.max(self.estimated_q_values))[0] # Get the best choices ie, the choices with the higher associated expected reward
            chosen_arm = np.random.choice(best_choices) # Pick one arm randomly between the bests
        return chosen_arm
    
    def agent_step(self, arms: Sequence[Arm]):
        """Réalise une action et atualise les q_values en fonction de l'action choisie.

        Args:
            arms (Sequence[Arm]): La séquence des bras du bandit.
        """
        chosen_arm = self.get_best_choice()
        self.action_taken[chosen_arm]+=1
        # update beliefs
        reward = arms[chosen_arm].generate_rewards()
        self.estimated_q_values[chosen_arm] += (reward - self.estimated_q_values[chosen_arm]) / self.action_taken[chosen_arm]
        self.rewards.append(reward)
        
    def n_steps(self, arms: Sequence[Arm], n: int = 1000):
        """Réalise n action et actualise les q_values à chaque action.

        Args:
            arms (Sequence[Arm]): La séquence des bras du bandit.
            n (int, optional): Le nombre d'actions à réaliser. Defaults to 1000.
        """
        for _ in range(n):
            self.agent_step(arms)