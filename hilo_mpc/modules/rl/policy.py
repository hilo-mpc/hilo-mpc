#
#   This file is part of HILO-MPC
#
#   HILO-MPC is a toolbox for easy, flexible and fast development of machine-learning-supported
#   optimal control and estimation problems
#
#   Copyright (c) 2021 Johannes Pohlodek, Bruno Morabito, Rolf Findeisen
#                      All rights reserved
#
#   HILO-MPC is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   HILO-MPC is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public License
#   along with HILO-MPC. If not, see <http://www.gnu.org/licenses/>.
#

from abc import ABCMeta, abstractmethod

import numpy as np


class Policy(metaclass=ABCMeta):
    """Abstract base class for RL action-selection policies."""

    @abstractmethod
    def select_action(self, q_values: np.ndarray) -> int:
        """
        Select an action index given Q-values for each action.

        :param q_values: Array of Q-values, one per action.
        :type q_values: np.ndarray
        :return: Index of the selected action.
        :rtype: int
        """
        pass

    def update(self) -> None:
        """
        Update internal policy state (e.g., decay exploration parameter).
        Called once per environment step. Override if needed.
        """
        pass


class EpsilonGreedyPolicy(Policy):
    """
    Epsilon-greedy exploration policy.

    With probability epsilon a random action is chosen; otherwise the
    action with the highest Q-value is selected (greedy).

    :param epsilon: Initial exploration probability.
    :type epsilon: float
    :param epsilon_min: Lower bound on epsilon after decay.
    :type epsilon_min: float
    :param epsilon_decay: Multiplicative decay factor applied each step.
    :type epsilon_decay: float
    """

    def __init__(self, epsilon: float = 0.1, epsilon_min: float = 0.01,
                 epsilon_decay: float = 1.0) -> None:
        """Constructor method"""
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

    def select_action(self, q_values: np.ndarray) -> int:
        """Select action with epsilon-greedy strategy."""
        if np.random.random() < self.epsilon:
            return int(np.random.randint(len(q_values)))
        return int(np.argmax(q_values))

    def update(self) -> None:
        """Decay epsilon towards epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class GreedyPolicy(Policy):
    """
    Pure greedy policy — always selects the action with the highest Q-value.
    Use for deployment after training.
    """

    def select_action(self, q_values: np.ndarray) -> int:
        """Select action with highest Q-value."""
        return int(np.argmax(q_values))


class SoftmaxPolicy(Policy):
    """
    Softmax (Boltzmann) exploration policy.

    Action probabilities are proportional to ``exp(Q(s,a) / temperature)``.
    Lower temperature approaches greedy; higher temperature approaches uniform.

    :param temperature: Temperature parameter controlling exploration.
    :type temperature: float
    """

    def __init__(self, temperature: float = 1.0) -> None:
        """Constructor method"""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def select_action(self, q_values: np.ndarray) -> int:
        """Sample action from softmax distribution over Q-values."""
        scaled = q_values / self.temperature
        # Subtract max for numerical stability
        shifted = scaled - np.max(scaled)
        probs = np.exp(shifted)
        probs /= probs.sum()
        return int(np.random.choice(len(q_values), p=probs))
