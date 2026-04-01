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

from typing import Tuple

import numpy as np


class ReplayBuffer:
    """
    Circular experience replay buffer for RL agents.

    Stores ``(state, action_index, reward, next_state, done)`` tuples and
    supports random sampling of minibatches for off-policy training.

    :param capacity: Maximum number of transitions to store.
    :type capacity: int
    """

    def __init__(self, capacity: int) -> None:
        """Constructor method"""
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._capacity = int(capacity)
        self._buffer = []
        self._position = 0

    def push(self, state: np.ndarray, action_idx: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """
        Store a transition.

        :param state: Current state vector.
        :param action_idx: Index of the action taken.
        :param reward: Scalar reward received.
        :param next_state: Next state vector after taking the action.
        :param done: Whether the episode ended after this transition.
        """
        if len(self._buffer) < self._capacity:
            self._buffer.append(None)
        self._buffer[self._position] = (
            np.array(state, dtype=np.float64),
            int(action_idx),
            float(reward),
            np.array(next_state, dtype=np.float64),
            bool(done),
        )
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray]:
        """
        Sample a random minibatch of transitions.

        :param batch_size: Number of transitions to sample.
        :type batch_size: int
        :return: Tuple of ``(states, action_indices, rewards, next_states, dones)``.
        :raises ValueError: If fewer transitions than batch_size are stored.
        """
        if len(self._buffer) < batch_size:
            raise ValueError(
                f"Buffer contains {len(self._buffer)} transitions but "
                f"batch_size={batch_size} was requested."
            )
        indices = np.random.choice(len(self._buffer), batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float64),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float64),
            np.array(next_states, dtype=np.float64),
            np.array(dones, dtype=np.float64),
        )

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return len(self._buffer)
