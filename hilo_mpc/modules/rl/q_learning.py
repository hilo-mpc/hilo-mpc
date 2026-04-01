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

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .base import RLBase
from .policy import EpsilonGreedyPolicy


class QLearningAgent(RLBase):
    """
    Tabular Q-learning agent for systems with discretized state-action spaces.

    The continuous state space is discretized into bins and a look-up table
    (Q-table) is maintained. Actions must be drawn from a finite set.

    **Interface** (mirrors MPC)::

        agent = QLearningAgent(model)
        agent.set_action_space([-1.0, 0.0, 1.0])       # discrete actions
        agent.set_state_space([(-5, 5), (-5, 5)], 20)  # state bounds + bins
        agent.set_reward_function(lambda x, u, xn: -float(x @ x))
        agent.learning_rate = 0.1
        agent.discount_factor = 0.99
        agent.setup()

        rewards = agent.train(x0, n_steps=200, n_episodes=500)
        u_opt = agent.optimize(x0)

    :param model: Dynamical system model used as the environment.
    :param id: Optional unique identifier.
    :type id: str, optional
    :param name: Optional human-readable name.
    :type name: str, optional
    :param plot_backend: Plotting backend (``'matplotlib'``, ``'bokeh'``, or ``None``).
    :type plot_backend: str, optional
    """

    def __init__(self, model, id: Optional[str] = None,
                 name: Optional[str] = None,
                 plot_backend: Optional[str] = None) -> None:
        """Constructor method"""
        super().__init__(model, id=id, name=name, plot_backend=plot_backend)
        self._state_bounds: Optional[List[Tuple[float, float]]] = None
        self._n_bins: Optional[List[int]] = None
        self._state_bins: Optional[List[np.ndarray]] = None
        self._q_table: Optional[np.ndarray] = None

    def _update_type(self) -> None:
        self._type = 'QLearning'

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_state_space(self, bounds: Sequence[Tuple[float, float]],
                        n_bins: Union[int, Sequence[int]]) -> None:
        """
        Define the discretized state space.

        :param bounds: List of ``(low, high)`` tuples, one per state dimension.
        :type bounds: sequence of (float, float)
        :param n_bins: Number of discretization bins per dimension.
                       A single integer applies the same count to all dimensions.
        :type n_bins: int or sequence of int
        """
        self._state_bounds = list(bounds)
        n_dims = len(self._state_bounds)
        if isinstance(n_bins, (int, np.integer)):
            self._n_bins = [int(n_bins)] * n_dims
        else:
            if len(n_bins) != n_dims:
                raise ValueError(
                    f"n_bins length ({len(n_bins)}) must match "
                    f"number of state dimensions ({n_dims})"
                )
            self._n_bins = [int(b) for b in n_bins]

        self._state_bins = [
            np.linspace(lo, hi, nb + 1)[1:-1]
            for (lo, hi), nb in zip(self._state_bounds, self._n_bins)
        ]

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, options: Optional[dict] = None) -> None:
        """
        Initialise the Q-table and validate configuration.

        Must be called after :meth:`set_action_space`,
        :meth:`set_state_space`, and :meth:`set_reward_function`.

        :param options: Reserved for future use.
        :type options: dict, optional
        :raises ValueError: If required configuration is missing.
        """
        if self._action_space is None:
            raise ValueError("Call set_action_space() before setup()")
        if self._state_bins is None:
            raise ValueError("Call set_state_space() before setup()")
        if self._reward_function is None:
            raise ValueError("Call set_reward_function() before setup()")

        shape = tuple(self._n_bins) + (len(self._action_space),)
        self._q_table = np.zeros(shape)

        if self._policy is None:
            self._policy = EpsilonGreedyPolicy(epsilon=0.1)

        self._setup_done = True

    # ------------------------------------------------------------------
    # Core RL methods
    # ------------------------------------------------------------------

    def optimize(self, x0) -> np.ndarray:
        """
        Return the greedy action for state *x0*.

        :param x0: Current system state.
        :return: Optimal action shaped ``(n_u, 1)``.
        :rtype: np.ndarray
        :raises RuntimeError: If :meth:`setup` has not been called.
        """
        if not self._setup_done:
            raise RuntimeError("Call setup() before optimize()")
        x = np.array(x0, dtype=np.float64).flatten()
        state_idx = self._discretize_state(x)
        action_idx = int(np.argmax(self._q_table[state_idx]))
        return self._action_space[action_idx].reshape(-1, 1)

    def update(self, state: np.ndarray, action: np.ndarray,
               reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Apply the Q-learning temporal-difference update.

        Update rule::

            Q(s, a) += lr * (r + γ * max_{a'} Q(s', a') - Q(s, a))

        :param state: State before the action.
        :param action: Action that was applied.
        :param reward: Scalar reward.
        :param next_state: Resulting state.
        :param done: Whether the episode terminated.
        """
        state_idx = self._discretize_state(np.array(state, dtype=np.float64).flatten())
        next_idx = self._discretize_state(np.array(next_state, dtype=np.float64).flatten())
        action_idx = self._action_to_index(np.array(action, dtype=np.float64).flatten())

        current_q = self._q_table[state_idx + (action_idx,)]
        max_next_q = 0.0 if done else float(np.max(self._q_table[next_idx]))
        td_error = reward + self._discount_factor * max_next_q - current_q
        self._q_table[state_idx + (action_idx,)] += self._learning_rate * td_error

        self._policy.update()

    def _select_action_train(self, x: np.ndarray) -> np.ndarray:
        """Choose an action using the exploration policy during training."""
        state_idx = self._discretize_state(x)
        q_values = self._q_table[state_idx]
        action_idx = self._policy.select_action(q_values)
        return self._action_space[action_idx].reshape(-1, 1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """
        Save the Q-table to a NumPy ``.npy`` file.

        :param filepath: Destination file path (without extension is fine;
                         NumPy adds ``.npy`` automatically).
        :type filepath: str
        """
        if self._q_table is None:
            raise RuntimeError("Q-table is empty — call setup() first")
        np.save(filepath, self._q_table)

    def load(self, filepath: str) -> None:
        """
        Load a previously saved Q-table.

        :param filepath: Path to the ``.npy`` file.
        :type filepath: str
        """
        self._q_table = np.load(filepath)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def q_table(self) -> Optional[np.ndarray]:
        """The Q-table array (read-only view)."""
        return self._q_table

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discretize_state(self, x: np.ndarray) -> Tuple:
        """Map a continuous state vector to Q-table indices."""
        return tuple(
            int(np.clip(np.digitize(xi, bins), 0, nb - 1))
            for xi, bins, nb in zip(x, self._state_bins, self._n_bins)
        )

    def _action_to_index(self, action_flat: np.ndarray) -> int:
        """Return the index of *action_flat* in the action space."""
        dists = [
            np.linalg.norm(self._action_space[i].flatten() - action_flat)
            for i in range(len(self._action_space))
        ]
        return int(np.argmin(dists))
