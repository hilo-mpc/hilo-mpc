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

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .base import RLBase
from .policy import EpsilonGreedyPolicy
from .replay_buffer import ReplayBuffer


class _SimpleNN:
    """
    Lightweight feedforward neural network implemented in NumPy.

    Used internally by :class:`DQNAgent` to avoid a hard dependency on
    PyTorch or TensorFlow.  Architecture: fully-connected layers with ReLU
    activations on hidden layers and a linear output layer.

    :param n_inputs: Number of input features.
    :param hidden_layers: Sizes of the hidden layers.
    :param n_outputs: Number of output units.
    """

    def __init__(self, n_inputs: int, hidden_layers: Tuple[int, ...],
                 n_outputs: int) -> None:
        """Constructor method"""
        sizes = [n_inputs] + list(hidden_layers) + [n_outputs]
        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        rng = np.random.default_rng()
        for i in range(len(sizes) - 1):
            # Xavier / Glorot initialisation
            scale = np.sqrt(2.0 / sizes[i])
            self._weights.append(rng.standard_normal((sizes[i], sizes[i + 1])) * scale)
            self._biases.append(np.zeros(sizes[i + 1]))
        self._n_layers = len(self._weights)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for a single sample."""
        a = np.array(x, dtype=np.float64).flatten()
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            a = a @ w + b
            if i < self._n_layers - 1:
                a = np.maximum(0.0, a)  # ReLU
        return a

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for a batch of samples."""
        a = np.array(X, dtype=np.float64)
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            a = a @ w + b
            if i < self._n_layers - 1:
                a = np.maximum(0.0, a)
        return a

    def train(self, X: np.ndarray, y_target: np.ndarray, lr: float) -> None:
        """
        One gradient descent step minimising MSE loss via backpropagation.

        :param X: Input batch of shape ``(batch, n_inputs)``.
        :param y_target: Target Q-values of shape ``(batch, n_outputs)``.
        :param lr: Learning rate.
        """
        n = len(X)
        # ---- forward pass (store activations and pre-activations) ----
        activations = [np.array(X, dtype=np.float64)]
        pre_acts = []
        a = activations[0]
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            z = a @ w + b
            pre_acts.append(z)
            a = np.maximum(0.0, z) if i < self._n_layers - 1 else z
            activations.append(a)

        # ---- backward pass (MSE gradient: d/dout = 2*(out - target)/n) ----
        delta = 2.0 * (activations[-1] - y_target) / n
        for i in range(self._n_layers - 1, -1, -1):
            dw = activations[i].T @ delta / n
            db = delta.mean(axis=0)
            self._weights[i] -= lr * dw
            self._biases[i] -= lr * db
            if i > 0:
                delta = delta @ self._weights[i].T
                # ReLU derivative: zero out where pre-activation <= 0
                delta = delta * (pre_acts[i - 1] > 0)

    def get_weights(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return a deep copy of all weight/bias pairs."""
        return [(w.copy(), b.copy()) for w, b in zip(self._weights, self._biases)]

    def set_weights(self, weights: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Overwrite weights from a list of ``(W, b)`` pairs."""
        for i, (w, b) in enumerate(weights):
            self._weights[i] = w.copy()
            self._biases[i] = b.copy()


class DQNAgent(RLBase):
    """
    Deep Q-Network (DQN) agent with experience replay and a target network.

    Uses a lightweight NumPy-based feedforward neural network so that no
    external ML framework (PyTorch, TensorFlow) is required.

    **Interface** (mirrors MPC)::

        agent = DQNAgent(model, hidden_layers=(64, 64))
        agent.set_action_space([-1.0, 0.0, 1.0])
        agent.set_reward_function(lambda x, u, xn: -float(x @ x))
        agent.learning_rate = 1e-3
        agent.setup()

        rewards = agent.train(x0, n_steps=200, n_episodes=300)
        u_opt = agent.optimize(x0)

    :param model: Dynamical system model used as the environment.
    :param id: Optional unique identifier.
    :type id: str, optional
    :param name: Optional human-readable name.
    :type name: str, optional
    :param plot_backend: Plotting backend (``'matplotlib'``, ``'bokeh'``, or ``None``).
    :type plot_backend: str, optional
    :param hidden_layers: Sizes of the hidden layers of the Q-network.
    :type hidden_layers: tuple of int
    :param batch_size: Minibatch size for each gradient update.
    :type batch_size: int
    :param buffer_size: Capacity of the experience replay buffer.
    :type buffer_size: int
    :param target_update_freq: Number of gradient steps between target network syncs.
    :type target_update_freq: int
    """

    def __init__(self, model, id: Optional[str] = None,
                 name: Optional[str] = None,
                 plot_backend: Optional[str] = None,
                 hidden_layers: Tuple[int, ...] = (64, 64),
                 batch_size: int = 32,
                 buffer_size: int = 10000,
                 target_update_freq: int = 100) -> None:
        """Constructor method"""
        super().__init__(model, id=id, name=name, plot_backend=plot_backend)
        self._hidden_layers = tuple(hidden_layers)
        self._batch_size = int(batch_size)
        self._buffer_size = int(buffer_size)
        self._target_update_freq = int(target_update_freq)
        self._network: Optional[_SimpleNN] = None
        self._target_network: Optional[_SimpleNN] = None
        self._replay_buffer: Optional[ReplayBuffer] = None
        self._step_count: int = 0

    def _update_type(self) -> None:
        self._type = 'DQN'

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, options: Optional[dict] = None) -> None:
        """
        Initialise the Q-network, target network, and replay buffer.

        Must be called after :meth:`set_action_space` and
        :meth:`set_reward_function`.

        :param options: Reserved for future use.
        :type options: dict, optional
        :raises ValueError: If required configuration is missing.
        """
        if self._action_space is None:
            raise ValueError("Call set_action_space() before setup()")
        if self._reward_function is None:
            raise ValueError("Call set_reward_function() before setup()")

        n_states = self._model.n_x
        n_actions = len(self._action_space)

        self._network = _SimpleNN(n_states, self._hidden_layers, n_actions)
        self._target_network = _SimpleNN(n_states, self._hidden_layers, n_actions)
        self._target_network.set_weights(self._network.get_weights())

        self._replay_buffer = ReplayBuffer(self._buffer_size)

        if self._policy is None:
            self._policy = EpsilonGreedyPolicy(
                epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995
            )

        self._step_count = 0
        self._setup_done = True

    # ------------------------------------------------------------------
    # Core RL methods
    # ------------------------------------------------------------------

    def optimize(self, x0) -> np.ndarray:
        """
        Return the greedy action (highest Q-value) for state *x0*.

        :param x0: Current system state.
        :return: Optimal action shaped ``(n_u, 1)``.
        :rtype: np.ndarray
        :raises RuntimeError: If :meth:`setup` has not been called.
        """
        if not self._setup_done:
            raise RuntimeError("Call setup() before optimize()")
        x = np.array(x0, dtype=np.float64).flatten()
        q_values = self._network.predict(x)
        action_idx = int(np.argmax(q_values))
        return self._action_space[action_idx].reshape(-1, 1)

    def update(self, state: np.ndarray, action: np.ndarray,
               reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Store a transition and, if the buffer is large enough, run a
        gradient update on the Q-network.

        :param state: State before the action.
        :param action: Action that was applied.
        :param reward: Scalar reward.
        :param next_state: Resulting state.
        :param done: Whether the episode terminated.
        """
        action_idx = self._action_to_index(np.array(action, dtype=np.float64).flatten())
        self._replay_buffer.push(
            np.array(state, dtype=np.float64).flatten(),
            action_idx,
            float(reward),
            np.array(next_state, dtype=np.float64).flatten(),
            done,
        )
        self._step_count += 1

        if len(self._replay_buffer) >= self._batch_size:
            self._train_step()

        if self._step_count % self._target_update_freq == 0:
            self._target_network.set_weights(self._network.get_weights())

        self._policy.update()

    def _select_action_train(self, x: np.ndarray) -> np.ndarray:
        """Choose an action using the exploration policy during training."""
        q_values = self._network.predict(x)
        action_idx = self._policy.select_action(q_values)
        return self._action_space[action_idx].reshape(-1, 1)

    def _train_step(self) -> None:
        """Sample a minibatch and perform one gradient descent step."""
        states, actions, rewards, next_states, dones = self._replay_buffer.sample(
            self._batch_size
        )
        # Compute TD targets using the target network
        next_q = self._target_network.predict_batch(next_states)
        targets_vec = rewards + self._discount_factor * np.max(next_q, axis=1) * (1.0 - dones)

        # Get current Q-value predictions and overwrite the taken actions
        current_q = self._network.predict_batch(states)
        current_q[np.arange(self._batch_size), actions] = targets_vec

        self._network.train(states, current_q, self._learning_rate)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def network(self) -> Optional[_SimpleNN]:
        """The online Q-network."""
        return self._network

    @property
    def replay_buffer(self) -> Optional[ReplayBuffer]:
        """The experience replay buffer."""
        return self._replay_buffer

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _action_to_index(self, action_flat: np.ndarray) -> int:
        """Return the index of *action_flat* in the action space."""
        dists = [
            np.linalg.norm(self._action_space[i].flatten() - action_flat)
            for i in range(len(self._action_space))
        ]
        return int(np.argmin(dists))
