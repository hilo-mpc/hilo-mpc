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

import casadi as ca
import numpy as np

from .base import RLBase
from .policy import EpsilonGreedyPolicy
from .replay_buffer import ReplayBuffer


class _CasADiNN:
    """
    Feedforward neural network built entirely with CasADi symbolic expressions.

    The network architecture is fully-connected with ReLU activations on all
    hidden layers and a linear output layer.  All weights are stored as a
    single flat :class:`casadi.DM` parameter vector so that CasADi's
    automatic differentiation can be applied directly to the MSE loss.

    Parameters are initialised with Xavier/Glorot scaling.  The forward pass
    and gradient are compiled once as :class:`casadi.Function` objects and
    then re-used efficiently for every prediction and update.

    :param n_inputs: Number of input features.
    :param hidden_layers: Sizes of the hidden layers.
    :param n_outputs: Number of output units (one per discrete action).
    """

    def __init__(self, n_inputs: int, hidden_layers: Tuple[int, ...],
                 n_outputs: int) -> None:
        """Constructor method"""
        sizes = [n_inputs] + list(hidden_layers) + [n_outputs]
        self._sizes = sizes
        self._n_layers = len(sizes) - 1

        # Compute flat-parameter slice indices for each layer: (W_start, W_end, b_start, b_end)
        self._param_slices: List[Tuple[int, int, int, int]] = []
        total_params = 0
        for i in range(self._n_layers):
            n_in, n_out = sizes[i], sizes[i + 1]
            w_start = total_params
            total_params += n_in * n_out
            b_start = total_params
            total_params += n_out
            self._param_slices.append((w_start, w_start + n_in * n_out,
                                       b_start, b_start + n_out))
        self._total_params = total_params

        # ---- Build symbolic forward pass --------------------------------
        x_sym = ca.SX.sym('x', n_inputs)
        params_sym = ca.SX.sym('params', total_params)

        h = x_sym
        for i, (ws, we, bs, be) in enumerate(self._param_slices):
            n_in, n_out = sizes[i], sizes[i + 1]
            W_flat = params_sym[ws:we]
            b = params_sym[bs:be]
            # Reshape column-major flat vector to (n_in, n_out) weight matrix
            W = ca.reshape(W_flat, n_in, n_out)
            z = W.T @ h + b
            # ReLU on hidden layers, linear on output
            h = ca.fmax(0, z) if i < self._n_layers - 1 else z

        q_sym = h  # shape (n_outputs,)
        self._f_forward = ca.Function(
            'q_net', [x_sym, params_sym], [q_sym],
            ['x', 'params'], ['q']
        )

        # ---- Build gradient function via automatic differentiation -------
        y_target_sym = ca.SX.sym('y_target', n_outputs)
        loss = ca.sumsqr(q_sym - y_target_sym) / n_outputs  # MSE
        grad = ca.gradient(loss, params_sym)
        self._f_grad = ca.Function(
            'q_grad', [x_sym, params_sym, y_target_sym], [grad],
            ['x', 'params', 'y_target'], ['grad']
        )

        # ---- Xavier / Glorot parameter initialisation -------------------
        rng = np.random.default_rng()
        params_init = np.zeros(total_params)
        for i, (ws, we, bs, be) in enumerate(self._param_slices):
            n_in = sizes[i]
            scale = np.sqrt(2.0 / n_in)
            params_init[ws:we] = rng.standard_normal(we - ws) * scale
            # biases initialised to zero
        self._params = ca.DM(params_init)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for a single sample, returns a NumPy array."""
        x_dm = ca.DM(np.array(x, dtype=np.float64).flatten())
        result = self._f_forward(x_dm, self._params)
        return np.array(result).flatten()

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass for a batch of samples.

        :param X: Input matrix of shape ``(batch_size, n_inputs)``.
        :return: Q-value matrix of shape ``(batch_size, n_outputs)``.
        """
        batch_size = X.shape[0]
        # map() expects inputs as columns: (n_inputs, batch_size)
        X_dm = ca.DM(np.array(X, dtype=np.float64).T)
        params_rep = ca.repmat(self._params, 1, batch_size)
        f_map = self._f_forward.map(batch_size)
        result = f_map(X_dm, params_rep)  # (n_outputs, batch_size)
        return np.array(result).T         # (batch_size, n_outputs)

    # ------------------------------------------------------------------
    # Gradient update
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray, y_target: np.ndarray, lr: float) -> None:
        """
        One gradient descent step minimising MSE loss.

        Gradients are computed via CasADi automatic differentiation applied
        to the symbolic forward pass, then averaged over the batch.

        :param X: Input batch of shape ``(batch_size, n_inputs)``.
        :param y_target: Target Q-values of shape ``(batch_size, n_outputs)``.
        :param lr: Learning rate.
        """
        batch_size = X.shape[0]
        X_dm = ca.DM(np.array(X, dtype=np.float64).T)          # (n_inputs, batch_size)
        y_dm = ca.DM(np.array(y_target, dtype=np.float64).T)   # (n_outputs, batch_size)
        params_rep = ca.repmat(self._params, 1, batch_size)

        f_grad_map = self._f_grad.map(batch_size)
        grad_batch = f_grad_map(X_dm, params_rep, y_dm)  # (total_params, batch_size)
        grad_mean = ca.sum2(grad_batch) / batch_size      # (total_params, 1)

        self._params = self._params - lr * grad_mean

    # ------------------------------------------------------------------
    # Weight serialisation (used for target network sync)
    # ------------------------------------------------------------------

    def get_weights(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return a list of ``(W_array, b_array)`` pairs (deep copy)."""
        params_np = np.array(self._params).flatten()
        result = []
        for i, (ws, we, bs, be) in enumerate(self._param_slices):
            n_in, n_out = self._sizes[i], self._sizes[i + 1]
            W = params_np[ws:we].reshape(n_in, n_out).copy()
            b = params_np[bs:be].copy()
            result.append((W, b))
        return result

    def set_weights(self, weights: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Overwrite parameters from a list of ``(W, b)`` pairs."""
        params_np = np.array(self._params).flatten().copy()
        for (W, b), (ws, we, bs, be) in zip(weights, self._param_slices):
            params_np[ws:we] = np.array(W).flatten()
            params_np[bs:be] = np.array(b).flatten()
        self._params = ca.DM(params_np)


class DQNAgent(RLBase):
    """
    Deep Q-Network (DQN) agent with experience replay and a target network.

    The Q-network is implemented using CasADi symbolic expressions so that
    all gradient computations are performed via automatic differentiation
    rather than manual backpropagation.  No external ML framework (PyTorch,
    TensorFlow) is required.

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
        self._network: Optional[_CasADiNN] = None
        self._target_network: Optional[_CasADiNN] = None
        self._replay_buffer: Optional[ReplayBuffer] = None
        self._step_count: int = 0

    def _update_type(self) -> None:
        self._type = 'DQN'

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, options: Optional[dict] = None) -> None:
        """
        Initialise the CasADi Q-network, target network, and replay buffer.

        The symbolic forward pass and gradient functions are compiled here
        once so they can be re-used efficiently during training.

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

        self._network = _CasADiNN(n_states, self._hidden_layers, n_actions)
        self._target_network = _CasADiNN(n_states, self._hidden_layers, n_actions)
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

        The Q-values are computed by evaluating the CasADi forward pass
        function with the current parameter vector.

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
        CasADi automatic-differentiation gradient update on the Q-network.

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
        """
        Sample a minibatch and perform one CasADi gradient descent step.

        TD targets are computed with the target network (numpy output), then
        the gradient update is performed entirely within CasADi via the
        pre-compiled ``q_grad`` function.
        """
        states, actions, rewards, next_states, dones = self._replay_buffer.sample(
            self._batch_size
        )
        # Compute TD targets using the target network
        next_q = self._target_network.predict_batch(next_states)
        targets_vec = rewards + self._discount_factor * np.max(next_q, axis=1) * (1.0 - dones)

        # Get current Q-value predictions and overwrite the taken actions
        current_q = self._network.predict_batch(states)
        current_q[np.arange(self._batch_size), actions] = targets_vec

        # Gradient step via CasADi automatic differentiation
        self._network.train(states, current_q, self._learning_rate)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def network(self) -> Optional[_CasADiNN]:
        """The online CasADi Q-network."""
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
