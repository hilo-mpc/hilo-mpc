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

from abc import abstractmethod
from typing import Callable, List, Optional, Sequence, Union

import numpy as np

from ..controller.base import Controller
from ..base import Base
from .policy import Policy


class RLBase(Controller, Base):
    """
    Abstract base class for reinforcement learning agents in HILO-MPC.

    Provides the common interface that mirrors the MPC controllers:

    - :meth:`setup` — configure and compile the agent before use
    - :meth:`optimize` — select an action for a given state (deployment)
    - :meth:`update` — incorporate one transition into the agent's learning
    - :meth:`train` — run a complete training loop using the attached model

    Sub-classes must implement :meth:`setup`, :meth:`optimize`, and
    :meth:`update`, as well as :meth:`_update_type` (inherited from
    :class:`~hilo_mpc.modules.controller.base.Controller`).

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
        super().__init__(id=id, name=name)
        if self._id is None:
            self._create_id()
        self._model = model
        self._plot_backend = plot_backend
        self._setup_done = False
        self._reward_function: Optional[Callable] = None
        self._action_space: Optional[np.ndarray] = None
        self._discount_factor: float = 0.99
        self._learning_rate: float = 0.01
        self._policy: Optional[Policy] = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def setup(self, options: Optional[dict] = None) -> None:
        """
        Configure and compile the agent.

        Must be called before :meth:`optimize` or :meth:`train`.

        :param options: Algorithm-specific configuration options.
        :type options: dict, optional
        """
        pass

    @abstractmethod
    def optimize(self, x0) -> np.ndarray:
        """
        Select the best action for the given state (exploitation).

        :param x0: Current system state.
        :return: Control action shaped ``(n_u, 1)``.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def update(self, state: np.ndarray, action: np.ndarray,
               reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Incorporate one environment transition into the agent's learning.

        :param state: State before the action.
        :param action: Action that was applied.
        :param reward: Scalar reward received.
        :param next_state: State after the action.
        :param done: Whether the episode terminated.
        """
        pass

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_reward_function(self, fn: Callable) -> None:
        """
        Set the reward function used during training.

        The function signature must be ``fn(x, u, x_next) -> float``
        where *x* and *x_next* are 1-D numpy arrays and *u* is the
        applied action.

        :param fn: Reward function.
        :type fn: callable
        """
        if not callable(fn):
            raise TypeError("reward function must be callable")
        self._reward_function = fn

    def set_action_space(self, actions: Union[List, np.ndarray]) -> None:
        """
        Define the discrete action space.

        Each element in *actions* represents one possible control input.
        For multi-dimensional inputs supply a list of arrays.

        :param actions: Sequence of actions (each action is a scalar or array).
        """
        arr = np.array(actions)
        # Ensure each action is at least 1-D so indexing is consistent
        if arr.ndim == 1:
            # Scalar actions — reshape to column vectors (n_actions, 1)
            self._action_space = arr.reshape(-1, 1)
        else:
            self._action_space = arr

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, x0, n_steps: int, n_episodes: int = 1,
              done_fn: Optional[Callable] = None) -> List[float]:
        """
        Run the closed-loop training loop.

        For each episode the model is reset to *x0*, then for each step
        an action is chosen with the current exploration policy, the model
        is stepped forward, the reward is computed, and :meth:`update` is
        called.

        :param x0: Initial state for every episode.
        :param n_steps: Maximum number of steps per episode.
        :type n_steps: int
        :param n_episodes: Number of training episodes.
        :type n_episodes: int
        :param done_fn: Optional termination function ``fn(x) -> bool``.
                        If it returns ``True`` the episode ends early.
        :type done_fn: callable, optional
        :return: List of total rewards per episode.
        :rtype: list of float
        """
        if not self._setup_done:
            raise RuntimeError("Call setup() before train()")
        if self._reward_function is None:
            raise RuntimeError("Call set_reward_function() before train()")

        episode_rewards = []
        for _ in range(n_episodes):
            self._model.set_initial_conditions(x0=list(np.array(x0).flatten()))
            x = np.array(x0, dtype=np.float64).flatten()
            total_reward = 0.0
            for _ in range(n_steps):
                u = self._select_action_train(x)
                u_list = u.flatten().tolist()
                self._model.simulate(u=u_list)
                x_next = np.array(
                    self._model.solution['x:f'], dtype=np.float64
                ).flatten()
                reward = float(self._reward_function(x, u.flatten(), x_next))
                total_reward += reward
                done = bool(done_fn(x_next)) if done_fn is not None else False
                self.update(x, u.flatten(), reward, x_next, done)
                x = x_next
                if done:
                    break
            episode_rewards.append(total_reward)
        return episode_rewards

    def _select_action_train(self, x: np.ndarray) -> np.ndarray:
        """
        Choose an action during training (with exploration).

        The default implementation uses the policy attached to the agent.
        Subclasses may override for algorithm-specific exploration.

        :param x: Current state.
        :return: Action vector shaped ``(n_u, 1)``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _select_action_train()"
        )

    # ------------------------------------------------------------------
    # SimpleControlLoop compatibility
    # ------------------------------------------------------------------

    def predict(self, x0) -> np.ndarray:
        """
        Alias for :meth:`optimize`.

        Allows the agent to be used inside
        :class:`~hilo_mpc.modules.control_loop.SimpleControlLoop` via the
        same pathway as ANN models (which expose ``predict``).
        """
        return self.optimize(x0)

    def is_setup(self) -> bool:
        """Return ``True`` if :meth:`setup` has been called successfully."""
        return self._setup_done

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def discount_factor(self) -> float:
        """Discount factor γ for future rewards (default 0.99)."""
        return self._discount_factor

    @discount_factor.setter
    def discount_factor(self, value: float) -> None:
        value = float(value)
        if not (0.0 <= value <= 1.0):
            raise ValueError("discount_factor must be in [0, 1]")
        self._discount_factor = value

    @property
    def learning_rate(self) -> float:
        """Learning rate α (default 0.01)."""
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        value = float(value)
        if value <= 0:
            raise ValueError("learning_rate must be positive")
        self._learning_rate = value

    @property
    def policy(self) -> Optional[Policy]:
        """The exploration/exploitation policy."""
        return self._policy

    @policy.setter
    def policy(self, value: Policy) -> None:
        if not isinstance(value, Policy):
            raise TypeError("policy must be an instance of Policy")
        self._policy = value

    @property
    def action_space(self) -> Optional[np.ndarray]:
        """Array of available actions shaped ``(n_actions, n_u)``."""
        return self._action_space
