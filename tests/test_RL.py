"""
Tests for the reinforcement learning module.

Covers:
- QLearningAgent: setup validation, Q-table initialisation, optimize, training
- DQNAgent: setup validation, network init, optimize, training
- Policy classes: action selection and epsilon decay
- ReplayBuffer: push, capacity, sampling
"""

import os
import tempfile
import unittest
from unittest import TestCase

import numpy as np

from hilo_mpc import Model, QLearningAgent, DQNAgent
from hilo_mpc import EpsilonGreedyPolicy, GreedyPolicy, SoftmaxPolicy
from hilo_mpc import ReplayBuffer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_integrator(dt: float = 0.1):
    """1D double integrator: dx = v, dv = u."""
    import casadi as ca
    model = Model(plot_backend=None)
    x = model.set_dynamical_states(['x', 'v'])
    u = model.set_inputs('F')
    model.set_equations(ode=[x[1], u])
    model.setup(dt=dt)
    return model


def _reward_fn(x, u, x_next):
    """Quadratic cost penalising state deviation."""
    return -float(x_next @ x_next)


def _make_ql_agent(model):
    """Return a fully configured (but not yet setup) QLearningAgent."""
    agent = QLearningAgent(model)
    agent.set_action_space([-1.0, 0.0, 1.0])
    agent.set_state_space([(-5.0, 5.0), (-5.0, 5.0)], n_bins=10)
    agent.set_reward_function(_reward_fn)
    return agent


def _make_dqn_agent(model):
    """Return a fully configured (but not yet setup) DQNAgent."""
    agent = DQNAgent(model, hidden_layers=(16, 16), batch_size=8,
                     buffer_size=100, target_update_freq=20)
    agent.set_action_space([-1.0, 0.0, 1.0])
    agent.set_reward_function(_reward_fn)
    return agent


# ---------------------------------------------------------------------------
# QLearningAgent tests
# ---------------------------------------------------------------------------

class TestQLearningAgent(TestCase):

    def setUp(self) -> None:
        self.model = _make_integrator()

    def test_type(self):
        """Agent type string is 'QLearning'."""
        agent = QLearningAgent(self.model)
        self.assertEqual(agent.type, 'QLearning')

    def test_setup_requires_action_space(self):
        """setup() raises ValueError when action_space is not set."""
        agent = QLearningAgent(self.model)
        agent.set_state_space([(-5.0, 5.0), (-5.0, 5.0)], n_bins=5)
        agent.set_reward_function(_reward_fn)
        with self.assertRaises(ValueError):
            agent.setup()

    def test_setup_requires_state_space(self):
        """setup() raises ValueError when state_space is not set."""
        agent = QLearningAgent(self.model)
        agent.set_action_space([-1.0, 0.0, 1.0])
        agent.set_reward_function(_reward_fn)
        with self.assertRaises(ValueError):
            agent.setup()

    def test_setup_requires_reward_function(self):
        """setup() raises ValueError when reward_function is not set."""
        agent = QLearningAgent(self.model)
        agent.set_action_space([-1.0, 0.0, 1.0])
        agent.set_state_space([(-5.0, 5.0), (-5.0, 5.0)], n_bins=5)
        with self.assertRaises(ValueError):
            agent.setup()

    def test_setup_success(self):
        """A fully configured agent sets up without error."""
        agent = _make_ql_agent(self.model)
        agent.setup()
        self.assertTrue(agent.is_setup())
        self.assertIsNotNone(agent.q_table)

    def test_q_table_shape(self):
        """Q-table has shape (n_bins_x0, n_bins_x1, n_actions)."""
        agent = _make_ql_agent(self.model)
        agent.setup()
        self.assertEqual(agent.q_table.shape, (10, 10, 3))

    def test_optimize_returns_valid_action(self):
        """optimize() returns an array whose value is in the action space."""
        agent = _make_ql_agent(self.model)
        agent.setup()
        x0 = [1.0, 0.0]
        u = agent.optimize(x0)
        self.assertEqual(u.shape, (1, 1))
        u_val = float(u.item())
        action_vals = [-1.0, 0.0, 1.0]
        self.assertIn(u_val, action_vals)

    def test_predict_alias(self):
        """predict() is an alias for optimize()."""
        agent = _make_ql_agent(self.model)
        agent.setup()
        x0 = [0.5, 0.0]
        u_opt = agent.optimize(x0)
        u_pred = agent.predict(x0)
        np.testing.assert_array_equal(u_opt, u_pred)

    def test_optimize_before_setup_raises(self):
        """optimize() raises RuntimeError before setup() is called."""
        agent = _make_ql_agent(self.model)
        with self.assertRaises(RuntimeError):
            agent.optimize([0.0, 0.0])

    def test_training_updates_q_table(self):
        """Q-table changes after calling train()."""
        agent = _make_ql_agent(self.model)
        agent.learning_rate = 0.5
        agent.setup()
        q_before = agent.q_table.copy()
        agent.train(x0=[1.0, 0.0], n_steps=50, n_episodes=5)
        self.assertFalse(np.allclose(agent.q_table, q_before))

    def test_training_returns_rewards(self):
        """train() returns a list of total rewards, one per episode."""
        agent = _make_ql_agent(self.model)
        agent.setup()
        rewards = agent.train(x0=[1.0, 0.0], n_steps=20, n_episodes=3)
        self.assertEqual(len(rewards), 3)
        self.assertTrue(all(isinstance(r, float) for r in rewards))

    def test_state_space_mismatch_raises(self):
        """Mismatched n_bins length raises ValueError."""
        agent = QLearningAgent(self.model)
        with self.assertRaises(ValueError):
            agent.set_state_space([(-5, 5), (-5, 5)], n_bins=[10])  # length 1 ≠ 2

    def test_save_load_roundtrip(self):
        """save() / load() preserve the Q-table."""
        agent = _make_ql_agent(self.model)
        agent.setup()
        agent.train(x0=[1.0, 0.0], n_steps=20, n_episodes=2)
        q_before = agent.q_table.copy()
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            path = f.name
        try:
            agent.save(path)
            agent2 = _make_ql_agent(self.model)
            agent2.setup()
            agent2.load(path)
            np.testing.assert_array_equal(agent2.q_table, q_before)
        finally:
            os.unlink(path)

    def test_discount_factor_validation(self):
        """discount_factor out of [0, 1] raises ValueError."""
        agent = QLearningAgent(self.model)
        with self.assertRaises(ValueError):
            agent.discount_factor = 1.5

    def test_learning_rate_validation(self):
        """Non-positive learning_rate raises ValueError."""
        agent = QLearningAgent(self.model)
        with self.assertRaises(ValueError):
            agent.learning_rate = -0.1


# ---------------------------------------------------------------------------
# DQNAgent tests
# ---------------------------------------------------------------------------

class TestDQNAgent(TestCase):

    def setUp(self) -> None:
        self.model = _make_integrator()

    def test_type(self):
        """Agent type string is 'DQN'."""
        agent = DQNAgent(self.model)
        self.assertEqual(agent.type, 'DQN')

    def test_setup_requires_action_space(self):
        """setup() raises ValueError when action_space is not set."""
        agent = DQNAgent(self.model)
        agent.set_reward_function(_reward_fn)
        with self.assertRaises(ValueError):
            agent.setup()

    def test_setup_requires_reward_function(self):
        """setup() raises ValueError when reward_function is not set."""
        agent = DQNAgent(self.model)
        agent.set_action_space([-1.0, 0.0, 1.0])
        with self.assertRaises(ValueError):
            agent.setup()

    def test_setup_success(self):
        """A fully configured DQN agent sets up without error."""
        agent = _make_dqn_agent(self.model)
        agent.setup()
        self.assertTrue(agent.is_setup())
        self.assertIsNotNone(agent.network)
        self.assertIsNotNone(agent.replay_buffer)

    def test_optimize_returns_valid_action(self):
        """optimize() returns a value from the action space."""
        agent = _make_dqn_agent(self.model)
        agent.setup()
        x0 = [1.0, 0.0]
        u = agent.optimize(x0)
        self.assertEqual(u.shape, (1, 1))
        u_val = float(u.item())
        self.assertIn(u_val, [-1.0, 0.0, 1.0])

    def test_predict_alias(self):
        """predict() is an alias for optimize()."""
        agent = _make_dqn_agent(self.model)
        agent.setup()
        x0 = [0.5, 0.0]
        np.testing.assert_array_equal(agent.optimize(x0), agent.predict(x0))

    def test_optimize_before_setup_raises(self):
        """optimize() raises RuntimeError before setup() is called."""
        agent = _make_dqn_agent(self.model)
        with self.assertRaises(RuntimeError):
            agent.optimize([0.0, 0.0])

    def test_training_runs_without_error(self):
        """train() completes without raising an exception."""
        agent = _make_dqn_agent(self.model)
        agent.setup()
        rewards = agent.train(x0=[1.0, 0.0], n_steps=30, n_episodes=3)
        self.assertEqual(len(rewards), 3)

    def test_epsilon_decays_during_training(self):
        """Epsilon-greedy policy epsilon decreases after training."""
        agent = _make_dqn_agent(self.model)
        agent.policy = EpsilonGreedyPolicy(epsilon=1.0, epsilon_min=0.01,
                                           epsilon_decay=0.9)
        agent.setup()
        eps_before = agent.policy.epsilon
        agent.train(x0=[1.0, 0.0], n_steps=20, n_episodes=2)
        self.assertLess(agent.policy.epsilon, eps_before)

    def test_replay_buffer_fills(self):
        """Replay buffer grows during training."""
        agent = _make_dqn_agent(self.model)
        agent.setup()
        agent.train(x0=[1.0, 0.0], n_steps=15, n_episodes=1)
        self.assertGreater(len(agent.replay_buffer), 0)

    def test_custom_hidden_layers(self):
        """DQN can be configured with a different network architecture."""
        agent = DQNAgent(self.model, hidden_layers=(32,), batch_size=8,
                         buffer_size=100, target_update_freq=10)
        agent.set_action_space([-1.0, 0.0, 1.0])
        agent.set_reward_function(_reward_fn)
        agent.setup()
        self.assertTrue(agent.is_setup())


# ---------------------------------------------------------------------------
# Policy tests
# ---------------------------------------------------------------------------

class TestPolicies(TestCase):

    def test_greedy_selects_argmax(self):
        """GreedyPolicy always selects the action with the highest Q-value."""
        policy = GreedyPolicy()
        q = np.array([0.1, 0.9, 0.5])
        self.assertEqual(policy.select_action(q), 1)

    def test_greedy_update_is_noop(self):
        """GreedyPolicy.update() does not raise and has no effect."""
        policy = GreedyPolicy()
        policy.update()  # should not raise

    def test_epsilon_greedy_zero_epsilon_is_greedy(self):
        """With epsilon=0 the epsilon-greedy policy is purely greedy."""
        policy = EpsilonGreedyPolicy(epsilon=0.0)
        q = np.array([0.1, 0.9, 0.5])
        for _ in range(20):
            self.assertEqual(policy.select_action(q), 1)

    def test_epsilon_greedy_full_random(self):
        """With epsilon=1 all actions are chosen (eventually)."""
        policy = EpsilonGreedyPolicy(epsilon=1.0)
        q = np.array([0.0, 1.0, 0.0])
        actions = {policy.select_action(q) for _ in range(200)}
        self.assertEqual(len(actions), 3)

    def test_epsilon_decay(self):
        """EpsilonGreedyPolicy epsilon decreases after update() calls."""
        policy = EpsilonGreedyPolicy(epsilon=1.0, epsilon_min=0.01,
                                     epsilon_decay=0.5)
        for _ in range(10):
            policy.update()
        self.assertLess(policy.epsilon, 1.0)
        self.assertGreaterEqual(policy.epsilon, 0.01)

    def test_epsilon_does_not_go_below_min(self):
        """Epsilon is clamped to epsilon_min after many decays."""
        policy = EpsilonGreedyPolicy(epsilon=1.0, epsilon_min=0.05,
                                     epsilon_decay=0.5)
        for _ in range(100):
            policy.update()
        self.assertAlmostEqual(policy.epsilon, 0.05)

    def test_softmax_probabilities_sum_to_one(self):
        """SoftmaxPolicy samples from a valid probability distribution."""
        policy = SoftmaxPolicy(temperature=1.0)
        q = np.array([1.0, 2.0, 3.0])
        counts = np.zeros(3)
        n = 3000
        for _ in range(n):
            counts[policy.select_action(q)] += 1
        # All actions should be selected at least once
        self.assertTrue(np.all(counts > 0))

    def test_softmax_low_temperature_nearly_greedy(self):
        """At very low temperature SoftmaxPolicy converges to greedy."""
        policy = SoftmaxPolicy(temperature=1e-6)
        q = np.array([0.1, 5.0, 0.5])
        for _ in range(50):
            self.assertEqual(policy.select_action(q), 1)

    def test_softmax_negative_temperature_raises(self):
        """Non-positive temperature raises ValueError."""
        with self.assertRaises(ValueError):
            SoftmaxPolicy(temperature=0.0)

    def test_invalid_policy_assignment(self):
        """Setting a non-Policy object as policy raises TypeError."""
        model = _make_integrator()
        agent = QLearningAgent(model)
        with self.assertRaises(TypeError):
            agent.policy = "not_a_policy"


# ---------------------------------------------------------------------------
# ReplayBuffer tests
# ---------------------------------------------------------------------------

class TestReplayBuffer(TestCase):

    def _make_transition(self, n_states: int = 2):
        s = np.random.randn(n_states)
        a = np.random.randint(3)
        r = float(np.random.randn())
        s_next = np.random.randn(n_states)
        done = False
        return s, a, r, s_next, done

    def test_push_and_len(self):
        """push() increments the buffer length."""
        buf = ReplayBuffer(capacity=10)
        self.assertEqual(len(buf), 0)
        for i in range(5):
            buf.push(*self._make_transition())
        self.assertEqual(len(buf), 5)

    def test_capacity_not_exceeded(self):
        """Buffer length never exceeds capacity."""
        buf = ReplayBuffer(capacity=5)
        for _ in range(20):
            buf.push(*self._make_transition())
        self.assertEqual(len(buf), 5)

    def test_sample_shape(self):
        """Sampled batch has the expected shapes."""
        buf = ReplayBuffer(capacity=50)
        for _ in range(20):
            buf.push(*self._make_transition(n_states=2))
        states, actions, rewards, next_states, dones = buf.sample(8)
        self.assertEqual(states.shape, (8, 2))
        self.assertEqual(actions.shape, (8,))
        self.assertEqual(rewards.shape, (8,))
        self.assertEqual(next_states.shape, (8, 2))
        self.assertEqual(dones.shape, (8,))

    def test_sample_too_large_raises(self):
        """Requesting more samples than stored raises ValueError."""
        buf = ReplayBuffer(capacity=50)
        for _ in range(3):
            buf.push(*self._make_transition())
        with self.assertRaises(ValueError):
            buf.sample(10)

    def test_zero_capacity_raises(self):
        """Zero capacity raises ValueError."""
        with self.assertRaises(ValueError):
            ReplayBuffer(capacity=0)

    def test_circular_overwrite(self):
        """Old transitions are overwritten when buffer is full."""
        buf = ReplayBuffer(capacity=3)
        # Fill buffer with identifiable transitions
        for i in range(3):
            buf.push(np.array([float(i), 0.0]), i, float(i), np.zeros(2), False)
        # Add one more — should overwrite position 0
        buf.push(np.array([99.0, 0.0]), 0, 99.0, np.zeros(2), False)
        self.assertEqual(len(buf), 3)
        # The buffer should contain the transition with state[0]=99.0
        found = any(buf._buffer[i][0][0] == 99.0 for i in range(3))
        self.assertTrue(found)


if __name__ == '__main__':
    unittest.main()
