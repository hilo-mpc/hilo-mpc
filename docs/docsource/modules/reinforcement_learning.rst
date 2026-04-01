
==============================
Reinforcement Learning module
==============================

This section describes the Reinforcement Learning (RL) module of HILO-MPC. The interface mirrors
the MPC controllers: after constructing an agent with a :class:`~hilo_mpc.Model`, the workflow is

1. Configure the agent (action space, reward function, hyper-parameters, policy).
2. Call :meth:`setup` to initialise internal data structures.
3. Train with :meth:`train`.
4. Deploy with :meth:`optimize` — the same call used for :class:`~hilo_mpc.NMPC`.

For a detailed description of the methods refer to :ref:`the API <rl_automodel>`.

The RL module contains the following classes:

- :class:`~hilo_mpc.QLearningAgent` — tabular Q-learning for discretised state-action spaces
- :class:`~hilo_mpc.DQNAgent` — deep Q-network with experience replay and a target network
- :class:`~hilo_mpc.EpsilonGreedyPolicy`, :class:`~hilo_mpc.GreedyPolicy`, :class:`~hilo_mpc.SoftmaxPolicy` — exploration/exploitation policies
- :class:`~hilo_mpc.ReplayBuffer` — circular experience replay buffer

----------------------------
Theoretical background
----------------------------

Markov Decision Process
-----------------------

The RL framework assumes the controlled system can be modelled as a **Markov Decision Process
(MDP)** :math:`(\mathcal{X}, \mathcal{U}, f, r, \gamma)` where

- :math:`\mathcal{X} \subseteq \mathbb{R}^{n_x}` is the state space,
- :math:`\mathcal{U} = \{u^{(1)}, \ldots, u^{(m)}\}` is a finite set of :math:`m` discrete actions,
- :math:`f : \mathcal{X} \times \mathcal{U} \to \mathcal{X}` is the (possibly stochastic) transition function,
- :math:`r : \mathcal{X} \times \mathcal{U} \times \mathcal{X} \to \mathbb{R}` is the scalar reward function,
- :math:`\gamma \in [0, 1)` is the discount factor.

At each discrete time step :math:`k`, the agent observes the current state :math:`x_k`, selects an
action :math:`u_k \in \mathcal{U}` according to a **policy** :math:`\pi : \mathcal{X} \to \mathcal{U}`,
receives a scalar reward :math:`r_k = r(x_k, u_k, x_{k+1})`, and transitions to
:math:`x_{k+1} = f(x_k, u_k)`.

The agent's goal is to find the **optimal policy** :math:`\pi^*` that maximises the expected
discounted cumulative reward (return):

.. math::

    G_k = \sum_{i=0}^{\infty} \gamma^i\, r_{k+i}

Action-value function
---------------------

The **Q-function** (action-value function) of a policy :math:`\pi` is defined as the expected return
when taking action :math:`u` in state :math:`x` and subsequently following :math:`\pi`:

.. math::

    Q^{\pi}(x, u) = \mathbb{E}_{\pi}\!\left[\,\sum_{i=0}^{\infty} \gamma^i\, r_{k+i}
    \;\middle|\; x_k = x,\; u_k = u\right]

The **Bellman optimality equation** gives the recursive characterisation of the optimal Q-function
:math:`Q^*`:

.. math::

    Q^*(x, u) = \mathbb{E}_{x'}\!\left[\,r(x, u, x') + \gamma \max_{u' \in \mathcal{U}} Q^*(x', u')
    \;\middle|\; x,\, u\right]

The optimal greedy policy is then recovered as

.. math::

    \pi^*(x) = \operatorname*{arg\,max}_{u \in \mathcal{U}}\; Q^*(x, u)

----------------------------
Q-Learning agent
----------------------------

Tabular Q-learning :cite:`Watkins1992` approximates :math:`Q^*` with a look-up table (the
*Q-table*) :math:`\hat{Q} \in \mathbb{R}^{n_1 \times \cdots \times n_{n_x} \times m}` whose entry
:math:`\hat{Q}[s_1, \ldots, s_{n_x}, j]` stores the estimated value of being in the discretised
state bin :math:`(s_1, \ldots, s_{n_x})` and taking action :math:`u^{(j)}`.

State discretisation
....................

Continuous states :math:`x \in \mathbb{R}^{n_x}` are mapped to discrete bin indices using
user-defined bin edges. For state dimension :math:`i` with :math:`n_i` bins defined by the edges
:math:`b_i^{(0)} < b_i^{(1)} < \cdots < b_i^{(n_i)}`:

.. math::

    s_i(x_i) = \max\!\left(0,\;\min\!\left(n_i - 1,\;
    \left\lfloor \frac{x_i - b_i^{(0)}}{b_i^{(n_i)} - b_i^{(0)}} n_i \right\rfloor\right)\right)

In HILO-MPC the bin edges are supplied via :meth:`~hilo_mpc.QLearningAgent.set_state_space`:

.. code-block:: python

    agent.set_state_space(
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],  # (low, high) per state dimension
        n_bins=10                             # number of bins per dimension
    )

TD update rule
..............

The Q-table is updated online with the **temporal-difference (TD) rule**:

.. math::

    \hat{Q}(x_k, u_k) \;\leftarrow\; \hat{Q}(x_k, u_k)
    + \alpha \underbrace{\left[\,r_k + \gamma \max_{u' \in \mathcal{U}} \hat{Q}(x_{k+1}, u')
    - \hat{Q}(x_k, u_k)\,\right]}_{\text{TD error }\delta_k}

where :math:`\alpha \in (0, 1]` is the learning rate. At termination (:math:`x_{k+1}` is a
terminal state), the bootstrap term :math:`\gamma \max_{u'} \hat{Q}(x_{k+1}, u')` is set to zero:

.. math::

    \hat{Q}(x_k, u_k) \;\leftarrow\; \hat{Q}(x_k, u_k) + \alpha \left[\,r_k - \hat{Q}(x_k, u_k)\,\right]

Usage
.....

.. code-block:: python

    from hilo_mpc import Model, QLearningAgent, EpsilonGreedyPolicy

    # Build and set up the plant model
    model = Model()
    # ... define dynamics ...
    model.setup(dt=0.1)

    # Create agent
    agent = QLearningAgent(model)
    agent.set_action_space([-10.0, 0.0, 10.0])           # discrete actions [N]
    agent.set_state_space([(-3.0, 3.0), (-5.0, 5.0)],    # state bounds
                          n_bins=10)                       # 10 bins per dim
    agent.set_reward_function(lambda x, u, xn: -float(xn @ xn))
    agent.learning_rate   = 0.1
    agent.discount_factor = 0.99
    agent.policy = EpsilonGreedyPolicy(epsilon=1.0, epsilon_min=0.05,
                                       epsilon_decay=0.995)
    agent.setup()

    # Training loop
    rewards = agent.train(x0=[1.0, 0.0], n_steps=200, n_episodes=500)

    # Deployment — identical call to NMPC.optimize
    u_opt = agent.optimize(x0)

The Q-table can be saved and restored:

.. code-block:: python

    agent.save('/tmp/qtable.npy')
    agent.load('/tmp/qtable.npy')

.. note::

    :class:`~hilo_mpc.QLearningAgent` requires no external machine-learning library.
    The Q-table is a plain NumPy array.

----------------------------
DQN agent
----------------------------

Deep Q-Networks (DQN) :cite:`Mnih2015` replace the Q-table with a parametric function
approximator — a feedforward neural network with parameters :math:`\theta`:

.. math::

    Q(x, u;\,\theta) \approx Q^*(x, u)

The DQN agent in HILO-MPC uses a lightweight NumPy-based network so that **no external ML
framework** (PyTorch, TensorFlow) is required.

Neural network architecture
...........................

The online Q-network :math:`Q(\cdot;\,\theta)` and a periodically-synchronised **target network**
:math:`Q(\cdot;\,\theta^-)` share the same architecture:

.. math::

    \begin{aligned}
        h^{(0)} &= x \\
        h^{(\ell)} &= \mathrm{ReLU}\!\left(W^{(\ell)}\, h^{(\ell-1)} + b^{(\ell)}\right),
        \quad \ell = 1, \ldots, L \\
        Q(x, \cdot\,;\,\theta) &= W^{(L+1)}\, h^{(L)} + b^{(L+1)}
    \end{aligned}

where :math:`L` is the number of hidden layers, :math:`W^{(\ell)}` and :math:`b^{(\ell)}` are the
weight matrix and bias vector of layer :math:`\ell`, and :math:`\mathrm{ReLU}(z) = \max(0, z)`.
Weights are initialised with **Xavier (Glorot) initialisation**:

.. math::

    W^{(\ell)}_{ij} \sim \mathcal{N}\!\left(0,\, \frac{2}{n^{(\ell-1)}}\right)

where :math:`n^{(\ell-1)}` is the number of units in layer :math:`\ell - 1`.

Experience replay
.................

Each transition :math:`(x_k, u_k, r_k, x_{k+1}, d_k)` — where :math:`d_k \in \{0,1\}` indicates
episode termination — is stored in a **circular replay buffer** :math:`\mathcal{D}` of capacity
:math:`C`:

.. math::

    \mathcal{D} = \left\{(x_k, u_k, r_k, x_{k+1}, d_k)\right\}_{k=1}^{|\mathcal{D}|}

At each gradient step a mini-batch of :math:`B` transitions is sampled uniformly at random from
:math:`\mathcal{D}`.

TD target and loss
..................

For each sampled transition the **TD target** is computed using the *frozen* target network:

.. math::

    y_k = r_k + \gamma\,(1 - d_k)\,\max_{u' \in \mathcal{U}} Q(x_{k+1}, u';\,\theta^-)

The network parameters :math:`\theta` are updated by minimising the **mean-squared Bellman error**
over the mini-batch:

.. math::

    \mathcal{L}(\theta) = \frac{1}{B}
    \sum_{k=1}^{B} \left(\,y_k - Q(x_k, u_k;\,\theta)\,\right)^2

Gradient descent is performed with one step of back-propagation per environment step:

.. math::

    \theta \;\leftarrow\; \theta - \alpha\, \nabla_\theta\, \mathcal{L}(\theta)

Target network update
.....................

Every :math:`C_{\text{target}}` gradient steps the target network is **hard-updated** by copying
the online network weights:

.. math::

    \theta^- \;\leftarrow\; \theta

This stabilises training by preventing the moving target problem that arises when both networks are
updated simultaneously :cite:`Mnih2015`.

Usage
.....

.. code-block:: python

    from hilo_mpc import Model, DQNAgent, EpsilonGreedyPolicy

    model = Model()
    # ... define dynamics ...
    model.setup(dt=0.1)

    agent = DQNAgent(
        model,
        hidden_layers=(64, 64),     # two hidden layers with 64 units each
        batch_size=32,
        buffer_size=10000,
        target_update_freq=100
    )
    agent.set_action_space([-10.0, 0.0, 10.0])
    agent.set_reward_function(lambda x, u, xn: -float(xn @ xn))
    agent.learning_rate   = 1e-3
    agent.discount_factor = 0.99
    agent.setup()

    rewards = agent.train(x0=[1.0, 0.0], n_steps=200, n_episodes=300)
    u_opt = agent.optimize(x0)

.. note::

    :class:`~hilo_mpc.DQNAgent` is implemented entirely in NumPy and requires no external
    ML library. The network is trained with a plain back-propagation loop.

----------------------------
Exploration policies
----------------------------

Policies control how actions are selected during training. All policy classes share the same
interface and can be swapped freely:

.. code-block:: python

    agent.policy = EpsilonGreedyPolicy(epsilon=0.5)
    # or
    agent.policy = SoftmaxPolicy(temperature=1.0)

Epsilon-greedy policy
.....................

The :class:`~hilo_mpc.EpsilonGreedyPolicy` selects a uniformly random action with probability
:math:`\epsilon` and the greedy action otherwise:

.. math::

    u_k =
    \begin{cases}
        \operatorname*{arg\,max}_{u \in \mathcal{U}} \hat{Q}(x_k, u)
            & \text{with probability } 1 - \epsilon \\[4pt]
        \text{uniform sample from } \mathcal{U}
            & \text{with probability } \epsilon
    \end{cases}

The exploration probability :math:`\epsilon` is decayed multiplicatively after each environment
step to gradually shift from exploration to exploitation:

.. math::

    \epsilon_{k+1} = \max\!\left(\epsilon_{\min},\; \epsilon_k \cdot \epsilon_{\text{decay}}\right)

Greedy policy
.............

The :class:`~hilo_mpc.GreedyPolicy` is the pure exploitation policy used for deployment
(no exploration):

.. math::

    u_k = \operatorname*{arg\,max}_{u \in \mathcal{U}}\; \hat{Q}(x_k, u)

Softmax (Boltzmann) policy
..........................

The :class:`~hilo_mpc.SoftmaxPolicy` samples actions from a Boltzmann distribution over
Q-values controlled by a temperature parameter :math:`\tau > 0`:

.. math::

    P(u \mid x_k) = \frac{\exp\!\left(Q(x_k, u) / \tau\right)}
                         {\displaystyle\sum_{u' \in \mathcal{U}} \exp\!\left(Q(x_k, u') / \tau\right)}

As :math:`\tau \to 0` the distribution concentrates on the greedy action; as
:math:`\tau \to \infty` the distribution becomes uniform.

----------------------------
Replay buffer
----------------------------

The :class:`~hilo_mpc.ReplayBuffer` is a **circular buffer** of fixed capacity :math:`C`. When the
buffer is full, the oldest transition is overwritten:

.. math::

    \text{position} = (k \bmod C)

Random sampling of a mini-batch of size :math:`B` is performed without replacement:

.. math::

    \mathcal{B} \sim \mathrm{Uniform}(\mathcal{D},\, B), \quad B \leq |\mathcal{D}|

.. code-block:: python

    from hilo_mpc import ReplayBuffer

    buf = ReplayBuffer(capacity=10000)
    buf.push(state, action_idx, reward, next_state, done)
    states, actions, rewards, next_states, dones = buf.sample(batch_size=32)

----------------------------
Training loop
----------------------------

The :meth:`~hilo_mpc.QLearningAgent.train` method runs the following closed-loop training
algorithm for both :class:`~hilo_mpc.QLearningAgent` and :class:`~hilo_mpc.DQNAgent`:

.. math::

    \begin{aligned}
    &\textbf{for } \text{episode} = 1, \ldots, N_{\text{ep}} \textbf{ do}\\
    &\quad x_0 \leftarrow x_{\text{init}} \\
    &\quad \textbf{for } k = 0, 1, \ldots, T-1 \textbf{ do}\\
    &\quad\quad u_k \leftarrow \pi_{\epsilon}(x_k)
        \quad\text{(exploration policy)}\\
    &\quad\quad x_{k+1} \leftarrow f(x_k, u_k)
        \quad\text{(model simulation)}\\
    &\quad\quad r_k \leftarrow r(x_k, u_k, x_{k+1}) \\
    &\quad\quad \text{update } \hat{Q} \text{ or } \theta \\
    &\quad\quad \textbf{if } d(x_{k+1}) \textbf{ then break}\\
    &\quad \textbf{end for}\\
    &\textbf{end for}
    \end{aligned}

The method returns a list of total rewards per episode that can be used to monitor convergence:

.. code-block:: python

    rewards = agent.train(x0=[1.0, 0.0], n_steps=200, n_episodes=500,
                          done_fn=lambda x: abs(x[2]) > 0.5)

    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total reward')

----------------------------
Comparison with MPC
----------------------------

The RL agents are designed to be **drop-in companions** to the MPC controllers in HILO-MPC.
The table below summarises the key differences and similarities.

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Feature
     - MPC (NMPC/LMPC)
     - RL (QLearning/DQN)
   * - Initialisation
     - :code:`NMPC(model)`
     - :code:`QLearningAgent(model)`
   * - Setup
     - :code:`nmpc.setup()`
     - :code:`agent.setup()`
   * - Action selection
     - :code:`nmpc.optimize(x0)`
     - :code:`agent.optimize(x0)`
   * - Requires model equations
     - Yes (for NLP)
     - Yes (for simulation)
   * - Online learning
     - No
     - Yes (via :meth:`train`)
   * - Requires solver (IPOPT, qpOASES)
     - Yes
     - No
   * - Handles constraints explicitly
     - Yes
     - No (implicit via reward)
   * - Prediction horizon
     - :math:`N` steps ahead
     - Discount factor :math:`\gamma`

Both agents expose a :meth:`predict` method (alias for :meth:`optimize`) so they can be
used inside :class:`~hilo_mpc.SimpleControlLoop`:

.. code-block:: python

    from hilo_mpc import SimpleControlLoop

    scl = SimpleControlLoop(model, agent)
    scl.run(steps=100)

----------------------------
Extending the framework
----------------------------

New RL algorithms (PPO, SAC, A2C, …) can be added by sub-classing
:class:`~hilo_mpc.RLBase` and implementing three abstract methods:

.. code-block:: python

    from hilo_mpc.modules.rl.base import RLBase

    class MySACAgent(RLBase):
        def _update_type(self) -> None:
            self._type = 'SAC'

        def setup(self, options=None) -> None:
            # initialise actor and critic networks
            self._setup_done = True

        def optimize(self, x0) -> np.ndarray:
            # return action from actor network
            ...

        def update(self, state, action, reward, next_state, done) -> None:
            # store transition and perform gradient updates
            ...

New policies can be added by sub-classing :class:`~hilo_mpc.modules.rl.policy.Policy`:

.. code-block:: python

    from hilo_mpc.modules.rl.policy import Policy

    class UCBPolicy(Policy):
        def select_action(self, q_values):
            # upper-confidence-bound selection
            ...

        def update(self) -> None:
            # update visit counts
            ...
