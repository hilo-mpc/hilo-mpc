.. _observer_module:

=========================
Observer module
=========================
This section describes the most important tools of the Observer module with examples. For a more detailed
description of the methods refer to :ref:`the API <observer_autodoc>`.

The Observer module contains several state (and parameter) observers. Observers are used to infer states and 
parameters from measurements, which is essential when not all states are directly measurable or when measurements 
are corrupted by noise.

The Observer module contains the following classes:

- Moving Horizon Estimator (MHE)
- Kalman Filter (KF)
- Extended Kalman Filter (EKF)
- Unscented Kalman Filter (UKF)
- Particle Filter (PF)

-----------------------------------
Moving Horizon Estimator
-----------------------------------
The class :class:`~hilo_mpc.MovingHorizonEstimator` (alias :class:`~hilo_mpc.MHE`) implements a Moving Horizon 
Estimator for state and parameter estimation. The MHE solves an optimization problem over a moving time window 
to estimate states and parameters by minimizing the mismatch between predicted and measured outputs.

For a complete example of using the MHE, see the :doc:`Chemical Reaction MHE Example <../examples/mhe_chemical_reaction>`.

Mathematical formulation
------------------------
The MHE is formulated for a discrete-time nonlinear system (HILO-MPC can work with both continuous-time and
discrete-time models, automatically discretizing continuous models):

.. math::

     \begin{aligned}
         x_{k+1} &= f(x_k, u_k, w_k, p), \\
         y_k &= h(x_k, u_k, p) + v_k
     \end{aligned}

where :math:`x_k \in \mathbb{R}^{n_x}` is the state, :math:`u_k \in \mathbb{R}^{n_u}` is the input,
:math:`w_k \in \mathbb{R}^{n_x}` is the process noise, :math:`p \in \mathbb{R}^{n_p}` are parameters,
:math:`y_k \in \mathbb{R}^{n_y}` are measurements, and :math:`v_k \in \mathbb{R}^{n_y}` is measurement noise.

At time :math:`t`, the Moving Horizon Estimator solves an optimization problem over a sliding window of length
:math:`N` to estimate the state trajectory :math:`\{x_{t-N}, \ldots, x_t\}` (and optionally parameters :math:`p`):

.. math::

    \begin{aligned}
    \min_{\substack{x_{t-N:t},\\ w_{t-N:t-1},\\ p}}\; & \underbrace{\Vert x_{t-N}-\bar{x}_{t-N} \Vert_{P_{t-N}^{-1}}^2}_{\text{arrival cost}}
    + \sum_{k=t-N}^{t-1} \underbrace{\Vert w_k \Vert_{Q_k^{-1}}^2}_{\text{process noise penalty}}
    + \sum_{k=t-N}^{t} \underbrace{\Vert y_k - h(x_k,u_k,p) \Vert_{R_k^{-1}}^2}_{\text{measurement residual}} \\
    \text{s.t.}\quad & x_{k+1} = f(x_k, u_k, w_k, p),\quad k=t-N,\ldots,t-1, \\
    & x_k \in \mathcal{X},\quad k=t-N,\ldots,t, \\
    & w_k \in \mathcal{W},\quad k=t-N,\ldots,t-1, \\
    & p \in \mathcal{P}, \\
    & c(x_k, u_k, p) \leq 0,\quad k=t-N,\ldots,t, \\
    & e(x_k, u_k, p) = 0,\quad k=t-N,\ldots,t.
    \end{aligned}

where :math:`\Vert z \Vert_M^2 = z^\top M z` denotes the weighted squared norm.

- :math:`N` is the horizon length (see ``horizon``).
- :math:`P_0` weights the arrival/prior term (see ``quad_arrival_cost.add_states(weights==...)``).
- :math:`R_k` weights measurement residuals (see ``quad_stage_cost.add_measurements(weights=...)``).
- :math:`Q_k` weights process or state noise (see ``quad_stage_cost.add_state_noise(weights=...)``).
- :math:`u_k` and :math:`y_k` are the inputs and measurements provided via ``add_measurements(y_meas=..., u_meas=...)``.
- :math:`\hat{x}_{t-N}` is the prior/arrival state supplied via ``estimate(x_arrival=...,p_arrival=...)``.
- :math:`x^{\min}, x^{\max}, w^{\min}, w^{\max}, p^{\min}, p^{\max}` are box constraints (see ``set_box_constraints(x_lb=..., x_ub=..., w_lb=..., w_ub=..., p_lb=..., p_ub=...)``).
- :math:`c(\cdot) \le 0` and :math:`e(\cdot) = 0` are nonlinear inequality and equality constraints (see ``stage_constraint.constraint``, ``stage_constraint.lb``, ``stage_constraint.ub``).

.. note::
    You can pass either a numpy matrix or a list or numpy array to the `weights`. If you pass a list or array, a matrix of appropriate dimensions will be created internally with the array in the diagonal and zero anywhere else.

.. note::
    All the parameter variables of the model are assumed to be unknown and will be estimated by the MHE. If you want to keep them constant, you can add equal lower and upper constraint, or substitute them with a actual number directly in the model.

The solution of this optimization problem provides the smoothed state estimate :math:`\hat{x}_t` at the current time,
along with estimates of parameters :math:`\hat{p}` if included in the optimization.

To set up an MHE properly you need *at least* to define:
- A dynamic model
- A horizon length using the :code:`horizon` property.
- A stage cost function (typically using :code:`quad_stage_cost`).
- An arrival cost function (typically using :code:`quad_arrival_cost`).

The MHE can be initialized as follows:

.. code-block:: python

    from hilo_mpc import Model, MHE

    # Initialize MHE with a model
    mhe = MHE(model)

Required information, like e.g. the model dynamics or the sampling time, will be automatically extracted from 
the :class:`~hilo_mpc.Model` instance.


Horizon length
--------------
The horizon length determines how many past measurements are considered in the estimation problem. A longer 
horizon typically provides better estimates but increases computational cost:

.. code-block:: python

    mhe = MHE(model)
    mhe.horizon = 10  # Use 10 past measurements

Cost functions
--------------
The MHE uses two types of costs:

- **Stage cost**: penalizes the mismatch between predicted and measured outputs at each time step
- **Arrival cost**: penalizes the deviation from the prior estimate at the beginning of the horizon

Stage cost
..........
The stage cost is typically defined to penalize measurement errors and state/parameter noise:

.. code-block:: python

    # Add measurement error terms
    mhe.quad_stage_cost.add_measurements(names=['y1', 'y2'], 
                                         weights=[10, 10])
    
    # Add state noise terms (if using state noise modeling)
    mhe.quad_stage_cost.add_state_noise(names=['x1', 'x2'], 
                                        weights=[1, 1])

Arrival cost
............
The arrival cost provides information about the initial state estimate at the beginning of the horizon:

.. code-block:: python

    # Set arrival cost for states
    mhe.quad_arrival_cost.add_states(names=['x1', 'x2'], 
                                     weights=[1, 1])

Adding measurements and estimation
-----------------------------------
Measurements are added to the MHE using the :meth:`~hilo_mpc.MovingHorizonEstimator.add_measurements` method, 
and estimation is performed using :meth:`~hilo_mpc.MovingHorizonEstimator.estimate`:

.. code-block:: python

    # Add measurement
    mhe.add_measurements(y_meas=[measurement_value], u_meas=[input_value])
    
    # Estimate states once horizon is reached
    x_estimated = mhe.estimate(x_arrival=x0)

Constraints
-----------
The MHE supports both box constraints and nonlinear constraints to enforce physical bounds and system constraints.

Box constraints
...............
Box constraints impose simple lower and upper bounds on states, process noise, and parameters:

.. code-block:: python

    # Set box constraints for states
    mhe.set_box_constraints(x_lb=[0, 0, 0], x_ub=[10, 10, 10])

    # Set constraints for states and process noise
    mhe.set_box_constraints(x_lb=[0, 0, 0], x_ub=[10, 10, 10],
                           w_lb=[-0.1, -0.1, -0.1], w_ub=[0.1, 0.1, 0.1])

    # Add parameter bounds when estimating parameters
    mhe.set_box_constraints(x_lb=[0, 0, 0], x_ub=[10, 10, 10],
                           p_lb=[0, 0], p_ub=[5, 5])

Nonlinear constraints
.....................
Nonlinear inequality and equality constraints can be added using the :obj:`stage_constraint` property:

.. code-block:: python

    # Add inequality constraint: 0.1 <= Ca <= 5
    mhe.stage_constraint.constraint = model.x[0]  # Ca
    mhe.stage_constraint.lb = 0.1
    mhe.stage_constraint.ub = 5

    # For equality constraints, set lb = ub
    # For inequality constraints c(x) >= a, use lb=a and ub=ca.inf
    # For inequality constraints c(x) <= b, use lb=-ca.inf and ub=b

Setup
-----
Before using the MHE, it must be set up using the :meth:`~hilo_mpc.MovingHorizonEstimator.setup` method:

.. code-block:: python

    mhe.setup(options={'integration_method': 'collocation'})

Non-uniform sampling intervals
-------------------------------
The MHE supports non-uniform sampling intervals, allowing measurements to be taken at irregular time intervals.
This is particularly useful when dealing with multi-rate sensors or event-triggered measurements.

In the mathematical program, this corresponds to time steps with variable lengths :math:`\Delta t_k`, which affect the
discretized dynamics constraint :math:`x_{k+1} = f_{\Delta t_k}(x_k, u_k, w_k)`. The chosen integration method
(``options={'integration_method': ...}``) determines how :math:`f_{\Delta t_k}(\cdot)` is formed from the continuous-time
model.

Multi-rate measurements
-----------------------
When different sensors provide measurements at different rates, the MHE can handle this naturally by accepting
measurements only when they are available.

-----------------------------------
Kalman Filter
-----------------------------------
The class :class:`~hilo_mpc.KalmanFilter` (alias :class:`~hilo_mpc.KF`) implements the Kalman filter developed 
by Rudolf E. Kálmán. The Kalman filter is optimal for linear systems with Gaussian noise and provides recursive 
state estimation with low computational cost.

Mathematical formulation
------------------------
The Kalman filter is designed for linear systems of the form:

.. math::

    \begin{aligned}
        x_{k+1} &= A x_k + B u_k + w_k, \\
        y_k &= C x_k + D u_k + v_k
    \end{aligned}

where :math:`w_k \sim \mathcal{N}(0, Q)` is process noise and :math:`v_k \sim \mathcal{N}(0, R)` is measurement noise.

The Kalman filter operates in two steps:

**Prediction step** (time update):

.. math::

    \begin{aligned}
        \hat{x}_{k|k-1} &= A \hat{x}_{k-1|k-1} + B u_k, \\
        P_{k|k-1} &= A P_{k-1|k-1} A^\top + Q
    \end{aligned}

**Update step** (measurement update):

.. math::

    \begin{aligned}
        K_k &= P_{k|k-1} C^\top (C P_{k|k-1} C^\top + R)^{-1}, \\
        \hat{x}_{k|k} &= \hat{x}_{k|k-1} + K_k (y_k - C \hat{x}_{k|k-1} - D u_k), \\
        P_{k|k} &= (I - K_k C) P_{k|k-1}
    \end{aligned}

where:

- :math:`\hat{x}_{k|k-1}` is the predicted state estimate
- :math:`\hat{x}_{k|k}` is the updated state estimate after incorporating measurement :math:`y_k`
- :math:`P_{k|k-1}` is the predicted error covariance
- :math:`P_{k|k}` is the updated error covariance
- :math:`K_k` is the Kalman gain matrix
- :math:`Q` is the process noise covariance → provided as :code:`Q` parameter in :code:`kf.estimate()`
- :math:`R` is the measurement noise covariance → provided as :code:`R` parameter in :code:`kf.estimate()`

To set up the Kalman filter you need an already set up :class:`~hilo_mpc.Model` instance with **linear dynamics**.
Additionally you might want to supply a plot backend (via the :obj:`plot_backend` keyword argument) in order to 
visualize the estimation results later on. At the moment only `Matplotlib <https://matplotlib.org/>`_ and 
`Bokeh <https://bokeh.org/>`_ are supported for plotting.

The Kalman filter can be initialized as follows:

.. code-block:: python

    from hilo_mpc import KF

    # Initialize Kalman filter
    kf = KF(model)

Required information, like e.g. the model dynamics or the sampling time, will be automatically extracted from 
the :class:`~hilo_mpc.Model` instance.

Setting noise covariances
--------------------------
The Kalman filter requires specification of process noise covariance (Q) and measurement noise covariance (R):

.. code-block:: python

    import numpy as np
    
    # Set process noise covariance
    Q = np.eye(n_states) * 0.01
    
    # Set measurement noise covariance
    R = np.eye(n_measurements) * 0.1

Setup and estimation
--------------------
After initialization, the Kalman filter must be set up before use:

.. code-block:: python

    # Setup the Kalman filter
    kf.setup()
    
    # Set initial state estimate and covariance
    x0 = [initial_state_estimate]
    P0 = np.eye(n_states)  # Initial covariance
    
    # Perform estimation
    x_estimated = kf.estimate(x0=x0, y=measurement, u=input, P=P0, Q=Q, R=R)

The :meth:`~hilo_mpc.KalmanFilter.estimate` method performs both the prediction and update steps of the 
Kalman filter algorithm.

-----------------------------------
Extended Kalman Filter
-----------------------------------
The class :class:`~hilo_mpc.ExtendedKalmanFilter` (alias :class:`~hilo_mpc.EKF`) implements the Extended 
Kalman Filter for **nonlinear systems**. The EKF linearizes the nonlinear system dynamics and measurement 
equations around the current state estimate at each time step.

Mathematical formulation
------------------------
The EKF is designed for nonlinear systems:

.. math::

    \begin{aligned}
        x_{k+1} &= f(x_k, u_k) + w_k, \\
        y_k &= h(x_k, u_k) + v_k
    \end{aligned}

where :math:`w_k \sim \mathcal{N}(0, Q)` and :math:`v_k \sim \mathcal{N}(0, R)`.

At each time step, the EKF linearizes the system around the current estimate :math:`\hat{x}_{k|k-1}`:

.. math::

    \begin{aligned}
        F_k &= \left. \frac{\partial f}{\partial x} \right|_{x=\hat{x}_{k-1|k-1}, u=u_k}, \\
        H_k &= \left. \frac{\partial h}{\partial x} \right|_{x=\hat{x}_{k|k-1}, u=u_k}
    \end{aligned}

**Prediction step**:

.. math::

    \begin{aligned}
        \hat{x}_{k|k-1} &= f(\hat{x}_{k-1|k-1}, u_k), \\
        P_{k|k-1} &= F_k P_{k-1|k-1} F_k^\top + Q
    \end{aligned}

**Update step**:

.. math::

    \begin{aligned}
        K_k &= P_{k|k-1} H_k^\top (H_k P_{k|k-1} H_k^\top + R)^{-1}, \\
        \hat{x}_{k|k} &= \hat{x}_{k|k-1} + K_k (y_k - h(\hat{x}_{k|k-1}, u_k)), \\
        P_{k|k} &= (I - K_k H_k) P_{k|k-1}
    \end{aligned}

The key difference from the linear Kalman filter is that:

1. The state prediction uses the nonlinear function :math:`f(\cdot)` directly
2. The Jacobian matrices :math:`F_k` and :math:`H_k` are computed at each step through automatic differentiation
3. These Jacobians replace the constant matrices :math:`A` and :math:`C` in the covariance updates

The EKF is suitable for systems where the dynamics are:

- Nonlinear but can be approximated well by linearization
- Subject to Gaussian or approximately Gaussian noise

Initialization is similar to the standard Kalman filter:

.. code-block:: python

    from hilo_mpc import EKF

    # Initialize Extended Kalman filter
    ekf = EKF(model)

The same methods and properties as for the :py:class:`Kalman filter <.KalmanFilter>` apply. The key difference 
is that the EKF automatically linearizes the nonlinear model at each estimation step.

Setup and estimation
--------------------

.. code-block:: python

    # Setup the EKF
    ekf.setup()
    
    # Perform estimation (same interface as KF)
    x_estimated = ekf.estimate(x0=x0, y=measurement, u=input, P=P0, Q=Q, R=R)

The EKF handles the linearization internally, so the user interface remains the same as the standard Kalman filter.

-----------------------------------
Unscented Kalman Filter
-----------------------------------
The class :class:`~hilo_mpc.UnscentedKalmanFilter` (alias :class:`~hilo_mpc.UKF`) implements the Unscented 
Kalman Filter, which uses the unscented transformation to handle nonlinearities. Unlike the EKF, the UKF does 
not require explicit linearization and can provide better estimates for highly nonlinear systems.

Mathematical formulation
------------------------
The UKF is also designed for nonlinear systems:

.. math::

    \begin{aligned}
        x_{k+1} &= f(x_k, u_k) + w_k, \\
        y_k &= h(x_k, u_k) + v_k
    \end{aligned}

where :math:`w_k \sim \mathcal{N}(0, Q)` and :math:`v_k \sim \mathcal{N}(0, R)`.

The UKF uses **sigma points** to capture the mean and covariance of the state distribution through nonlinear
transformations. For an :math:`n_x`-dimensional state, the UKF generates :math:`2n_x + 1` sigma points:

.. math::

    \begin{aligned}
        \mathcal{X}_{k-1}^{(0)} &= \hat{x}_{k-1|k-1}, \\
        \mathcal{X}_{k-1}^{(i)} &= \hat{x}_{k-1|k-1} + \left(\sqrt{(n_x + \lambda) P_{k-1|k-1}}\right)_i, \quad i=1,\ldots,n_x, \\
        \mathcal{X}_{k-1}^{(i)} &= \hat{x}_{k-1|k-1} - \left(\sqrt{(n_x + \lambda) P_{k-1|k-1}}\right)_{i-n_x}, \quad i=n_x+1,\ldots,2n_x
    \end{aligned}

where :math:`\lambda = \alpha^2 (n_x + \kappa) - n_x` is a scaling parameter.

**Prediction step**:

1. Propagate sigma points through the dynamics:

.. math::

    \mathcal{X}_{k|k-1}^{(i)} = f(\mathcal{X}_{k-1}^{(i)}, u_k), \quad i=0,\ldots,2n_x

2. Compute predicted mean and covariance:

.. math::

    \begin{aligned}
        \hat{x}_{k|k-1} &= \sum_{i=0}^{2n_x} W_i^{(m)} \mathcal{X}_{k|k-1}^{(i)}, \\
        P_{k|k-1} &= \sum_{i=0}^{2n_x} W_i^{(c)} (\mathcal{X}_{k|k-1}^{(i)} - \hat{x}_{k|k-1})(\mathcal{X}_{k|k-1}^{(i)} - \hat{x}_{k|k-1})^\top + Q
    \end{aligned}

**Update step**:

1. Generate new sigma points and propagate through measurement function:

.. math::

    \mathcal{Y}_{k|k-1}^{(i)} = h(\mathcal{X}_{k|k-1}^{(i)}, u_k), \quad i=0,\ldots,2n_x

2. Compute predicted measurement and innovation covariance:

.. math::

    \begin{aligned}
        \hat{y}_{k|k-1} &= \sum_{i=0}^{2n_x} W_i^{(m)} \mathcal{Y}_{k|k-1}^{(i)}, \\
        P_{yy} &= \sum_{i=0}^{2n_x} W_i^{(c)} (\mathcal{Y}_{k|k-1}^{(i)} - \hat{y}_{k|k-1})(\mathcal{Y}_{k|k-1}^{(i)} - \hat{y}_{k|k-1})^\top + R, \\
        P_{xy} &= \sum_{i=0}^{2n_x} W_i^{(c)} (\mathcal{X}_{k|k-1}^{(i)} - \hat{x}_{k|k-1})(\mathcal{Y}_{k|k-1}^{(i)} - \hat{y}_{k|k-1})^\top
    \end{aligned}

3. Compute Kalman gain and update state:

.. math::

    \begin{aligned}
        K_k &= P_{xy} P_{yy}^{-1}, \\
        \hat{x}_{k|k} &= \hat{x}_{k|k-1} + K_k (y_k - \hat{y}_{k|k-1}), \\
        P_{k|k} &= P_{k|k-1} - K_k P_{yy} K_k^\top
    \end{aligned}

The weights are given by:

.. math::

    \begin{aligned}
        W_0^{(m)} &= \frac{\lambda}{n_x + \lambda}, \\
        W_0^{(c)} &= \frac{\lambda}{n_x + \lambda} + (1 - \alpha^2 + \beta), \\
        W_i^{(m)} &= W_i^{(c)} = \frac{1}{2(n_x + \lambda)}, \quad i=1,\ldots,2n_x
    \end{aligned}

The UKF uses sigma points to capture the mean and covariance of the state distribution through nonlinear
transformations. This approach often provides more accurate estimates than the EKF, especially for systems with 
strong nonlinearities.

Initialization
--------------

.. code-block:: python

    from hilo_mpc import UKF

    # Initialize Unscented Kalman filter
    ukf = UKF(model)

The UKF has additional tuning parameters that control the distribution of sigma points:

- :math:`\alpha` (``ukf.alpha``) - Spread of sigma points around the mean (typically :math:`10^{-3} \leq \alpha \leq 1`)
- :math:`\beta` (``ukf.beta``) - Incorporate prior knowledge of distribution (2 is optimal for Gaussian distributions)
- :math:`\kappa` (``ukf.kappa``) - Secondary scaling parameter (typically 0 or :math:`3-n_x`)

.. code-block:: python

    # Set UKF-specific parameters
    ukf.alpha = 0.001  # Spread of sigma points (typically 0.001 to 1)
    ukf.beta = 2.0     # Incorporate prior knowledge of distribution (2 is optimal for Gaussian)
    ukf.kappa = 0.0    # Secondary scaling parameter (typically 0 or 3-n where n is state dimension)

Setup and estimation
--------------------

.. code-block:: python

    # Setup the UKF
    ukf.setup()
    
    # Perform estimation
    x_estimated = ukf.estimate(x0=x0, y=measurement, u=input, P=P0, Q=Q, R=R)

The UKF provides the same interface as the KF and EKF but uses the unscented transformation internally.

-----------------------------------
Particle Filter
-----------------------------------
The class :class:`~hilo_mpc.ParticleFilter` (alias :class:`~hilo_mpc.PF`) implements a particle filter 
(Sequential Monte Carlo method) for state estimation. The particle filter can handle:

- Highly nonlinear dynamics
- Non-Gaussian noise distributions
- Multimodal state distributions

Mathematical formulation
------------------------
The particle filter represents the probability distribution of the state using a set of :math:`N_s` particles
(samples) :math:`\{x_k^{(i)}, w_k^{(i)}\}_{i=1}^{N_s}`, where :math:`x_k^{(i)}` is the :math:`i`-th particle
and :math:`w_k^{(i)}` is its associated weight.

Consider the nonlinear system:

.. math::

    \begin{aligned}
        x_{k+1} &= f(x_k, u_k) + w_k, \\
        y_k &= h(x_k, u_k) + v_k
    \end{aligned}

The particle filter algorithm consists of the following steps:

**1. Initialization** (:math:`k=0`):

Sample :math:`N_s` particles from the initial distribution:

.. math::

    x_0^{(i)} \sim p(x_0), \quad w_0^{(i)} = \frac{1}{N_s}, \quad i=1,\ldots,N_s

**2. Prediction step**:

Propagate each particle through the system dynamics:

.. math::

    x_{k}^{(i)} = f(x_{k-1}^{(i)}, u_k) + w_k^{(i)}, \quad w_k^{(i)} \sim p(w)

**3. Update step**:

Update particle weights based on the likelihood of the measurement:

.. math::

    \begin{aligned}
        \tilde{w}_k^{(i)} &= w_{k-1}^{(i)} \cdot p(y_k | x_k^{(i)}, u_k), \\
        w_k^{(i)} &= \frac{\tilde{w}_k^{(i)}}{\sum_{j=1}^{N_s} \tilde{w}_k^{(j)}}
    \end{aligned}

where the likelihood is typically:

.. math::

    p(y_k | x_k^{(i)}, u_k) = \frac{1}{\sqrt{(2\pi)^{n_y}|R|}} \exp\left(-\frac{1}{2}\Vert y_k - h(x_k^{(i)}, u_k) \Vert_R^2\right)

**4. State estimate**:

The state estimate is computed as the weighted mean of particles:

.. math::

    \hat{x}_k = \sum_{i=1}^{N_s} w_k^{(i)} x_k^{(i)}

**5. Resampling**:

To prevent particle degeneracy (all weight concentrated on few particles), resampling is performed when the
effective sample size falls below a threshold:

.. math::

    N_{\text{eff}} = \frac{1}{\sum_{i=1}^{N_s} (w_k^{(i)})^2}

Various resampling schemes exist (multinomial, systematic, stratified). After resampling, weights are reset to
:math:`w_k^{(i)} = 1/N_s`.

**Optional enhancements**:

- **Roughening**: Add small random noise to particles after resampling to prevent sample impoverishment
- **Prior editing**: Reject particles that are inconsistent with known constraints before weight update

The particle filter represents the probability distribution of the state using a set of particles (samples),
making it very flexible but more computationally intensive than Kalman-based filters.

Initialization
--------------

.. code-block:: python

    from hilo_mpc import PF

    # Initialize Particle filter
    pf = PF(model, plot_backend='bokeh')

Setting the number of particles
--------------------------------
The number of particles :math:`N_s` affects both accuracy and computational cost:

.. code-block:: python

    # Set number of particles
    pf.sample_size = 1000  # More particles = better accuracy but higher cost

Variants and options
--------------------
The particle filter supports different variants and options:

.. code-block:: python

    # Enable roughening to prevent particle depletion
    pf = PF(model, roughening=True, K=0.2)
    
    # Enable prior editing
    pf = PF(model, prior_editing=True)

Setup and estimation
--------------------

.. code-block:: python

    # Setup the particle filter
    pf.setup()
    
    # Perform estimation
    x_estimated = pf.estimate(y=measurement, u=input, Q=Q, R=R)

The particle filter automatically handles particle propagation, weight computation, and resampling.

-----------------------------------
Choosing the Right Observer
-----------------------------------

The choice of observer depends on several factors:

**System Characteristics:**

- **Linear systems with Gaussian noise**: Use :class:`~hilo_mpc.KalmanFilter` (KF) for optimal performance
- **Mildly nonlinear systems**: Use :class:`~hilo_mpc.ExtendedKalmanFilter` (EKF) for good balance of accuracy and speed
- **Highly nonlinear systems**: Use :class:`~hilo_mpc.UnscentedKalmanFilter` (UKF) or :class:`~hilo_mpc.ParticleFilter` (PF)
- **Non-Gaussian noise or multimodal distributions**: Use :class:`~hilo_mpc.ParticleFilter` (PF)

**Computational Resources:**

- **Low computational cost**: KF (fastest) or EKF
- **Moderate computational cost**: UKF
- **High computational cost acceptable**: PF or :class:`~hilo_mpc.MovingHorizonEstimator` (MHE)

**Estimation Requirements:**

- **Real-time recursive estimation**: Use Kalman-based filters (KF, EKF, UKF) or PF
- **Parameter estimation**: Use :class:`~hilo_mpc.MovingHorizonEstimator` (MHE)
- **Constrained estimation**: Use :class:`~hilo_mpc.MovingHorizonEstimator` (MHE)
- **Best possible accuracy with full measurement history**: Use :class:`~hilo_mpc.MovingHorizonEstimator` (MHE)

**Summary Table:**

.. list-table:: Observer overview
     :header-rows: 1
     :widths: 24 14 18 18 16

     * - Observer Type
         - System Type
         - Computational Cost
         - Handles Constraints
         - Parameter Estimation
     * - Kalman Filter (KF)
         - Linear
         - Very Low
         - No
         - Future
     * - Extended KF (EKF)
         - Nonlinear
         - Low
         - No
         - Future
     * - Unscented KF (UKF)
         - Nonlinear
         - Moderate
         - No
         - Future
     * - Particle Filter (PF)
         - Nonlinear
         - High
         - No
         - Limited
     * - Moving Horizon Est (MHE)
         - Any
         - High
         - Yes
         - Yes

