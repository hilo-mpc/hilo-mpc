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

To set up an MHE properly you need *at least* to define:

- A horizon length using the :code:`horizon` property.
- A stage cost function (typically using :code:`quad_stage_cost`).
- An arrival cost function (typically using :code:`quad_arrival_cost`).

The MHE can be initialized as follows:

.. code-block:: python

    from hilo_mpc import Model, MHE

    # Initialize MHE with a model
    mhe = MHE(model, plot_backend='bokeh')

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

Setup
-----
Before using the MHE, it must be set up using the :meth:`~hilo_mpc.MovingHorizonEstimator.setup` method:

.. code-block:: python

    mhe.setup(options={'integration_method': 'collocation'})

Non-uniform sampling intervals
-------------------------------
The MHE supports non-uniform sampling intervals, allowing measurements to be taken at irregular time intervals.
This is particularly useful when dealing with multi-rate sensors or event-triggered measurements.

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

To set up the Kalman filter you need an already set up :class:`~hilo_mpc.Model` instance with **linear dynamics**. 
Additionally you might want to supply a plot backend (via the :obj:`plot_backend` keyword argument) in order to 
visualize the estimation results later on. At the moment only `Matplotlib <https://matplotlib.org/>`_ and 
`Bokeh <https://bokeh.org/>`_ are supported for plotting.

The Kalman filter can be initialized as follows:

.. code-block:: python

    from hilo_mpc import KF

    # Initialize Kalman filter
    kf = KF(model, plot_backend='bokeh')

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

The EKF is suitable for systems where the dynamics are:

- Nonlinear but can be approximated well by linearization
- Subject to Gaussian or approximately Gaussian noise

Initialization is similar to the standard Kalman filter:

.. code-block:: python

    from hilo_mpc import EKF

    # Initialize Extended Kalman filter
    ekf = EKF(model, plot_backend='bokeh')

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

The UKF uses sigma points to capture the mean and covariance of the state distribution through nonlinear 
transformations. This approach often provides more accurate estimates than the EKF, especially for systems with 
strong nonlinearities.

Initialization
--------------

.. code-block:: python

    from hilo_mpc import UKF

    # Initialize Unscented Kalman filter
    ukf = UKF(model, plot_backend='bokeh')

The UKF has additional tuning parameters that control the distribution of sigma points:

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
The number of particles affects both accuracy and computational cost:

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

======================== =========== ================= =================== =====================
Observer Type            System Type Computational Cost Handles Constraints Parameter Estimation
======================== =========== ================= =================== =====================
Kalman Filter (KF)       Linear      Very Low          No                  Planned
Extended KF (EKF)        Nonlinear   Low               No                  Planned
Unscented KF (UKF)       Nonlinear   Moderate          No                  Planned
Particle Filter (PF)     Nonlinear   High              No                  Limited
Moving Horizon Est (MHE) Any         High              Yes                 Yes
======================== =========== ================= =================== =====================
