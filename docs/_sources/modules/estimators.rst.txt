.. _observer_module:

=========================
Observer module
=========================
The |project_name| Observer module contains several state (and parameter) observers.
Observer are used to infer states and parameters from measurements. For a more detailed
description of the methods refer to :ref:`the API <observer_autodoc>`.

The Observer module contains the following classes:

- Moving Horizon Estimator (MHE)
- Kalman Filter (KF)
- Extended Kalman Filter (EKF)
- Unscented Kalman Filter (UKF)
- Particle Filter (PF)

-----------------------------------
Moving Horizon Estimator
-----------------------------------

Non-uniform sampling intervals
-------------------------------

Multi-rate measurements
-------------------------

-----------------------------------
Kalman Filter
-----------------------------------
The class :class:`~hilo_mpc.KalmanFilter` (alias :class:`~hilo_mpc.KF`) implements the Kalman filter developed by Rudolf E. Kálmán. To set up the Kalman filter you need an already set up :class:`~hilo_mpc.Model` instance. Additionally you might want to supply a plot backend (via the :obj:`plot_backend` keyword argument) in order to visualize the estimation results later on. At the moment only `Matplotlib <https://matplotlib.org/>`_ and `Bokeh <https://bokeh.org/>`_ are supported for plotting. The Kalman filter can be initialized as follows:

.. code-block:: python

    from hilo_mpc import KF


    # Initialize Kalman filter
    kf = KF(model, plot_backend='bokeh')

Required information, like e.g. the model dynamics or the sampling time, will be automatically extracted from the :class:`~hilo_mpc.Model` instance.

-----------------------------------
Extended Kalman Filter
-----------------------------------

-----------------------------------
Unscented Kalman Filter
-----------------------------------

-----------------------------------
Particle Filter
-----------------------------------
