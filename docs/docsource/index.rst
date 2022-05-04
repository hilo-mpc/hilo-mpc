.. HILO-MPC documentation master file, created by
   sphinx-quickstart on Wed Apr 14 16:52:41 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HILO-MPC Documentation
===================================

HILO-MPC is a toolbox for **easy, flexible and fast development of machine-learning-supported optimal control and
estimation problems**.

This tool can leverage `Tensorflow <https://www.tensorflow.org/>`_ and `PyTorch <https://pytorch.org/>`_ to create
machine learning models, and the `CasADi <https://web.casadi.org/>`_ framework to efficiently build control and
estimation problems.

.. image:: images/overview.svg

At the moment HILO-MPC supports:

1. Control

   1. Nonlinear and linear model predictive control
   2. Model predictive control for path following
   3. Model predictive control for trajectory tracking
   4. PID controller and linear quadratic regulator

2. Machine Learning

   1. Artificial neural networks
   2. Gaussian processes

3. Estimation

   1. Moving horizon estimation
   2. Kalman filter (including nonlinear extensions)
   3. Particle filter

4. Modeling

   1. Ordinary differential equations
   2. Differential algebraic equations

5. Embedded

   1. :math:`\mu\text{AO-MPC}` (code generation software for linear model predictive control)

On the pipeline we have:

1. Control

   1. Mixed-integer linear model predictive control (work in progress)
   2. Tube model predictive control (work in progress)
   3. Multi-mode model predictive control

2. Machine Learning

   1. Recurrent neural networks
   2. Physics-informed training of neural networks
   3. Reinforcement learning

3. Estimation

   1. Multi-rate moving horizon estimation

4. Embedded

   1. SAM (solver for Al'brekht's method)


.. toctree::
   :maxdepth: 1
   :caption: Quick start:

   installation/installation
   license
   citation
   about

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   .. examples/nmpc_hybrid_bio
   .. examples/mhe_chemical_reaction
   .. examples/mpc_formula1
   .. examples/mpc_pendulum
   .. examples/path_following_mpc
   .. examples/greenhouse

   .. examples/CSTR_Example
   .. examples/gp_mpc_linear_mass_spring_damper
   .. examples/learn_mpc


.. toctree::
   :maxdepth: 5
   :caption: Modules:

   .. modules/controllers
   .. modules/machinelearnings
   .. modules/modellings
   .. modules/observers


.. toctree::
   :maxdepth: 5
   :caption: API:

   .. api/controllers
   .. api/learning
   .. api/modelling
   .. api/observers


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. some definitions

.. This creates a python inline coder highliter environment
.. role:: python(code)
   :language: python
