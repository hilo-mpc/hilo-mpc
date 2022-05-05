.. _modelling_module:

=========================
Modelling module
=========================
The main component of the |project_name| Modelling module is the :class:`~hilo_mpc.Model` class. For a more detailed
description of the methods refer to :ref:`the API <modelling_autodoc>`.

Models are used to describe the dynamics of systems. The |project_name| :class:`~hilo_mpc.Model` class supports the following system properties:

* **linear/nonlinear** system dynamics
* **continuous/discrete** time dynamics
* **time-invariant/time-variant** systems

In addition to ordinary differential equations (ODEs), systems with continuous time dynamics can also be modelled using differential algebraic equations (DAEs). Systems with discrete time dynamics can be designed analogously. Optionally, measurement equations mapping the states of a model to some measurable quantity of the system can be supplied as well.

.. note::

    :ref:`Observers <observer_module>` usually require measurement equations to be present. If no measurement equations were supplied during model setup the observers will assume that all states can be measured.

A general **time-invariant** **continuous**-time **nonlinear** system can be described by the following DAEs

.. math::

    \dot{x}(t)&=f(x(t),z(t),u(t_k),p),\\
    0&=g(x(t),z(t),u(t_k),p),\\
    y&=h(x(t),z(t),u(t_k),p),

where :math:`x\in\mathbb{R}^{n_x}` is the state vector, :math:`z\in\mathbb{R}^{n_z}` the vector of algebraic variables, :math:`u\in\mathbb{R}^{n_u}` is the input vector and :math:`p\in\mathbb{R}^{n_p}` is a vector of parameters. Since the system is time-invariant only the states :math:`x`, algebraic variables :math:`z` and the input :math:`u` depend on the time :math:`t`. Due to practical reasons the input :math:`u` is assumed to be piecewise constant (hence the index :math:`k`). The function :math:`f\colon\mathbb{R}^{n_x+n_z+n_u+n_p}\mapsto\mathbb{R}^{n_x}` represents the ODEs of the model and the function :math:`g\colon\mathbb{R}^{n_x+n_z+n_u+n_p}\mapsto\mathbb{R}^{n_z}` depicts the algebraic equations forming a semi-explicit DAE system overall.

.. note::

    The |project_name| :class:`~hilo_mpc.Model` class only supports semi-explicit DAE systems of index 1. The index of 1 indicates that the DAE system can be transformed into a pure ODE system by differentiating the algebraic equations once.

The function :math:`h\colon\mathbb{R}^{n_x+n_z+n_u+n_p}\mapsto\mathbb{R}^{n_y}` describes the measurement equations mapping the states and algebraic variables to some measurable quantities.

In the **time-variant** case the functions :math:`f`, :math:`g` and :math:`h` additionally depend explicitly on the time as well

.. math::

    \dot{x}(t)&=f(x(t),z(t),u(t_k),p,t),\\
    0&=g(x(t),z(t),u(t_k),p,t),\\
    y&=h(x(t),z(t),u(t_k),p,t).

A general **time-invariant** **discrete**-time **nonlinear** system can be described by the following set of equations

.. math::

    x_{k+1}&=\tilde{f}(x_k,z_k,u_k,p),\\
    0&=\tilde{g}(x_k,z_k,u_k,p),\\
    y_k&=h(x_k,z_k,u_k,p),

-----------------------------------
Model Setup
-----------------------------------

-----------------------------------
Running Simulations
-----------------------------------

-----------------------------------
Model Solution
-----------------------------------

-----------------------------------
Model Discretization
-----------------------------------

-----------------------------------
Model Linearization
-----------------------------------
