HILO-MPC 
========

[![python](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9-informational)](https://www.python.org/downloads/)
[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](https://hilo-mpc.github.io/hilo-mpc/)
[![tests](https://github.com/hilo-mpc/hilo-mpc/workflows/Tests/badge.svg)](https://github.com/hilo-mpc/hilo-mpc/actions?query=workflow%3ATests)
<!--[![codecov](https://codecov.io/gh/hilo-mpc/hilo-mpc/branch/master/graph/badge.svg?token=7U83P1M0H4)](https://codecov.io/gh/hilo-mpc/hilo-mpc)-->
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/hilo-mpc/hilo-mpc.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/hilo-mpc/hilo-mpc/context:python)
[![doi](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2203.13671-informational)](https://doi.org/10.48550/arXiv.2203.13671)
[![Github license](https://img.shields.io/github/license/hilo-mpc/hilo-mpc.svg)](https://github.com/hilo-mpc/hilo-mpc/blob/master/LICENSE)
[![Github release](https://img.shields.io/github/release/hilo-mpc/hilo-mpc.svg)](https://GitHub.com/hilo-mpc/hilo-mpc/releases/)

HILO-MPC is a Python toolbox for **easy, flexible and fast realization of machine-learning-supported optimal control, and 
estimation problems** developed mainly at the [Control and Cyber-Physical Systems Laboratory, TU Darmstadt](https://www.ccps.tu-darmstadt.de), and the [Laboratory for Systems Theory and Control, Otto von Guericke University](http://ifatwww.et.uni-magdeburg.de/syst/). It can be used for model predictive control, moving horizon estimation, Kalman filters, solving optimal control problems and has interfaces to embedded model predictive control tools.

HILO-MPC can interface directly to [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/)
to create machine learning models and the [CasADi](https://web.casadi.org/) framework to efficiently
build control and estimation problems. The machine learning models can be used (almost) everywhere in the setup
of these problems. 

![plot](docs/docsource/images/overview.svg)

Currently the following machine learning models are supported:

- Feedforward neural networks
- Gaussian processes
 
The following machine learning models are currently under development:
 
- Bayesian neural network
- Recurrent neural network

At the moment the following MPC and optimal control problems can be solved:

- Reference tracking nonlinear MPC
- Trajectory tracking nonlinear MPC
- Path following nonlinear MPC
- Economic nonlinear MPC
- Linear MPC
- Traditional optimal control problems

All the nonlinear MPCs support soft constraints, time-variant systems, time-varying parameters and can be used to solve 
minimum-time problems. They work for continuous-time and discrete-time models, in DAE or ODE form. Linear MPC is currently limited towards discrete-time models. 

A rich set of [examples](https://github.com/hilo-mpc/examples) is available, spanning:
- NMPC for bioreactors using hybrid first principle and learned models
- Trajectory tracking and path following model predictive control with learning and obstacle avoidance
- Output feedback MPC of a continuous stirred tank reactor with a Gaussian process prediction model
- Learning NMPC control using a neural network
- Simple LQR, PID 
- Moving horizon estimation, extended Kalman filter, unscented Kalman filter, and particle filter for a continuous stirred tank reactor

Installation
------------
You can use pip to install HILO-MPC as follows 

```shell
pip install hilo-mpc
```

Additional Packages
-------------------
If you want to make use of the complete functionality of the toolbox, you may want to install one of the following 
packages

| Package                                          | Version          | Usage                                          |
|--------------------------------------------------|------------------|------------------------------------------------|
| [TensorFlow](https://www.tensorflow.org)         | \>=2.3.0, <2.8.0 | Training of neural networks                    |
| [PyTorch](https://pytorch.org)                   | \>=1.2.0         | Training of neural networks                    |
| [scikit-learn](https://scikit-learn.org/stable/) | \>=0.19.2        | Normalizing of training data                   |
| [Bokeh](https://bokeh.org)                       | \>=2.3.0         | Plotting                                       |
| [Matplotlib](https://matplotlib.org)             | \>=3.0.0         | Plotting                                       |
| [pandas](https://pandas.pydata.org)              | \>=1.0.0         | Providing data for training of neural networks |

Documentation
-------------
A preliminary documentation can be found [here](https://hilo-mpc.github.io/hilo-mpc/). Note that this documentation is
not complete and will be updated over time.

Citing HILO-MPC
---------------
If you use HILO-MPC for your research, please cite the following publication:

* J. Pohlodek, B. Morabito, C. Schlauch, P. Zometa, R. Findeisen. **[Flexible development and evaluation of 
machine-learning-supported optimal control and estimation methods via HILO-MPC](https://arxiv.org/abs/2203.13671)**. 
arXiv. 2022.

```
@misc{pohlodek2022hilompc,
    title = {Flexible development and evaluation of machine-learning-supported optimal control and estimation methods via {HILO-MPC}},
    author = {Pohlodek, Johannes and Morabito, Bruno and Schlauch, Christian and Zometa, Pablo and Findeisen, Rolf},
    publisher = {arXiv},
    year = {2022},
    doi = {10.48550/ARXIV.2203.13671}
}
```
