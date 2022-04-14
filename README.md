HILO-MPC 
=========

HILO-MPC is a toolbox for **easy, flexible and fast development of machine-learning supported optimal control and estimation problems**.
It can be used for model predictive control, moving horizon estimation, Kalman filters and has interfaces to embedded am 
embedded MPC software. 

HILO-MPC can leverage `Tensorflow <https://www.tensorflow.org/>`_ and `PyTorch <https://pytorch.org/>`_
to create machine learning models, and the  `CasADi <https://web.casadi.org/>`_ framework to efficiently
build control and estimation problems. The machine learning models can be used (almost) during in the setup
of the estimation or control problems 

![plot](doc/source/images/overview.png)
The following machine learning models are currently supported:

- Artificial feed-forward neural networks
- Gaussian processes
 
 The following machine learning models are currently under development
 
- Bayesian neural network
- Recurrent neural network

At the moment the following MPC problems can be solved. 

- Reference tracking nonlinear MPC
- Trajectory tracking nonlinear MPC
- Path-following nonlinear MPC
- Economic nonlinear MPC
- Linear MPC

All the nonlinear MPCs support soft constraints, time-varying 
systems, time-varying parameters and can be used to solve minimum-time problems. They work for time-continuous
and time-discrete models, in DAE or ODE form. The Linear MPC works only with discrete-time models. 

Installation
-------------
You can use pip to install HILO-MPC as follows 

``
pip install hilo-mpc
``

Documentation
-------------
The link to the documentation will appear soon.

Citing HILO-MPC
---------------
If you use HILO-MPC for your research, please cite it using the following bibtex entry:

``
    @article{pohlodek2022hilompc,
      title={Flexible development and evaluation of machine-learning-supported optimal control and estimation methods via HILO-MPC},
      author={Pohlodek, Johannes and Morabito, Bruno and Schlauch, Christian, and Zometa, Pablo and Findeisen, Rolf},
      journal={arXiv preprint arXiv:2203.13671},
      year={2022}
    }
``
