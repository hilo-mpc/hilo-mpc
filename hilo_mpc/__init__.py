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

import hilo_mpc.modules.dynamic_model.dynamic_model as dyn_mod
import hilo_mpc.modules.controller.mpc as mpc
import hilo_mpc.modules.controller.ocp as ocp
import hilo_mpc.modules.controller.lqr as lqr
import hilo_mpc.modules.controller.pid as pid
import hilo_mpc.modules.estimator.mhe as mhe
import hilo_mpc.modules.estimator.kf as kf
import hilo_mpc.modules.estimator.pf as pf
import hilo_mpc.modules.machine_learning.nn.layer as layer
import hilo_mpc.modules.machine_learning.nn.nn as nn
import hilo_mpc.modules.machine_learning.gp.mean as mean
import hilo_mpc.modules.machine_learning.gp.kernel as kernel
import hilo_mpc.modules.machine_learning.gp.gp as gp
import hilo_mpc.modules.control_loop as cl
import hilo_mpc.modules.optimizer as opti
import hilo_mpc.util.plotting as plotting
import hilo_mpc.util.session as session


Model = dyn_mod.Model
NMPC = mpc.NMPC
LMPC = mpc.LMPC
OptimalControlProblem = ocp.OptimalControlProblem
OCP = OptimalControlProblem
LinearQuadraticRegulator = lqr.LinearQuadraticRegulator
LQR = LinearQuadraticRegulator
PID = pid.PID
MovingHorizonEstimator = mhe.MovingHorizonEstimator
MHE = MovingHorizonEstimator
KalmanFilter = kf.KalmanFilter
KF = KalmanFilter
ExtendedKalmanFilter = kf.ExtendedKalmanFilter
EKF = ExtendedKalmanFilter
UnscentedKalmanFilter = kf.UnscentedKalmanFilter
UKF = UnscentedKalmanFilter
ParticleFilter = pf.ParticleFilter
PF = ParticleFilter
Layer = layer.Layer
Dense = layer.Dense
Dropout = layer.Dropout
ArtificialNeuralNetwork = nn.ArtificialNeuralNetwork
ANN = ArtificialNeuralNetwork
Mean = mean.Mean
ConstantMean = mean.ConstantMean
ZeroMean = mean.ZeroMean
OneMean = mean.OneMean
PolynomialMean = mean.PolynomialMean
LinearMean = mean.LinearMean
Kernel = kernel.Kernel
ConstantKernel = kernel.ConstantKernel
SquaredExponentialKernel = kernel.SquaredExponentialKernel
MaternKernel = kernel.MaternKernel
ExponentialKernel = kernel.ExponentialKernel
Matern32Kernel = kernel.Matern32Kernel
Matern52Kernel = kernel.Matern52Kernel
RationalQuadraticKernel = kernel.RationalQuadraticKernel
PiecewisePolynomialKernel = kernel.PiecewisePolynomialKernel
DotProductKernel = kernel.DotProductKernel
PolynomialKernel = kernel.PolynomialKernel
LinearKernel = kernel.LinearKernel
NeuralNetworkKernel = kernel.NeuralNetworkKernel
PeriodicKernel = kernel.PeriodicKernel
GaussianProcess = gp.GaussianProcess
GP = GaussianProcess
GPArray = gp.GPArray
SimpleControlLoop = cl.SimpleControlLoop
LinearProgram = opti.LinearProgram
LP = LinearProgram
QuadraticProgram = opti.QuadraticProgram
QP = QuadraticProgram
NonlinearProgram = opti.NonlinearProgram
NLP = NonlinearProgram
Session = session.Session

get_plot_backend = plotting.get_plot_backend
set_plot_backend = plotting.set_plot_backend


__all__ = [
    'Model',
    'NMPC',
    'LMPC',
    'OptimalControlProblem',
    'OCP',
    'LinearQuadraticRegulator',
    'LQR',
    'PID',
    'MovingHorizonEstimator',
    'MHE',
    'KalmanFilter',
    'KF',
    'ExtendedKalmanFilter',
    'EKF',
    'UnscentedKalmanFilter',
    'UKF',
    'ParticleFilter',
    'PF',
    'Layer',
    'Dense',
    'Dropout',
    'ArtificialNeuralNetwork',
    'ANN',
    'Mean',
    'ConstantMean',
    'ZeroMean',
    'OneMean',
    'PolynomialMean',
    'LinearMean',
    'Kernel',
    'ConstantKernel',
    'SquaredExponentialKernel',
    'MaternKernel',
    'ExponentialKernel',
    'Matern32Kernel',
    'Matern52Kernel',
    'RationalQuadraticKernel',
    'PiecewisePolynomialKernel',
    'DotProductKernel',
    'PolynomialKernel',
    'LinearKernel',
    'NeuralNetworkKernel',
    'PeriodicKernel',
    'GaussianProcess',
    'GP',
    'GPArray',
    'SimpleControlLoop',
    'LinearProgram',
    'LP',
    'QuadraticProgram',
    'QP',
    'NonlinearProgram',
    'NLP',
    'Session',
    'get_plot_backend',
    'set_plot_backend'
]
