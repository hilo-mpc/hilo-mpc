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

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, TypeVar, Union
import warnings

import casadi as ca
import numpy as np
from scipy import stats

from .inference import Inference
from .likelihood import Likelihood
from .mean import Mean
from .kernel import Kernel
from ..base import LearningBase
from ...base import Series, TimeSeries, Equations
from ...optimizer import NonlinearProgram
from ....util.machine_learning import Parameter, Hyperparameter, register_hyperparameters
from ....util.util import convert, is_list_like


Numeric = Union[int, float]
Values = Union[Numeric, np.ndarray]
Bound = Union[Numeric, str, Tuple[Numeric], Tuple[Numeric, Numeric]]
Bounds = Dict[str, Union[str, Tuple[Numeric, Numeric]]]
Array = TypeVar('Array', np.ndarray, ca.DM, ca.SX, ca.MX)
Param = TypeVar('Param', bound=Parameter)
Inf = TypeVar('Inf', bound=Inference)
Lik = TypeVar('Lik', bound=Likelihood)
Mu = TypeVar('Mu', bound=Mean)
Cov = TypeVar('Cov', bound=Kernel)


@dataclass
class Data:
    """"""
    values: np.ndarray = np.array([[]])
    SX: ca.SX = ca.SX()
    MX: ca.MX = ca.MX()


class _GPSeries(Series):
    """"""
    def __init__(self, backend: str) -> None:
        super().__init__(backend=backend)

        self._abscissa = 'x'

    def _update_dimensions(self) -> None:
        """

        :return:
        """
        pass


class GaussianProcess(LearningBase):
    """
    Gaussian Process Regression

    :Note: The Gaussian process regressor currently does not use the Cholesky factorization for training and
        prediction. This will be implemented at a later time.

    :param features: names of the features
    :type features: list of strings
    :param labels: names of the labels
    :type labels: list of strings
    :param inference:
    :type inference:
    :param likelihood:
    :type likelihood:
    :param kernel: Specifying the covariance function of the GP. The kernel hyperparameters are optimized during
        training, defaults to :class:`hilo_mpc.core.learning.kernels.SquaredExponential`.
    :type kernel: kernel
    :param noise_variance:
    :type noise_variance:
    :param hyperprior:
    :type hyperprior:
    :param id:
    :type id: str, optional
    :param name: name of the GP
    :type name: str, optional
    :param solver:
    :type solver: str, optional
    :param solver_options: Options to the solver, e.g. ipopt.
    :type solver_options: dict
    :param kwargs:
    """
    def __init__(
            self,
            features: Union[str, list[str]],
            labels: Union[str, list[str]],
            inference: Optional[Union[str, Inf]] = None,
            likelihood: Optional[Union[str, Lik]] = None,
            mean: Optional[Mu] = None,
            kernel: Optional[Cov] = None,
            noise_variance: Numeric = 1.,
            hyperprior: Optional[str] = None,  # Maybe we should be able to access other hyperpriors from here as well
            id: Optional[str] = None,
            name: Optional[str] = None,
            solver: Optional[str] = None,
            solver_options: Optional[dict] = None,  # TODO: Switch to mapping?
            **kwargs
    ) -> None:
        """Constructor method"""
        if not is_list_like(features):
            features = [features]
        if not is_list_like(labels):
            labels = [labels]
        if len(labels) > 1:
            raise ValueError("Training a GP on multiple labels is not supported. Please use 'MultiOutputGP' to train "
                             "GPs on multiple labels.")
        super().__init__(features, labels, id=id, name=name)

        if likelihood is None:
            likelihood = Likelihood.gaussian()
        elif isinstance(likelihood, str):
            name = likelihood.replace("'", "").replace(' ', '_').lower()
            if name == 'gaussian':
                likelihood = Likelihood.gaussian()
            elif name == 'logistic':
                likelihood = Likelihood.logistic()
            elif name == 'laplacian':
                likelihood = Likelihood.laplacian()
            elif name == 'students_t':
                likelihood = Likelihood.students_t()
            else:
                raise ValueError(f"Likelihood '{likelihood}' not recognized")
        self.likelihood = likelihood

        if inference is None:
            inference = Inference.exact()
        elif isinstance(inference, str):
            name = inference.replace(' ', '_').lower()
            if name == 'exact':
                inference = Inference.exact()
            elif name == 'laplace':
                inference = Inference.laplace()
            elif name == 'expectation_propagation':
                inference = Inference.expectation_propagation()
            elif name == 'variational_bayes':
                inference = Inference.variational_bayes()
            elif name == 'kullback_leibler':
                inference = Inference.kullback_leibler()
            else:
                raise ValueError(f"Inference '{inference}' not recognized")
        self.inference = inference

        if mean is None:
            mean = Mean.zero()
        self.mean = mean

        if kernel is None:
            kernel = Kernel.squared_exponential()
        self.kernel = kernel

        hyper_kwargs = {}
        if hyperprior is not None:
            hyper_kwargs['prior'] = hyperprior
        hyperprior_parameters = kwargs.get('hyperprior_parameters')
        if hyperprior_parameters is not None:
            hyper_kwargs['prior_parameters'] = hyperprior_parameters
        bounds = kwargs.get('bounds')
        if bounds is not None:  # pragma: no cover
            variance_bounds = bounds.get('noise_variance')
            if variance_bounds is not None:
                if variance_bounds == 'fixed':
                    hyper_kwargs['fixed'] = True
                else:
                    hyper_kwargs['bounds'] = bounds
        self.noise_variance = Hyperparameter('GP.noise_variance', value=noise_variance, **hyper_kwargs)

        self._hyp_is_log = {'GP.noise_variance': True}
        self._hyp_is_log.update({parameter.name: False for parameter in self.mean.hyperparameters})
        self._hyp_is_log.update({parameter.name: True for parameter in self.kernel.hyperparameters})

        register_hyperparameters(self, [parameter.id for parameter in self.hyperparameters])

        # bounds = self.noise_variance.log_bounds
        # unbounded_noise = bounds[0] == -ca.inf and bounds[1] == ca.inf
        # unconstrained_op = all([not self.kernel.is_bounded(), unbounded_noise])
        unconstrained_op = True

        if solver is None:
            if unconstrained_op:
                solver = 'Newton-CG'
            else:  # pragma: no cover
                # TODO: Test GP for constrained hyperparameters
                solver = 'L-BFGS-B'
        self._solver = solver

        if solver == 'ipopt':
            if solver_options is None:
                solver_options = {'print_time': False, 'ipopt.suppress_all_output': 'yes'}
            # solver_options.update({'ipopt.hessian_approximation': 'limited-memory'})
        else:
            if solver_options is None:
                solver_options = {}
        self._solver_options = solver_options

        self._X_train = Data()
        self._y_train = Data()

        epsilon = kwargs.get('epsilon')
        if epsilon is None:
            epsilon = 1e-8
        self._epsilon = epsilon

        self._where_is_what = {}
        self._log_marginal_likelihood = None
        self._gp_solver = None
        self._gp_args = {}
        self._optimization_stats = {}

    def __str__(self) -> str:
        """String representation method"""
        message = "Gaussian process with \n"
        for attribute, value in self.__dict__.items():
            if attribute[0] != '_':
                message += f"\t {attribute}: {value} \n"
        return message

    def _initialize_solver(self) -> None:
        """

        :return:
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self._gp_solver = NonlinearProgram(solver=self._solver, solver_options=self._solver_options)
        for warning in w:
            if issubclass(warning.category, UserWarning):
                if warning.message.args[0] == 'Plots are disabled, since no backend was selected.':
                    continue
            warnings.warn(warning.message, warning.category)

    def _update_gp_args(
            self,
            name: str,
            value: Values,
            bounds: Optional[Bound]
    ) -> None:
        """

        :param name:
        :param value:
        :param bounds:
        :return:
        """
        if self._gp_solver is not None:
            x_or_p, index = self._where_is_what[name]
            is_slice = isinstance(index, slice)

            if hasattr(value, 'shape'):
                shape = value.shape
                if not shape:
                    n_val = 1
                else:
                    n_dim = len(shape)
                    if n_dim == 1:
                        n_val = shape[0]
                    else:
                        if n_dim == 2:
                            if 1 in shape:
                                n_val = shape[0] * shape[1]
                            else:
                                raise ValueError("Dimension mismatch. Expected 1 dimension, got 2.")
                        else:
                            raise ValueError(f"Dimension mismatch. Expected 1 dimension, got {n_dim}.")
            else:
                n_val = 1
            if (is_slice and n_val != index.stop - index.start) or (is_slice and n_val == 1) or (
                    not is_slice and n_val > 1):
                # TODO: Raise error here?
                msg = f"Dimension of hyperparameter {name} changed. Run the setup() method again"
                warnings.warn(msg)
            else:
                if self._hyp_is_log[name]:
                    if 'variance' in name:
                        self._gp_args[x_or_p][index] = ca.log(value) / 2
                    else:
                        self._gp_args[x_or_p][index] = ca.log(value)
                else:
                    self._gp_args[x_or_p][index] = value

            if bounds is not None:  # pragma: no cover
                if bounds == 'fixed':
                    if n_val > 1:
                        raise ValueError(f"Dimension mismatch between values and bounds. Supplied values have a "
                                         f"dimension bigger than 1.")
                    lb = value
                    ub = value
                elif isinstance(bounds, (int, float)):
                    lb = bounds
                    ub = ca.inf
                elif len(bounds) == 1:
                    lb = bounds[0]
                    ub = ca.inf
                elif len(bounds) == 2:
                    lb = bounds[0]
                    ub = bounds[1]
                    if lb > ub:
                        lb, ub = ub, lb
                else:
                    raise ValueError("Bounds of unsupported type")
                if lb is not None:
                    self._gp_args['lbx'][index] = lb
                if ub is not None:
                    self._gp_args['ubx'][index] = ub

    @property
    def X_train(self) -> Data:
        """
        The tuple contains the training data matrix as ndarray of shape (dimensions_input, number_observations) and
        its respective CasADi SX and MX symbols of same dimensions that are used to define the CasADi functions during
        fitting and prediction. It can be changed by just assigned an ndarray and its setter method will automatically
        adjust the symbols.

        :return:
        """
        return self._X_train

    @X_train.setter
    def X_train(self, value: np.ndarray) -> None:
        shape = value.shape
        if shape[0] != self._n_features:
            raise ValueError(f"Dimension mismatch. Supplied dimension for the features is {shape[0]}, but "
                             f"required dimension is {self._n_features}.")
        X_train_SX = ca.SX.sym('X', *shape)
        X_train_MX = ca.MX.sym('X', *shape)
        self._X_train = Data(values=value, SX=X_train_SX, MX=X_train_MX)

    @property
    def y_train(self) -> Data:
        """
        The tuple contains the training target matrix as ndarray of shape (dimensions_input, number_observations)
        and its respective CasADi SX and MX symbols of same dimensions that are used to define the CasADi functions
        during fitting and prediction. It can be changed by just assigned an ndarray and its setter method will
        automatically adjust the symbols.

        :return:
        """
        return self._y_train

    @y_train.setter
    def y_train(self, value: np.ndarray) -> None:
        shape = value.shape
        if shape[0] != self._n_labels:
            raise ValueError(f"Dimension mismatch. Supplied dimension for the labels is {shape[0]}, but required "
                             f"dimension is {self._n_labels}.")
        y_train_SX = ca.SX.sym('y', *shape)
        y_train_MX = ca.MX.sym('y', *shape)
        self._y_train = Data(values=value, SX=y_train_SX, MX=y_train_MX)

    def set_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Sets the training matrix and its target vector

        :param X:
        :param y:
        :return:
        """
        if self._function is not None:
            warnings.warn("Gaussian process was already executed. Use the fit_model() method again to optimize with "
                          "respect to the newly set training data.")

        new_X_shape = X.shape
        new_y_shape = y.shape
        if new_X_shape[1] != new_y_shape[1]:
            raise ValueError("Number of observations in training matrix and target vector do not match!")
        old_X_shape = self._X_train.values.shape
        old_y_shape = self._y_train.values.shape
        self.X_train = X
        self.y_train = y

        if self._gp_solver is not None:
            if old_X_shape == new_X_shape and old_y_shape == new_y_shape:
                n = new_X_shape[1]
                n *= self._n_features
                self._gp_args['p'][:n] = X.flatten()
                self._gp_args['p'][n:2 * n] = y.flatten()
            else:
                warnings.warn("Dimensions of training data set changed. Please run setup() method again.")

    @property
    def hyperparameters(self) -> list[Param]:
        """
        Includes all hyperparameters of the kernel as well as the sigma_noise hyperparameter

        :return:
        """
        return [self.noise_variance] + self.mean.hyperparameters + self.kernel.hyperparameters

    @property
    def hyperparameter_names(self) -> list[str]:
        """

        :return:
        """
        return [self.noise_variance.name] + self.mean.hyperparameter_names + self.kernel.hyperparameter_names

    def get_hyperparameter_by_name(self, name: str) -> Param:
        """

        :param name:
        :return:
        """
        return [parameter for parameter in self.hyperparameters if parameter.name == name][0]

    def update(self, *args) -> None:
        """

        :param args:
        :return:
        """
        self._update_gp_args(*args)

    def update_hyperparameters(
            self,
            names: Sequence[str],
            values: Optional[Union[Sequence[Optional[Values]], np.ndarray]] = None,
            bounds: Optional[Sequence[Optional[Bounds]]] = None
    ) -> None:
        """
        Updates hyperparameter values and boundaries

        If names of hyperparameters are found in the given dictionaries, their respective value and/or boundaries will
        be updated to the value and/or boundaries defined in the dictionary.

        :Note: Instead of boundary values the string 'fixed' can be passed which flags the hyperparameter, keeping him
            constant during any optimization routine. This flag will be automatically set back if a list of two floats
            is passed.

        :param names:
        :param values:
        :param bounds:
        :return:
        """
        for name in names:
            parameter = self.get_hyperparameter_by_name(name)
            _, index = self._where_is_what[name]
            if values is not None:
                if hasattr(values, 'ndim') and values.ndim != 1:
                    raise ValueError("At the moment only flat NumPy arrays are supported for updating the "
                                     "hyperparameter values.")
                value = values[index]
                if value is not None:
                    if 'variance' in name:
                        value *= 2
                    if not parameter.fixed:
                        if self._hyp_is_log[name]:
                            value = np.exp(value)
                    parameter.value = value
            if bounds is not None:  # pragma: no cover
                bound = bounds[index]
                if bound is not None:
                    parameter.bounds = bound

    def initialize(
            self,
            features: Union[str, list[str]],
            labels: Union[str, list[str]],
            inference: Optional[Union[str, Inf]] = None,
            likelihood: Optional[Union[str, Lik]] = None,
            mean: Optional[Mu] = None,
            kernel: Optional[Cov] = None,
            noise_variance: Numeric = 1.,
            hyperprior: Optional[str] = None,  # Maybe we should be able to access other hyperpriors from here as well
            id: Optional[str] = None,
            name: Optional[str] = None,
            solver: Optional[str] = None,
            solver_options: Optional[dict] = None,  # TODO: Switch to mapping?
            **kwargs
    ) -> None:
        """

        :param features:
        :param labels:
        :param inference:
        :param likelihood:
        :param mean:
        :param kernel:
        :param noise_variance:
        :param hyperprior:
        :param id:
        :param name:
        :param solver:
        :param solver_options:
        :param kwargs:
        :return:
        """
        # NOTE: Maybe we could abuse the __call__-method and check in there whether the class was already instantiated
        #  (https://stackoverflow.com/questions/40457599/how-to-check-if-an-object-has-been-initialized-in-python/40459563,
        #  https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python).
        #  If the object was already instantiated, we execute the normal __call__-method.
        self.__init__(features, labels, inference=inference, likelihood=likelihood, mean=mean, kernel=kernel,
                      noise_variance=noise_variance, hyperprior=hyperprior, id=id, name=name, solver=solver,
                      solver_options=solver_options, **kwargs)

    def setup(self, **kwargs) -> None:
        """

        :param kwargs:
        :return:
        """
        X_sym = self._X_train.SX
        if X_sym.is_empty():
            raise RuntimeError("The training data has not been set. Please run the method set_training_data() to "
                               "proceed.")

        y_sym = self._y_train.SX
        if y_sym.is_empty():
            # NOTE: We shouldn't get here
            raise RuntimeError("The training data has not been set. Please run the method set_training_data() to "
                               "proceed.")

        n, D = X_sym.shape
        X = ca.SX.sym('X', n)

        posterior = self.inference(X_sym, y_sym, X, self.noise_variance.SX, self.likelihood, self.mean, self.kernel)
        log_marginal_likelihood = posterior['log_marginal_likelihood']
        mean = posterior['mean']
        var = posterior['var']

        hyperparameters_to_optimize = [parameter for parameter in self.hyperparameters if not parameter.fixed]
        w = []
        w0 = []
        # lbw = []
        # ubw = []
        k = 0
        for parameter in hyperparameters_to_optimize:
            name = parameter.name
            param = parameter.SX

            hyperprior = parameter.prior
            if hyperprior is not None:
                log_marginal_likelihood += hyperprior(param, log=True)

            w.append(param)
            n_p = param.numel()
            if self._hyp_is_log[name]:
                if 'variance' in name:
                    w0.append(parameter.log / 2)
                else:
                    w0.append(parameter.log)
                # lbw.append(n_p * [parameter.log_bounds[0]])
                # ubw.append(n_p * [parameter.log_bounds[1]])
            else:
                w0.append(parameter.value)
                # lbw.append(n_p * [parameter.bounds[0]])
                # ubw.append(n_p * [parameter.bounds[1]])
            if n_p == 1:
                self._where_is_what[name] = ('x0', k)
            else:
                self._where_is_what[name] = ('x0', slice(k, k + n_p))
            k += n_p

        hyperparameters_fixed = [parameter for parameter in self.hyperparameters if parameter.fixed]
        p = []
        p0 = []
        k = 2 * D
        for parameter in hyperparameters_fixed:
            name = parameter.name
            p.append(parameter.SX)
            n_p = parameter.SX.numel()
            if self._hyp_is_log[name]:
                if 'variance' in name:
                    p0.append(parameter.log / 2)
                else:
                    p0.append(parameter.log)
            else:
                p0.append(parameter.value)
            if n_p == 1:
                self._where_is_what[name] = ('p', k)
            else:
                self._where_is_what[name] = ('p', slice(k, k + n_p))
            k += n_p

        w = ca.vertcat(*w)
        w0 = ca.vertcat(*w0)
        # lbw = ca.vertcat(*lbw)
        # ubw = ca.vertcat(*ubw)
        p = ca.vertcat(X_sym.T[:], y_sym.T[:], *p)
        # TODO: Check if this is actually the same as what is done for the SX variables
        p0 = np.concatenate([self.X_train.values.flatten(), self.y_train.values.flatten(), p0])

        self._initialize_solver()

        # for some reason this doesn't result in free variables
        # self._gp_solver.set_decision_variables(w, lower_bound=lbw, upper_bound=ubw)
        self._gp_solver.set_decision_variables(w)
        self._gp_solver.set_parameters(p)
        self._gp_solver.set_objective(-log_marginal_likelihood)

        self._log_marginal_likelihood = ca.Function('log_marginal_likelihood',
                                                    [w, p],
                                                    [log_marginal_likelihood],
                                                    ['x0', 'p'],
                                                    ['log_marg_lik'])

        self._function = ca.Function(
            'prediction',
            [X, w, p],
            [mean, var],
            ['X', 'x0', 'p'],
            ['mean', 'variance']
        )

        self._gp_solver.setup()

        self._gp_solver.set_initial_guess(w0)
        self._gp_solver.set_parameter_values(p0)

        self._gp_args.update({
            'x0': w0,
            'p': p0,
            # 'lbx': lbw,
            # 'ubx': ubw
        })

    def is_setup(self) -> bool:
        """

        :return:
        """
        if self._gp_solver is not None:
            return True
        else:
            return False

    def log_marginal_likelihood(self) -> float:
        """

        :return:
        """
        return float(self._log_marginal_likelihood(x0=self._gp_args['x0'], p=self._gp_args['p'])['log_marg_lik'])

    def fit_model(self) -> None:
        """
        Optimizes the hyperparameters by minimizing the negative log marginal likelihood.

        The model fit is the training phase in the GP regression. Given the design data (training observations and
        their targets) and suitable start values and bounds for the hyperparameters, the hyperparameters will be
        adapted by minimizing the negative log marginal likelihood.

        :Note: Instead of boundary values the string 'fixed' can be passed which flags the hyperparameter, keeping it
            constant during the optimization routine.

        :return:
        """
        if self._gp_solver is None:
            raise RuntimeError("The GP has not been set up yet. Please run the setup() method before fitting.")

        self._gp_solver.solve()
        solution = self._gp_solver.solution

        names = [parameter.name for parameter in self.hyperparameters if not parameter.fixed]
        values = solution.get_by_id('x:f').full().flatten()
        self.update_hyperparameters(names, values=values)
        self._optimization_stats = self._gp_solver.stats()
        if not self._optimization_stats['success']:  # pragma: no cover
            if self._solver == 'ipopt':
                return_status = self._optimization_stats['return_status']
                if return_status == 'Infeasible_Problem_Detected':
                    message = 'Infeasible problem detected'
                elif return_status == 'Restoration_Failed':
                    message = 'Restoration failed'
                elif return_status == 'Maximum_Iterations_Exceeded':
                    message = 'Maximum iterations exceeded'
                else:
                    raise RuntimeError(f"Unrecognized return status: {return_status}")
            else:
                message = self._optimization_stats['message']
            warnings.warn(f"Fitting of GP didn't terminate successfully\nSolver message: {message}\n"
                          f"Try to use a different solver")

    def predict(self, X_query: Array, noise_free: bool = False) -> (Array, Array):
        """

        :param X_query:
        :param noise_free:
        :return:
        """
        if self._function is None:
            raise RuntimeError("The GP has not been set up yet. Please run the setup() method before predicting.")

        prediction = self._function(X=X_query, x0=self._gp_args['x0'], p=self._gp_args['p'])
        mean, var = prediction['mean'], prediction['variance']
        if not noise_free:
            var += self.noise_variance.value

        if isinstance(X_query, np.ndarray):
            mean = mean.full()
            var = var.full()

        return mean, var

    def predict_quantiles(
            self,
            quantiles: Optional[tuple[Numeric, Numeric]] = None,
            X_query: Optional[np.ndarray] = None,
            mean: Optional[np.ndarray] = None,
            var: Optional[np.ndarray] = None
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """

        :param quantiles:
        :param X_query:
        :param mean:
        :param var:
        :return:
        """
        if quantiles is None:
            quantiles = (2.5, 97.5)

        if X_query is not None:
            mean, var = self.predict(X_query, noise_free=True)
        elif mean is None and var is None:
            return None

        predictive_quantiles = [stats.norm.ppf(q / 100) * np.sqrt(var + float(self.noise_variance.value)) + mean for q
                                in quantiles]

        return predictive_quantiles[0], predictive_quantiles[1]

    def plot(self, X_query: Array, backend: Optional[str] = None, **kwargs) -> None:
        """

        :param X_query:
        :param backend:
        :param kwargs:
        :return:
        """
        X_train = self._X_train.values
        y_train = self._y_train.values
        n_x = self._n_features

        post_mean, _ = self.predict(X_query, noise_free=True)

        solution = _GPSeries(backend=backend)
        if backend is None:
            backend = solution.plot_backend

        names = ['x', 'y']
        vector = {
            'x': {
                'values_or_names': self._features,
                'description': n_x * [''],
                'labels': self._features,
                'units': n_x * [''],
                'shape': (n_x, 0),
                'data_format': ca.DM
            },
            'y': {
                'values_or_names': ['post. mean at test points'],
                'description': [''],
                'labels': self._labels,
                'units': [''],
                'shape': (1, 0),
                'data_format': ca.DM
            }
        }
        solution.setup(*names, **vector)

        solution.set('x', X_query)
        solution.set('y', post_mean)

        plots = self._n_labels * tuple((feature, 'post. mean at test points') for feature in self._features)

        ext_data_x = []
        ext_data_y = []
        for i in range(self._n_labels):
            for j in range(n_x):
                ext_data_x.append({'data': X_train[j, :], 'subplot': i * n_x + j, 'label': 'training points'})
                ext_data_y.append(
                    {
                        'data': y_train[i, :],
                        'subplot': i * n_x + j,
                        'label': 'training points',
                        'kind': 'scatter',
                        'marker': '+',
                        'marker_size': 20 if backend == 'bokeh' else 60
                    }
                )

        plot_kwargs = kwargs.copy()
        if backend == 'bokeh':
            if kwargs.get("output_notebook", False):
                plot_kwargs["figsize"] = kwargs.get("figsize", (300, 300))
            else:
                plot_kwargs["figsize"] = kwargs.get("figsize", (500, 500))
            plot_kwargs["major_label_text_font_size"] = kwargs.get("major_label_text_font_size", "12pt")
            plot_kwargs["axis_label_text_font_size"] = kwargs.get("axis_label_text_font_size", "12pt")
        plot_kwargs['marker'] = kwargs.get('marker', 'o')
        if backend == 'bokeh':
            plot_kwargs['marker_size'] = kwargs.get('marker_size', 5)
        elif backend == 'matplotlib':
            plot_kwargs['marker_size'] = kwargs.get('marker_size', 20)
        plot_kwargs['title'] = kwargs.get('title', n_x * ('', ))
        plot_kwargs['legend'] = kwargs.get('legend', True)
        plot_kwargs['kind'] = 'scatter'

        solution.plot(*plots, x_data=ext_data_x, y_data=ext_data_y, **plot_kwargs)

    def plot_1d(
            self,
            quantiles: Optional[tuple[Numeric, Numeric]] = None,
            resolution: Optional[int] = 200,
            backend: Optional[str] = None,
            **kwargs
    ) -> None:
        """

        :param quantiles:
        :param resolution:
        :param backend:
        :param kwargs:
        :return:
        """
        X_train = self._X_train.values
        y_train = self._y_train.values
        n_x = X_train.shape[0]

        if n_x > 1:
            raise RuntimeError("The method plot_1d is only supported for Gaussian processes with one feature.")

        x_min = X_train.min()
        x_max = X_train.max()
        x_min, x_max = x_min - 0.25 * (x_max - x_min), x_max + 0.25 * (x_max - x_min)
        X = np.linspace([float(x_min)], [float(x_max)], resolution, axis=1)

        y, var = self.predict(X, noise_free=True)
        lb, ub = self.predict_quantiles(quantiles=quantiles, mean=y, var=var)

        solution = _GPSeries(backend=backend)
        if backend is None:
            backend = solution.plot_backend

        names = ['x', 'y']
        vector = {
            'x': {
                'values_or_names': self._features,
                'description': [''],
                'labels': self._features,
                'units': [''],
                'shape': (1, 0),
                'data_format': ca.DM
            },
            'y': {
                'values_or_names': ['mean'],
                'description': [''],
                'labels': self._labels,
                'units': [''],
                'shape': (1, 0),
                'data_format': ca.DM
            }
        }
        solution.setup(*names, **vector)

        solution.set('x', X)
        solution.set('y', y)

        ext_data_x = [{'data': X_train, 'subplot': 0, 'label': 'data'}]
        ext_data_y = [{'data': y_train, 'subplot': 0, 'label': 'data', 'kind': 'scatter'}]

        plot_kwargs = kwargs.copy()
        if backend == 'bokeh':
            if kwargs.get("output_notebook", False):
                plot_kwargs["figsize"] = kwargs.get("figsize", (300, 300))
            else:
                plot_kwargs["figsize"] = kwargs.get("figsize", (500, 500))

            plot_kwargs["line_width"] = kwargs.get("line_width", 2)
            plot_kwargs["major_label_text_font_size"] = kwargs.get("major_label_text_font_size", "12pt")
            plot_kwargs["axis_label_text_font_size"] = kwargs.get("axis_label_text_font_size", "12pt")
        plot_kwargs['color'] = kwargs.get('color', ['#3465a4', 'k'])
        plot_kwargs['marker'] = kwargs.get('marker', 'o')
        if backend == 'bokeh':
            plot_kwargs['marker_size'] = kwargs.get('marker_size', 15)
        elif backend == 'matplotlib':
            plot_kwargs['marker_size'] = kwargs.get('marker_size', 60)
        plot_kwargs['fill_between'] = [
            {'x': self._features[0], 'lb': lb, 'ub': ub, 'label': 'confidence', 'line_color': '#204a87',
             'line_width': .5, 'fill_color': '#729fcf', 'fill_alpha': .2}
        ]
        plot_kwargs['title'] = kwargs.get('title', "GP regression")
        plot_kwargs['legend'] = kwargs.get('legend', True)

        solution.plot((self._features[0], 'mean'), x_data=ext_data_x, y_data=ext_data_y, **plot_kwargs)

    def plot_prediction_error(
            self,
            X_query: Array,
            y_query: Array,
            noise_free: bool = False,
            backend: Optional[str] = None,
            **kwargs
    ) -> None:
        """

        :param X_query:
        :param y_query:
        :param noise_free:
        :param backend:
        :param kwargs:
        :return:
        """
        post_mean, post_var = self.predict(X_query, noise_free=noise_free)
        error = np.absolute(post_mean - y_query)
        post_std = np.sqrt(post_var)
        n_samples = X_query.shape[1]

        solution = TimeSeries(backend=backend)
        if backend is None:
            backend = solution.plot_backend

        names = ['t', 'x']
        vector = {
            't': {
                'values_or_names': ['t'],
                'description': [''],
                'labels': [''],
                'units': [''],
                'shape': (1, 0),
                'data_format': ca.DM
            },
            'x': {
                'values_or_names': ['error', 'post_std'],
                'description': 2 * [''],
                'labels': 2 * [''],
                'units': 2 * [''],
                'shape': (2, 0),
                'data_format': ca.DM
            }
        }
        solution.setup(*names, **vector)

        solution.set('t', np.linspace([0.], [n_samples - 1], n_samples, axis=1))
        solution.set('x', np.append(error, post_std, axis=0))

        plot_kwargs = kwargs.copy()
        plot_kwargs['title'] = ('prediction error', 'prediction function standard deviation')
        plot_kwargs['xlabel'] = ('', '')
        plot_kwargs['ylabel'] = ('', '')
        plot_kwargs['kind'] = 'scatter'
        plot_kwargs['marker'] = 'o'
        plot_kwargs['marker_size'] = 5 if backend == 'bokeh' else 20

        solution.plot(('t', 'error'), ('t', 'post_std'), **plot_kwargs)


class GPArray:
    """"""
    def __init__(self, n_gps: int) -> None:
        """Constructor method"""
        self._n_gps = n_gps
        self._gps = np.empty((self._n_gps, 1), dtype=object)

    __array_ufunc__ = None  # https://stackoverflow.com/questions/38229953/array-and-rmul-operator-in-python-numpy

    def __len__(self) -> int:
        """Length method"""
        return self._n_gps

    def __iter__(self) -> Optional[GaussianProcess]:
        """Item iteration method"""
        for k in range(self._n_gps):
            gp = self._gps[k, 0]
            if gp is None:
                gp = object.__new__(GaussianProcess)
                self._gps[k, 0] = gp
            yield gp

    def __getitem__(self, item: int) -> Optional[GaussianProcess]:
        """Item getter method"""
        return self._gps[item, 0]

    def __rmatmul__(self, other: Array) -> GPArray:
        """Multiplication method (from the right side)"""
        other = convert(other, ca.DM)
        features = []
        labels = ca.SX.zeros(self._n_gps)
        for k in range(self._n_gps):
            gp = self._gps[k, 0]

            query = [ca.SX.sym(k) for k in gp.features]
            out, _ = gp.predict(ca.vertcat(*query))

            features += query
            labels[k] += out
        # NOTE: @-operator doesn't seem to work because we set __array_ufunc__ to None
        equations = Equations(ca.SX, {'matmul': ca.mtimes(other, labels)})
        return equations.to_function(f'{type(GaussianProcess).__name__}_matmul', *features)

    # def append(self, gp: GaussianProcess) -> None:
    #     """
    #
    #     :param gp:
    #     :return:
    #     """
    #
    # def pop(self) -> GaussianProcess:
    #     """
    #
    #     :return:
    #     """
    #
    # def remove(self, item) -> None:
    #     """
    #
    #     :param item:
    #     :return:
    #     """


__all__ = [
    'GaussianProcess',
    'GPArray'
]
