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

# TODO: Typing hints

from typing import TypeVar
import warnings

import casadi as ca
import numpy as np

from .util import check_and_wrap_to_list, check_if_has_duplicates, check_if_list_of_type, check_if_list_of_none, \
    check_if_square, who_am_i


Symbolic = TypeVar('Symbolic', ca.SX, ca.MX)


class GenericCost:
    """Class for generic cost functions"""
    def __init__(self, model, use_sx=True):
        """Constructor method"""
        self._cost = 0
        self._is_set = False
        self._model = model
        self._use_sx = use_sx

        # Number of time-varying references. Use for traj. tracking and path following
        self._n_tv_ref = 0

    def __repr__(self):
        """Representation method"""
        pass

    @property
    def cost(self):
        """

        :return:
        """
        return self._cost

    @cost.setter
    def cost(self, arg):
        # TODO check if the arg has the same SX variables that appear in the model
        if not isinstance(arg, ca.SX) and not isinstance(arg, ca.MX):
            raise TypeError('The cost function must be a casadi SX/MX expression.')

        if not arg.shape[0] == arg.shape[0] == 1:
            raise TypeError('The cost must be a one-dimensional function.')

        # Substitute the measurement equation if the user imputed it
        for i in range(self._model.n_y):
            arg = ca.substitute(arg, self._model._y[i], self._model.meas[i])

        self._is_set = True
        self._cost += arg

    @property
    def n_tv_ref(self):
        """
        Number of time-varying references

        :return:
        """
        return self._n_tv_ref


class QuadraticCost(GenericCost):
    """"""
    def __init__(self, model, use_sx=True):
        """Constructor method"""
        super().__init__(model, use_sx=use_sx)

        self._paths_list = []
        self._trajectories_list = []
        self._references_list = []
        self._input_change_list = []

        self._has_trajectory_following = False
        self._has_path_following = False
        # Number of time-varying references. Use for traj. tracking and path following
        self._n_tv_ref = 0

        # Placeholders for the references, needed for trajectory-tracking problems
        self.ref_placeholder = ca.SX.sym('foo', 0)

        # Placeholders for the inputs changes
        self.input_change_placeholder = []

        # Vector that stores the indices of the inputs that enter the change input cost term
        self._ind_input_changes = []

        # Added variable - used for checking
        # TODO: use them in the different add things
        self._added_states = []

        # Store the name of all the varying trajectories
        self._varying_trajectories_names = []

        # Store the name of all the varying trajectories for which the trajectory has not yet been provided
        self._open_varying_trajectories_names = []

    @staticmethod
    def _check_dimensions(var_list, var_weight, var_ref, type):
        """
        Run some sanity checks on the user supplied values

        :param var_list:
        :param var_weight:
        :param var_ref:
        :param type:
        :return:
        """
        if var_list is not None and var_weight is None:
            raise ValueError(f"You passed the following {type}: {var_list} to the cost function, but I do not have any "
                             f"weights for it/them. Please pass me the weights.")

        if var_list is None and var_weight is not None:
            raise ValueError(f"You the passed the {type} weights {var_weight} but without states names. Please pass me "
                             f"the states names with quad_stage_cost.x.")

        if not check_if_list_of_none([var_list, var_weight]):
            if not len(var_list) == var_weight.shape[0]:
                raise ValueError(f"{type} and weights dimensions must be compatible. The {type} vector you passed me is"
                                 f" {len(var_list)} long while cost {var_weight.shape}.")

        if var_ref is not None:
            if isinstance(var_ref, ca.SX) or isinstance(var_ref, ca.MX):
                var_ref_dim = var_ref.shape[0]
            else:
                var_ref_dim = len(var_ref)

            if not len(var_list) == var_ref_dim:
                raise ValueError(f"{type} and reference dimensions must be compatible. The states vector you passed me "
                                 f"is {len(var_list)} long while cost {var_ref_dim}.")

    @staticmethod
    def _create_weight_matrix(arg, name):
        """

        :param arg:
        :param name:
        :return:
        """
        if isinstance(arg, np.ndarray) or isinstance(arg, ca.DM):
            if check_if_square(arg):
                W = arg
            elif arg.ndim == 1:
                W = np.diag(arg)
            else:
                raise TypeError(f"{name} must be a square matrix,a 1-D array or a list of real numbers.")
        elif isinstance(arg, list):
            W = np.diag(arg)
        elif isinstance(arg, float) or isinstance(arg, int):
            W = np.diag([arg])
        else:
            raise TypeError(f"The {name} must be a list of floats, numpy array or casadi DM.")
        return W

    def _check_options(self, trajectory_tracking, path_following, names, references, type):
        """

        :param trajectory_tracking:
        :param path_following:
        :param names:
        :param references:
        :param type:
        :return:
        """
        if trajectory_tracking and path_following:
            raise TypeError("You cannot have both trajectory tracking and path following activated. Please chose one of"
                            " them.")
        elif trajectory_tracking and not path_following:
            self._n_tv_ref += len(names)
            self._has_trajectory_following = True
        elif path_following and not trajectory_tracking:
            if not isinstance(references, ca.SX) and not isinstance(references, ca.MX):
                raise TypeError(f"If path_following=True you need to pass an casadi SX or MX expression to the "
                                f"{type}_ref  parameter.")
            self._has_path_following = True
        elif not path_following and not trajectory_tracking and (
                isinstance(references, ca.SX) or isinstance(references, ca.MX)):
            raise ValueError(f"You passed an {references.type_name()} expression to the {type}_ref. In this case you "
                             f"need to tell me if this is a path following or trajectory tracking problem. \n Please "
                             f"set either 'path_following=True' or trajectory_tracking=True.")
        if check_if_has_duplicates(self.name_varying_trajectories):
            raise TypeError("Two different varying trajectory for the same states are not allowed.")

    def _add_cost_term(self, var, names, W, ref, path_following, trajectory_tracking, ind, type):
        """

        :param var:
        :param names:
        :param W:
        :param ref:
        :param path_following:
        :param trajectory_tracking:
        :param ind:
        :param type:
        :return:
        """
        ref, _ = self._transform_reference(ref)

        self._check_options(trajectory_tracking, path_following, names, ref, type)

        W = self._create_weight_matrix(W, who_am_i())

        self._check_dimensions(names, W, ref, type)

        if ref is None and not trajectory_tracking:
            if type == 'inputs_change':
                if self._use_sx:
                    u_old = ca.SX.sym('_'.join(names) + '_old', var.shape[0])
                else:
                    u_old = ca.MX.sym('_'.join(names) + '_old', var.shape[0])
                self._cost += ca.mtimes((var - u_old).T, ca.mtimes(W, (var - u_old)))
                self.input_change_placeholder = ca.vertcat(self.input_change_placeholder, u_old)
                self._input_change_list.append({'ref': ref, 'names': names, 'placeholder': u_old, 'ind': ind,
                                                'type': type})
            else:
                # Then is a stabilization problem
                self._cost += ca.mtimes(var.T, ca.mtimes(W, var))
        else:
            # Can be either a reference tracking, trajectory tracking or path following problem
            if path_following:
                # If path following, simply substitute the SX/MX expression of the path inside the cost
                p_ref = ref

                # extract the theta variable
                theta = p_ref[0].dep()

                # The paths are saved for plotting
                self._paths_list.append({'ref': ref, 'names': names, 'ind': ind, 'type': type, 'path_fun': None,
                                         'theta': theta, 'ind_theta': None})
            else:
                # For trajectory tracking and reference tracking problems, put a placeholder.
                # This will be scaled and substituted with the reference inside the MPC
                if trajectory_tracking:
                    if self._use_sx:
                        p_ref = ca.SX.sym('_'.join(names) + 'ref', len(names))
                    else:
                        p_ref = ca.MX.sym('_'.join(names) + 'ref', len(names))
                    self.ref_placeholder = ca.vertcat(self.ref_placeholder, p_ref)

                    self._trajectories_list.append({'ref': ref, 'names': names, 'placeholder': p_ref, 'ind': ind,
                                                    'type': type, 'traj_fun': None})
                else:
                    ref = check_and_wrap_to_list(ref)
                    if self._use_sx:
                        p_ref = ca.SX.sym('_'.join(names) + 'ref', len(names))
                    else:
                        p_ref = ca.MX.sym('_'.join(names) + 'ref', len(names))
                    self._references_list.append({'ref': ref, 'names': names, 'placeholder': p_ref, 'ind': ind,
                                                  'type': type})

            self._cost += ca.mtimes((var - p_ref).T, ca.mtimes(W, (var - p_ref)))

        self._is_set = True

    def _setup(self, x_scale=None, u_scale=None, y_scale=None, path_variables=None, time_variable=None):
        """

        :param x_scale:
        :param u_scale:
        :param y_scale:
        :param path_variables:
        :param time_variable:
        :return:
        """
        for i, ref in enumerate(self._references_list):
            # This is for the reference tracking problem. Scale and substitute the real value of the reference.
            ind = ref['ind']
            type = ref['type']
            if type == 'states':
                scale = x_scale
            elif type == 'measurements':
                scale = y_scale
            elif type == 'inputs':
                scale = u_scale
            else:
                raise TypeError(f"Type {type} no available.")

            self._cost = ca.substitute(self._cost, ref['placeholder'], ref['ref'] / ca.DM(scale)[ind])

        for i, traj in enumerate(self._trajectories_list):
            # This is for the trajectory tracking problem. Just scale, the reference will be substituted later.
            ind = traj['ind']
            type = traj['type']

            # Create casadi functions out of the functions used for trajectory-tracking problems. Necessary for plotting
            if isinstance(traj['ref'], ca.SX) or isinstance(traj['ref'], ca.MX):
                traj['traj_fun'] = ca.Function('traj_fun', [time_variable], [traj['ref']])

            if type == 'states':
                scale = x_scale
            elif type == 'measurements':
                scale = y_scale
            elif type == 'inputs':
                scale = u_scale
            else:
                raise TypeError(f"Type {type} no available.")
            self._cost = ca.substitute(self._cost, traj['placeholder'], traj['placeholder'] / ca.DM(scale)[ind])

        # Create casadi functions out of the functions used for path-following problems. Necessary for plotting
        theta_tot = []
        for path_var in path_variables:
            theta_tot.append(path_var['theta'])
        for path in self._paths_list:
            path['path_fun'] = ca.Function('path_fun', [ca.vertcat(*theta_tot)], [path['ref']])

        for i, clist in enumerate(self._input_change_list):
            names = clist['names']
            for name in names:
                indx = self._model.input_names.index(name)
                if indx not in self._input_change_list:
                    self._ind_input_changes.append(indx)

    @staticmethod
    def _transform_reference(ref):
        """

        :param ref:
        :return:
        """
        if isinstance(ref, float) or isinstance(ref, int):
            # Allow the user to input also float or integer as ref. This can be the case if just one variable is passed
            ref = [ref]
        elif check_if_list_of_type(ref, ca.MX) or check_if_list_of_type(ref, ca.SX):
            ref = ca.vertcat(*ref)

        use_sx = True
        if isinstance(ref, ca.MX):
            use_sx = False

        return ref, use_sx

    def add_states(self, names, weights, ref=None, path_following=False, trajectory_tracking=False):
        """

        :param names:
        :param weights:
        :param ref:
        :param path_following:
        :param trajectory_tracking:
        :return:
        """
        names = check_and_wrap_to_list(names)
        ind_x = []
        for i in names:
            try:
                ind_x.append(self._model.dynamical_state_names.index(i))
            except ValueError:
                raise ValueError(f"The state {i} does not exist. The available states are {self._model._x._names}")

        x = self._model._x[ind_x]
        self._add_cost_term(x, names, weights, ref, path_following, trajectory_tracking, ind_x, 'states')

    def add_measurements(self, names, weights, ref=None, path_following=False, trajectory_tracking=False):
        """

        :param names:
        :param weights:
        :param ref:
        :param path_following:
        :param trajectory_tracking:
        :return:
        """
        names = check_and_wrap_to_list(names)
        ind_y = []
        for i in names:
            try:
                ind_y.append(self._model.measurement_names.index(i))
            except ValueError:
                raise ValueError(f"The measurement {i} does not exist. The available measurements are "
                                 f"{self._model._y._names}")

        y = self._model._y[ind_y]
        # Substitute the measurement variables with the measurement equations. These are function of states,inputs or
        # parameters
        m_eq = ca.substitute(y, y, self._model.meas[ind_y])
        self._add_cost_term(m_eq, names, weights, ref, path_following, trajectory_tracking, ind_y, 'measurements')

    def add_inputs(self, names, weights, ref=None, path_following=False, trajectory_tracking=False):
        """

        :param names:
        :param weights:
        :param ref:
        :param path_following:
        :param trajectory_tracking:
        :return:
        """
        names = check_and_wrap_to_list(names)
        ind_u = []
        for i in names:
            try:
                ind_u.append(self._model._u._names.index(i))
            except ValueError:
                raise ValueError(f"The state {i} does not exist. The available states are {self._model._u._names}")

        u = self._model._u[ind_u]
        self._add_cost_term(u, names, weights, ref, path_following, trajectory_tracking, ind_u, 'inputs')

    def add_inputs_change(self, names, weights):
        """

        :param names:
        :param weights:
        :return:
        """
        ref = None
        path_following = False
        trajectory_tracking = False
        names = check_and_wrap_to_list(names)
        ind_u = []
        for i in names:
            try:
                ind_u.append(self._model._u._names.index(i))
            except ValueError:
                raise ValueError(f"The state {i} does not exist. The available states are {self._model._u._names}")

        u = self._model._u[ind_u]
        self._add_cost_term(u, names, weights, ref, path_following, trajectory_tracking, ind_u, 'inputs_change')

    @property
    def n_of_paths(self):
        """
        Number of paths. It's different from zero if path following is active.

        :return:
        """
        return len(self._paths_list)

    @property
    def n_of_trajectories(self):
        """
        Number of paths. It's different from zero if path following is active.

        :return:
        """
        return len(self._trajectories_list)

    @property
    def name_varying_trajectories(self):
        """

        :return:
        """
        for l in self._trajectories_list:
            self._varying_trajectories_names.extend(l['names'])
        return self._varying_trajectories_names

    @property
    def name_open_varying_trajectories(self):
        """

        :return:
        """
        for l in self._trajectories_list:
            if l['ref'] is None:
                self._open_varying_trajectories_names.extend(l['names'])
        return self._open_varying_trajectories_names

    @property
    def Q(self):
        """

        :return:
        """
        # TODO: generalize this
        warnings.warn("This property is supposed to be used only if muAO-MPC is used to solve the linear MPC problem.")
        return self._Q

    @Q.setter
    def Q(self, arg):
        # TODO: generalize this
        warnings.warn("This setter is supposed to be used only if muAO-MPC is used to solve the linear MPC problem.")
        self._Q = arg

    @property
    def R(self):
        """

        :return:
        """
        # TODO: generalize this
        warnings.warn("This property is supposed to be used only if muAO-MPC is used to solve the linear MPC problem.")
        return self._Q

    @R.setter
    def R(self, arg):
        # TODO: generalize this
        self._R = arg

    @property
    def P(self):
        """

        :return:
        """
        # TODO: generalize this
        warnings.warn("This property is supposed to be used only if muAO-MPC is used to solve the linear MPC problem.")
        return self._P

    @P.setter
    def P(self, arg):
        # TODO: generalize this
        warnings.warn("This setter is supposed to be used only if muAO-MPC is used to solve the linear MPC problem.")
        self._P = arg


class MHEQuadraticCost(GenericCost):
    """"""
    def __init__(self, model, use_sx=True):
        """Constructor method"""
        super().__init__(model, use_sx)

        # List of references that have a fixed values for all MHE iterations
        self._fixed_ref_list = []

        # List of references that change at every time step of the MHE horizon
        self._tv_ref_list = []

        # List of references that change every at every iteration, but not withing the MHE horizon:
        self._iter_ref_list = []

        # Placeholders for the references, needed for trajectory-tracking problems
        self.ref_placeholder = []
        self._has_state_noise = False
        self._w = []
        self._x_guess = None
        self._p_guess = None
        self._n_tv_refs = 0
        self._n_iter_var_refs = 0
        self._name_long_to_short = {'states': 'x', 'inputs': 'u', 'measurements': 'y', 'parameters': 'p',
                                    'state_noise': 'w'}

    @staticmethod
    def _check_dimensions(var_list, var_weight, var_ref, type):
        """
        Run some sanity checks on the user supplied values

        :return:
        """
        if var_list is not None and var_weight is None:
            raise ValueError(
                f"You passed the following {type}: {var_list} to the cost function, "
                f"but I do not have any weights for it/them. "
                f"Please pass me the weights."
            )

        if var_list is None and var_weight is not None:
            raise ValueError(f"You passed the {type} weights {var_weight} but without state names. "
                             f"Please pass me the state names with quad_stage_cost.x.")

        if not check_if_list_of_none([var_list, var_weight]):
            if not len(var_list) == var_weight.shape[0]:
                raise ValueError(f"{type} and weights dimensions must be compatible."
                                 f"The {type} vector you passed me is {len(var_list)} long "
                                 f"while cost {var_weight.shape}.")

        if var_ref is not None:
            if isinstance(var_ref, ca.SX):
                var_ref_dim = var_ref.shape[0]
            else:
                var_ref_dim = len(var_ref)

            if not len(var_list) == var_ref_dim:
                raise ValueError(f"{type} and reference dimensions must be compatible."
                                 f"The states vector you passed me is {len(var_list)} long while cost {var_ref_dim}.")

    @staticmethod
    def _create_weight_matrix(arg, name):
        """

        :param arg:
        :param name:
        :return:
        """
        if isinstance(arg, np.ndarray) or isinstance(arg, ca.DM):
            if check_if_square(arg):
                W = arg
            elif arg.ndim == 1:
                W = np.diag(arg)
            else:
                raise TypeError(f"{name} must be a square matrix, a 1-D array or a list of real numbers.")
        elif isinstance(arg, list):
            W = np.diag(arg)
        elif isinstance(arg, float) or isinstance(arg, int):
            W = np.diag([arg])
        else:
            raise TypeError(f"The {name} must be a list of floats, numpy array or CasADi DM.")
        return W

    def _add_cost_term(self, var, names, W, ref, ind, type, ref_type=None):
        """

        :param var:
        :param names:
        :param W:
        :param ref:
        :param ind:
        :param type:
        :param ref_type:
        :return:
        """
        name = self._name_long_to_short[type]
        if isinstance(ref, float) or isinstance(ref, int):
            # Allow the user to input also float or integer as ref. This can be the case if just one variable is passed
            ref = [ref]

        W = self._create_weight_matrix(W, who_am_i())

        self._check_dimensions(names, W, ref, type)

        if ref is None:
            # Then is a stabilization problem
            self._cost += ca.mtimes(var.T, ca.mtimes(W, var))
        else:
            if ref_type == 'tv_ref':
                # Time varying references. These reference change within the prediction horizon, i.e. they are function
                # of the states and inputs
                p_ref = ca.SX.sym(f'{name}_ref', ref.shape[0])
                self._n_tv_refs += ref.shape[0]
                self._tv_ref_list.append({'ref': ref, 'names': names, 'placeholder': p_ref, 'ind': ind, 'type': type})
            elif ref_type == 'iter_var_ref':
                # Iteration varying references. These reference change only once for every MHE iteration
                ref = check_and_wrap_to_list(ref)
                p_ref = ca.SX.sym(f'{name}_ref', len(ref))
                self._n_iter_var_refs += len(ref)
                self._iter_ref_list.append({'ref': ref, 'names': names, 'placeholder': p_ref, 'ind': ind, 'type': type})
            elif ref_type == 'fixed_ref':
                # Fixed references. The values of these reference do not change within the prediction horizon
                ref = check_and_wrap_to_list(ref)
                p_ref = ca.SX.sym(f'{name}_ref', len(ref))
                self._fixed_ref_list.append(
                    {'ref': ref, 'names': names, 'placeholder': p_ref, 'ind': ind, 'type': type}
                )
            else:
                raise ValueError(f"{ref_type} is not known.")

            self._cost += ca.mtimes((var - p_ref).T, ca.mtimes(W, (var - p_ref)))

        self._is_set = True

    def _setup(self, x_scale=None, w_scale=None, p_scale=None, u_scale=None):
        """
        This scales the references of path following and trajectory tracking.

        :param x_scale:
        :param w_scale:
        :param p_scale:
        :param u_scale:
        :return:
        """
        for fixed_ref in self._fixed_ref_list:
            self._cost = ca.substitute(self._cost, fixed_ref['placeholder'], fixed_ref['ref'])
        self._cost = ca.substitute(self._cost, self._model.x, self._model.x * ca.DM(x_scale))
        if self._model.n_p > 0:
            self._cost = ca.substitute(self._cost, self._model.p, self._model.p * ca.DM(p_scale))
        if self._has_state_noise:
            self._cost = ca.substitute(self._cost, self._w, self._w * ca.DM(w_scale))

    def add_measurements(self, weights, names=None):
        """

        :param weights:
        :param names:
        :return:
        """
        if names is None:
            names = self._model.measurement_names
            ind_y = list(range(self._model.n_y))
        else:
            names = check_and_wrap_to_list(names)
            ind_y = []
            for i in names:
                try:
                    ind_y.append(self._model._y._names.index(i))
                except ValueError:
                    raise ValueError(
                        f"The measurement {i} does not exist. The available states are {self._model._y._names}"
                    )

        y = self._model.y[ind_y]

        # Substitute the measurement variables with the measurement equations.
        # These are function of states,inputs or parameters
        m_eq = ca.substitute(y, y, self._model.meas[ind_y])
        self._add_cost_term(m_eq, names, weights, y, ind_y, 'measurements', 'tv_ref')

    def add_inputs(self, names, weights):
        """

        :param names:
        :param weights:
        :return:
        """
        names = check_and_wrap_to_list(names)
        ind_u = []
        for i in names:
            try:
                ind_u.append(self._model._u._names.index(i))
            except ValueError:
                raise ValueError(
                    f"The state {i} does not exist. The available states are {self._model._u._names}"
                )

        u = self._model.u[ind_u]
        ref = ca.SX.sym('u_meas', len(ind_u))
        self._add_cost_term(u, names, weights, ref, ind_u, 'inputs', 'tv_ref')

    def add_state_noise(self, weights):
        """

        :param weights:
        :return:
        """
        self._w = ca.SX.sym('w', self._model.n_x)
        names = [f'w_{i}' for i in self._model.dynamical_state_names]
        ind_w = list(range(self._model.n_x))
        self._has_state_noise = True
        self._add_cost_term(self._w, names, weights, None, ind_w, 'state_noise')

    def add_states(self, weights, guess):
        """

        :param weights:
        :param guess:
        :return:
        """
        names = self._model.dynamical_state_names
        guess = check_and_wrap_to_list(guess)
        if len(guess) != self._model.n_x:
            raise ValueError(f"The guess must have the same dimension of the model states."
                             f"There are {self._model.n_x} states but guess has {len(guess)} values.")
        ind_x = list(range(self._model.n_x))
        self.x_guess = guess
        self._add_cost_term(self._model.x, names, weights, guess, ind_x, 'states', 'iter_var_ref')

    def add_parameters(self, weights, guess):
        """

        :param weights:
        :param guess:
        :return:
        """
        names = self._model.parameter_names
        ind_p = list(range(self._model.n_p))
        guess = check_and_wrap_to_list(guess)
        if len(guess) != self._model.n_p:
            raise ValueError(f"The guess must have the same dimension of the model parameters."
                             f"There are {self._model.n_p} states but guess has {len(guess)} values.")
        self.p_guess = guess
        self._add_cost_term(self._model.p, names, weights, guess, ind_p, 'parameters', 'iter_var_ref')

    @property
    def w(self):
        """

        :return:
        """
        return self._w

    @property
    def x_guess(self):
        """

        :return:
        """
        return self._x_guess

    @x_guess.setter
    def x_guess(self, arg):
        self._x_guess = arg

    @property
    def p_guess(self):
        """

        :return:
        """
        return self._p_guess

    @p_guess.setter
    def p_guess(self, arg):
        self._p_guess = arg

    @property
    def n_iter_var_refs(self):
        """

        :return:
        """
        return self._n_iter_var_refs


class GenericConstraint:
    """Class for generic constraints"""
    def __init__(self, model, name='constraint'):
        """Constructor method"""
        self._is_soft = False
        self._ub = None
        self._lb = None
        self._function = None
        self._name = name
        self._formatted_name = name.replace(' ', '_')
        self._model = model
        self._is_set = False
        self._size = 0
        self._cost = None
        self._weight = None
        self.e_soft_value = 0
        self._max_violation = ca.inf

    def _check_and_setup(self, x_scale=None, u_scale=None, y_scale=None, p_scale=None):
        """

        :param x_scale:
        :param u_scale:
        :param y_scale:
        :param p_scale:
        :return:
        """
        if self.constraint is not None:
            for i in range(self._model.n_x):
                self.constraint = ca.substitute(self.constraint, self._model.x[i], self._model.x[i] * x_scale[i])

            for i in range(self._model.n_u):
                self.constraint = ca.substitute(self.constraint, self._model.u[i], self._model.u[i] * u_scale[i])

            for i in range(self._model.n_y):
                self.constraint = ca.substitute(self.constraint, self._model.y[i], self._model.y[i] * y_scale[i])

        if self.is_set:
            if self.size != len(self._ub):
                raise ValueError("The dimensions of the terminal constraint function and its upper bound are not "
                                 "compatible. The terminal constraint must have dimension (n x 1) where n is the number"
                                 " of upper bounds.")
            if self.size != len(self._lb):
                raise ValueError("The dimensions of the terminal constraint function and its lower bound are not "
                                 "compatible. The terminal constraint must have dimension (n x 1) where n is the lower "
                                 "of upper bounds.")
            if self.is_soft:
                e = ca.SX.sym('e', self.size)
                self._cost = ca.Function(f'cost_{self._formatted_name}', [e],
                                         [ca.mtimes(ca.mtimes(e.T, self._weight), e)])
        else:
            if self._function is not None:
                if self.is_soft:
                    if self._weight is None:
                        self._weight = np.diag(np.ones(self.size) * 10000)
                    e = ca.SX.sym('e', self.size)
                    self._cost = ca.Function(f'cost_{self._formatted_name}', [e],
                                             [ca.mtimes(ca.mtimes(e.T, self._weight), e)])

                if self.lb is None:
                    self.lb = -np.inf * np.ones(self._size)

                if self.ub is None:
                    self.ub = np.inf * np.ones(self._size)

                self._is_set = True

    @property
    def cost(self):
        """

        :return:
        """
        if self.is_set:
            return self._cost

    @property
    def constraint(self):
        """

        :return:
        """
        return self._function

    @constraint.setter
    def constraint(self, arg):
        if arg is not None:
            if not isinstance(arg, ca.SX) and not isinstance(arg, ca.MX):
                raise TypeError(f"The {self._name} must be of type casadi SX/MX or None.")
        self._function = arg

    @property
    def ub(self):
        """

        :return:
        """
        return self._ub

    @ub.setter
    def ub(self, arg):
        if arg is not None:
            arg = check_and_wrap_to_list(arg)
        self._ub = arg

    @property
    def lb(self):
        """

        :return:
        """
        return self._lb

    @lb.setter
    def lb(self, arg):
        if arg is not None:
            arg = check_and_wrap_to_list(arg)
        self._lb = arg

    @property
    def is_soft(self):
        """

        :return:
        """
        return self._is_soft

    @is_soft.setter
    def is_soft(self, arg):
        if not isinstance(arg, bool):
            raise TypeError("is_soft must be of type bool")
        self._is_soft = arg

    @property
    def max_violation(self):
        """

        :return:
        """
        return self._max_violation

    @max_violation.setter
    def max_violation(self, arg):
        if arg is not None:
            arg = check_and_wrap_to_list(arg)
        self._max_violation = arg

    @property
    def is_set(self):
        """

        :return:
        """
        if self.is_soft:
            if all(v is not None for v in [self._weight, self._function, self._ub, self._lb]):
                return True
            else:
                return False
        else:
            if all(v is not None for v in [self._function, self._ub, self._lb]):
                return True
            else:
                return False

    @property
    def size(self):
        """

        :return:
        """
        if self.constraint is not None:
            self._size = self.constraint.size1()
        return self._size

    @property
    def weight(self):
        """

        :return:
        """
        return self._weight

    @weight.setter
    def weight(self, arg):
        self._weight = arg


EXPLICIT_METHODS = {
    'forward_euler': {
        'A': np.array([[0.]]),
        'b': np.array([1.]),
        'c': np.array([0.])
    },
    'midpoint': {
        'A': np.array([[0., 0.],
                       [.5, 0.]]),
        'b': np.array([0., 1.]),
        'c': np.array([0., .5])
    },
    'heun': {
        '2': {
            'A': np.array([[0., 0.],
                           [1., 0.]]),
            'b': np.array([.5, .5]),
            'c': np.array([0., 1.])
        },
        '3': {
            'A': np.array([[0., 0., 0.],
                           [1 / 3, 0., 0.],
                           [0., 2 / 3, 0.]]),
            'b': np.array([.25, 0., .75]),
            'c': np.array([0., 1 / 3, 2 / 3])
        }
    },
    'ralston': {
        '2': {
            'A': np.array([[0., 0.],
                           [2 / 3, 0.]]),
            'b': np.array([.25, .75]),
            'c': np.array([0., 2 / 3])
        },
        '3': {
            'A': np.array([[0., 0., 0.],
                           [.5, 0., 0.],
                           [0., .75, 0.]]),
            'b': np.array([2 / 9, 1 / 3, 4 / 9]),
            'c': np.array([0., .5, .75])
        },
        '4': {}
    },
    'generic': {
        '2': {},
        '3': {}
    },
    'kutta': {
        'A': np.array([[0., 0., 0.],
                       [.5, 0., 0.],
                       [-1., 2., 0.]]),
        'b': np.array([1 / 6, 2 / 3, 1 / 6]),
        'c': np.array([0., .5, 1.])
    },
    'ssprk3': {
        'A': np.array([[0., 0., 0.],
                       [1., 0., 0.],
                       [.25, .25, 0.]]),
        'b': np.array([1 / 6, 1 / 6, 2 / 3]),
        'c': np.array([0., 1., .5])
    },
    'classic': {
        'A': np.array([[0., 0., 0., 0.],
                       [.5, 0., 0., 0.],
                       [0., .5, 0., 0.],
                       [0., 0., 1., 0.]]),
        'b': np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6]),
        'c': np.array([0., .5, .5, 1.])
    },
    '3_8_rule': {
        'A': np.array([[0., 0., 0., 0.],
                       [1 / 3, 0., 0., 0.],
                       [-1 / 3, 0., 0., 0.],
                       [1., -1., 1., 0.]]),
        'b': np.array([1 / 8, 3 / 8, 3 / 8, 1 / 8]),
        'c': np.array([0., 1 / 3, 2 / 3, 1.])
    }
}


class RungeKutta:
    """"""
    @staticmethod
    def _construct_polynomial_basis(
            degree: int,
            method: str,
            h: Symbolic
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """

        :param degree:
        :param method:
        :return:
        """
        B = np.zeros(degree + 1)
        C = np.zeros((degree + 1, degree + 1))
        D = np.zeros(degree + 1)
        # T = np.zeros(degree + 1)
        T = h.zeros(degree + 1)
        tau = [0.] + ca.collocation_points(degree, method)

        for i in range(degree + 1):
            L = np.poly1d([1.])
            for j in range(degree + 1):
                if j != i:
                    L *= np.poly1d([1., -tau[j]]) / (tau[i] - tau[j])

            D[i] = L(1.)

            Ldot = np.polyder(L)
            for j in range(degree + 1):
                C[i, j] = Ldot(tau[j])

            Lint = np.polyint(L)
            B[i] = Lint(1.)

            T[i] = h * tau[i]

        return B, C, D, T

    @classmethod
    def _collocation(cls, problem, **opts):
        """

        :param problem:
        :param opts:
        :return:
        """
        # TODO: Add support for algebraic variables (DAE)
        ode = problem.get('ode')
        alg = problem.get('alg')
        quad = problem.get('quad')
        meas = problem.get('meas')
        t = problem.get('t')
        x = problem.get('x')
        z = problem.get('z')
        u = problem.get('u')
        p = problem.get('p')
        dt = problem.get('dt')

        if meas is not None:
            if not isinstance(meas, (ca.SX, ca.MX)):
                # TODO: What is this supposed to do?
                if not meas.is_empty():
                    meas = ca.Function('meas', [x, z, u, p], [meas])
                else:
                    meas = ca.SX()

        degree = opts.get('degree', None)
        collocation_points = opts.get('collocation_points', None)
        if degree is None:
            degree = 2
        if collocation_points is None:
            collocation_points = 'radau'
        n_x = x.size1()
        n_z = z.size1()
        function = ca.Function('function', [t, x, z, u, p], [ode, alg, quad])

        B, C, D, T = cls._construct_polynomial_basis(degree, collocation_points, dt)  # h instead of dt

        J = 0.
        ce = []

        xc = []
        zc = []  # not sure about this
        for k in range(degree):
            xk = x.sym('x_' + str(k), n_x)
            xc.append(xk)
            zk = z.sym('z_' + str(k), n_z)  # not sure about this
            zc.append(zk)  # not sure about this

        xf = D[0] * x
        zf = 0.  # not sure about this
        for i in range(1, degree + 1):
            xp = C[0, i] * x
            for j in range(degree):
                xp += C[j + 1, i] * xc[j]

            fi, gi, qi = function(t + T[i - 1], xc[i - 1], zc[i - 1], u, p)  # not sure about this
            # (zc[i - 1] instead of z)
            ce.append(dt * fi - xp)  # h instead of dt
            ce.append(gi)  # not sure about this

            xf += D[i] * xc[i - 1]
            zf += D[i] * zc[i - 1]  # not sure about this

            J += B[i] * qi * dt  # h instead of dt

        # xc = ca.vertcat(*xc)
        # zc = ca.vertcat(*zc)
        ce = ca.vertcat(*ce)

        if meas is not None:
            if isinstance(meas, ca.Function):
                meas = meas(xf, zf, u, p)  # not sure about this (zf instead of z)

        problem['ode'] = xf
        problem['alg'] = zf  # not sure about this (zf instead of alg)
        problem['quad'] = J
        problem['meas'] = meas
        problem['collocation_points_ode'] = xc
        problem['collocation_points_alg'] = zc  # not sure about this
        problem['collocation_equations'] = ce

    @classmethod
    def _explicit(cls, problem, **opts):
        """

        :param problem:
        :param opts:
        :return:
        """
        # TODO: Integrate algebraic variables
        # TODO: Deal with None values
        ode = problem.get('ode')
        alg = problem.get('alg')
        quad = problem.get('quad')
        h = problem.get('dt')
        t = problem.get('t')
        x = problem.get('x')
        z = problem.get('z')
        u = problem.get('u')
        p = problem.get('p')

        order = opts.get('order')
        if order is None:
            order = 1
        dyn = ca.Function('dyn', [t, x, z, u, p], [ode, quad])
        alg = ca.Function('alg', [t, x, z, u, p], [alg])
        butcher_tableau = opts.get('butcher_tableau', None)
        if butcher_tableau is None:
            if order == 1:
                butcher_tableau = 'forward_euler'
            elif order == 2:
                butcher_tableau = 'midpoint'
            elif order == 3:
                butcher_tableau = 'kutta'
            elif order == 4:
                butcher_tableau = 'classic'
            else:
                raise NotImplementedError(f"Explicit Runge-Kutta discretization for order {order} is not yet "
                                          f"implemented.")
        if butcher_tableau in ['heun', 'ralston', 'generic']:
            butcher_tableau = EXPLICIT_METHODS[butcher_tableau][str(order)]
        else:
            butcher_tableau = EXPLICIT_METHODS[butcher_tableau]
        A = butcher_tableau['A']
        b = butcher_tableau['b']
        c = butcher_tableau['c']
        k = []
        ka = []
        kq = []
        Z = z.sym('Z', (z.size1(), order))
        for i in range(order):
            ki = 0
            for j in range(i):
                ki += A[i, j] * k[j]
            dk, dkq = dyn(t + h * c[i], x + h * ki, Z[:, i], u, p)
            k.append(dk)
            ka.append(alg(t + h * c[i], dk, Z[:, i], u, p))
            kq.append(dkq)

        ode = x
        quad = 0
        for i in range(order):
            ode += h * b[i] * k[i]
            quad += h * b[i] * kq[i]
        alg = ca.vertcat(*ka)

        problem['discretization_points'] = Z[:]
        problem['ode'] = ode
        problem['alg'] = alg
        problem['quad'] = quad

    @classmethod
    def discretize(cls, problem, **opts):
        """

        :param problem:
        :param opts:
        :return:
        """
        class_ = opts.pop('class', 'explicit')
        method = opts.pop('method', 'rk')

        if class_ == 'explicit':
            if method == 'rk':
                cls._explicit(problem, **opts)
        elif class_ == 'implicit':
            if method == 'collocation':
                cls._collocation(problem, **opts)
        else:
            raise ValueError(f"Runge-Kutta class {class_} not recognized")


def continuous2discrete(problem, category='runge-kutta', **opts):
    """

    :param problem:
    :param category:
    :param opts:
    :return:
    """
    # TODO: Maybe add inplace feature? (If inplace=False, then a new dictionary will be returned)
    if category == 'runge-kutta':
        RungeKutta.discretize(problem, **opts)
