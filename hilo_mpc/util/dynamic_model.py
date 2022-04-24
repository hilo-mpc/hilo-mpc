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

import warnings

import casadi as ca
import numpy as np

from .util import check_and_wrap_to_list, check_if_has_duplicates, check_if_list_of_type, check_if_list_of_none, \
    check_if_square, who_am_i


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
