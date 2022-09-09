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

from copy import deepcopy
import time
import warnings

import casadi as ca
import casadi.tools as catools
import numpy as np

from .base import Controller
from ..optimizer import DynamicOptimization
from ...util.modeling import GenericCost, QuadraticCost, GenericConstraint, continuous2discrete
from ...util.optimizer import IpoptDebugger
from ...util.util import check_and_wrap_to_list, check_and_wrap_to_DM, scale_vector


class NMPC(Controller, DynamicOptimization):
    """Class for Nonlinear Model Predictive Control"""
    def __init__(self, model, id=None, name=None, plot_backend=None, use_sx=True, stats=False):
        """Constructor method"""
        # TODO: when discrete_u or discrete_x is given to the opts structure, but NMPC is used, raise an error saying
        #  that MINMPC should be used
        super().__init__(model, id=id, name=name, plot_backend=plot_backend, stats=stats, use_sx=use_sx)

        self._may_term_flag = False
        self._lag_term_flag = False
        self._prediction_horizon_is_set = False
        self._control_horizon_is_set = False
        self._change_input_term_flag = False
        self._mixed_integer_flag = False
        self._trajectory_following_flag = False
        self._mpc_options_is_set = False
        # self._tau = ca.SX.sym('tau', 0)
        self._paths_var_list = []
        self._time = 0

        # Initialize the costs
        self.stage_cost = GenericCost(self._model, use_sx=use_sx)
        self.terminal_cost = GenericCost(self._model, use_sx=use_sx)
        self.quad_stage_cost = QuadraticCost(self._model, use_sx=use_sx)
        self.quad_terminal_cost = QuadraticCost(self._model, use_sx=use_sx)
        self._time_var = []

        # Initialize generic stage and terminal constraints
        self.stage_constraint = GenericConstraint(self._model)
        self.terminal_constraint = GenericConstraint(self._model)

        self._lag_term = 0
        self._may_term = 0

        self._minimize_final_time_flag = False
        self._time_varying_parameters_ind = []
        self._x_ub = []
        self._x_lb = []
        self._u_ub = []
        self._u_lb = []
        self.time_varying_parameters = []
        self._x_scaling = []
        self._u_scaling = []
        self._y_scaling = []
        # self._sampling_interval = []
        self._prediction_horizon = []
        self._control_horizon = []

        self._e_soft_stage_ind = []
        self._e_term_stage_ind = []

        # In some cases, for example path following with interpolated path using spline, the objective
        # function needs to be defined in MX variable instead of SX, because the CasADi spline interpolation
        # is defined only in MX functions.
        self._use_sx = use_sx

        # Save dimensions of the model. This can be useful because the MPC can change these later, but we still need
        # them
        self._n_x = deepcopy(model.n_x)
        self._n_u = deepcopy(model.n_u)
        self._n_y = deepcopy(model.n_y)
        self._n_z = deepcopy(model.n_z)
        self._n_p = deepcopy(model.n_p)

    def __str__(self):
        """String representation method"""
        if not self._nlp_setup_done:
            raise ValueError(f'Howdy! You need to setup {self.__class__.__name__} before printing. '
                             f'Run {self.__class__.__name__}.setup().')

        from prettytable import PrettyTable
        import textwrap

        x_box_const = PrettyTable()
        x_box_const.field_names = ["State", "LB", "UB"]
        for k, name in enumerate(self._model.dynamical_state_names):
            x_box_const.add_row([name, self._x_lb[k], self._x_ub[k]])

        u_box_const = PrettyTable()
        u_box_const.field_names = ["Input", "LB", "UB"]
        for k, name in enumerate(self._model.input_names):
            u_box_const.add_row([name, self._u_lb[k], self._u_ub[k]])

        args = "=================================================================\n"
        args += "             Nonlinear Model Predictive Control                 \n"
        if self._id is not None:
            args += f"id='{self._id}'"
        if self.name is not None:
            args += f", name='{self.name}'"
        args += "-----------------------------------------------------------------\n"
        args += f"Prediction horizon: {self.prediction_horizon} \n"
        args += f"Control horizon: {self.control_horizon} \n"
        args += f"Model name: {self._model.name} \n"
        args += f"Sampling interval: {self.sampling_interval}\n"
        if self.quad_stage_cost.n_of_paths > 0 or self.quad_terminal_cost.n_of_paths > 0:
            args += f"Path following problem\n"
        if self.quad_stage_cost.n_of_trajectories > 0 or self.quad_terminal_cost.n_of_trajectories > 0:
            args += f"Trajectory tracking problem\n"

        args += "\n-----------------------------------------------------------------\n"
        args += "Objective function"
        args += "\n-----------------------------------------------------------------\n"
        args += "Lagrange (stage) cost:\n"
        args += textwrap.fill(f"{self._lag_term}", width=80, subsequent_indent='\t\t...')
        args += "\n"
        args += "Mayor (terminal) cost:\n"
        args += textwrap.fill(f"{self._may_term}", width=80, subsequent_indent='\t\t...')

        args += "\n-----------------------------------------------------------------\n"
        args += "Box constraints"
        args += "\n-----------------------------------------------------------------\n"
        args += "States \n"
        args += x_box_const.get_string()
        args += "\n"
        args += "Inputs \n"
        args += u_box_const.get_string()

        if self.quad_terminal_cost._has_trajectory_following:
            tf_table = PrettyTable()
            tf_table.field_names = ["Variable", "Trajectory"]
            args += "\t Trajectory following active:"
            for element in self.quad_stage_cost._trajectories_list:
                for k, name in enumerate(element['names']):
                    tf_table.add_row([name, element['ref'][k]])
            args += tf_table.get_string()
            args += "\n"

        args += "\n-----------------------------------------------------------------\n"
        args += "Initial guesses"
        args += "\n-----------------------------------------------------------------\n"

        x_guess_table = PrettyTable()
        x_guess_table.field_names = ["State", "Guess"]
        for k, name in enumerate(self._model.dynamical_state_names):
            x_guess_table.add_row([name, self._x_guess[k]])
        args += "States \n"
        args += x_guess_table.get_string()
        args += "\n"

        u_guess_table = PrettyTable()
        u_guess_table.field_names = ["Input", "Guess"]
        for k, name in enumerate(self._model.input_names):
            u_guess_table.add_row([name, self._u_guess[k]])
        args += "Inputs \n"
        args += u_guess_table.get_string()
        args += "\n"

        args += "\n-----------------------------------------------------------------\n"
        args += "Numerical solution and optimization stats"
        args += "\n-----------------------------------------------------------------\n"
        args += f"Integration: {self._nlp_options['integration_method']} method. \n"
        args += f"NLP solver: {self._solver_name}. \n"
        args += f"Number of optimization variables: {self._n_v}. \n"
        args += "\n=================================================================\n"

        return args

    def _update_type(self) -> None:
        """

        :return:
        """
        self._type = 'NMPC'

    def _define_cost_terms(self):
        """
        Creates the SX expressions for the stage or terminal cost

        :return:
        """
        if self.stage_cost._is_set:
            self._lag_term += self.stage_cost.cost
            self._lag_term_flag = True
        if self.terminal_cost._is_set:
            self._may_term += self.terminal_cost.cost
            self._may_term_flag = True

        self.quad_stage_cost._setup(x_scale=self._x_scaling, u_scale=self._u_scaling, y_scale=self._y_scaling,
                                    time_variable=self.time_var, path_variables=self._paths_var_list)
        if self.quad_stage_cost._is_set:
            self._lag_term += self.quad_stage_cost.cost
            self._lag_term_flag = True

        self.quad_terminal_cost._setup(x_scale=self._x_scaling, u_scale=self._u_scaling, y_scale=self._y_scaling,
                                       time_variable=self.time_var, path_variables=self._paths_var_list)
        if self.quad_terminal_cost._is_set:
            self._may_term += self.quad_terminal_cost.cost
            self._may_term_flag = True

        if self.quad_stage_cost._has_trajectory_following:
            for i in range(self.quad_stage_cost.n_of_trajectories):
                fun_sx = self.quad_stage_cost._trajectories_list[i]['ref']
                if fun_sx is not None:
                    self._lag_term = ca.substitute(self._lag_term,
                                                   self.quad_stage_cost._trajectories_list[i]['placeholder'],
                                                   self.quad_stage_cost._trajectories_list[i]['ref'])

        if self.quad_terminal_cost._has_trajectory_following:
            for i in range(self.quad_terminal_cost.n_of_trajectories):
                fun_sx = self.quad_terminal_cost._trajectories_list[i]['ref']
                if fun_sx is not None:
                    self._may_term = ca.substitute(self._may_term,
                                                   self.quad_terminal_cost._trajectories_list[i]['placeholder'],
                                                   self.quad_terminal_cost._trajectories_list[i]['ref'])

    def _scale_problem(self):
        """

        :return:
        """
        self._u_ub = scale_vector(self._u_ub, self._u_scaling)
        self._u_lb = scale_vector(self._u_lb, self._u_scaling)
        self._u_guess = scale_vector(self._u_guess, self._u_scaling)

        self._x_ub = scale_vector(self._x_ub, self._x_scaling)
        self._x_lb = scale_vector(self._x_lb, self._x_scaling)
        self._x_guess = scale_vector(self._x_guess, self._x_scaling)

        # ... ode ...
        self._model.scale(self._u_scaling, id='u')
        self._model.scale(self._x_scaling, id='x')

    def _check_mpc_is_well_posed(self):
        """

        :return:
        """
        if self._lag_term_flag is False and self._may_term_flag is False and self._minimize_final_time_flag is False:
            raise ValueError("You need to define a cost function before setting up the mpc.")
        if not self._prediction_horizon_is_set:
            raise ValueError("You must set a prediction horizon length before")
        if not self._control_horizon_is_set:
            raise ValueError("You must set a control horizon length before.")

        # Check tvp and initialize horizon of tvp values
        if self._time_varying_parameters_values is not None:
            self._time_varying_parameters_horizon = ca.DM.zeros((self._n_tvp), self.prediction_horizon)
            tvp_counter = 0
            for key, value in self._time_varying_parameters_values.items():
                if len(value) < self.prediction_horizon:
                    raise TypeError(
                        f"When passing time-varying parameters, you need to pass a number of values at least "
                        f"as long as the prediction horizon. The parameter {key} has {len(value)} values but the MPC "
                        f"has a prediction horizon length of {self._prediction_horizon}."
                    )

                value = self._time_varying_parameters_values[key]
                self._time_varying_parameters_horizon[tvp_counter, :] = value[0:self._prediction_horizon]
                tvp_counter += 1

    def _get_tvp_parameters_values(self, tvp):
        """
        Given the current iteration, returns the value for every sampling time of all time-varying parameter

        :return:
        """
        ci = self._n_iterations

        # Shift the horizon of tvp one step back
        if ci > 0:
            self._time_varying_parameters_horizon[:, 0:-1] = self._time_varying_parameters_horizon[:, 1:]
            for k, name in enumerate(self._time_varying_parameters):
                value = self._time_varying_parameters_values[name]

                if ci + self.prediction_horizon > len(value):

                    warnings.warn("The prediction horizon is predicting outside the values of the time varying "
                                  "parameters. I am now taking looping back the values and start from there. "
                                  "See documentation for info.")

                    n_of_overtakes = int(np.floor(ci / self.prediction_horizon))

                    self._time_varying_parameters_horizon[k, -1] = value[
                        ci - self.prediction_horizon * n_of_overtakes]

                else:
                    self._time_varying_parameters_horizon[k, -1] = value[ci + self.prediction_horizon - 1]
        else:
            self._time_varying_parameters_horizon = ca.DM.zeros((self._n_tvp), self.prediction_horizon)
            tvp_counter = 0
            for key, value in tvp.items():
                if len(value) < self.prediction_horizon:
                    raise TypeError(
                        f"When passing time-varying parameters, you need to pass a number of values at least "
                        f"as long as the prediction horizon. The parameter {key} has {len(value)} values but the MPC "
                        f"has a prediction horizon length of {self._prediction_horizon}."
                    )

                value = tvp[key]
                self._time_varying_parameters_horizon[tvp_counter, :] = value[0:self._prediction_horizon]
                tvp_counter += 1

    def _parse_tvp_parameters_values(self, tvp):
        """

        :param tvp:
        :return:
        """
        # Get horizon of tvp values
        if tvp is not None:
            self._time_varying_parameters_horizon = ca.DM.zeros((self._n_tvp), self.prediction_horizon)
            tvp_counter = 0
            for key, value in tvp.items():
                if len(value) < self.prediction_horizon:
                    raise TypeError(
                        f"When passing time-varying parameters, you need to pass a number of values at least "
                        f"as long as the prediction horizon. The parameter {key} has {len(value)} values but the MPC "
                        f"has a prediction horizon length of {self._prediction_horizon}."
                    )

                value = tvp[key]
                self._time_varying_parameters_horizon[tvp_counter, :] = value[0:self._prediction_horizon]
                tvp_counter += 1

        elif self._time_varying_parameters_values is not None:
            self._get_tvp_parameters_values(self._time_varying_parameters_values)
        else:
            raise ValueError(
                f"Mate, I know there are {self._n_tvp} time varying parameters but you did not pass me any."
                f"Please provide me with the values of the parameters, either to the optimize() method or to the "
                f"set_time_varying_parameters() method."
            )

    def _parse_trajectory_values(self, param, **kwargs):
        """

        :param param:
        :param kwargs:
        :return:
        """
        if self.quad_stage_cost._has_trajectory_following:
            # See if references have been passed to the optimize method
            ref_sc = kwargs.get('ref_sc')
            if ref_sc is not None:
                if not isinstance(ref_sc, dict):
                    raise TypeError(
                        f"The trajectory must be a dict with as key the name of the variables that have a trajectory.")

                for key in ref_sc.keys():
                    if key not in self.quad_stage_cost.name_open_varying_trajectories:
                        raise ValueError(
                            f"I cannot find the variable {key} in the variables with varying trajectory or the "
                            f"trajectory has already been provided as a function. The trajectories without reference "
                            f"are {', '.join(self.quad_stage_cost.name_open_varying_trajectories)}"
                        )

                for traj in self.quad_stage_cost._trajectories_list:
                    for name in traj['names']:
                        traj_i = ref_sc[name]
                        traj_i = check_and_wrap_to_list(traj_i)
                        if len(traj_i) == 1:
                            # The reference is not time varying
                            traj_i = traj_i * self.prediction_horizon
                            param[name + '_sr'] = traj_i
                            traj['ref'] = traj_i
                        else:
                            if self.n_iterations + self.prediction_horizon >= len(traj_i):
                                # The reference is time varying
                                raise ValueError(
                                    f"The length of the varying reference must be one or longer than than the "
                                    f"simulation time plus the prediction horizon. Please supply data points. The "
                                    f"trajectory is long {len(traj_i)} but I am predicting at least up to the "
                                    f"{self.n_iterations + self.prediction_horizon + 1} step."
                                )
                            if self._nlp_options['objective_function'] != 'discrete':
                                raise AssertionError(
                                    "Since you are passing time series as reference points, the objective function has"
                                    "to be set to discrete. Pass `options={'objective_function': 'discrete'}` to the "
                                    "NMPC setup."
                                )
                            traj_i = traj_i[self.n_iterations:self.n_iterations + self.prediction_horizon]
                            param[name + '_sr'] = traj_i
                            traj['ref'] = traj_i
            else:
                for traj in self.quad_stage_cost._trajectories_list:
                    if traj['ref'] is None:
                        raise ValueError(
                            f"Mate, it looks like the variable(s) {traj['names']} must follow a reference, but"
                            f"you did not pass any. Please pass a reference as a function in the stage cost "
                            f"or as values in the setup of the MPC."
                        )

        if self.quad_terminal_cost._has_trajectory_following:
            ref_tc = kwargs.get('ref_tc')
            if ref_tc is not None:
                if not isinstance(ref_tc, dict):
                    raise TypeError(
                        f"The trajectory must be a dict with as key the name of the variables that have a trajectory.")
                for key in ref_tc.keys():
                    if key not in self.quad_terminal_cost.name_open_varying_trajectories:
                        raise ValueError(
                            f"I cannot find the variable {key} in the variables with varying trajectory or the "
                            f"trajectory has already been provided as a function. The trajectories without reference "
                            f"are {', '.join(self.quad_terminal_cost.name_open_varying_trajectories)}"
                        )

                for traj in self.quad_terminal_cost._trajectories_list:
                    for name in traj['names']:
                        traj_i = ref_tc[name]
                        traj_i = check_and_wrap_to_list(traj_i)
                        if len(traj_i) == 1:
                            # The reference is not time varying
                            param[name + '_tr'] = traj_i
                            traj['ref'] = traj_i
                        else:
                            if self.n_iterations + self.prediction_horizon >= len(traj_i):
                                # The reference is time varying
                                raise ValueError(
                                    f"The length of the verying reference must be one or longer than than the "
                                    f"simulation time plus the prediction horizon. Please supply datapoints. The "
                                    f"trajectory is long {len(traj_i)} but I am predicting at least up to the "
                                    f"{self.n_iterations + self.prediction_horizon + 1} step."
                                )
                            if self._nlp_options['objective_function'] != 'discrete':
                                raise AssertionError(
                                    "Since you are passing time series of reference points, the objective function has "
                                    "to be set to discrete. Check the documentation to see how to set the objective "
                                    "function to discrete."
                                )
                            traj_i = traj_i[self.n_iterations + self.prediction_horizon]
                            param[name + '_tr'] = traj_i
                            traj['ref'] = traj_i
            else:
                for traj in self.quad_terminal_cost._trajectories_list:
                    if traj['ref'] is None:
                        raise ValueError(
                            f"Mate, it looks like the variable(s) {traj['names']} must follow a reference, but"
                            f"you did not pass any. Please pass a reference as a function in the terminal "
                            f"cost or as values in the setup of the MPC."
                        )

    def _get_nlp_parameters(self, cp, tvp, **kwargs):
        """
        This arranges parameters, time-varying parameters and time-varying references in the parameters structure that
        goes in the optimization.

        :param cp:
        :param tvp:
        :param kwargs:
        :return:
        """
        param = self._param_npl_mpc(0)

        # Substitute value to the parameters of the nonlinear program
        # Input change term
        if len(self.quad_stage_cost._ind_input_changes) > 0:
            if self._nlp_solution is not None:
                param['u_old'] = [self._nlp_solution['x'][self._u_ind[0]][i] for i in
                                  self.quad_stage_cost._ind_input_changes]
            else:
                param['u_old'] = [self._u_guess[i] for i in self.quad_stage_cost._ind_input_changes]

        # Constant parameters term
        if cp is not None:
            param['c_p'] = cp

        # Time-varying parameters term
        if self._n_tvp != 0:
            self._parse_tvp_parameters_values(tvp)
            param['tv_p'] = self._time_varying_parameters_horizon

        # Reference for trajectory trackin problems term
        self._parse_trajectory_values(param, **kwargs)

        # TODO when varying sampling times are implemented, give the possibility to provide varying sampling times.
        # Sampling intervals, for unequal sampling times
        dt_grid = None
        if dt_grid is None:
            param['dt'] = ca.repmat(self.sampling_interval, self.prediction_horizon)

        param['time'] = self._time

        return param

    def _save_references(self, param):
        """
        Saves references into the solution object. For plotting purposes

        :return:
        """
        # Save current reference. Used for plotting purposes
        x_ref = ca.DM.nan(self._model_orig.n_x, self.prediction_horizon)
        u_ref = ca.DM.nan(self._model_orig.n_u, self.control_horizon)
        if self.quad_stage_cost._has_trajectory_following:
            for i, name in enumerate(self._model_orig.dynamical_state_names):
                if f'{name}_sr' in param.keys():
                    x_ref[i, :] = param[f'{name}_sr']
            for i, name in enumerate(self._model_orig.input_names):
                if f'{name}_sr' in param.keys():
                    u_ref[i, :] = param[f'{name}_sr']
            for traj in self.quad_stage_cost._trajectories_list:
                if traj['traj_fun'] is not None:
                    if traj['type'] == 'states':
                        x_ref[traj['ind'], :] = traj['traj_fun'](self.solution['t'][0:self.prediction_horizon])
                    elif traj['type'] == 'inputs':
                        u_ref[traj['ind'], :] = traj['traj_fun'](self.solution['t'][0:self.control_horizon])

        for ref in self.quad_stage_cost._references_list:
            if ref['type'] == 'states':
                x_ref[ref['ind'], :] = ca.repmat(ca.DM(ref['ref']).T, self.prediction_horizon).T
            elif ref['type'] == 'inputs':
                u_ref[ref['ind'], :] = ca.repmat(ca.DM(ref['ref']).T, self.control_horizon).T

        for path in self.quad_stage_cost._paths_list:
            if path['type'] == 'states':
                x_ref[path['ind'], :] = path['path_fun'](self.solution['thetapfo'][:, :self.prediction_horizon])
            elif path['type'] == 'inputs':
                u_ref[path['ind'], :] = path['path_fun'](self.solution['thetapfo'][:, :self.control_horizon])

        self.solution.add('u_ref', u_ref)
        self.solution.add('x_ref', x_ref)

    def _save_predictions(self):
        """
        Store prediction in the solution object. Necessary for plotting.

        :return:
        """
        # Save prediction in solution
        x_pred, u_pred, dt_pred = self.return_prediction()

        # Save solution in the solution class
        self.solution.add('x', x_pred[:self._n_x, :])
        self.solution.add('u', u_pred[:self._n_u, :])
        self.solution.add('thetapfo', x_pred[self._n_x:self._n_x+self.n_of_path_vars, :])
        if dt_pred is not None:
            dt_sum = 0
            for i in range(self.prediction_horizon):
                self.solution.add('t', self._time + dt_sum)
                dt_sum += dt_pred[i]
        else:
            t_pred = np.linspace(self._time, self._time + (self.prediction_horizon) * self.sampling_interval,
                                 self.prediction_horizon + 1)
            self.solution.add('t', ca.DM(t_pred).T)

    @property
    def x_lb(self):
        """

        :return:
        """
        return self._x_lb

    @property
    def x_ub(self):
        """

        :return:
        """
        return self._x_ub

    @property
    def u_lb(self):
        """

        :return:
        """
        return self._u_lb

    @property
    def u_ub(self):
        """

        :return:
        """
        return self._u_ub

    def set_box_constraints(self, x_ub=None, x_lb=None, u_ub=None, u_lb=None, y_ub=None, y_lb=None, z_ub=None,
                            z_lb=None):
        """
        Set box constraints to the model's variables. These look like

        .. math::
                                    x_{lb} \leq x \leq x_{ub}

        :param x_ub: upper bound on states.
        :type x_ub: list, numpy array or CasADi DM array
        :param x_lb: lower bound on  states
        :type x_lb: list, numpy array or CasADi DM array
        :param u_ub: upper bound on inputs
        :type u_ub: list, numpy array or CasADi DM array
        :param u_lb: lower bound on inputs
        :type u_lb: list, numpy array or CasADi DM array
        :param y_ub: upper bound on measurements
        :type y_ub: list, numpy array or CasADi DM array
        :param y_lb: lower bound on measurements
        :type y_lb: list, numpy array or CasADi DM array
        :param z_ub: upper bound on algebraic states
        :type z_ub: list, numpy array or CasADi DM array
        :param z_lb: lower bound on algebraic states
        :type z_lb: list, numpy array or CasADi DM array
        :return:
        """
        if x_ub is not None:
            x_ub = deepcopy(x_ub)
            x_ub = check_and_wrap_to_list(x_ub)
            if len(x_ub) != self._n_x:
                raise TypeError(f"The model has {self._n_x} states. You need to pass the same number of bounds.")
            self._x_ub = x_ub
        else:
            self._x_ub = self._model.n_x * [ca.inf]

        if x_lb is not None:
            x_lb = deepcopy(x_lb)
            x_lb = check_and_wrap_to_list(x_lb)
            if len(x_lb) != self._n_x:
                raise TypeError(f"The model has {self._n_x} states. You need to pass the same number of bounds.")
            self._x_lb = x_lb
        else:
            self._x_lb = self._model.n_x * [-ca.inf]

        # Input constraints
        if u_ub is not None:
            u_ub = deepcopy(u_ub)
            u_ub = check_and_wrap_to_list(u_ub)
            if len(u_ub) != self._n_u:
                raise TypeError(f"The model has {self._n_u} inputs. You need to pass the same number of bounds.")
            self._u_ub = u_ub
        else:
            self._u_ub = self._model.n_u * [ca.inf]

        if u_lb is not None:
            u_lb = deepcopy(u_lb)
            u_lb = check_and_wrap_to_list(u_lb)
            if len(u_lb) != self._n_u:
                raise TypeError(f"The model has {self._n_u} inputs. You need to pass the same number of bounds.")
            self._u_lb = u_lb
        else:
            self._u_lb = self._model.n_u * [-ca.inf]

        # Algebraic constraints
        if z_ub is not None:
            z_ub = deepcopy(z_ub)
            z_ub = check_and_wrap_to_list(z_ub)
            if len(z_ub) != self._n_z:
                raise TypeError(f"The model has {self._n_z} algebraic states. You need to pass the same number of "
                                f"bounds.")
            self._z_ub = z_ub
        else:
            self._z_ub = self._model.n_z * [ca.inf]

        if z_lb is not None:
            z_lb = deepcopy(z_lb)
            z_lb = check_and_wrap_to_list(z_lb)
            if len(z_lb) != self._n_z:
                raise TypeError(f"The model has {self._n_z} algebraic states. You need to pass the same number of "
                                f"bounds.")
            self._z_lb = z_lb
        else:
            self._z_lb = self._model.n_z * [-ca.inf]

        if y_lb is not None or y_ub is not None:
            # Measurement box constraints can be added by an extra stage and terminal constraint (possibly nonlinear)
            self.set_stage_constraints(stage_constraint=self._model.meas, ub=deepcopy(y_ub), lb=deepcopy(y_lb),
                                       name='measurement_constraint')
            self.set_terminal_constraints(terminal_constraint=self._model.meas, ub=deepcopy(y_ub), lb=deepcopy(y_lb),
                                          name='measurement_constraint')

        self._box_constraints_is_set = True

    def _optimize(self, v0, runs, param, **kwargs):
        """

        :param v0:
        :param runs:
        :param param:
        :param kwargs:
        :return:
        """
        if runs == 0:
            sol = self._solver(x0=v0, lbx=self._v_lb, ubx=self._v_ub, lbg=self._g_lb, ubg=self._g_ub, p=param)
            self._nlp_solution = sol
            u_opt = sol['x'][self._u_ind[0]]
            if self._nlp_options['warm_start']:
                self._v0 = sol['x']
        else:
            pert_factor = kwargs.get('pert_factor', 0.1)
            f_r_better = np.inf
            v00 = v0
            for r in range(runs):
                sol = self._solver(x0=v00, lbx=self._v_lb, ubx=self._v_ub, lbg=self._g_lb, ubg=self._g_ub, p=param)
                if sol['f'] < f_r_better:
                    f_r_better = sol['f']
                    self._nlp_solution = sol
                    u_opt = sol['x'][self._u_ind[0]]
                    if self._nlp_options['warm_start']:
                        self._v0 = sol['x']
                v00 = v0 + v0 * (1 - 2 * np.random.rand(self._n_v)) * pert_factor

        return u_opt

    def optimize(self, x0, cp=None, tvp=None, v0=None, runs=0, fix_x0=True, **kwargs):
        """
        Solves the MPC problem

        :param x0: current system state
        :type x0: list or casadi DM
        :param cp: constant system parameters (these will be assumed constant along the prediction horizon)
        :type cp: list or casadi DM, optional
        :param tvp: time-varying system parameters (these can change during prediction horizon, entire parameter
            history must be passed in)
        :type tvp: dict, optional
        :param v0: initial guess of the optimal vector
        :type v0: list or casadi DM, optional
        :param runs: number of optimizations to run. If different from zero will run very optimization will perturb the
            initial guess v0 randomly.
            ACHTUNG: This could cause problems with the integrators or give something outside constraints. The output
            will be the solution with the minimum objective function (default 0)
        :type runs: int
        :param fix_x0: If True, the first state is fixed as the measured states. This is the classic MPC approach. If
            False, also the initial state is optimized.
        :type fix_x0: bool
        :return: u_opt: first piece of optimal control sequence
        """
        if not self._nlp_setup_done:
            raise ValueError("Howdy! You need to setup the MPC before optimizing. Run .setup() on the MPC object.")

        # Check the constant parameters
        if self._model.n_p - self._n_tvp != 0:
            if cp is not None:
                cp = check_and_wrap_to_DM(cp)

            if cp is None or cp.size1() != self._model.n_p - self._n_tvp:
                raise ValueError(
                    f"The model has {self._model.n_p - self._n_tvp} constant parameter(s): "
                    f"{self._model.parameter_names}. You must pass me the value of these before running the "
                    f"optimization to the 'cp' parameter."
                )
        else:
            if cp is not None:
                warnings.warn("You are passing a parameter vector in the optimizer, but the model has no defined "
                              "parameters. I am ignoring the vector.")

        if self._nlp_options['ipopt_debugger']:
            # Reset the solution of the debugger before next optimization
            self.debugger.reset_solution()
        # Check the state
        x0 = check_and_wrap_to_DM(x0)

        if x0.shape[0] != self._n_x:
            raise ValueError(
                f"We have an issue mate, the x0 you supplied has dimension {x0.shape[0]} but the model has {self._n_x} "
                f"states."
            )
        if fix_x0 is True:
            # Force the initial state to be the measured state.
            # Note that the path following problem modifies the model dimensions,
            # so only the first n_x positions need to be forced where n_x is state number of the original problem
            self._v_lb[self._x_ind[0][0:self._n_x]] = x0 / ca.DM(self._x_scaling[0:self._n_x])
            self._v_ub[self._x_ind[0][0:self._n_x]] = x0 / ca.DM(self._x_scaling[0:self._n_x])
        else:
            x0_lb = kwargs.get('x0_lb', self._x_lb)
            x0_ub = kwargs.get('x0_ub', self._x_ub)
            self._v_lb[self._x_ind[0][0:self._n_x]] = x0_lb / ca.DM(self._x_scaling[0:self._n_x])
            self._v_ub[self._x_ind[0][0:self._n_x]] = x0_ub / ca.DM(self._x_scaling[0:self._n_x])

        # Check and prepare parameters
        param = self._get_nlp_parameters(cp, tvp, **kwargs)

        if v0 is None:
            v0 = self._v0

        if self._stats:
            start = time.time()

        # Compute the MPC
        u_opt = self._optimize(v0, runs, param)

        # Get the status of the solver
        self._solver_status_wrapper()

        # Print output
        self._print_message()

        # Reset the solution
        self.reset_solution()

        # Populate solution with new results
        if self._stats:
            elapsed_time = time.time() - start
            self.solution.add('extime', elapsed_time)
            self.solution.add('niterations', self._n_iterations)
            self.solution.add('solvstatus', self._solver_status_code)

        if self.stage_constraint.is_set and self.stage_constraint.is_soft:
            self.stage_constraint.e_soft_value = self._nlp_solution['x'][self._e_soft_stage_ind[0]]

        if self.terminal_constraint.is_set and self.terminal_constraint.is_soft:
            self.terminal_constraint.e_soft_value = self._nlp_solution['x'][self._e_soft_term_ind[0]]

        # Save predictions in the solution object. Done for plotting purposes.
        self._save_predictions()

        # Save the references in the solution object. Done for plotting purposes.
        self._save_references(param)

        # Update the time clock - Useful for time-varying systems and references.
        self._time += self.sampling_interval

        # Interation counter
        self._n_iterations += 1

        # Extract first input from the optimal sequence
        uopt = u_opt[0:self._n_u] * self._u_scaling[0:self._n_u]
        return uopt

    def minimize_final_time(self, weight=1):
        """

        :param weight:
        :return:
        """
        self._minimize_final_time_flag = True
        self._minimize_final_time_weight = weight

    def plot_prediction(self, save_plot=False, plot_dir=None, name_file='mpc_prediction.html', show_plot=True,
                        extras=None, extras_names=None, title=None, format_figure=None, **kwargs):
        """
        Plots the MPC predicted values.

        :param save_plot: if True plot will be saved under 'plot_dir/name_file.html' if they are declared, otherwise in
            current directory
        :type save_plot: bool
        :param plot_dir: path to the folder where plots are saved (default = None)
        :type plot_dir: str
        :param name_file: name of the file where plot will be saved  (default = mpc_prediction.html)
        :type name_file: str
        :param show_plot: if True, shows plots (default = False)
        :type show_plot: bool
        :param extras: dictionary with values that will be plotted over the predictions if keys are equal to predicted
            states/inputs.
        :type extras: dict
        :param extras_names: tags that will be attached to the extras in the legend
        :type extras_names: list
        :param title: title of the plots
        :type title: str
        :param format_figure: python function that modifies the format of the figure
        :type format_figure: python function taking a bokeh figure object as an input
        :return:
        """
        import os

        from bokeh.io import output_file, output_notebook, show, save
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, DataTable, TableColumn, CellFormatter, Div
        from bokeh.layouts import gridplot, column, row, grid
        from bokeh.palettes import Spectral4 as palettespectral

        if self._nlp_solution is None:
            raise RuntimeError("You need to run the MPC at least once to see the plots")
        if save_plot:
            if plot_dir is not None:
                output_file(os.path.join(plot_dir, name_file))
            else:
                output_file(name_file)
        else:
            if kwargs.get('output_notebook', False):
                output_notebook()

        if extras is None:
            extras = []
        if extras_names is None:
            extras_names = []
        if isinstance(extras, dict):
            extras = [extras]
        elif not isinstance(extras, list) and not isinstance(extras, dict):
            raise ValueError("The extras options should be a dictionary or a list of dictionaries")

        res_extras_list = [0 for i in range(len(extras))]
        for i in range(len(extras)):
            res_extras_list[i] = {}
            for k, name in enumerate(self._model.dynamical_state_names):
                if name in extras[i].keys():
                    res_extras_list[i][name] = extras[i][name]

        if len(extras) != len(extras_names):
            raise ValueError("The length of the extra and extras_names must be the same.")

        time = self._time
        # TODO: consider time step for the x axis
        x_pred, u_pred, dt_pred = self.return_prediction()

        if dt_pred is None:
            time_vector = np.linspace(time, time + (self._prediction_horizon) * self.sampling_interval,
                                      self._prediction_horizon + 1)
        else:
            time_vector = [self._time]
            for i in range(self._prediction_horizon):
                time_vector.append(time_vector[i] + dt_pred[i])

        input_dict = {i: [] for i in self._model.input_names}
        states_dict = {i: [] for i in self._model.dynamical_state_names}

        for k, name in enumerate(self._model.input_names):
            input_dict[name] = u_pred[k, :]

        for k, name in enumerate(self._model.dynamical_state_names):
            states_dict[name] = x_pred[k, :]

        p1 = [figure(title=title, background_fill_color="#fafafa") for i in range(self._model.n_x)]

        for s, name in enumerate(self._model.dynamical_state_names):
            p1[s].line(x=time_vector,
                       y=states_dict[name],
                       legend_label=name + '_pred', line_width=2)
            for i in range(len(self.quad_stage_cost._references_list)):
                if name in self.quad_stage_cost._references_list[i]['names']:
                    position = self.quad_stage_cost._references_list[i]['names'].index(name)
                    value = self.quad_stage_cost._references_list[i]['ref'][position]
                    p1[s].line([time_vector[0], time_vector[-1]], [value, value], legend_label=name + '_ref',
                               line_dash='dashed', line_color="red", line_width=2)

            p1[s].yaxis.axis_label = name
            p1[s].xaxis.axis_label = 'time'
            if format_figure is not None:
                p1[s] = format_figure(p1[s])

            for i in range(len(extras)):
                if name in list(res_extras_list[i].keys()):
                    p1[s].line(x=time_vector,
                               y=res_extras_list[i][name], line_width=2, color=palettespectral[i + 1],
                               legend_label=name + '_' + extras_names[i])
                    p1[s].yaxis.axis_label = name
                    p1[s].xaxis.axis_label = 'time'
                    if format_figure is not None:
                        p1[s] = format_figure(p1[s])

        p2 = [figure(title=title, background_fill_color="#fafafa") for i in range(self._model.n_u)]
        for s, name in enumerate(self._model.input_names):
            p2[s].step(x=time_vector[:-1], y=input_dict[name],
                       legend_label=name + '_pred', mode="after", line_width=2)
            p2[s].yaxis.axis_label = name
            p2[s].xaxis.axis_label = 'time'
            if format_figure is not None:
                p2[s] = format_figure(p2[s])

        # Create some data to print statistics
        variables = []
        values = []
        if self.stage_constraint.is_soft:
            variables.append('Slack soft constraint')
            values.append(float(np.array(self.stage_constraint.e_soft_value).squeeze()))

        heading = Div(text="MPC stats", height=80, sizing_mode="stretch_width", align='center',
                      style={'font-size': '200%'})
        # heading fills available width
        data = dict(
            variables=variables,
            values=values,
        )
        source = ColumnDataSource(data)

        columns = [
            TableColumn(field="variables", title="Variables"),
            TableColumn(field="values", title="Values", formatter=CellFormatter()),
        ]
        data_table = DataTable(source=source, columns=columns, width=400, height=280)

        grid_states = gridplot(p1, ncols=3, sizing_mode="stretch_width")
        grid_inputs = gridplot(p2, ncols=3, sizing_mode="stretch_width")

        if show_plot:
            states_header = Div(text="Predicted States", height=10, sizing_mode="stretch_width", align='center',
                                style={'font-size': '200%'})
            inputs_header = Div(text="Predicted Inputs", height=10, sizing_mode="stretch_width", align='center',
                                style={'font-size': '200%'})
            layout = row(column(states_header, grid_states, inputs_header, grid_inputs), column(heading, data_table))
            show(layout)
        else:
            if save_plot:
                save(grid)

    def create_path_variable(self, name='theta', u_pf_lb=0.0001, u_pf_ub=1, u_pf_ref=None, u_pf_weight=10,
                             theta_guess=0, theta_lb=0, theta_ub=ca.inf):
        """
        Set the path following variable. This must be used for building SX expression of the path.

        :param name: name of the variable, default = 'theta'
        :type name: string
        :param u_pf_lb: lower bound on the path virtual input :math:`vel_{lb} \leq \dot{ \\theta }`, default = 0.0001
        :type u_pf_lb: float
        :param u_pf_ub: upper bound on the path virtual input :math:`vel_{ub} \geq \dot{ \\theta }`, default = 1
        :type u_pf_ub: float
        :param u_pf_ref: Reference for the path virtual input, default = None
        :type u_pf_ref: float
        :param u_pf_weight: Weight for the path virtual input , default = 10
        :type u_pf_weight: float
        :param theta_guess:
        :param theta_lb:
        :param theta_ub:
        :return: casadi.SX
        """
        if self._use_sx:
            theta = ca.SX.sym(name)
        else:
            theta = ca.MX.sym(name)
        self._paths_var_list.append(
            {'theta': theta, 'u_pf_lb': u_pf_lb, 'u_pf_ub': u_pf_ub, 'u_pf_ref': u_pf_ref, 'u_pf_weight': u_pf_weight,
             'theta_guess': theta_guess, 'theta_lb': theta_lb, 'theta_ub': theta_ub}
        )
        return theta

    def get_time_variable(self):
        """
        Useful for trajectory tacking

        :return:
        """
        self._time_var = self._model.t
        return self._time_var

    def set_quadratic_stage_cost(self, states=None, cost_states=None, states_references=None,
                                 inputs=None, cost_inputs=None, inputs_references=None):
        """
        More compact way to set the quadratic cost for the MPC. Mostly left for backwards compatibility.
        To use only for set-point-tracking problems.

        :param states: list of states name that will appear in the quadratic cost
        :type states: list
        :param cost_states:  weights values that will be multiplied by the states
        :type cost_states: list, numpy array, or casADi DM array
        :param states_references:  list of reference values
        :type states_references: list
        :param inputs:  list of inputs names that will be multiplied by the states
        :type inputs: list
        :param cost_inputs:  list of inputs weights that will be multiplied by the states
        :type cost_inputs: list, numpy array, or casADi DM array
        :param inputs_references:  list of inputs reference values
        :type inputs_references: list
        :return:
        """
        self.quad_stage_cost.add_states(names=states, weights=cost_states, ref=states_references)
        self.quad_stage_cost.add_inputs(names=inputs, weights=cost_inputs, ref=inputs_references)

    def set_quadratic_terminal_cost(self, states=None, cost=None, references=None):
        """
        More compact way to set the quadratic cost for the MPC. Mostly left for backwards compatibility.
        To use only for set-point-tracking problems.

        :param states: list of states name that will appear in the quadratic cost
        :type states: list
        :param cost: list of weights values that will be multiplied by the states
        :type cost: list, numpy array or CasADi DM
        :param references: list of references values for the states
        :type references:
        :return:
        """
        self.quad_terminal_cost.add_states(names=states, weights=cost, ref=references)

    def set_terminal_constraints(self,terminal_constraint,name='terminal_constraint', lb=None, ub=None, is_soft=False, max_violation=ca.inf,
                                 weight=None):
        """
        Allows to add a (nonlinear) terminal constraint.

        :param terminal_constraint:  It has to contain variables of the model
        :type terminal_constraint: CasADi SX expression
        :param lb:  Lower bound on the constraint
        :type lb: list of float, integer or casadi.DM.
        :param ub:  Upper bound on the constraint
        :type ub: list of float, integer or casadi.DM.
        :param is_soft:  if True soft constraints are used (default False)
        :type is_soft: bool
        :param max_violation: (optional) Maximum violation if constraint is soft. Default: inf
        :type max_violation: list float,integer or casadi.DM
        :param weight: (optional) matrix of appropriate dimension. If is_soft=True it will be used to weight the soft
            constraint in the objective function using a quadratic cost.
        :type weight: casadi.DM
        :return:
        """
        # TODO check if all the variables used in the function are in the model
        # TODO allow to pass either only the lower or the upper bound
        self.terminal_constraint.constraint = terminal_constraint
        self.terminal_constraint.lb = lb
        self.terminal_constraint.ub = ub
        self.terminal_constraint.is_soft = is_soft
        self.terminal_constraint.max_violation = max_violation
        self.terminal_constraint.weight = weight
        self.terminal_constraint.name = name

    def setup(self, options=None, solver_options=None) -> None:

        """
        Sets up the corresponding optimization problem (OP) of the MPC. This must be run before attempting to solve
        the MPC.

        :param options: Options for MPC. See documentation.
        :type options: dict
        :param solver_options: Dictionary with options for the optimizer. These options are solver specific. Refer to
            the CasADi Documentation https://web.casadi.org/python-api/#nlp
        :type solver_options: dict
        :return: None
        """
        if not self._scaling_is_set:
            self.set_scaling()
        if not self._time_varying_parameters_is_set:
            self.set_time_varying_parameters()
        if not self._box_constraints_is_set:
            self.set_box_constraints()
        if not self._initial_guess_is_set:
            self.set_initial_guess()

        if not self._nlp_options_is_set:
            self.set_nlp_options(options)
        if not self._solver_options_is_set:
            self.set_solver_opts(solver_options)

        if not self._sampling_time_is_set:
            self.set_sampling_interval()

        self._populate_solution()
        # Path following
        self._x_scaling_orig = deepcopy(self._x_scaling)
        self._u_scaling_orig = deepcopy(self._u_scaling)

        self._x_lb_orig = deepcopy(self._x_lb)
        self._x_ub_orig = deepcopy(self._x_ub)
        self._u_lb_orig = deepcopy(self._u_lb)
        self._u_ub_orig = deepcopy(self._u_ub)

        for i in range(self.n_of_path_vars):
            theta = self._paths_var_list[i]['theta']
            theta_vel_ub = self._paths_var_list[i]['u_pf_ub']
            theta_vel_lb = self._paths_var_list[i]['u_pf_lb']
            theta_lb = self._paths_var_list[i]['theta_lb']
            theta_ub = self._paths_var_list[i]['theta_ub']
            theta_guess = self._paths_var_list[i]['theta_guess']
            # If path following the model increases of dimension to allow the path variable
            self._model.add_dynamical_states(theta)
            if self._use_sx:
                u_theta = ca.SX.sym('u_theta')
            else:
                u_theta = ca.MX.sym('u_theta')
            self._model.add_inputs(u_theta)

            if self._model.discrete:
                # TODO: here I am doing simply explicit euler. Maybe one could use the same discretization method of
                #  the model
                self._model.add_dynamical_equations(theta + self.sampling_interval * u_theta)
            else:
                self._model.add_dynamical_equations(u_theta)
            self._x_guess.append(theta_guess)
            self._u_guess.append(theta_vel_lb + 0.0001)
            self._u_lb.append(theta_vel_lb)
            self._u_ub.append(theta_vel_ub)
            self._x_lb.append(theta_lb)
            self._x_ub.append(theta_ub)
            self._u_scaling.append(1)
            self._x_scaling.append(1)
            if self._paths_var_list[i]['u_pf_ref'] is not None:
                self.stage_cost.cost = (u_theta - self._paths_var_list[i]['u_pf_ref']) ** 2 * self._paths_var_list[i][
                    'u_pf_weight']

        # Define cost terms
        self._define_cost_terms()

        # Scaling...
        self._scale_problem()

        # Define custom stage and terminal constraints
        self.stage_constraint._check_and_setup(x_scale=self._x_scaling, u_scale=self._u_scaling,
                                               y_scale=self._y_scaling)
        self.terminal_constraint._check_and_setup(x_scale=self._x_scaling, u_scale=self._u_scaling,
                                                  y_scale=self._y_scaling)

        self._check_mpc_is_well_posed()

        if self._nlp_options['ipopt_debugger']:
            # This dict saves the po
            self._g_indices = {'dynamics_collocation': [],
                               'dynamics_multiple_shooting': [],
                               'nonlin_stag_const': [],
                               'nonlin_term_const': [],
                               'time_const.': []}

        if self._nlp_setup_done is False:
            model = self._model

            # ... objective function.
            if self._may_term_flag:
                # Substiture references
                if self.quad_terminal_cost.ref_placeholder.shape[0] != 0:
                    references = self.quad_terminal_cost.ref_placeholder
                else:
                    if self._use_sx:
                        references = ca.SX.sym('r', 0)
                    else:
                        references = ca.MX.sym('r', 0)

                self._may_term_fun = ca.Function('mayor_term',
                                                 [model.t, model.x,
                                                  references],
                                                 [self._may_term])

            # Check time varying parameters
            # tvp_ind = []
            #TODO this should be moved in the optimiziation method
            if len(self._time_varying_parameters) != 0:
                p_names = model.parameter_names
                for tvp in self._time_varying_parameters:
                    assert tvp in p_names, f"The time-varying parameter {tvp} is not in the model parameter. " \
                                           f"The model parameters are {p_names}."

            if self.terminal_constraint.is_set:
                if self.terminal_constraint.is_soft:
                    e_terminal = model.t.sym('e_terminal', self.terminal_constraint.constraint.size1())
                    self._terminal_constraints_fun = ca.Function('soft_terminal_const_term',
                                                                 [model.t, model.x, model.z, model.p, e_terminal],
                                                                 [ca.vertcat(self.terminal_constraint.constraint -
                                                                             e_terminal,
                                                                             -self.terminal_constraint.constraint -
                                                                             e_terminal)])
                else:
                    self._terminal_constraints_fun = ca.Function('terminal_const_term',
                                                                 [model.t, model.x, model.z, model.p],
                                                                 [self.terminal_constraint.constraint])

            # Define casadi function for auxiliary stage nonlinear constraints
            if self.stage_constraint.is_set:
                if self.stage_constraint.is_soft:
                    e_stage = model.t.sym('e_stage', self.stage_constraint.constraint.size1())
                    self._stage_constraints_fun = ca.Function('soft_stage_nl_constr',
                                                              [model.t, model.x, model.u, model.z, model.p, e_stage],
                                                              [ca.vertcat(self.stage_constraint.constraint - e_stage,
                                                                          -self.stage_constraint.constraint - e_stage)])
                else:
                    self._stage_constraints_fun = ca.Function('stage_nl_constr',
                                                              [model.t, model.x, model.u, model.z, model.p],
                                                              [self.stage_constraint.constraint])

            if self._lag_term_flag:
                model.set_quadrature_function(self._lag_term)
                references = self.quad_stage_cost.ref_placeholder
                u_old = self.quad_stage_cost.input_change_placeholder
            else:
                references = []
                u_old = []

            problem = dict(model)

            if self._lag_term_flag:
                if self.quad_stage_cost.ref_placeholder.shape[0] != 0:
                    references = self.quad_stage_cost.ref_placeholder
                else:
                    if self._use_sx:
                        references = ca.SX.sym('r', 0)
                    else:
                        references = ca.MX.sym('r', 0)

                u_old = self.quad_stage_cost.input_change_placeholder
                self._lag_term_fun = ca.Function('lagrange_term',
                                                 [problem['t'], problem['x'],
                                                  problem['u'], problem['z'], problem['p'],
                                                  references, u_old],
                                                 [self._lag_term])

            if self._nlp_options['integration_method'] == 'collocation':
                continuous2discrete(problem, **self._nlp_options)

                # Slack for soft constraints
                if self._use_sx:
                    ek = ca.SX.sym('e', self.stage_constraint.size)
                else:
                    ek = ca.MX.sym('e', self.stage_constraint.size)

                # Add all constraints to the collocation points
                #   box constraints
                # TODO is here options['degree']+ 1 or problem['ode']???
                n_xik = (self._nlp_options['degree']) * (model.n_x)
                n_zik = (self._nlp_options['degree']) * (model.n_z)

                x_ik_guess = np.tile(self._x_guess, self._nlp_options['degree'])
                x_ik_ub = np.tile(self._x_ub, self._nlp_options['degree'])
                x_ik_lb = np.tile(self._x_lb, self._nlp_options['degree'])

                if model.n_z > 0:
                    z_ik_guess = np.tile(self._z_guess, self._nlp_options['degree'])
                    z_ik_ub = np.tile(self._z_ub, self._nlp_options['degree'])
                    z_ik_lb = np.tile(self._z_lb, self._nlp_options['degree'])

                #   nonlinear constraints
                # Constraints in the control interval
                gk_col = []
                gk_col_lb = []
                gk_col_ub = []
                for k in range(self._nlp_options['degree']):
                    x_col = problem['collocation_points_ode'][k]
                    z_col = problem['collocation_points_alg'][k]
                    if self.stage_constraint.is_set:
                        if self.stage_constraint.is_soft:
                            residual = self._stage_constraints_fun(problem['t'], x_col, problem['u'], z_col,
                                                                   problem['p'], ek)
                            gk_col.append(residual)
                            gk_col_lb.append(np.repeat(-np.inf, self.stage_constraint.size * 2))
                            gk_col_ub.append(self.stage_constraint.ub)
                            gk_col_ub.append([-lb for lb in self.stage_constraint.lb])
                        else:
                            residual = self._stage_constraints_fun(problem['t'], x_col, problem['u'], z_col,
                                                                   problem['p'])
                            gk_col.append(residual)
                            gk_col_lb.append(self.stage_constraint.lb)
                            gk_col_ub.append(self.stage_constraint.ub)

                gk_col.append(problem['collocation_equations'])
                gk_col_lb.append(np.zeros(problem['collocation_equations'].shape[0]))
                gk_col_ub.append(np.zeros(problem['collocation_equations'].shape[0]))

                # Create function
                int_dynamics_fun = ca.Function("integrator_collocation",
                                               [problem['t'],  # time variable (for time varying systems)
                                                problem['dt'],  # dt variable (for possibly different sampling time)
                                                ca.vertcat(*problem['collocation_points_ode']),
                                                # x at collocation points
                                                problem['x'],  # x at the begining of the interval
                                                problem['u'],  # input of the interval
                                                ca.vertcat(*problem['collocation_points_alg']),  # alg states at coll.
                                                problem['p'],  # parameters (constant over the interval at least)
                                                ek,  # slack variable for (possibly) soft constrained systems
                                                references,
                                                u_old],
                                               [ca.vertcat(*gk_col), problem['ode'], problem['quad']])

            elif self._nlp_options['integration_method'] == 'discrete':
                n_zik = model.n_z
                z_ik_guess = self._z_guess
                z_ik_ub = self._z_ub
                z_ik_lb = self._z_lb

                int_dynamics_fun = ca.Function('integrator_discrete',
                                               [problem['t'],
                                                problem['dt'],
                                                problem['x'],
                                                problem['u'],
                                                problem['z'],
                                                problem['p'],
                                                ],
                                               [problem['ode']])

            elif self._nlp_options['integration_method'] in ['rk', 'rk4']:
                continuous2discrete(problem, **self._nlp_options)

                n_zik = (self._nlp_options['order']) * model.n_z

                z_ik_guess = np.tile(self._z_guess, self._nlp_options['order'])
                z_ik_ub = np.tile(self._z_ub, self._nlp_options['order'])
                z_ik_lb = np.tile(self._z_lb, self._nlp_options['order'])

                # Create function
                int_dynamics_fun = ca.Function("integrator_collocation",
                                               [
                                                   problem['t'],  # time variable (for time varying systems)
                                                   problem['dt'],  # dt variable (for possibly different sampling time)
                                                   problem['x'],  # x at the beginning of the interval
                                                   problem['u'],  # input of the interval
                                                   problem['discretization_points'],  # algebraic variables
                                                   problem['z'],
                                                   problem['p'],  # parameters (constant over the interval at least)
                                                   references,
                                                   u_old
                                               ],
                                               [problem['alg'], problem['ode'], problem['quad']])

            elif self._nlp_options['integration_method'] in ['idas', 'cvodes']:
                n_zik = model.n_z
                z_ik_guess = self._z_guess
                z_ik_ub = self._z_ub
                z_ik_lb = self._z_lb
                if model.n_z == 0:
                    dae = {'t': problem['t'],
                           'x': problem['x'],
                           'p': ca.vertcat(problem['u'], problem['p'], u_old, references, problem['dt']),
                           'ode': model.ode,
                           'quad': self._lag_term}
                else:
                    dae = {'t': problem['t'],
                           'x': ca.vertcat(problem['x'], problem['z']),
                           'p': ca.vertcat(problem['u'], problem['p'], u_old, references, problem['dt']),
                           'ode': ca.vertcat(model.ode, model.alg),
                           'quad': self._lag_term}

                opts = {'abstol': 1e-10, 'reltol': 1e-10, 'tf': self._sampling_interval}
                int_dynamics_fun = ca.integrator('integrator_ms', self._nlp_options['integration_method'], dae, opts)

            else:
                raise ValueError(f"Integration {self._nlp_options['integration_method']} not defined.")

            # Total number of optimization variable
            n_v = self._control_horizon * model.n_u + (self._prediction_horizon + 1) * (model.n_x + model.n_z)

            if self._nlp_options['integration_method'] == 'collocation':
                n_v += self._prediction_horizon * (n_xik + n_zik)
            if self._nlp_options['integration_method'] in ['rk', 'rk4']:
                n_v += self._prediction_horizon * (n_zik)
            if self.stage_constraint.is_soft:
                n_v += self.stage_constraint.constraint.size1()
            if self.terminal_constraint.is_soft:
                n_v += self.terminal_constraint.constraint.size1()
            if self._custom_constraint_is_soft_flag:
                n_v += self._custom_constraint_size
            if self._minimize_final_time_flag:
                n_v += self._prediction_horizon

            v = ca.MX.sym('v', n_v)

            v_lb = np.zeros(n_v)
            v_ub = np.zeros(n_v)
            v_guess = np.zeros(n_v)
            offset = 0

            # Predefine optimization variable - Those always exist, independently of the method used
            x = np.resize(np.array([], dtype=ca.MX), (self._prediction_horizon + 1, 1))
            x_ind = []
            for ii in range(self._prediction_horizon + 1):
                x[ii, 0] = v[offset:offset + model.n_x]
                x_ind.append([j for j in range(offset, offset + model.n_x)])
                v_guess[offset:offset + model.n_x] = self._x_guess
                v_lb[offset:offset + model.n_x] = self._x_lb
                v_ub[offset:offset + model.n_x] = self._x_ub
                if self._mixed_integer_flag:
                    self._discrete_variables_bool.extend([False] * model._n_x)
                offset += model.n_x

            u = np.resize(np.array([], dtype=ca.MX), (self._control_horizon, 1))
            u_ind = []
            for ii in range(self._control_horizon):
                u[ii, 0] = v[offset:offset + model.n_u]
                u_ind.append([j for j in range(offset, offset + model.n_u)])
                v_guess[offset:offset + model.n_u] = self._u_guess
                v_lb[offset:offset + model.n_u] = self._u_lb
                v_ub[offset:offset + model.n_u] = self._u_ub
                if self._mixed_integer_flag:
                    self._discrete_variables_bool.extend(model.discrete_u)
                offset += model.n_u

            if model.n_z > 0:
                z = np.resize(np.array([], dtype=ca.MX), (self._prediction_horizon + 1, 1))
                z_ind = []
                for ii in range(self._prediction_horizon + 1):
                    z[ii, 0] = v[offset:offset + model.n_z]
                    z_ind.append([j for j in range(offset, offset + model.n_z)])
                    v_guess[offset:offset + model.n_z] = self._z_guess
                    v_lb[offset:offset + model.n_z] = self._z_lb
                    v_ub[offset:offset + model.n_z] = self._z_ub
                    offset += model.n_z

            # Some other variables must be added, depending on the approximation method used
            if self._nlp_options['integration_method'] == 'collocation':
                ip = np.resize(np.array([], dtype=ca.MX), (self._prediction_horizon, 1))
                zp = np.resize(np.array([], dtype=ca.MX), (self._prediction_horizon, 1))

                for ii in range(self._prediction_horizon):
                    ip[ii, 0] = v[offset:offset + n_xik]
                    v_guess[offset:offset + n_xik] = x_ik_guess
                    v_lb[offset:offset + n_xik] = x_ik_lb
                    v_ub[offset:offset + n_xik] = x_ik_ub

                    if self._mixed_integer_flag:
                        self._discrete_variables_bool.extend([False] * n_xik)
                    offset += n_xik

                    if model.n_z > 0:
                        zp[ii, 0] = v[offset:offset + n_zik]
                        v_guess[offset:offset + n_zik] = z_ik_guess
                        v_lb[offset:offset + n_zik] = z_ik_lb
                        v_ub[offset:offset + n_zik] = z_ik_ub
                        offset += n_zik

            elif self._nlp_options['integration_method'] in ['rk', 'rk4', 'discrete', 'idas', 'cvodes']:
                zp = np.resize(np.array([], dtype=ca.MX), (self._prediction_horizon, 1))
                for ii in range(self._prediction_horizon):
                    zp[ii, 0] = v[offset:offset + n_zik]
                    v_guess[offset:offset + n_zik] = z_ik_guess
                    v_lb[offset:offset + n_zik] = z_ik_lb
                    v_ub[offset:offset + n_zik] = z_ik_ub
                    offset += n_zik

            if self.stage_constraint.is_soft:
                e_soft_stage = v[offset:offset + self.stage_constraint.size]
                self._e_soft_stage_ind = [j for j in range(offset, offset + self.stage_constraint.size)]
                v_lb[offset:offset + self.stage_constraint.size] = np.zeros(self.stage_constraint.size)
                v_ub[offset:offset + self.stage_constraint.size] = self.stage_constraint.max_violation
                v_guess[offset:offset + self.stage_constraint.size] = np.zeros(self.stage_constraint.size)
                if self._mixed_integer_flag:
                    self._discrete_variables_bool.extend([False])
                offset += self.stage_constraint.size
            else:
                e_soft_stage = ca.MX.sym('e_soft_stage', 0)

            if self.terminal_constraint.is_soft:
                e_soft_term = v[offset:offset + self.terminal_constraint.size]
                self._e_soft_term_ind = [j for j in range(offset, offset + self.terminal_constraint.size)]
                v_lb[offset:offset + self.terminal_constraint.size] = np.zeros(self.terminal_constraint.size)
                v_ub[offset:offset + self.terminal_constraint.size] = self.terminal_constraint.max_violation
                v_guess[offset:offset + self.terminal_constraint.size] = np.zeros(self.terminal_constraint.size)
                if self._mixed_integer_flag:
                    self._discrete_variables_bool.extend([False])
                offset += self.terminal_constraint.size

            if self._custom_constraint_is_soft_flag:
                e_cus = v[offset:offset + self._custom_constraint_size]
                v_lb[offset:offset + self._custom_constraint_size] = np.zeros(self._custom_constraint_size)
                v_ub[offset:offset + self._custom_constraint_size] = self._custom_constraint_maximum_violation
                v_guess[offset:offset + self._custom_constraint_size] = self._custom_constraint_size
                offset += self._custom_constraint_size

            # Create an entry for every reference
            tv_stage_ref_list = []
            for ii in range(len(self.quad_stage_cost._trajectories_list)):
                for name in self.quad_stage_cost._trajectories_list[ii]['names']:
                    if not isinstance(self.quad_stage_cost._trajectories_list[ii]['ref'], ca.SX) and not isinstance(
                            self.quad_stage_cost._trajectories_list[ii]['ref'], ca.MX):
                        tv_stage_ref_list.append(catools.entry(name + '_sr', shape=(1, self._prediction_horizon)))

            tv_term_ref_list = []
            for ii in range(len(self.quad_terminal_cost._trajectories_list)):
                for name in self.quad_terminal_cost._trajectories_list[ii]['names']:
                    if not isinstance(self.quad_terminal_cost._trajectories_list[ii]['ref'], ca.SX) and not isinstance(
                            self.quad_terminal_cost._trajectories_list[ii]['ref'], ca.MX):
                        tv_term_ref_list.append(catools.entry(name + '_tr', shape=1))

            # Predefine parameters (those are fixed and not optimized)
            param_npl_mpc = catools.struct_symMX([catools.entry("tv_p", shape=(self._n_tvp, self._prediction_horizon)),
                                                  catools.entry("c_p", shape=model.n_p - self._n_tvp),
                                                  catools.entry('u_old',
                                                                shape=len(self.quad_stage_cost._ind_input_changes)),
                                                  catools.entry('dt', shape=self._prediction_horizon),
                                                  catools.entry('time', shape=1)] +
                                                 tv_term_ref_list + tv_stage_ref_list)

            tv_p = param_npl_mpc['tv_p']
            c_p = param_npl_mpc['c_p']
            u_old = param_npl_mpc['u_old']

            # Concat all the time-varying trajectories
            tv_ref_sc = []
            for ii in range(len(self.quad_stage_cost._trajectories_list)):
                for name in self.quad_stage_cost._trajectories_list[ii]['names']:
                    if self.quad_stage_cost._trajectories_list[ii]['ref'] is None:
                        tv_ref_sc.append(param_npl_mpc[name + '_sr'])
            if len(tv_ref_sc) > 0:
                tv_ref_sc = ca.vertcat(*tv_ref_sc)
            else:
                tv_ref_sc = ca.MX.sym('bla', (0, self._prediction_horizon))

            tv_ref_tc = []
            for ii in range(len(self.quad_terminal_cost._trajectories_list)):
                for name in self.quad_terminal_cost._trajectories_list[ii]['names']:
                    if self.quad_terminal_cost._trajectories_list[ii]['ref'] is None:
                        tv_ref_tc.append(param_npl_mpc[name + '_tr'])
            if len(tv_ref_tc) > 0:
                tv_ref_tc = ca.vertcat(*tv_ref_tc)
            else:
                tv_ref_tc = ca.MX.sym('bla', 0)

            t_ind = []
            if self._minimize_final_time_flag is False:
                _dt = param_npl_mpc['dt']
            else:
                _dt = v[offset:offset + self.prediction_horizon]
                t_ind = [j for j in range(offset, offset + self.prediction_horizon)]
                v_lb[offset:offset + self.prediction_horizon] = np.zeros(self.prediction_horizon)
                v_ub[offset:offset + self.prediction_horizon] = ca.inf * np.ones(self.prediction_horizon)
                v_guess[offset:offset + self.prediction_horizon] = self._sampling_interval * np.ones(
                    self.prediction_horizon)
                offset += self.prediction_horizon

            # Constraint function for the NLP
            g = []
            g_lb = []
            g_ub = []
            J = 0
            ind_g = 0
            # get the current sampling time
            time = param_npl_mpc['time']
            for ii in range(self._prediction_horizon):
                x_ii = x[ii, 0]
                if ii < self._control_horizon:
                    u_ii = u[ii, 0]
                    if ii >= 1:
                        u_old0 = [u_ii[jj] for jj in self.quad_stage_cost._ind_input_changes]
                        u_old0 = ca.vertcat(*u_old0)
                    else:
                        u_old0 = u_old

                p_ii = self._rearrange_parameters(tv_p[:, ii], c_p)
                tv_ref_sc_ii = tv_ref_sc[:, ii]
                dt_ii = _dt[ii]

                if self._nlp_options['integration_method'] in ['idas', 'cvodes']:
                    sol = int_dynamics_fun(x0=ca.vertcat(x_ii, zp[ii, 0]),
                                           p=ca.vertcat(u_ii, p_ii, u_old0, tv_ref_sc_ii, dt_ii))
                    x_ii_1 = sol['xf']
                    quad = sol['qf']
                elif self._nlp_options['integration_method'] in ['rk4', 'rk']:
                    [alg, x_ii_1, quad] = int_dynamics_fun(
                        time, dt_ii, x_ii, u_ii, zp[ii, 0], p_ii, tv_ref_sc_ii, e_soft_stage, u_old0)
                    g.append(alg)
                    g_lb.append(np.zeros(alg.size1()))
                    g_ub.append(np.zeros(alg.size1()))
                    if self._nlp_options['ipopt_debugger']:
                        self._g_indices['dynamics_collocation'].append([ind_g, ind_g + alg.size1()])
                        ind_g += alg.size1()
                elif self._nlp_options['integration_method'] == 'collocation':
                    [g_coll, x_ii_1, quad] = int_dynamics_fun(
                        time, dt_ii, ip[ii, 0], x_ii, u_ii, zp[ii, 0], p_ii, e_soft_stage, tv_ref_sc_ii, u_old0)
                    g.append(g_coll)
                    g_lb.extend(gk_col_lb)
                    g_ub.extend(gk_col_ub)
                    if self._nlp_options['ipopt_debugger']:
                        self._g_indices['dynamics_collocation'].append([ind_g, ind_g + g_coll.size1()])
                        ind_g += g_coll.size1()
                elif self._nlp_options['integration_method'] == 'discrete':
                    x_ii_1 = int_dynamics_fun(time, dt_ii, x_ii, u_ii, zp[ii, 0], p_ii)

                g.append(x[ii + 1, 0] - x_ii_1)
                g_lb.append(np.zeros(model.n_x))
                g_ub.append(np.zeros(model.n_x))
                if self._nlp_options['ipopt_debugger']:
                    self._g_indices['dynamics_multiple_shooting'].append([ind_g, ind_g + x_ii_1.shape[0]])
                    ind_g += x_ii_1.size1()

                # Add lagrange term
                # TODO check if call is necessary
                if self._lag_term_flag:
                    if self._nlp_options['integration_method'] == 'discrete' or \
                            self._nlp_options['objective_function'] == 'discrete':
                        quad = self._lag_term_fun(time, x_ii, u_ii, zp[ii, 0], p_ii, tv_ref_sc_ii, u_old0)
                    J += quad
                if self._may_term_flag and ii == self._prediction_horizon - 1:
                    J += self._may_term_fun(time + dt_ii, x_ii_1, tv_ref_tc)
                if self.terminal_constraint.is_set and ii == self._prediction_horizon - 1:
                    if self.terminal_constraint.is_soft:
                        residual = self._terminal_constraints_fun(time, x_ii, zp[ii, 0], p_ii, e_soft_term)
                        J += self.terminal_constraint.cost(e_soft_term)
                        g.append(residual)
                        g_lb.append([-ca.inf] * self.terminal_constraint.size * 2)
                        g_ub.append([ub for ub in self.terminal_constraint.ub])
                        g_ub.append([-lb for lb in self.terminal_constraint.lb])
                        if self._nlp_options['ipopt_debugger']:
                            self._g_indices['nonlin_term_const'].append(
                                [ind_g, ind_g + self.terminal_constraint.size * 2])
                            ind_g += self.terminal_constraint.size * 2
                    else:
                        residual = self._terminal_constraints_fun(time, x_ii_1, zp[ii, 0], p_ii)
                        g.append(residual)
                        g_lb.append(self.terminal_constraint.lb)
                        g_ub.append(self.terminal_constraint.ub)
                        if self._nlp_options['ipopt_debugger']:
                            self._g_indices['nonlin_term_const'].append(
                                [ind_g, ind_g + residual.size1()])
                            ind_g += residual.size1()

                if self.stage_constraint.is_set:
                    if self.stage_constraint.is_soft:
                        residual = self._stage_constraints_fun(time, x_ii, u_ii, zp[ii, 0], p_ii, e_soft_stage)
                        J += self.stage_constraint.cost(e_soft_stage)
                        g.append(residual)
                        g_lb.append([-ca.inf] * self.stage_constraint.size * 2)
                        g_ub.append([ub for ub in self.stage_constraint.ub])
                        g_ub.append([-lb for lb in self.stage_constraint.lb])
                        if self._nlp_options['ipopt_debugger']:
                            self._g_indices['nonlin_stag_const'].append(
                                [ind_g, ind_g + residual.size1()])
                            ind_g += residual.size1()
                    else:
                        residual = self._stage_constraints_fun(time, x_ii, u_ii, zp[ii, 0], p_ii)
                        g.append(residual)
                        g_lb.append(self.stage_constraint.lb)
                        g_ub.append(self.stage_constraint.ub)
                        if self._nlp_options['ipopt_debugger']:
                            self._g_indices['nonlin_stag_const'].append(
                                [ind_g, ind_g + residual.size1()])
                            ind_g += residual.size1()

                # update time in the horizon
                time += dt_ii

            if self._custom_constraint_flag:
                if self._custom_constraint_is_soft_flag:
                    W = np.diag([10000] * self._custom_constraint_size)
                    J += ca.mtimes(e_cus.T, ca.mtimes(W, e_cus))
                    g.append(self._custom_constraint_fun(v, x_ind, u_ind) - e_cus)
                    g_lb.append([-ca.inf] * self._custom_constraint_size)
                    g_ub.append(self._custom_constraint_fun_ub)

                    g.append(self._custom_constraint_fun(v, x_ind, u_ind) + e_cus)
                    g_lb.append(self._custom_constraint_fun_lb)
                    g_ub.append([ca.inf] * self._custom_constraint_size)
                else:
                    g.append(self._custom_constraint_fun(v, x_ind, u_ind))
                    g_lb.append(self._custom_constraint_fun_lb)
                    g_ub.append(self._custom_constraint_fun_ub)

            if self._minimize_final_time_flag:
                # Force all the sampling time to be equal
                for kkk in range(self.prediction_horizon - 1):
                    g.append(_dt[kkk] - _dt[kkk + 1])
                    g_lb.append(0)
                    g_ub.append(0)

                # Add the time to the objective function
                J += ca.sum1(_dt) * self._minimize_final_time_weight

            g = ca.vertcat(*g)
            self._g_lb = ca.DM(ca.vertcat(*g_lb))
            self._g_ub = ca.DM(ca.vertcat(*g_ub))
            self._v0 = ca.DM(v_guess)
            self._v_lb = ca.DM(v_lb)
            self._v_ub = ca.DM(v_ub)
            self._u_ind = u_ind
            self._x_ind = x_ind
            self._dt_ind = t_ind
            self._J = J
            self._v = v
            self._param_npl_mpc = param_npl_mpc
            self._g = g
            self._nlp_setup_done = True
            self._n_v = n_v

            if self._nlp_options['ipopt_debugger']:
                # Adds the callback debugger.
                debugger = IpoptDebugger('ipopt_debugger', self._n_v, self._g.shape[0], 0, 0, self._x_ind, self._u_ind)
                self.debugger = debugger
                self._nlp_opts.update({'iteration_callback': debugger})

            nlp_dict = {'f': self._J, 'x': self._v, 'p': self._param_npl_mpc, 'g': self._g}
            if self._solver_name in self._solver_name_list_nlp:
                solver = ca.nlpsol('solver', self._solver_name, nlp_dict, self._nlp_opts)
            elif self._solver_name in self._solver_name_list_qp:
                solver = ca.qpsol('solver', self._solver_name, nlp_dict, self._nlp_opts)
            else:
                raise ValueError(
                    f"The solver {self._solver_name} does no exist. The possible solver are {self._solver_name_list}."
                )
            self._solver = solver

    def return_prediction(self):
        """
        Returns the mpc prediction.

        :return: x_pred, u_pred, t_pred
        """

        if self._nlp_solution is not None:
            x_pred = np.zeros((self._model.n_x, self._prediction_horizon + 1))
            u_pred = np.zeros((self._model.n_u, self._control_horizon))
            dt_pred = np.zeros(self._prediction_horizon)
            for ii in range(self._prediction_horizon + 1):
                x_pred[:, ii] = np.asarray(self._nlp_solution['x'][self._x_ind[ii]]).squeeze() * self._x_scaling
            for ii in range(self._control_horizon):
                u_pred[:, ii] = np.asarray(self._nlp_solution['x'][self._u_ind[ii]]).squeeze() * self._u_scaling
            if len(self._dt_ind) > 0:
                for ii in range(self._prediction_horizon):
                    dt_pred[ii] = np.asarray(self._nlp_solution['x'][self._dt_ind[ii]]).squeeze()
            else:
                dt_pred = None
            return x_pred, u_pred, dt_pred

        else:
            warnings.warn("There is still no mpc solution available. Run mpc.optimize() to get one.")
            return None, None, None

    def get_reference(self, time):
        """
        Returns reference values at a given time. If works for path following, trajectory tracking and reference
        tracking problems. For the reference tracking problem it returns always a (constant) reference.

        :param time:
        :return:
        """
        references = []
        for ref in self.quad_stage_cost._references_list:
            value = ref['ref']
            references.append({'type': ref['type'], 'value': value, 'ind': ref['ind']})
        for ref in self.quad_stage_cost._trajectories_list:
            traj_i = ref['ref']
            if traj_i is not None:
                if isinstance(traj_i, ca.SX):
                    # In this case the trajectory is given as a function, which has to be evaluated at
                    # the current time
                    value = ca.DM(ca.substitute(traj_i, self.time_var, time))
                else:
                    # In this case, the trajectory is just a series of values
                    value = traj_i[self.n_iterations]
                references.append({'type': ref['type'], 'value': value, 'ind': ref['ind']})
            else:
                raise ValueError("You did not supply a reference trajectory for the trajectory tracking problem."
                                 "Use set_trajectory to set it up or add a function trajectory in the cost function."
                                 )

        for _ in self.quad_stage_cost._paths_list:
            for i, path_var in enumerate(self._controller._paths_var_list):
                theta = path_var['theta']
                path_i = ca.substitute(path_i, theta, self._controller.solution[f'thetapf{i}'][0])

        return references

    @property
    def obj_fun(self):
        """

        :return: Objective function (MX)
        """
        if self._nlp_setup_done is False:
            raise AttributeError(
                "You need to setup the NLP by running mpc.setup() method before accessing the objective function."
            )
        return self._J

    @property
    def prediction_horizon(self):
        """

        :return:
        """
        return self._prediction_horizon

    @prediction_horizon.setter
    def prediction_horizon(self, arg):
        if isinstance(arg, int) and arg >= 1:
            self._prediction_horizon = arg
            self._prediction_horizon_is_set = True
        else:
            raise ValueError("The horizon numer must be a positive nonzero integer")

    @property
    def control_horizon(self):
        """

        :return:
        """
        return self._control_horizon

    @control_horizon.setter
    def control_horizon(self, arg):
        if isinstance(arg, int) and arg >= 1:
            self._control_horizon = arg
            self._control_horizon_is_set = True
        else:
            raise ValueError("The horizon numer must be a positive nonzero integer")

    @property
    def n_of_path_vars(self):
        """
        Number of path variables used for path following

        :return:
        """
        return len(self._paths_var_list)

    @property
    def n_tvp(self):
        """

        :return:
        """
        return self._n_tvp

    @property
    def time_var(self):
        """

        :return:
        """
        return self._time_var


class LMPC(Controller, DynamicOptimization):
    """"""
    def __init__(self, model, id=None, name=None, plot_backend=None, use_sx=True):
        """Constructor method"""
        super().__init__(model, id=id, name=name, plot_backend=plot_backend)
        if not model.is_linear():
            raise TypeError("The model must be linear. Use the NMPC class instead.")
        if not model.discrete:
            raise TypeError("The model not discrete-time. Use the NMPC class instead.")

        self._may_term_flag = False
        self._lag_term_flag = False
        self._prediction_horizon_is_set = False
        self._control_horizon_is_set = False
        self._change_input_term_flag = False
        self._mixed_integer_flag = False
        self._trajectory_following_flag = False
        self._mpc_options_is_set = False
        self._paths_var_list = []
        self._time = 0
        self._n_iterations = 0

        # Initialize the costs
        self._time_var = []

        self._lag_term = 0
        self._may_term = 0

        self._minimize_final_time_flag = False
        self._time_varying_parameters_ind = []
        self.x_ub = []
        self.x_lb = []
        self.u_ub = []
        self.u_lb = []
        self.time_varying_parameters = []
        self._x_scaling = []
        self._u_scaling = []
        self._y_scaling = []
        self._prediction_horizon = []
        self._control_horizon = []

        self._e_soft_stage_ind = []
        self._e_term_stage_ind = []

        # In some cases, for example path following with interpolated path using spline, the objective
        # function needs to be defined in MX variable instead of SX, because the CasADi spline interpolation
        # is defined only in MX functions.
        self._use_sx = use_sx

        # Save dimensions of the model. This can be useful because the MPC can change these later, but we still need
        # them
        self._n_x = deepcopy(model.n_x)
        self._n_u = deepcopy(model.n_u)
        self._n_y = deepcopy(model.n_y)
        self._n_z = deepcopy(model.n_z)
        self._n_p = deepcopy(model.n_p)
        self._n_m = 1
        self._options = {}

        self._solver_name = 'qpoases'

    def _update_type(self) -> None:
        """

        :return:
        """
        self._type = 'LMPC'

    def _parse_tvp_parameters_values(self, tvp):
        """

        :param tvp:
        :return:
        """
        # Get horizon of tvp values
        if tvp is not None:
            self._time_varying_parameters_horizon = ca.DM.zeros(self._n_tvp, self.prediction_horizon)
            tvp_counter = 0
            for key, value in tvp.items():
                if len(value) < self.prediction_horizon:
                    raise TypeError(
                        "When passing time-varying parameters, you need to pass a number of values at least "
                        f"as long as the prediction horizon. The parameter {key} has {len(value)} values but the MPC "
                        f"has a prediction horizon length of {self._prediction_horizon}."
                    )

                value = tvp[key]
                self._time_varying_parameters_horizon[tvp_counter, :] = value[:self._prediction_horizon]
                tvp_counter += 1
        elif self._time_varying_parameters_values is not None:
            self._get_tvp_parameters_values()
        else:
            raise ValueError(
                f"Mate, I know there are {self._n_tvp} time varying parameters but you did not pass me any."
                f"Please provide me with the values of the parameters, either to the optimize() method or to the "
                f"set_time_varying_parameters() method."
            )

    def setup(self, options=None, solver_options=None, nlp_solver='qpoases'):
        """

        :param options:
        :param solver_options:
        :param nlp_solver:
        :return:
        """
        if not self._scaling_is_set:
            self.set_scaling()
        if not self._time_varying_parameters_is_set:
            self.set_time_varying_parameters()
        if not self._box_constraints_is_set:
            self.set_box_constraints()
        if not self._initial_guess_is_set:
            self.set_initial_guess()

        if not self._nlp_solver_is_set:
            self.set_nlp_solver(nlp_solver)
        if not self._sampling_time_is_set:
            self.set_sampling_interval()

        n_x = self._n_x
        n_u = self._n_u

        A = self._model.state_matrix
        B = self._model.input_matrix
        Q = self.Q
        R = self.R

        dim_states = n_x * (self._horizon + 1)
        dim_control = n_u * self._horizon

        # Build Adis
        aux1 = np.eye(self._horizon, self._horizon + 1)
        Abar1 = ca.kron(aux1, A)
        aux2 = np.zeros((self._horizon, self._horizon + 1))

        # Save indices variables
        u_ind = []
        x_ind = []

        offset_x = 0
        x_ind.append([j for j in range(offset_x, offset_x + n_x)])
        offset_x += n_x

        offset_u = (self._horizon + 1) * n_x
        for i in range(self._horizon):
            aux2[i, i + 1] = -1
            x_ind.append([j for j in range(offset_x, offset_x + n_x)])
            u_ind.append([j for j in range(offset_u, offset_u + n_u)])
            offset_x += n_x
            offset_u += n_u

        Abar2 = ca.kron(aux2, ca.DM.eye(n_x))
        aux3 = np.eye(self._horizon, self._horizon)
        Abar3 = ca.kron(aux3, B)

        # Add constraints for the ode
        Adis = ca.horzcat(Abar1 + Abar2, Abar3)

        # generate parameters
        bdis = ca.kron(ca.DM.zeros(self._model.n_x), ca.DM.ones(self.horizon))
        # Add constraints for the polytope constraints
        # TODO, they should be added to A

        H_states = ca.kron(ca.DM.eye(self._horizon + 1), Q)
        H_control = ca.kron(ca.DM.eye(self._horizon), R)

        H = ca.diagcat(H_states, H_control)

        g = ca.DM.zeros((dim_states + dim_control, 1))
        lb_states = ca.kron(ca.DM.ones((self._horizon + 1, 1)), self._x_lb)
        ub_states = ca.kron(ca.DM.ones((self._horizon + 1, 1)), self._x_ub)

        lb_control = ca.kron(ca.DM.ones((self._horizon, 1)), self._u_lb)
        ub_control = ca.kron(ca.DM.ones((self._horizon, 1)), self._u_ub)

        v_lb = ca.vertcat(lb_states, lb_control)
        v_ub = ca.vertcat(ub_states, ub_control)

        qp = {
            'h': H.sparsity(),
            'a': Adis.sparsity()
        }

        # TODO move this check of the solver into the setup_solver method
        if self._solver_name in self._solver_name_list_qp:
            solver = ca.conic("solver", self._solver_name, qp, self._nlp_opts)
        elif self._solver_name == 'muaompc':
            try:
                from ..embedded.muaompc import setup_solver
            except ImportError as err:
                message = f"{err}."
                message += "\nTo use nlp_solver='muaompc' first install muaompc."
                message += " Try:\n    pip install muaompc"
                raise ImportError(message)

            solver = setup_solver(self)
        else:
            raise ValueError(
                f"The solver {self._solver_name} does no exist. The possible solver are {self._solver_name_list_qp}."
            )
        self._solver = solver

        self._H = ca.DM(H)
        self._g = g
        self._Ad = Adis
        self._Ad_lb = bdis
        self._Ad_ub = bdis
        self._v_lb = v_lb
        self._v_ub = v_ub
        self._x_ind = x_ind
        self._u_ind = u_ind

    def optimize(self, x0, tvp=None, cp=None):
        """

        :param x0:
        :param tvp:
        :param cp:
        :return:
        """
        # TODO: substitute all the values of all the variables remaining in the matrices
        if self._solver_name == 'muaompc':
            return self._solver(x0)

        if self._model.n_p - self._n_tvp != 0:
            if cp is not None:
                cp = check_and_wrap_to_DM(cp)

            if cp is None or cp.size1() != self._model.n_p - self._n_tvp:
                raise ValueError(
                    f"The model has {self._model.n_p - self._n_tvp} constant parameter(s): "
                    f"{self._model.parameter_names}. You must pass me the value of these before running the "
                    f"optimization to the 'cp' parameter."
                )
        else:
            if cp is not None:
                warnings.warn(
                    "You are passing a parameter vector in the optimizer, but the model has no defined parameters. "
                    "I am ignoring the vector."
                )

        x0 = check_and_wrap_to_DM(x0)

        if self._n_tvp > 0:
            self._parse_tvp_parameters_values(tvp)

        Ad = self._Ad
        Ad_ub = self._Ad_ub
        Ad_lb = self._Ad_lb

        Ad = ca.substitute(Ad, self._model.dt, self._sampling_interval)
        Ad_ub = ca.substitute(Ad_ub, self._model.dt, self._sampling_interval)
        Ad_lb = ca.substitute(Ad_lb, self._model.dt, self._sampling_interval)

        if cp is not None:
            ind_cp_par = [i for i in range(self._model.n_p) if i not in self._time_varying_parameters_ind]
            Ad = ca.substitute(Ad, self._model.p[ind_cp_par], cp)
            Ad_ub = ca.substitute(Ad_ub, self._model.p[ind_cp_par], cp)
            Ad_lb = ca.substitute(Ad_lb, self._model.p[ind_cp_par], cp)

        self._v_lb[self._x_ind[0][0:self._n_x]] = x0
        self._v_ub[self._x_ind[0][0:self._n_x]] = x0

        if self._n_tvp:
            Ad_ub = ca.substitute(Ad_ub, self._tvp, self._time_varying_parameters_horizon)
            Ad_lb = ca.substitute(Ad_lb, self._tvp, self._time_varying_parameters_horizon)

        Ad = ca.DM(Ad)
        Ad_ub = ca.DM(Ad_ub)
        Ad_lb = ca.DM(Ad_lb)

        sol = self._solver(h=self._H, g=self._g, a=Ad, lbx=self._v_lb, ubx=self._v_ub, lba=Ad_lb, uba=Ad_ub)

        self._nlp_solution = sol
        u_opt = sol['x'][self._u_ind[0]]

        self._n_iterations += 1

        return u_opt

    @property
    def prediction_horizon(self):
        """

        :return:
        """
        return self._horizon


__all__ = [
    'NMPC',
    'LMPC'
]
