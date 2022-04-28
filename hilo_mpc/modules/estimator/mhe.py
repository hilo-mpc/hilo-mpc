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
import warnings

import casadi as ca
import casadi.tools as catools
import numpy as np

from .base import Estimator
from ..optimizer import DynamicOptimization
from ...util.modeling import MHEQuadraticCost, GenericConstraint, continuous2discrete
from ...util.util import check_and_wrap_to_list, check_and_wrap_to_DM, check_if_list_of_none, check_if_list_of_string, \
    scale_vector, who_am_i


class MovingHorizonEstimator(Estimator, DynamicOptimization):
    """
    Moving Horizon Estimator (MHE) class for state and parameter estimation.

    :param model: HILO-MPC model object that will be used in the estimator.
    :type model: HILO-MPC Model
    :param id: id name
    :type id:  str, optional
    :param name: The name of the MHE. By default the MHE has no name.
    :type name:  str, optional
    :param plot_backend: Plot package. Can be 'bokeh' (recommended) or 'matplotlib' (default: matplotlib).
    :type plot_backend:  str, optional
    :param time: Initial time. Useful for time-varying systems (default: 0 ).
    :type time:  float or int, optional


    mhe_opts: options list

    =================== =============================
    Opts                Description
    =================== =============================
    arrival_cost_update Can be smoothing or filtering
    =================== =============================

    """
    def __init__(self, model, id=None, name=None, plot_backend=None, time=0) -> None:
        """Constructor method"""
        super().__init__(model, id=id, name=name, plot_backend=plot_backend)

        self._nlp_options_is_set = False
        self._arrival_term_flag = False
        self._stage_term_flag = False
        self._state_noise_flag = False
        self._horizon_is_set = False
        self._box_constraints_is_set = False
        self._initial_guess_is_set = False
        self._integration_opts_is_set = False
        self._nlp_opts_is_set = False
        self._nlp_solver_is_set = False
        self._stage_constraints_flag = False
        self._custom_constraints_flag = False
        self._change_mode_term_flat = False
        self._scaling_is_set = False
        self._time_varying_parameters_is_set = False
        self._sampling_time_is_set = False
        self._soft_constraints_flag = False
        self._mixed_integer_flag = False
        self._mode_guess_is_set = False
        self._stage_term = 0
        self._arrival_term = 0
        self._horizon_is_reached = False

        # Initialize the costs
        self.quad_stage_cost = MHEQuadraticCost(self._model)
        self.quad_arrival_cost = MHEQuadraticCost(self._model)

        self._horizon = int()
        # This will be populated with CasADi symbolics and represent the state and measurement noise respectively
        self._y_meas = ca.SX.sym('y', 0)
        self._w = ca.SX.sym('w', 0)

        self._time_varying_parameters_ind = None
        # Clock time
        self._time = time

        if self._model.n_y == 0:
            warnings.warn(
                f"The model has no measurement equations, I am assuming measurements of all states "
                f"{self._model.dynamical_state_names} are available."
            )
            self._meas_eq_exist = False
        else:
            self._meas_eq_exist = True

        if self._meas_eq_exist:
            meas_names = self._model.measurement_names
        else:
            meas_names = self._model.dynamical_state_names

        for s in meas_names:
            s_w = ca.SX.sym(s + '_meas')
            self._y_meas = ca.vertcat(*[self._y_meas, s_w])

        # Here the measurements of input, output and time varying parameters
        self._y_history = []
        self._u_history = []
        self._tvp_history = []
        # Counter that counts the number of measurements collected
        self._meas_counter = 0

        self.meas_num = len(meas_names)

        self.change_input_term = []
        self.x_ub = []
        self.x_lb = []
        self.x_guess = []
        self.w_ub = []
        self.w_lb = []
        self.w_guess = []
        self._p_ind = []
        self._x_ind = []
        self._w_ind = []

        # Initialize generic stage and terminal constraints
        self.stage_constraint = GenericConstraint(self._model)

        self.time_varying_parameters = []
        self.x_scaling = []
        self.w_scaling = []

    def _update_type(self) -> None:
        """

        :return:
        """
        self._type = 'MHE'

    def _define_cost_terms(self):
        """
        Creates the SX expressions for the stage or terminal cost

        :return:
        """
        self._state_noise_flag = self.quad_stage_cost._has_state_noise

        if self._state_noise_flag:
            self._w = self.quad_stage_cost.w

        self.quad_stage_cost._setup(x_scale=self._x_scaling, u_scale=self._u_scaling,
                                    p_scale=self._p_scaling,
                                    w_scale=self._w_scaling)
        if self.quad_stage_cost._is_set:
            self._stage_term += self.quad_stage_cost.cost
            self._stage_term_flag = True

        self.quad_arrival_cost._setup(x_scale=self._x_scaling, u_scale=self._u_scaling,
                                      p_scale=self._p_scaling,
                                      w_scale=self._w_scaling)
        if self.quad_arrival_cost._is_set:
            self._arrival_term += self.quad_arrival_cost.cost
            self._arrival_term_flag = True

    def _check_measurements(self, y, u):
        """

        :param y:
        :param u:
        :return:
        """
        y = check_and_wrap_to_list(y)

        if not len(y) == self.meas_num:
            raise ValueError(
                f'You passed {len(y)} output measurements but the measurement equation has {self.meas_num} output')

        if u is not None:
            u = check_and_wrap_to_list(u)
            if not len(u) == self._model.n_u:
                raise ValueError(
                    f'You passed {len(u)} input measurements but the model has {self._model.n_u} inputs')

        return y, u

    def _check_mhe_is_well_posed(self):
        """

        :return:
        """
        # TODO make sure all the important things are checked
        if self._stage_term_flag is False and self._arrival_term_flag is False:
            raise ValueError("You need to define a cost function before setting up the MHE.")
        if self._horizon is None:
            raise ValueError("You must set a prediction horizon length before")

        if len(self._x_ub) != len(self._x_lb):
            raise ValueError("x_ub has not the same dimension of x_lb.")

        if len(self._x_ub) != self._model.n_x:
            raise ValueError(f"x_ub has not the same dimension of the state of the model. "
                             f"x_ub has dimension {len(self._x_ub)} while"
                             f"the model has {self._model.n_x} states")

        if len(self._x_lb) != self._model.n_x:
            raise ValueError(f"x_lb has not the same dimension of the state of the model. "
                             f"x_lb has dimension {len(self._x_lb)} while"
                             f"the model has {self._model.n_x} states")

    def _scale_problem(self):
        """

        :return:
        """
        self._x_ub = scale_vector(self._x_ub, self._x_scaling)
        self._x_lb = scale_vector(self._x_lb, self._x_scaling)
        self._x_guess = scale_vector(self._x_guess, self._x_scaling)

        self._w_ub = scale_vector(self._w_ub, self._w_scaling)
        self._w_lb = scale_vector(self._w_lb, self._w_scaling)
        self._w_guess = scale_vector(self._w_guess, self._w_scaling)

        if self._model.n_p > 0:
            self._p_ub = scale_vector(self._p_ub, self._p_scaling)
            self._p_lb = scale_vector(self._p_lb, self._p_scaling)
            self._p_guess = scale_vector(self._p_guess, self._p_scaling)

        # ... ode ...
        self._model.scale(self._u_scaling, id='u')
        self._model.scale(self._x_scaling, id='x')
        self._model.scale(self._p_scaling, id='p')

    def _update_arrival_states(self):
        """
        The state guess in the arrival cost can be updated automatically depending on the method.

        :return:
        """
        if self._nlp_solution is not None:
            if self._nlp_options['arrival_guess_update'] == 'smoothing':
                return self._nlp_solution['x'][self._x_ind[2]]
            elif self._nlp_options['arrival_guess_update'] == 'filtering':
                raise NotImplementedError("The filtering update is not yet implemented.")
        else:
            return self.quad_arrival_cost.x_guess

    def _update_arrival_param(self):

        """
        The parameter guess in the arrival cost can be updated automatically

        :return:
        """
        if self._nlp_solution is not None:
            return self._nlp_solution['x'][self._p_ind[0]]
        else:
            return self.quad_arrival_cost.p_guess

    def add_measurements(self, y_meas, u_meas=None):
        """
        This adds measurements that will appended to the measurements history.
        At the moment it is assumed that measurements are available ata  constant time interval.

        :param y_meas:
        :param u_meas:
        :return:
        """
        if not self._nlp_setup_done:
            raise RuntimeError(f"You need to setup the MHE by running .setup() before running {who_am_i()}")

        # Add measurements to the history
        y_meas = check_and_wrap_to_list(y_meas)
        if u_meas is not None:
            u_meas = check_and_wrap_to_list(u_meas)

        # Quality check
        y_meas, u_meas = self._check_measurements(y_meas, u_meas)

        self._y_history.append(y_meas)
        if u_meas is not None:
            self._u_history.append(u_meas)

        self._meas_counter += 1

        if self._meas_counter == self._horizon:
            # The horizon has been reached. The MHE can start
            self._horizon_is_reached = True

        if self._meas_counter > self._horizon:
            # Shift horizon
            # The last measurements will be forgotten
            self._y_history.pop(0)
            if u_meas is not None:
                self._u_history.pop(0)

    def estimate(self, x_arrival=None, p_arrival=None, v0=None, runs=0, **kwargs):
        """
        Compute MHE

        :param x_arrival:
        :param p_arrival:
        :param v0: initial guess of the optimal vector
        :param runs: number of optimizations to run. If different than zero will run very optimization will perturb the
            initial guess v0 randomly.
            ACHTUNG: this could cause problems with the integrators or give something outside constraints.
            The output will be the solution with the minimum objective function (default 0)
        :param kwargs:
        :return: u_opt: first piece of optimal control sequence
        """
        # TODO Check the shape of p0, x0
        # TODO to test
        if not self._nlp_setup_done:
            raise ValueError("You need to setup the nlp before optimizing. Type *mheObject*.setup()")

        if not isinstance(runs, int) and not runs > 0:
            raise TypeError("the 'runs' parameter must be a positive integer")

        # Update the time
        self._time += self._sampling_interval

        if self._horizon_is_reached:
            # Get external parameters
            param = self._ext_parameters(0)

            if x_arrival is not None:
                x_arrival = check_and_wrap_to_DM(x_arrival)
                if len(x_arrival) != self._model.n_x:
                    raise ValueError(
                        'The model has {} states(s): {}. You must pass me a guess of the values before '
                        'running the optimization'.format(
                            self._model.n_x, self._model.dynamical_state_names))
            else:
                x_arrival = self._update_arrival_states()

            param['x_arrival'] = x_arrival

            if self._model.n_p > 0:
                if p_arrival is not None:
                    p_arrival = check_and_wrap_to_DM(p_arrival)
                else:
                    p_arrival = self._update_arrival_param()
                param['p_arrival'] = p_arrival

            if self._model.n_y > 0:
                param['y_meas'] = np.array(self._y_history).T  # y_meas must be self._horizon, self.meas_num
            if self._model.n_u > 0:
                param['u_meas'] = np.array(self._u_history).T  # y_meas must be self._horizon, self._model.n_u

            # TODO when varying sampling times are implemented, give the possibility to provide varying sampling times.
            dt_grid = None
            if dt_grid is None:
                param['dt'] = ca.repmat(self.sampling_interval, self.horizon)

            param['time'] = self._time

            if v0 is None:
                v0 = self._v0

            if runs == 0:
                sol = self._solver(x0=v0, lbx=self._v_lb, ubx=self._v_ub, lbg=self._g_lb, ubg=self._g_ub, p=param)
                self._nlp_solution = sol

                if self._model.n_p > 0:
                    p_opt = sol['x'][self._p_ind[0]] * self._p_scaling

                # NOTE: this returns the one step-ahead prediction.
                #  To return the filtered prediction one need to access -2, but remember
                #  that the time that enters the solution object must be one time step back!
                x_opt = sol['x'][self._x_ind[-1]] * self._x_scaling
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
                        if self._model.n_p:
                            p_opt = sol['x'][self._p_ind[0]] * self._p_scaling
                        x_opt = sol['x'][self._x_ind[-1]] * self._x_scaling
                        self._v0 = sol['x']
                    v00 = v0 + v0 * (1 - 2 * np.random.rand(self._n_v)) * pert_factor

            # Get the status of the solver
            self._solver_status_wrapper()

            # print output
            self._print_message()
            if self._model.n_p > 0:
                self._solution.add('p', p_opt)
                self._solution.add('x', x_opt)
                self._solution.add('t', self._time)
                return x_opt, p_opt
            else:
                self._solution.add('x', x_opt)
                self._solution.add('t', self._time)
                return x_opt, None
        else:
            return None, None

    def setup(self, options=None, nlp_opts=None, solver='ipopt'):
        """

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
        if not self._stage_constraints_flag:
            self.set_aux_nonlinear_constraints()

        if not self._nlp_options_is_set:
            self.set_nlp_options(options)
        if not self._nlp_solver_is_set:
            self.set_nlp_solver(solver)  # solver ty-e
        if not self._solver_options_is_set:
            self.set_solver_opts()  # solver ipopt.blabla

        if not self._sampling_time_is_set:
            self.set_sampling_interval()

        # Setup the solution TimeSeries
        self._populate_solution()

        # Define cost terms
        self._define_cost_terms()

        # Scaling...
        self._scale_problem()

        # Check if NLP is well posed
        self._check_mhe_is_well_posed()

        # Get all the variables of the time-varying references. These need to be passed to the stage function
        t_ref_placeholder = ca.SX.sym('t_ref', 0)
        for tv_ref in self.quad_stage_cost._tv_ref_list:
            t_ref_placeholder = ca.vertcat(t_ref_placeholder, tv_ref['placeholder'])

        # Get all the variables of the references that change once per iteration. These need to be passed to the stage
        # function
        p_arrival = ca.SX.sym('p_arrival', 0)
        x_arrival = ca.SX.sym('x_arrival', 0)
        for iter_ref in self.quad_arrival_cost._iter_ref_list:
            if iter_ref['type'] == 'parameters':
                p_arrival = iter_ref['placeholder']
            elif iter_ref['type'] == 'states':
                x_arrival = iter_ref['placeholder']

        if self._nlp_setup_done is False:
            model = self._model

            # ... objective function.
            if self._arrival_term_flag:
                self._arrival_term_fun = ca.Function('arrival_term',
                                                     [model.x, model.p, x_arrival, p_arrival],
                                                     [self._arrival_term])

            if self._stage_term_flag:
                self._stage_term_fun = ca.Function('stage_term',
                                                   [self._w, model.x, t_ref_placeholder],
                                                   [self._stage_term])

            # Check time varying parameters
            tvp_ind = []
            if len(self._time_varying_parameters) != 0:
                p_names = model.parameter_names
                for tvp in self._time_varying_parameters:
                    assert tvp in p_names, f"The time-varying parameter {tvp} is not in the model parameter. " \
                                           f"The model parameters are {p_names}."
                    tvp_ind.append(p_names.index(tvp))

                self._time_varying_parameters_ind = tvp_ind
            self._n_tvp = len(tvp_ind)

            # Define CasADi function of auxiliary nonlinear constraints
            if self.stage_constraint.is_set:
                if self.stage_constraint.is_soft:
                    e_stage = ca.SX.sym('e_stage', self.stage_constraint.constraint.size1())
                    self._stage_constraints_fun = ca.Function('soft_aux_nl_constr',
                                                              [model.x, self._w, model.p, e_stage],
                                                              [ca.vertcat(self.stage_constraint.constraint - e_stage,
                                                                          -self.stage_constraint.constraint - e_stage)])
                else:
                    self._stage_constraints_fun = ca.Function('soft_aux_nl_constr',
                                                              [model.x, self._w, model.p],
                                                              [self.stage_constraint.constraint])

            problem = dict(self._model)

            if self._nlp_options['integration_method'] == 'collocation':
                continuous2discrete(problem, **self._nlp_options)

                # Slack for soft constraints
                ek = ca.SX.sym('e', self.stage_constraint.size)

                # Add all constraints to the collocation points box constraints
                # TODO is here options['degree']+ 1 or problem['ode']???
                n_xik = (self._nlp_options['degree']) * model.n_x
                n_zik = (self._nlp_options['degree']) * model.n_z

                x_ik_guess = np.tile(self._x_guess, self._nlp_options['degree'])
                x_ik_ub = np.tile(self._x_ub, self._nlp_options['degree'])
                x_ik_lb = np.tile(self._x_lb, self._nlp_options['degree'])

                if model.n_z > 0:
                    z_ik_guess = np.tile(self._z_guess, self._nlp_options['degree'])
                    z_ik_ub = np.tile(self._z_ub, self._nlp_options['degree'])
                    z_ik_lb = np.tile(self._z_lb, self._nlp_options['degree'])

                # nonlinear constraints
                # Constraints in the control interval
                gk_col = []
                gk_col_lb = []
                gk_col_ub = []
                for k in range(self._nlp_options['degree']):
                    x_col = problem['collocation_points_ode'][k]
                    if self.stage_constraint.is_set:
                        if self.stage_constraint.is_soft:
                            residual = self._stage_constraints_fun(x_col, self._w, problem['p'], ek)
                            gk_col.append(residual)
                            gk_col_lb.append(np.repeat(-np.inf, self.stage_constraint.size * 2))
                            gk_col_ub.append(self.stage_constraint.ub)
                            gk_col_ub.append([-lb for lb in self.stage_constraint.lb])
                        else:
                            residual = self._stage_constraints_fun(x_col, self._w, problem['p'])
                            gk_col.append(residual)
                            gk_col_lb.append(self.stage_constraint.lb)
                            gk_col_ub.append(self.stage_constraint.ub)

                gk_col.append(problem['collocation_equations'])
                gk_col_lb.append(np.zeros(problem['collocation_equations'].shape[0]))
                gk_col_ub.append(np.zeros(problem['collocation_equations'].shape[0]))

                # Create function
                int_dynamics_fun = ca.Function('integrator_collocation',
                                               [problem['t'],  # time variable (for time varying systems)
                                                problem['dt'],  # dt variable (for possibly different sampling time)
                                                ca.vertcat(*problem['collocation_points_ode']),
                                                # x at collocation points
                                                problem['x'],  # x at the beginning of the interval
                                                problem['u'],  # input of the interval
                                                ca.vertcat(*problem['collocation_points_alg']),  # alg states at coll.
                                                problem['p'],  # parameters (constant over the interval at least)
                                                ek,  # slack variable for (possibly) soft constrained systems
                                                # self._w,
                                                t_ref_placeholder],
                                               [ca.vertcat(*gk_col), problem['ode']])
            elif self._nlp_options['integration_method'] == 'discrete':
                int_dynamics_fun = ca.Function('integrator_discrete',
                                               [problem['t'],
                                                problem['dt'],
                                                problem['x'],
                                                problem['u'],
                                                problem['p'],
                                                ],
                                               [problem['ode']])
            elif self._nlp_options['integration_method'] == 'multiple_shooting':
                x = problem['x']
                u = problem['u']
                p = problem['p']
                z = problem['z']
                if model.n_z == 0:
                    dae = {'x': x, 'p': ca.vertcat(u, p, t_ref_placeholder), 'ode': model.ode}
                else:
                    dae = {'x': ca.vertcat(x, z), 'p': ca.vertcat(u, p, t_ref_placeholder),
                           'ode': ca.vertcat(model.ode, model.alg)}

                opts = {'abstol': 1e-10, 'reltol': 1e-10, 'tf': self._sampling_interval}
                int_dynamics_fun = ca.integrator("integrator_ms", model.solver, dae, opts)
            else:
                raise ValueError(f"Integration {self._nlp_options['integration_method']} not defined.")

            # Total number of optimization variable
            n_v = model.n_x * (self._horizon + 1) + self._model._n_p
            if self._state_noise_flag:
                n_v += self._horizon * model.n_x
            if self._nlp_options['integration_method'] == 'collocation':
                n_v += self._horizon * (n_xik + n_zik)
            if self.stage_constraint.is_soft:
                n_v += self.stage_constraint.constraint.size1()

            v = ca.MX.sym('v', n_v)

            # All variables with bounds and initial guess
            v_lb = np.zeros(n_v)
            v_ub = np.zeros(n_v)
            v_guess = np.zeros(n_v)
            offset = 0

            # Get parameters
            if self._model.n_p > 0:
                p_ind = self._p_ind
                p = v[offset:offset + self._model.n_p]
                p_ind.append([j for j in range(offset, offset + self._model.n_p)])
                v_lb[offset:offset + self._model.n_p] = self._p_lb
                v_ub[offset:offset + self._model.n_p] = self._p_ub
                v_guess[offset:offset + self._model.n_p] = self._p_guess
                offset += self._model.n_p
            else:
                p = []

            # Predefine optimization variable
            x = np.resize(np.array([], dtype=ca.MX), (self._horizon + 1, 1))
            x_ind = self._x_ind
            for ii in range(self._horizon + 1):
                x[ii, 0] = v[offset:offset + model.n_x]
                x_ind.append([j for j in range(offset, offset + model.n_x)])
                v_guess[offset:offset + model.n_x] = self._x_guess
                v_lb[offset:offset + model.n_x] = self._x_lb
                v_ub[offset:offset + model.n_x] = self._x_ub
                offset += model.n_x

            # Predefine optimization variable
            if self._state_noise_flag:
                w = np.resize(np.array([], dtype=ca.MX), (self._horizon, 1))
                w_ind = self._w_ind
                for ii in range(self._horizon):
                    w[ii, 0] = v[offset:offset + model.n_x]
                    w_ind.append([j for j in range(offset, offset + model.n_x)])
                    v_guess[offset:offset + model.n_x] = self._w_guess
                    v_lb[offset:offset + model.n_x] = self._w_lb
                    v_ub[offset:offset + model.n_x] = self._w_ub
                    offset += model.n_x

            if self._nlp_options['integration_method'] == 'collocation':
                ip = np.resize(np.array([], dtype=ca.MX), (self._prediction_horizon, 1))
                zp = np.resize(np.array([], dtype=ca.MX), (self._prediction_horizon, 1))

                for ii in range(self._prediction_horizon):
                    ip[ii, 0] = v[offset:offset + n_xik]
                    v_guess[offset:offset + n_xik] = x_ik_guess
                    v_lb[offset:offset + n_xik] = x_ik_lb
                    v_ub[offset:offset + n_xik] = x_ik_ub

                    offset += n_xik

                    if model.n_z > 0:
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
                offset += self.stage_constraint.size
            else:
                e_soft_stage = ca.MX.sym('e_soft_stage', 0)

            # Those are parameters that need to be passed to the solver before running the optimization
            ext_parameters = catools.struct_symMX([catools.entry('tv_p', shape=(self._n_tvp, self._horizon)),
                                                   catools.entry('p_arrival', shape=p_arrival.shape[0]),
                                                   catools.entry('x_arrival', shape=x_arrival.shape[0]),
                                                   catools.entry('u_meas', shape=(self._model.n_u, self._horizon)),
                                                   catools.entry('y_meas',
                                                                 shape=(
                                                                     self.quad_stage_cost._n_tv_refs, self._horizon)),
                                                   catools.entry('dt', shape=self._horizon),
                                                   catools.entry('time', shape=1)])

            u_meas = ext_parameters['u_meas']
            y_meas = ext_parameters['y_meas']
            p_arrival = ext_parameters['p_arrival']
            x_arrival = ext_parameters['x_arrival']
            _dt = ext_parameters['dt']
            # Constraint function for the NLP
            g = []
            g_lb = []
            g_ub = []
            J = 0
            # current time
            time_now = ext_parameters['time']

            # Time at the beginning of the horizon (in the past)
            time = time_now - ca.sum1(_dt)

            for ii in range(self._horizon):
                x_ii = x[ii, 0]
                if self._state_noise_flag:
                    w_ii = w[ii, 0]
                else:
                    w_ii = []
                u_ii = u_meas[:, ii]
                y_ii = y_meas[:, ii]
                dt_ii = _dt[ii]

                if self._nlp_options['integration_method'] == 'multiple_shooting':
                    # TODO: here somewhere t has to enter
                    sol = int_dynamics_fun(x0=x_ii, p=ca.vertcat(u_ii, p, y_ii))
                    # add state noise. I assume the state noise is in discrete form so I need to add it here
                    if self._state_noise_flag:
                        x_ii_1 = sol['xf'] + w_ii
                elif self._nlp_options['integration_method'] == 'collocation':
                    [g_coll, x_ii_1] = int_dynamics_fun(
                        time + dt_ii, dt_ii, ip[ii, 0], x_ii, u_ii, zp[ii, 0], p, e_soft_stage, y_ii)
                    g.append(g_coll)
                    g_lb.extend(gk_col_lb)
                    g_ub.extend(gk_col_ub)
                    # add state noise. I assume the state noise is in discrete form so I need to add it here
                    if self._state_noise_flag:
                        x_ii_1 = x_ii_1 + w_ii
                elif self._nlp_options['integration_method'] == 'discrete':
                    x_ii_1 = int_dynamics_fun(time + dt_ii, dt_ii, x_ii, u_ii, p)
                    if self._state_noise_flag:
                        x_ii_1 = x_ii_1 + w_ii

                g.append(x[ii + 1, 0] - x_ii_1)
                g_lb.append(np.zeros(model.n_x))
                g_ub.append(np.zeros(model.n_x))

                # Add lagrange term
                if ii == 0:
                    if self._arrival_term_flag:
                        J += self._arrival_term_fun(x_ii, p, x_arrival, p_arrival)
                else:
                    if self._stage_term_flag:
                        J += self._stage_term_fun(w_ii, x_ii, y_ii)

                if self.stage_constraint.is_set:
                    if self.stage_constraint.is_soft:
                        residual = self._stage_constraints_fun(x_ii, w_ii, p, e_soft_stage)
                        J += self.stage_constraint.cost(e_soft_stage)
                        g.append(residual)
                        g_lb.append([-ca.inf] * self.stage_constraint.size * 2)
                        g_ub.append(self.stage_constraint.ub)
                        g_ub.append([-lb for lb in self.stage_constraint.lb])
                    else:
                        residual = self._stage_constraints_fun(x_ii, w_ii, p)
                        g.append(residual)
                        g_lb.append(self.stage_constraint.lb)
                        g_ub.append(self.stage_constraint.ub)

                # update time in the horizon
                time += dt_ii

            if self._custom_constraints_flag:
                raise NotImplementedError('Custom constraints are not yet implemented')

            if self._stage_constraints_flag:
                raise NotImplementedError('Auxiliary constraints are not yet implemented')

            g = ca.vertcat(*g)
            self._g_lb = ca.DM(ca.vertcat(*g_lb))
            self._g_ub = ca.DM(ca.vertcat(*g_ub))
            self._v0 = ca.DM(v_guess)
            self._v_lb = ca.DM(v_lb)
            self._v_ub = ca.DM(v_ub)
            self._J = J
            self._v = v
            self._param_npl_mhe = ext_parameters
            self._g = g
            self._nlp_setup_done = True
            self._n_v = n_v
            self._ext_parameters = ext_parameters

            nlp_dict = {'f': self._J, 'x': self._v, 'p': self._param_npl_mhe, 'g': self._g}
            if self._solver_name == 'ipopt':
                solver = ca.nlpsol("solver", 'ipopt', nlp_dict, self._nlp_opts)
            elif self._solver_name == 'qpsol':
                solver = ca.qpsol("solver", 'qpoases', nlp_dict, self._nlp_opts)
            else:
                raise ValueError(f"The solver {self._solver_name} does no exist. The possible solver are",
                                 self._solver_name_list)
            self._solver = solver

    def set_nlp_options(self, *args, **kwargs):
        """
        Sets the options that modify how the mpc problem is set

        :param args:
        :param kwargs:
        :return:
        """
        # TODO: when multiple-shooting and irk are implemented/tested add them to the list
        possible_choices = {}
        possible_choices['integration_method'] = ['collocation', 'rk4', 'erk', 'discrete', 'multiple_shooting']  # 'irk'
        possible_choices['collocation_points'] = ['radau', 'legendre']
        possible_choices['degree'] = None
        possible_choices['print_level'] = [0, 1]

        possible_choices['arrival_guess_update'] = ['filtering', 'smoothing']

        option_list = list(possible_choices.keys())

        default_opts = {
            'integration_method': 'collocation',
            'collocation_points': 'radau',
            'degree': 3,
            'print_level': 1,
            'arrival_guess_update': 'smoothing'
        }

        opts = {}
        if len(args) != 0:
            if isinstance(args[0], dict):
                opts = args[0]
        else:
            if len(kwargs) != 0:
                opts = kwargs

        for key, value in opts.items():
            if key not in option_list:
                raise ValueError(f"The option named {key} does not exist. Possible options are {option_list}.")
            if possible_choices[key] is not None and value not in possible_choices[key]:
                raise ValueError(
                    f"The option {key} is set to value {value} put the only allowed values are {possible_choices[key]}."
                )
            else:
                default_opts[key] = value

        if default_opts.get('integration_method') != 'discrete' and self._model.discrete is True:
            warnings.warn(
                f"The integration method is set to {default_opts.get('integration_method')} but I notice that the model"
                f" is in discrete time. I am overwriting and using discrete mode."
            )
            default_opts['integration_method'] = 'discrete'

        # Integration methods. Those are necessary for the RungeKutta class
        if default_opts['integration_method'] == 'rk4':
            default_opts['class'] = 'explicit'
            default_opts['method'] = 'rk'
            default_opts['order'] = 4
            default_opts['category'] = 'runge-kutta'
        elif default_opts['integration_method'] == 'erk':
            default_opts['class'] = 'explicit'
            default_opts['method'] = 'rk'
            default_opts['category'] = 'runge-kutta'
        elif default_opts['integration_method'] == 'irk':
            default_opts['class'] = 'implicit'
            default_opts['method'] = 'rk'
            default_opts['category'] = 'runge-kutta'
        elif default_opts['integration_method'] == 'collocation':
            default_opts['class'] = 'implicit'
            default_opts['method'] = 'collocation'
            default_opts['category'] = 'runge-kutta'

        self._nlp_options_is_set = True
        self._nlp_options = default_opts

    def set_scaling(self, x_scaling=None, w_scaling=None, p_scaling=None, u_scaling=None):
        """
        Scales the states and input. This is important for systems with large difference of order of magnitude in
        states and inputs.

        :param x_scaling: list of scaling factors
        :param w_scaling: list of scaling factors
        :param p_scaling: list of scaling factors
        :param u_scaling: list of scaling factors
        :return:
        """
        if x_scaling is None:
            self._x_scaling = self._model.n_x * [1]
        else:
            if isinstance(x_scaling, list) or isinstance(x_scaling, np.ndarray):
                self._x_scaling = x_scaling
            else:
                raise ValueError("Scaling factors x_scaling must be a list or nd.arrays")

        if w_scaling is None:
            self._w_scaling = self._model.n_x * [1]
        else:
            if isinstance(w_scaling, list) or isinstance(w_scaling, np.ndarray):
                self._w_scaling = w_scaling
            else:
                raise ValueError("Scaling factors u_scaling must be a list or nd.arrays")

        if p_scaling is None:
            self._p_scaling = self._model.n_p * [1]
        else:
            if isinstance(p_scaling, list) or isinstance(p_scaling, np.ndarray):
                self._p_scaling = p_scaling
            else:
                raise ValueError("Scaling factors x_scaling must be a list or nd.arrays")

        if u_scaling is None:
            self._u_scaling = self._model.n_u * [1]
        else:
            if isinstance(u_scaling, list) or isinstance(u_scaling, np.ndarray):
                self._u_scaling = u_scaling
            else:
                raise ValueError("Scaling factors u_scaling must be a list or nd.arrays")

        self._scaling_is_set = True

    def set_time_varying_parameters(self, time_varying_parameters=None):
        """
        Sets the time-varying parameters for the estimator.
        The estimator will expect these as an input at every iteration

        :param time_varying_parameters: list of strings with time varying parameter names
        :return:
        """
        # TODO check if it is the same as the parent class, if yes delete
        if time_varying_parameters is None:
            self._time_varying_parameters = []
        else:
            if not check_if_list_of_string(time_varying_parameters):
                raise ValueError("tvp must be a list of strings with the parameters name that are time varying")

            for tvp in time_varying_parameters:
                if tvp not in self._model.p:
                    raise ValueError(f"The time-varying parameter {tvp} is not in the model0 parameter. "
                                     f"The model0 parameters are {self._model.parameter_names}.")

            self._time_varying_parameters = time_varying_parameters
        self._time_varying_parameters_is_set = True

    def set_box_constraints(self, x_ub=None, x_lb=None, w_ub=None, w_lb=None, p_lb=None, p_ub=None, z_ub=None,
                            z_lb=None):
        """

        :param x_ub:
        :param x_lb:
        :param w_ub:
        :param w_lb:
        :param p_lb:
        :param p_ub:
        :param z_ub:
        :param z_lb:
        :return:
        """
        if x_ub is not None:
            self._x_ub = check_and_wrap_to_list(x_ub)
        else:
            self._x_ub = self._model.n_x * [ca.inf]

        if x_lb is not None:
            self._x_lb = check_and_wrap_to_list(x_lb)
        else:
            self._x_lb = self._model.n_x * [-ca.inf]

        if w_ub is not None:
            self._w_ub = check_and_wrap_to_list(w_ub)
        else:
            self._w_ub = self._model.n_x * [ca.inf]
        if w_lb is not None:
            self._w_lb = check_and_wrap_to_list(w_lb)
        else:
            self._w_lb = self._model.n_x * [-ca.inf]

        if self._model._n_p > 0:
            if p_ub is not None:
                self._p_ub = check_and_wrap_to_list(p_ub)
            else:
                warnings.warn("Some parameters are uncertain but you did not give me the upper bounds. I am assuming "
                              "there are infinite.")
                self._p_ub = self._model._n_p * [ca.inf]

            if p_lb is not None:
                self._p_lb = check_and_wrap_to_list(p_lb)
            else:
                warnings.warn(
                    "Some parameters are uncertain but you did not give me the lower bounds. "
                    "I am assuming there are -infinite."
                )
                self._p_lb = self._model._n_p * [-ca.inf]

        # Algebraic constraints
        if z_ub is not None:
            z_ub = check_and_wrap_to_list(z_ub)
            if len(z_ub) != self._model.n_z :
                raise TypeError(f"The model has {self._n_z} algebraic states. "
                                f"You need to pass the same number of bounds.")
            self._z_ub = z_ub
        else:
            self._z_ub = self._model.n_z * [ca.inf]

        if z_lb is not None:
            z_lb = check_and_wrap_to_list(z_lb)
            if len(z_lb) != self._model.n_z :
                raise TypeError(f"The model has {self._n_z} algebraic states. "
                                f"You need to pass the same number of bounds.")
            self._z_lb = z_lb
        else:
            self._z_lb = self._model.n_z * [-ca.inf]

        self._box_constraints_is_set = True

    def set_initial_guess(self, x_guess=None, w_guess=None, p_guess=None, z_guess=None):
        """
        Sets initial guess for the optimizer when no other information of the states or inputs are available.

        :param x_guess: list of optimal dynamical state guess
        :param w_guess: list of optimal input guess
        :param p_guess: list of optimal parameter guess
        :param z_guess: list of optimal algebraic state guess
        :return:
        """
        if x_guess is not None:
            self._x_guess = check_and_wrap_to_list(x_guess)

            if len(self._x_guess) != self._model.n_x:
                raise ValueError(
                    f"x_guess dimension and model input dimension do not match. Model input has dimension "
                    f"{len(self._x_guess)} while x_guess had dimension {self._model.n_x}"
                )
        else:
            if ca.inf not in self._x_lb and ca.inf not in self._x_ub:
                self._x_guess = [(self._x_ub[i] - self._x_lb[i]) / 2 for i in range(len(self._x_ub))]
            else:
                self._x_guess = self._model.n_x * [0]

        if w_guess is not None:
            self._w_guess = check_and_wrap_to_list(w_guess)
        else:
            self._w_guess = self._model.n_x * [0]
        if len(self._w_guess) != self._model.n_x:
            raise ValueError(
                f"w_guess dimension and dimension of the model state do not match. "
                f"The state vector has dimension {self._model.n_x} while w_guess {len(self._w_guess)}"
            )

        if self._model.n_p > 0:
            if p_guess is not None:
                self._p_guess = check_and_wrap_to_list(p_guess)
            else:
                if ca.inf not in self._p_lb and ca.inf not in self._p_ub:
                    self._p_guess = [(self._p_ub[i] - self._p_lb[i]) / 2 for i in range(len(self._p_ub))]
                else:
                    self._p_guess = self._model.n_p * [0]
            if len(self._p_guess) != self._model._n_p:
                raise ValueError(
                    f"p_guess dimension and the dimension  of the uncertain parameters do not match. "
                    f"The uncertain parameters are {self._model._n_p} while p_guess is {len(self._p_guess)} long."
                )

        if z_guess is not None:
            self._z_guess = check_and_wrap_to_list(deepcopy(z_guess))

            if len(self._z_guess) != self._model.n_z:
                raise ValueError(
                    f"z_guess dimension and model u dimension do not match. Model z has dimension {len(self._z_guess)} "
                    f"while z_guess{self._model.n_z}"
                )
        else:
            try:
                if ca.inf not in self._z_lb and ca.inf not in self._z_ub:
                    self._z_guess = [(u - l) / 2 for u in self._z_ub for l in self._z_lb]
            except AttributeError:
                self._z_guess = self._model.n_z * [0]

        self._initial_guess_is_set = True

    def set_aux_nonlinear_constraints(self, aux_nl_const=None, ub=None, lb=None):
        """

        :param aux_nl_const:
        :param ub:
        :param lb:
        :return:
        """
        if None not in [aux_nl_const, ub, lb]:
            self._stage_constraints = aux_nl_const
            self._stage_constraints_lb = lb
            self._stage_constraints_ub = ub
            self._stage_constraints_flag = True
        elif check_if_list_of_none([aux_nl_const, ub, lb]):
            pass
        else:
            raise ValueError("When passing nonlinear constraints, you must pass"
                             "the nonlinear constraint function, lower and upper bound")

    def plot_mhe_estimation(self, save_plot=False, plot_dir=None, name_file='mhe_estimation.html',
                            show_plot=True, extras=None, extras_names=None, title=None, format_figure=None):
        """

        :param save_plot: if True plot will be saved under 'plot_dir/name_file.html' if they are declared, otherwise in
            current directory
        :param plot_dir: path to the folder where plots are saved (default = None)
        :param name_file: name of the file where plot will be saved  (default = mpc_prediction.html)
        :param show_plot: if True, shows plots (default = False)
        :param extras: dictionary with values that will be plotted over the predictions if keys are equal to predicted
            states/inputs
        :param extras_names: tags that will be attached to the extras in the legend
        :param title: title of the plots
        :param format_figure: python function that modifies the format of the figure
        :return:
        """
        import os

        from bokeh.io import output_file, show, save
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, DataTable, TableColumn, CellFormatter, Div
        from bokeh.layouts import gridplot, row, column

        time = self.current_time - self.sampling_interval * self._horizon
        if self._nlp_solution is None:
            raise RuntimeError("You need to run the MPC at least once to see the plots")
        if save_plot:
            if plot_dir is not None:
                output_file(os.path.join(plot_dir, name_file))
            else:
                output_file(name_file)

        if extras is None:
            extras = []
        if extras_names is None:
            extras_names = []
        if isinstance(extras, dict):
            extras = [extras]
        elif not isinstance(extras, list) and not isinstance(extras, dict):
            raise ValueError("The extras options should be a dictionary or a list of dictionaries")

        res_extras_list = len(extras) * [0]
        for i in range(len(extras)):
            res_extras_list[i] = {}
            for k, name in enumerate(self._model.dynamical_state_names):
                if name in extras[i].keys():
                    res_extras_list[i][name] = extras[i][name]

        if len(extras) != len(extras_names):
            raise ValueError("The length of the extra and extras_names must be the same.")

        # TODO: consider time step for the x axis
        x_pred, w_pred = self.return_mhe_estimation()

        noise_dict = {i: [] for i in self._model.dynamical_state_names}
        states_dict = {i: [] for i in self._model.dynamical_state_names}

        for k, name in enumerate(self._model.dynamical_state_names):
            noise_dict[name] = w_pred[k, :]

        for k, name in enumerate(self._model.dynamical_state_names):
            states_dict[name] = x_pred[k, :]

        p1 = [figure(title=title, background_fill_color="#fafafa") for i in range(self._model.n_x)]

        time_vector = np.linspace(time, time + self._horizon * self.sampling_interval,
                                  self._horizon + 1)

        for s, name in enumerate(self._model.dynamical_state_names):
            p1[s].line(x=time_vector,
                       y=states_dict[name],
                       legend_label=name + '_est', line_width=2)
            p1[s].yaxis.axis_label = name
            p1[s].xaxis.axis_label = 'time'
            if format_figure is not None:
                p1[s] = format_figure(p1[s])

        if self._state_noise_flag:
            p2 = self._model.n_x * [figure(title=title, background_fill_color="#fafafa")]
            for s, name in enumerate(self._model.dynamical_state_names):
                p2[s].step(x=time_vector[:-1], y=noise_dict[name],
                           legend_label=name + "_noise", mode='after', line_width=2)
                p2[s].yaxis.axis_label = name
                p2[s].xaxis.axis_label = "time"
                if format_figure is not None:
                    p2[s] = format_figure(p2[s])

        # Create some data to print statistics
        variables = []
        values = []

        heading = Div(text="MHE stats", height=80, sizing_mode='stretch_width', align='center',
                      style={'font-size': '200%'})
        # heading fills available width
        data = dict(
            variables=variables,
            values=values,
        )
        source = ColumnDataSource(data)

        columns = [
            TableColumn(field='variables', title="Variables"),
            TableColumn(field='values', title="Values", formatter=CellFormatter()),
        ]
        data_table = DataTable(source=source, columns=columns, width=400, height=280)

        grid_states = gridplot(p1, ncols=3, sizing_mode='stretch_width')

        states_header = Div(text="Predicted States", height=10, sizing_mode='stretch_width', align='center',
                            style={'font-size': '200%'})

        if self._state_noise_flag:
            grid_noise = gridplot(p2, ncols=3, sizing_mode='stretch_width')
            inputs_header = Div(text="Predicted Noise", height=10, sizing_mode='stretch_width', align='center',
                                style={'font-size': '200%'})
            layout = row(column(states_header, grid_states, inputs_header, grid_noise), column(heading, data_table))
        else:
            layout = row(column(states_header, grid_states), column(heading, data_table))

        if show_plot:
            show(layout, browser='google-chrome')
        else:
            if save_plot:
                save(layout)

    def return_mhe_estimation(self):
        """

        :return:
        """
        if self._nlp_solution is not None:
            x_pred = np.zeros((self._model.n_x, self._horizon + 1))
            w_pred = np.zeros((self._model.n_x, self._horizon))
            for ii in range(self._horizon + 1):
                x_pred[:, ii] = np.asarray(self._nlp_solution['x'][self._x_ind[ii]]).squeeze() * self._x_scaling
            if self._state_noise_flag:
                for ii in range(self._horizon):
                    w_pred[:, ii] = np.asarray(self._nlp_solution['x'][self._w_ind[ii]]).squeeze() * self._w_scaling
            else:
                w_pred = None
            return x_pred, w_pred
        else:
            warnings.warn("There is still no mpc solution available. Run mpc.optimize() to get one.")
            return None, None

    @property
    def has_state_noise(self):
        """

        :return:
        """
        return self._state_noise_flag

    @has_state_noise.setter
    def has_state_noise(self, arg):
        if not isinstance(arg, bool):
            raise TypeError("has_state_noise accepts True or False")
        self._state_noise_flag = arg


__all__ = [
    'MovingHorizonEstimator'
]
