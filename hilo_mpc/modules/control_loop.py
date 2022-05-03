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

from typing import Optional, TypeVar, Union

import casadi as ca
import numpy as np

from .dynamic_model.dynamic_model import Model
from .controller.base import Controller
from .machine_learning.base import LearningBase
from .estimator.base import Estimator


Control = TypeVar('Control', bound=Controller)
ML = TypeVar('ML', bound=LearningBase)
Estimate = TypeVar('Estimate', bound=Estimator)


_step = 0


class SimpleControlLoop:
    """
    This class creates a simple feedback control loop with a plant, controller and an optional observer.

    :param plant:
    :param controller:
    :param observer:
    """
    def __init__(self, plant: Model, controller: Union[Control, ML], observer: Optional[Estimate] = None) -> None:
        """Constructor method"""
        if not plant.is_setup():
            plant.setup()
        self._plant = plant
        self._ind_map_controller = {'x': [], 'u': [], 'y': [], 'z': []}
        if not controller.is_setup():
            controller.setup()
        self._controller = controller
        self._controller_is_mpc = False
        self._controller_is_ocp = False
        self._controller_is_pid = False
        self._controller_is_ann = hasattr(self._controller, 'predict')
        if not self._controller_is_ann:
            self._controller_is_mpc = self._controller.type in ['NMPC', 'LMPC']
            self._controller_is_ocp = self._controller.type == 'OCP'
            if not self._controller_is_mpc:
                if self._controller.type == 'PID':
                    self._controller_is_pid = True

        if self._controller_is_ocp:
            self._u_sequence = None

        if self._controller_is_mpc or self._controller_is_ocp:
            # TODO do the same for the estimator
            name_set = set(self._plant.dynamical_state_names)
            self._ind_map_controller['x'] = [i for i, e in enumerate(self._controller._model_orig.dynamical_state_names)
                                             if
                                             e in name_set]
            name_set = set(self._plant.input_names)
            self._ind_map_controller['u'] = [i for i, e in enumerate(self._controller._model_orig.input_names) if
                                             e in name_set]

            name_set = set(self._plant.measurement_names)
            self._ind_map_controller['y'] = [i for i, e in enumerate(self._controller._model_orig.measurement_names) if
                                             e in name_set]

            name_set = set(self._plant.algebraic_state_names)
            self._ind_map_controller['z'] = [i for i, e in enumerate(self._controller._model_orig.algebraic_state_names)
                                             if
                                             e in name_set]

        if observer is not None:
            if not observer.is_setup():
                observer.setup()
            self._optimization_based_observer = hasattr(observer, 'estimate')
            self._learning_based_observer = hasattr(observer, 'predict')
        self._observer = observer

    def _initialize_solution(self, solution):
        """

        :param solution:
        :return:
        """
        # TODO: Process estimates from the observer and predicted states from MPC
        # Process control bounds
        self._update_bounds(solution, ignore_inputs=True)

    def _finalize_solution(self, solution):
        """

        :param solution:
        :return:
        """
        # TODO: Improve the behavior of the solution object (Series class) for updating values at specified indices
        #  (here the initial values)
        self._update_references(solution, index=1)
        self._update_bounds(solution, ignore_states=True)

    def _update_references(self, solution, ignore_states=False, ignore_inputs=False, index=0):
        """

        :param solution:
        :param ignore_states:
        :param ignore_inputs:
        :param index:
        :return:
        """
        if self._controller_is_mpc:
            x_ref = ca.DM.nan(self._plant.n_x)
            if not ignore_states:
                x_ref[self._ind_map_controller['x']] = self._controller.solution.get_by_id('x_reference')[:, index]

            u_ref = ca.DM.nan(self._plant.n_u)
            if not ignore_inputs:
                u_ref[self._ind_map_controller['u']] = self._controller.solution.get_by_id('u_reference')[:, index]
            kwargs = {
                'x_ref': x_ref,
                'u_ref': u_ref
            }
            solution.update(**kwargs)

    def _update_bounds(self, solution, ignore_states=False, ignore_inputs=False):
        """

        :param solution:
        :param ignore_states:
        :param ignore_inputs:
        :return:
        """
        # NOTE: For the path following case, the controller has more bounds than states of the system, for
        #  this we take only the first self._plant.n_x/self._plant.n_u bounds for states and inputs, respectively.
        if self._controller_is_mpc:
            kwargs = {}
            if not ignore_states:
                x_lb = self._controller._x_lb_orig
                if x_lb:
                    kwargs['x_lb'] = ca.DM.nan(self._plant.n_x)
                    kwargs['x_lb'][self._ind_map_controller['x']] = x_lb
                x_ub = self._controller._x_ub_orig
                if x_ub:
                    kwargs['x_ub'] = ca.DM.nan(self._plant.n_x)
                    kwargs['x_ub'][self._ind_map_controller['x']] = x_ub
            if not ignore_inputs:
                u_lb = self._controller._u_lb_orig
                if u_lb:
                    kwargs['u_lb'] = ca.DM.nan(self._plant.n_u)
                    kwargs['u_lb'][self._ind_map_controller['u']] = u_lb
                u_ub = self._controller._u_ub_orig
                if u_ub:
                    kwargs['u_ub'] = ca.DM.nan(self._plant.n_u)
                    kwargs['u_ub'][self._ind_map_controller['u']] = u_ub
            solution.update(**kwargs)

    @staticmethod
    def _reset_step():
        """

        :return:
        """
        global _step
        _step = 0

    def _live_plot(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        solution = self._plant.solution
        if solution.plot_backend == 'latex':
            raise NotImplementedError("Live animations using the backend 'latex' are not supported")
        elif solution.plot_backend == 'bokeh':
            self._live_plot_bokeh(solution, *args, **kwargs)
        elif solution.plot_backend == 'matplotlib':
            self._live_plot_matplotlib(solution, *args, **kwargs)
        else:
            raise NotImplementedError(f"Plot backend '{solution.plot_device}' not available")

    def _live_plot_bokeh(self, solution, *args, **kwargs):
        """
        Runs a live plot

        :param args:
        :param kwargs:
        :return:
        """
        steps = args[0]
        p = kwargs.get('p')
        # TODO: Check, if we can use solution.get_by_name('t:0') here
        x_range = [float(solution.get_by_name('t')[0]), steps * solution.dt]
        # NOTE: self._plant.n_x * [['line']] will generate a list of self._plant.n_x pointers to one and the same list,
        #  so we use list comprehension here.
        # kind = [['line'] for _ in range(self._plant.n_x)] + \
        #        [['line'] for _ in range(self._plant.n_y)] + \
        #        [['line'] for _ in range(self._plant.n_z)] + \
        #        [['step'] for _ in range(self._plant.n_u)] + \
        #        [['step'] for _ in range(self._plant.n_p)]

        plot, sources, Server, theme = self.plot(server=True, x_range=x_range)  # kind=kind
        browser = kwargs.get('browser')

        # TODO: Improve this (right now it only works if no arguments are supplied, so we plot everything)
        names = solution.get_names()[1:]  # Ignore time 't'

        def bkapp(doc):
            """

            :param doc:
            :return:
            """
            def callback():
                """

                :return:
                """
                # TODO: What are unlocked callbacks? (see https://docs.bokeh.org/en/latest/docs/user_guide/server.html)
                global _step

                x0 = solution.get_by_id('x:f')
                self._run(x0, _step, p=p)
                _step += 1

                # BEST PRACTICE --- update .data in one step with a new dict
                # TODO: Needs to be updated according to plot specifications.
                # NOTE: This workflow assumes, that we only have one variable per subplot and the first entry of the
                #  list 'source' is always the solution and the second entry is always the reference. This will
                #  obviously not work with more complicated plot arrangements the way it is implemented right now.
                #  Needs to be updated in the future.
                for k, source in enumerate(sources):
                    x = solution.get_by_name('t').full().flatten()
                    y = solution.get_by_name(names[k]).full().flatten()
                    if x.size > y.size:  # in case of inputs -> step plot
                        x = x[:-1]
                    new_data = {
                        'x': x,
                        'y0': y
                    }
                    source[0].data = new_data  # In the current setup first source is always the data

                    # reference = solution.get_ref_by_name(names[k])
                    # if reference is not None and not reference.is_empty():
                    #     new_data['y0'] = reference.full().flatten()
                    #     source[1].data = new_data  # In the current setup second source is always the reference

                if _step >= steps:
                    doc.remove_periodic_callback(pc_id)

            button = plot.children[0]
            # button.label = "Start live animation"
            # button.on_click(callback)
            button.visible = False

            doc.add_root(plot)
            pc_id = doc.add_periodic_callback(callback, 200)

            # doc.theme = theme

        server = Server({'/': bkapp})  # num_procs=4
        server.start()
        server.io_loop.add_callback(server.show, "/", browser=browser)
        server.io_loop.start()
        # TODO: How to shutdown server

    def _live_plot_matplotlib(self, solution, *args, **kwargs):
        """

        :param solution:
        :param args:
        :param kwargs:
        :return:
        """
        steps = args[0]
        iterations = args[0]
        p = kwargs.get('p')
        # TODO: Check, if we can use solution.get_by_name('t:0') here
        x_range = [float(solution.get_by_name('t')[0]), steps * solution.dt]

        plt, fig, axes, anim = self.plot(anim=True, x_range=x_range)
        axes = axes.flatten()
        lines = [ax.get_lines() for ax in axes]

        # TODO: Improve this (right now it only works if no arguments are supplied, so we plot everything)
        names = solution.get_names()[1:]  # Ignore time 't'

        def update(frame):
            """

            :param frame:
            :return:
            """
            nonlocal steps

            if steps > 0:
                x0 = solution.get_by_id('x:f')
                self._run(x0, iterations - steps, p=p)

                for k, line in enumerate(lines):
                    x = solution.get_by_name('t').full().flatten()
                    y = solution.get_by_name(names[k]).full().flatten()
                    if x.size > y.size:  # in case of inputs -> step plot
                        x = x[:-1]
                    line[0].set_data(x, y)  # In the current setup first line is always the data

                    # reference = solution.get_ref_by_name(names[k])
                    # if reference is not None and not reference.is_empty():
                    #     ref = reference.full().flatten()
                    #     line[1].set_data(x, ref)  # In the current setup second line is always the reference

                    axes[k].relim()
                    axes[k].autoscale_view()

                steps -= 1

            return np.concatenate(lines)

        # NOTE: For whatever reason just writing anim(...) won't work
        ani = anim(fig, update, frames=np.arange(steps), blit=True)
        plt.show()

    def _run(self, x0, iteration, p=None, **kwargs):
        """

        :param x0:
        :param iteration:
        :param p:
        :param kwargs:
        :return:
        """
        # Controller step
        if self._controller_is_mpc or self._controller_is_ocp:
            # states of model could be different from the states of the plant.
            state_names = self._controller._model_orig.dynamical_state_names
            ind_states = [self._plant.dynamical_state_names.index(name) for name in state_names]

            if len(self._controller._time_varying_parameters) > 0:
                ind_param = [name for name in self._plant.parameter_names if
                             name not in self._controller._time_varying_parameters]
                cp = [p[i] for i in ind_param]
            else:
                cp = p

            if self._controller_is_mpc:
                u = self._controller.optimize(x0[ind_states], cp=cp, **kwargs)
            else:
                if iteration == 0:
                    _ = self._controller.optimize(x0[ind_states], cp=cp, **kwargs)
                    self._u_sequence = self._controller.solution['u']
                u = self._u_sequence[:, iteration]
        elif self._controller_is_ann:
            u = self._controller.predict(x0)
        elif self._controller_is_pid:
            u = self._controller.call(pv=x0)
        else:
            u = self._controller.call(x=x0, p=p)

        # Update reference
        # TODO: Improve processing of references (something similar to get_function_args for references?)
        # TODO: References supplied by user
        self._update_references(self._plant.solution)
        self._update_bounds(self._plant.solution)

        # Simulate plant
        self._plant.simulate(u=u, p=p)

        # Observer step
        if self._observer is not None:
            if self._optimization_based_observer:
                # TODO: Further processing
                # It's an MHE or KF
                self._observer.estimate()
            elif self._learning_based_observer:
                # TODO: Further processing
                # It's a learning-based 'observer'
                self._observer.predict()

    def run(self, steps, p=None, live_animation=False, browser=None, **kwargs):
        """

        :param steps: Number of simulation steps
        :type steps: int
        :param p:
        :type p:
        :param live_animation:
        :type live_animation: bool
        :param browser:
        :param kwargs:
        :return:
        """
        if self._controller_is_ocp:
            if not self._controller.prediction_horizon >= steps:
                raise ValueError(f"To solve an OCP problem, the prediction horizon must be larger than the number of "
                                 f"steps. The horizon length is {self._controller.prediction_horizon}, while the steps"
                                 f" are {steps}.")

        if _step != 0:
            self._reset_step()

        if live_animation:
            self._live_plot(steps, p=p, browser=browser)
        else:
            self._plant.reset_solution()
            solution = self._plant.solution
            self._initialize_solution(solution)
            x0 = solution.get_by_id('x:0')
            for k in range(steps):
                self._run(x0, k, p=p, **kwargs)
                x0 = solution.get_by_id('x:f')
            self._finalize_solution(solution)

    def plot(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        new_kwargs = kwargs
        if self._plant.solution.plot_backend == 'bokeh':
            if kwargs.get("output_notebook", False):
                new_kwargs["figsize"] = kwargs.get("figsize", (300, 300))
            else:
                new_kwargs["figsize"] = kwargs.get("figsize", (400, 400))
            new_kwargs["line_width"] = kwargs.get("line_width", 2)
        elif self._plant.solution.plot_backend == 'matplotlib':
            new_kwargs["line_width"] = kwargs.get("line_width", 1)

        # NOTE: The following 2 lines will be ignored for backend 'matplotlib'
        new_kwargs["major_label_text_font_size"] = kwargs.get("major_label_text_font_size", "12pt")
        new_kwargs["axis_label_text_font_size"] = kwargs.get("axis_label_text_font_size", "12pt")
        new_kwargs["layout"] = kwargs.get("layout", (-1, 3))

        return self._plant.solution.plot(*args, **new_kwargs)
