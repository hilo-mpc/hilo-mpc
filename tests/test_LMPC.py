import unittest
from hilo_mpc import Model, LMPC, SimpleControlLoop, NMPC
import numpy as np
import casadi as ca


class MyTestCase(unittest.TestCase):
    def test_already_linear_model(self):
        model = Model(plot_backend='bokeh', discrete=True)

        Ts = 0.5
        x0 = [1, 1]
        model.A = np.array([[1, model.dt], [0, 1]])
        model.B = np.array([[model.dt ** 2 / 2], [model.dt]])

        model.setup(dt=Ts)
        model.set_initial_conditions(x0=x0)
        mpc = LMPC(model)
        mpc.Q = np.eye(2)
        mpc.R = 1
        mpc.horizon = 10
        mpc.set_initial_guess(x_guess=x0, u_guess=[0])
        mpc.set_box_constraints(x_lb=[-5, -5], x_ub=[5, 5], u_lb=[-1], u_ub=[1])
        mpc.setup()

        for i in range(100):
            u = mpc.optimize(x0=x0)
            model.simulate(u=u)
            x0 = model.solution['x:f']

        # model.solution.plot(output_file='results/test_lmpc.html')

        model.reset_solution(keep_initial_conditions=True)

        nmpc = NMPC(model)
        nmpc.horizon = 10
        nmpc.quad_stage_cost.add_states(names=['x_0', 'x_1'], weights=[1, 1])
        nmpc.quad_stage_cost.add_inputs(names=['u'], weights=[1])
        nmpc.set_initial_guess(x_guess=x0, u_guess=[0])
        nmpc.set_box_constraints(x_lb=[-5, -5], x_ub=[5, 5], u_lb=[-1], u_ub=[1])
        nmpc.setup()

        for i in range(10):
            u = nmpc.optimize(x0=x0)
            model.simulate(u=u)
            x0 = model.solution['x:f']

        # model.solution.plot(output_file='results/test_mpc.html')

    def test_linearized_model(self):

        model = Model(plot_backend='bokeh')

        states = model.set_dynamical_states(['px', 'py', 'v', 'phi'])
        inputs = model.set_inputs(['a', 'delta'])

        # Unwrap states
        px = states[0]
        py = states[1]
        v = states[2]
        phi = states[3]

        # Unwrap states
        a = inputs[0]
        delta = inputs[1]

        # Parameters
        lr = 1.4  # [m]
        lf = 1.8  # [m]
        beta = ca.arctan(lr / (lr + lf) * ca.tan(delta))

        # ODE
        dpx = v * ca.cos(phi + beta)
        dpy = v * ca.sin(phi + beta)
        dv = a
        dphi = v / lr * ca.sin(beta)

        model.set_dynamical_equations([dpx, dpy, dv, dphi])
        model.discretize(method='rk4', inplace=True)
        model = model.linearize()
        dt = 0.05
        model.setup(dt=dt)

        mpc = LMPC(model)
        mpc.horizon = 10
        mpc.Q = np.eye(model.n_x)
        mpc.R = np.eye(model.n_u)
        mpc.setup()
        x0 = [0, 0, 0, 0]

        mpc.optimize(x0=x0)

    def test_plot_prediction(self):
        model = Model(plot_backend='bokeh')

        states = model.set_dynamical_states(['px', 'py', 'v', 'phi'])
        inputs = model.set_inputs(['a', 'delta'])

        # Unwrap states
        px = states[0]
        py = states[1]
        v = states[2]
        phi = states[3]

        # Unwrap states
        a = inputs[0]
        delta = inputs[1]

        # Parameters
        lr = 1.4  # [m]
        lf = 1.8  # [m]
        beta = ca.arctan(lr / (lr + lf) * ca.tan(delta))

        # ODE
        dpx = v * ca.cos(phi + beta)
        dpy = v * ca.sin(phi + beta)
        dv = a
        dphi = v / lr * ca.sin(beta)

        model.set_dynamical_equations([dpx, dpy, dv, dphi])
        model.discretize(method='rk4', inplace=True)
        model = model.linearize()
        dt = 0.05
        model.setup(dt=dt)

        mpc = LMPC(model)
        mpc.horizon = 10
        mpc.Q = np.eye(model.n_x)
        mpc.R = np.eye(model.n_u)
        mpc.setup()
        x0 = [0.5, 0, 0, 0]

        mpc.optimize(x0=x0)
        mpc.solution.plot()



if __name__ == '__main__':
    unittest.main()
