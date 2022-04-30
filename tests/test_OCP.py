import unittest
from hilo_mpc import OptimalControlProblem as OCP
from hilo_mpc import SimpleControlLoop, Model
import casadi as ca


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        model = Model(plot_backend='bokeh')
        # Constants
        M = 5.
        m = 1.
        l = 1.
        h = .5
        g = 9.81

        # States and algebraic variables
        x = model.set_dynamical_states(['x', 'v', 'theta', 'omega'])
        model.set_measurements(['yx', 'yv', 'ytheta', 'tomega'])
        model.set_measurement_equations([x[0], x[1], x[2], x[3]])
        # y = model.set_algebraic_variables(['y'])
        v = x[1]
        theta = x[2]
        omega = x[3]
        # Inputs
        F = model.set_inputs(['F1','F2'])

        # ODE
        dx = v
        dv = 1. / (M + m - m * ca.cos(theta)) * (m * g * ca.sin(theta) - m * l * ca.sin(theta) * omega ** 2 + F[0])
        dtheta = omega
        domega = 1. / l * (dv * ca.cos(theta) + g * ca.sin(theta)) + F[1]

        model.set_equations(ode=[dx, dv, dtheta, domega])

        # Initial conditions
        x0 = [2.5, 0., 0.1, 0.]
        u0 = [0., 0.]

        # Create model and run simulation
        dt = .1
        model.setup(dt=dt)

        self.model = model
        self.dt = dt
        self.x0 = x0
        self.u0 = u0

    def test_ocp(self):
        " Test normal nonlinear MPC for using a pendulum model. This test checks the normal problem setup"
        x0 = self.x0
        u0 = self.u0
        model = self.model
        ocp = OCP(model)
        ocp.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
        ocp.quad_stage_cost.add_inputs(names=['F1','F2'], weights=[0.1, 0.1])
        ocp.horizon = 25
        ocp.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])
        ocp.set_initial_guess(x_guess=x0, u_guess=u0)
        ocp.setup()

        n_steps = 25
        model.set_initial_conditions(x0=x0)
        scl = SimpleControlLoop(model, ocp)
        scl.run(steps=n_steps)
        scl.plot()


if __name__ == '__main__':
    unittest.main()
