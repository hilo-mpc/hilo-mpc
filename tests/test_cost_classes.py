import unittest
import casadi as ca
from hilo_mpc import Model
from hilo_mpc.util.modeling import QuadraticCost


class TestQuadraticCost(unittest.TestCase):
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
        F = model.set_inputs('F')

        # ODE
        dx = v
        dv = 1. / (M + m - m * ca.cos(theta)) * (m * g * ca.sin(theta) - m * l * ca.sin(theta) * omega ** 2 + F)
        dtheta = omega
        domega = 1. / l * (dv * ca.cos(theta) + g * ca.sin(theta))

        model.set_equations(ode=[dx, dv, dtheta, domega])

        # Initial conditions
        x0 = [2.5, 0., 0.1, 0.]
        z0 = ca.sqrt(3.) / 2.
        u0 = 0.

        # Create model and run simulation
        dt = .1
        model.setup(dt=dt)

        self.model = model
        self.dt = dt
        self.x0 = x0
        self.u0 = u0

    def test_Q(self):
        cost = QuadraticCost(model=self.model)
        cost.add_states(names=['x', 'v'], weights=[10, 5])
        Q = ca.DM([[10, 0, 0, 0], [0, 5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        for i in range(self.model.n_x):
            for j in range(self.model.n_x):
                if cost.Q[i, j] != Q[i, j]:
                    raise ValueError

    def test_R(self):
        cost = QuadraticCost(model=self.model)
        cost.add_inputs(names=['F'], weights=[10])
        R = ca.DM([[10]])
        for i in range(self.model.n_u):
            for j in range(self.model.n_u):
                if cost.R[i, j] != R[i, j]:
                    raise ValueError

if __name__ == '__main__':
    unittest.main()
