import unittest
import casadi as ca
import numpy as np
from hilo_mpc import Model, SMPC, GP, SMPCUKF
import scipy

class TestIO(unittest.TestCase):
    def setUp(self):
        plant = Model(plot_backend='matplotlib')

        states = plant.set_dynamical_states(['px'])
        inputs = plant.set_inputs(['a'])
        # parameter = plant.set_parameters(['p'])

        # Unwrap states
        px = states[0]

        # Unwrap states
        a = inputs[0]

        # dpx = parameter*a
        dpx = a

        plant.set_dynamical_equations([dpx])

        # Initial conditions
        x0 = [15]

        # Create plant and run simulation
        dt = 1
        plant.setup(dt=dt)
        plant.set_initial_conditions(x0=x0)
        model = plant.copy(setup=False)
        model.discretize('erk', order=1, inplace=True)
        model.setup(dt=dt)
        model.set_initial_conditions(x0=x0)
        self.model = model

        # Setup GP
        gp = GP(['px'], 'z', solver='ipopt')
        X_train = np.array([[0., .5, 1. / np.sqrt(2.), np.sqrt(3.) / 2., 1., 0.]])
        y_train = np.array([[0., np.pi / 6., np.pi / 4., np.pi / 3., np.pi / 2., np.pi]])
        gp.set_training_data(X_train, y_train)
        gp.setup()
        gp.fit_model()
        self.gp = gp
        # Matrix
        self.B = np.array([[1]])
        self.x0 = x0

        def dlqr(A, B, Q, R, N=None):
            """Solve the discrete time infinite-horizon lqr controller.

            x[k+1] = A x[k] + B u[k]

            cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
            """
            if N is None:

                # ref Bertsekas, p.151

                # first, try to solve the Riccati equation
                X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))

                # compute the LQR gain
                K = np.matrix(scipy.linalg.inv(B.T * X * B + R) * (B.T * X * A))

                eigVals, eigVecs = scipy.linalg.eig(A - B * K)

                return K, X, eigVals

            else:
                P = Q
                K = []
                for i in range(N):
                    if isinstance(A, list) and isinstance(B, list):
                        A_i = A[i]
                        B_i = B[i]
                    else:
                        A_i = A
                        B_i = B
                    K.append(-np.linalg.inv(R + B_i.T @ P @ B_i) @ (B_i.T @ P @ A_i))
                    P = A_i.T @ P @ A_i - (A_i.T @ P @ B_i) @ np.linalg.inv(B_i.T @ P @ B_i) @ (B_i.T @ P @ A_i) + Q

                return K, None, None

    def test_box_constraints(self):
        smpc = SMPC(self.model, self.gp, self.B)
        smpc.set_box_chance_constraints(x_lb=[10])

    def test_box_constraints_1(self):
        smpc = SMPC(self.model, self.gp, self.B)
        self.assertRaises(TypeError, smpc.set_box_chance_constraints, x_lb=[10], x_lb_p=2)

    def test_setup(self):
        smpc = SMPC(self.model, self.gp, self.B)
        smpc.horizon = 10
        smpc.quad_stage_cost.add_states(names='px', ref=1, weights=10)
        smpc.set_box_chance_constraints(x_lb=[10], x_lb_p=0.9)
        smpc.setup(options={'chance_constraints': 'prs'})

    def test_one_iter(self):
        smpc = SMPC(self.model, self.gp, self.B)
        smpc.horizon = 10
        smpc.quad_stage_cost.add_states(names='px', ref=1, weights=10)
        smpc.set_box_chance_constraints(x_lb=[10], x_lb_p=0.9)
        smpc.setup(options={'chance_constraints': 'prs'})
        smpc.optimize(x0=self.x0, cov_x0=[0], Kgain=0)

    def test_not_passing_k0(self):
        smpc = SMPC(self.model, self.gp, self.B)
        smpc.horizon = 10
        smpc.quad_stage_cost.add_states(names='px', ref=1, weights=10)
        smpc.set_box_chance_constraints(x_lb=[10], x_lb_p=0.9)
        smpc.setup(options={'chance_constraints': 'prs'})
        self.assertRaises(ValueError, smpc.optimize, x0=self.x0, Kgain=0)

    def test_plot(self):
        smpc = SMPC(self.model, self.gp, self.B)
        smpc.horizon = 10
        smpc.quad_stage_cost.add_states(names='px', ref=1, weights=10)
        smpc.quad_terminal_cost.add_states(names='px', ref=1, weights=10)
        smpc.set_box_chance_constraints(x_lb=[0], x_lb_p=0.9)
        smpc.setup(options={'chance_constraints': 'prs'})
        smpc.optimize(x0=self.x0, cov_x0=[0], Kgain=0)
        smpc.plot_prediction()


class TestMIMOSystem(unittest.TestCase):
    def setUp(self):
        plant = Model(plot_backend='matplotlib')

        states = plant.set_dynamical_states(['px', 'py'])
        inputs = plant.set_inputs(['ax', 'ay'])
        # parameter = plant.set_parameters(['p'])

        # unwrap states
        px = states[0]
        py = states[1]

        # unwrap states
        a1 = inputs[0]
        a2 = inputs[1]

        # dpx = parameter*a
        dpx = a1
        dpy = a2

        plant.set_dynamical_equations([dpx, dpy])

        # initial conditions
        x0 = [15, 10]

        # create plant and run simulation
        dt = 1
        plant.setup(dt=dt)
        plant.set_initial_conditions(x0=x0)
        model = plant.copy(setup=False)
        model.discretize('erk', order=1, inplace=True)
        model.setup(dt=dt)
        model.set_initial_conditions(x0=x0)
        self.model = model

        # setup gp
        gp = GP(['px'], 'z', solver='ipopt')
        x_train = np.array([[0., .5, 1. / np.sqrt(2.), np.sqrt(3.) / 2., 1., 0.]])
        y_train = np.array([[0., np.pi / 6., np.pi / 4., np.pi / 3., np.pi / 2., np.pi]])
        gp.set_training_data(x_train, y_train)
        gp.setup()
        gp.fit_model()
        self.gp = gp
        # matrix
        self.B = np.array([[1, 1]]).T
        self.x0 = x0

    def test_simple_mimo(self):
        smpc = SMPC(self.model, self.gp, self.B)
        smpc.horizon = 10
        smpc.quad_stage_cost.add_states(names=['px', 'py'], ref=[1, 1], weights=[10, 10])
        smpc.quad_terminal_cost.add_states(names=['px', 'py'], ref=[1, 1], weights=[10, 10])
        smpc.set_box_chance_constraints(x_lb=[0, 0], x_lb_p=[0.97, 0.97])
        smpc.setup(options={'chance_constraints': 'prs'})
        cov_x0 = np.array([[0, 0], [0, 0]])
        Kgain = np.array([[0, 0], [0, 0]])
        smpc.optimize(x0=self.x0, cov_x0=cov_x0, Kgain=Kgain)
        smpc.plot_prediction()

    def test_simple_negative_bounds(self):
        smpc = SMPC(self.model, self.gp, self.B)
        smpc.horizon = 15
        smpc.quad_stage_cost.add_states(names=['px'], ref=[1], weights=[10])
        smpc.quad_terminal_cost.add_states(names=['px'], ref=[1], weights=[10])
        smpc.set_box_chance_constraints(x_lb=[-100, 0], x_lb_p=[0.95, 0.95], x_ub=[100, 30], x_ub_p=[0.95, 0.95])
        # smpc.set_box_chance_constraints(x_ub=[100, 30], x_ub_p=[0.95,0.95])
        smpc.setup(options={'chance_constraints': 'prs'})
        cov_x0 = np.array([[0, 0], [0, 0]])
        Kgain = np.array([[0, 0], [0, 0]])
        smpc.optimize(x0=self.x0, cov_x0=cov_x0, Kgain=Kgain)
        smpc.plot_prediction()

    def test_K_gain(self):
        from hilo_mpc import LQR

        lqr = LQR(self.model)
        lqr.horizon = 10
        lqr.setup()
        lqr.Q = 10 * np.eye(2)
        lqr.R = 10 * np.eye(2)
        lqr.feedback_gain

    def test_lqr(self):
        import numpy as np

        from hilo_mpc import Model, LQR, SimpleControlLoop

        # Initialize empty model
        model = Model(discrete=True, time_unit='', plot_backend='bokeh')

        # Set model matrices
        model.A = np.array([[1., 1.], [0., 1.]])
        model.B = np.array([[0.], [1.]])

        # Set up model
        model.setup()

        # Initialize LQR
        lqr = LQR(model, plot_backend='bokeh')

        # Set LQR horizon for finite horizon formulation
        lqr.horizon = 5

        # Set up LQR
        lqr.setup()

        # Initial conditions of the model
        model.set_initial_conditions([2, 1])

        # Set LQR matrices
        lqr.Q = 2. * np.ones((2, 2))
        lqr.R = 2.

        # Run simulation
        scl = SimpleControlLoop(model, lqr)
        scl.run(20)
        scl.plot()



class TestUKF(unittest.TestCase):

    def test_initialization_ukf(self):
        model = Model(plot_backend='bokeh')
        # Constants
        M = 5.
        m = 1.
        l = 1.
        g = 9.81

        # States and algebraic variables
        x = model.set_dynamical_states(['x', 'v', 'theta', 'omega'])
        model.set_measurements(['yx', 'yv', 'ytheta', 'tomega'])
        model.set_measurement_equations([x[0], x[1], x[2], x[3]])
        h= model.set_parameters(['h'])
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
        model.discretize(method='rk4', inplace=True)
        model.setup(dt=dt)
        smpc = SMPCUKF(model, plot_backend='bokeh')


if __name__ == '__main__':
    unittest.main()
