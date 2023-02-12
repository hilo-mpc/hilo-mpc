from bokeh.plotting import figure, show
from bokeh.layouts import row, gridplot, grid
from bokeh.io import output_file
from bokeh.palettes import Category20b
import unittest
import casadi as ca
import numpy as np
from hilo_mpc import Model, SMPC, GP, SMPCUKF
import scipy
import itertools


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

    def initial_condition_covariances(self, model):
        C = np.diag(np.ones(model.n_x) * 1e-6).tolist()
        return list(itertools.chain.from_iterable(C))

    def bounds_covariances(self, model, v_ub, cv_lb=-1e6, cv_ub=1e6):
        C_lb = np.ones((model.n_x, model.n_x)) * cv_lb
        np.fill_diagonal(C_lb, 0)

        C_ub = np.ones((model.n_x, model.n_x)) * cv_ub
        np.fill_diagonal(C_ub, v_ub)

        C_lb = C_lb.tolist()
        C_ub = C_ub.tolist()

        return list(itertools.chain.from_iterable(C_lb)), list(itertools.chain.from_iterable(C_ub))

    def get_mean_sigma_points(self, model, smpc):
        xp_mean = {}
        for state in model.dynamical_state_names:
            xp_mean[state] = np.average(smpc.solution[state].toarray(), weights=smpc._weights[0, :], axis=0)
        return xp_mean

    def test_initialization_ukf(self):

        model = Model(plot_backend='bokeh')
        # Constants

        # States and algebraic variables
        xx = model.set_dynamical_states(['px', 'vx', 'py', 'vy'])
        model.set_measurements(['y_x', 'y_vx', 'y_y', 'y_vy'])
        model.set_measurement_equations([xx[0], xx[1], xx[2], xx[3]])
        M = model.set_parameters(['M'])
        x = xx[0]
        vx = xx[1]
        y = xx[2]
        vy = xx[3]
        # Inputs
        F = model.set_inputs(['Fx', 'Fy'])
        Fx = F[0]
        Fy = F[1]
        # ODEs
        dd1 = vx
        dd2 = Fx / M
        dd3 = vy
        dd4 = Fy / M

        model.set_dynamical_equations([dd1, dd2, dd3, dd4])

        # Initial conditions
        x0 = [1, 0, 2, 0]
        u0 = [0., 0.]

        # Create model and run simulation
        dt = 0.1
        model.discretize(method='rk4', inplace=True)
        model.setup(dt=dt)

        # Initial conditions
        # x0c = initial_condition_covariances(model)
        x0_new = x0 * (2 * 5 + 1)
        # x0_new = x0
        u0 = [0., 0.]

        # lower and upper bounds
        x_lb_c, x_ub_c = self.bounds_covariances(model, [1e6, 1e6, 1e6, 1e6])
        # Create model and run simulation

        smpc = SMPCUKF(model, plot_backend='bokeh', alpha=0.9)
        smpc.quad_stage_cost.add_states(names=['px', 'py'], ref=[2, 2], weights=[10, 5])
        smpc.quad_terminal_cost.add_states(names=['px', 'py'], ref=[2, 2], weights=[10, 5])
        smpc.quad_stage_cost.add_inputs(names=['Fx', 'Fy'], weights=[4, 4])
        smpc.horizon = 20
        smpc.robust_horizon = 3
        smpc.covariance_states = np.eye(model.n_x) * 0.001
        smpc.covariance_states_noise = np.eye(model.n_x) * 0.00001
        smpc.covariance_parameters = np.eye(model.n_p) * 0.001
        # smpc.set_box_constraints(x_ub=x_ub_c + [10, 10, 10, 10] * (2 * 5 + 1),
        #                          x_lb=x_lb_c + [-10, -10, -10, -10] * (2 * 5 + 1))
        # smpc.set_initial_guess(x_guess=x0_new, u_guess=u0)
        smpc.setup(solver_options={'ipopt.print_level': 5}, options={'integration_method': 'rk4'})

        smpc.optimize(x0=x0_new, cp=[5])
        # smpc.plot_iterations(plot_last=True)
        # Get the sigma
        sigma_pred = np.zeros((model.n_x, model.n_x, smpc.robust_horizon))
        for ii in range(smpc.robust_horizon):
            sigma_pred[:, :, ii] = np.asarray(smpc._nlp_solution['x'][smpc._sigma_ind[ii]]).reshape(
                (model.n_x, model.n_x)) @ np.asarray(smpc._nlp_solution['x'][smpc._sigma_ind[ii]]).reshape(
                (model.n_x, model.n_x)).T

        # Extend sigma with the last sigma until the prediciton horizon
        sigma_pred = np.concatenate((sigma_pred, np.dstack(
            [sigma_pred[:, :, -1] for i in range(smpc.prediction_horizon + 1 - smpc.robust_horizon)])), axis=2)
        x_mean = self.get_mean_sigma_points(model, smpc)

        p_states = []
        color_list = ['blue', 'green', 'magenta', 'red']
        for k, state in enumerate(model.dynamical_state_names):
            p = figure(background_fill_color='#fafafa')

            p.varea(smpc.solution['t'].toarray().squeeze(),
                    x_mean[state] - np.sqrt(sigma_pred[k, k, :].squeeze()),
                    x_mean[state] + np.sqrt(sigma_pred[k, k, :].squeeze()),
                    fill_alpha=0.5, fill_color=color_list[k], legend_label=state)
            p.line(smpc.solution['t'].toarray().squeeze(), x_mean[state],
                   legend_label=state,
                   line_color=color_list[k])
            p_states.append(p)
        grid = gridplot([[p_states[0], p_states[1]], [p_states[2], p_states[3]]])
        show(grid)

    def test_bio(self):
        def logtransform_matrix(model, mean, cov):
            new_cov = np.zeros((model.n_x, model.n_x))
            for i in range(model.n_x):
                for j in range(model.n_x):
                    new_cov[i, j] = np.exp(mean[i] + mean[j] + (cov[i, i] + cov[j, j]) / 2) * (np.exp(cov[i, j]) - 1)

            return new_cov
        def inverse_transform(x,a,log_transform=False):
            if log_transform:
                return ca.exp(x)+a
            else:
                return x


        theta0 = [0.08, 0.4, 0.04, 0.2]
        Cglu = 50
        dt = 3
        covariance_states = np.eye(4) * 0.0001
        covariance_state_noise = np.eye(4) * 0.00001
        log_transform = True
        model = Model(name='simple_bio', plot_backend='bokeh')
        x = model.set_dynamical_states(['Glu', 'Bio', 'Lac', 'V'])
        u = model.set_inputs(['Fglu'])
        theta = model.set_parameters(['v_max', 'km', 'v_maxl', 'km_l'])
        v_max = theta[0]
        km = theta[1]
        v_maxl = theta[2]
        kml = theta[3]
        x0 = [20.5, 5, 2, 2.]
        a = 1e-6

        if log_transform:
            x0 = ca.log(ca.DM(x0) - a).toarray().squeeze().tolist()
            log_glu = x[0]
            log_bio = x[1]
            log_lac = x[2]
            log_v = x[3]

            r = (v_max * (ca.exp(log_glu) + a)) / (km + (ca.exp(log_glu) + a))
            rl = (v_maxl * (ca.exp(log_glu) + a)) / (kml + (ca.exp(log_glu) + a))

            dbio = 1 / (ca.exp(log_bio)) * (r * (ca.exp(log_bio) + a) - u / (ca.exp(log_v) + a) * (ca.exp(log_bio) + a))
            dglu = 1 / (ca.exp(log_glu)) * (
                    -r * (ca.exp(log_bio) + a) - rl * (ca.exp(log_bio) + a) + u / (ca.exp(log_v) + a) * (
                    Cglu - (ca.exp(log_glu) + a)))
            dlac = 1 / (ca.exp(log_lac)) * (
                    rl * (ca.exp(log_bio) + a) - u / (ca.exp(log_v) + a) * (ca.exp(log_lac) + a))
            dv = 1 / (ca.exp(log_v)) * u
            x_lb = [ca.log(1e-5 - a)] * 4
            x_ub = [ca.log(50 - a)] * 4
        else:

            glu = x[0]
            bio = x[1]
            lac = x[2]
            v = x[3]

            r = (v_max * glu) / (km + glu)
            rl = (v_maxl * glu) / (kml + glu)

            dbio = r * bio - u / v * bio
            dglu = -r * bio - rl * bio + u / v * (Cglu - glu)
            dlac = rl * bio - u / v * lac
            dv = u
            x_lb = [1e-5] * 4
            x_ub = None

        model.set_dynamical_equations([dglu, dbio, dlac, dv])
        model.discretize(method='rk4', inplace=True)
        model.setup(dt=dt)

        # Initial conditions
        x0_new = x0 * (2 * (model.n_x + model.n_p) + 1)
        u0 = [0.]

        # lower and upper bounds
        smpc = SMPCUKF(model, plot_backend='bokeh', alpha=0.9)
        smpc.quad_stage_cost.add_states(names=['Lac'], weights=[-10])
        smpc.quad_terminal_cost.add_states(names=['Lac'], weights=[-10])
        smpc.quad_stage_cost.add_inputs(names=['Fglu'], weights=[4])
        smpc.horizon = 6
        smpc.robust_horizon = 3
        smpc.covariance_states = covariance_states
        smpc.covariance_states_noise = covariance_state_noise
        smpc.covariance_parameters = np.eye(model.n_p) * 0.0001
        smpc.set_box_chance_constraints(u_ub=[0.5], u_lb=[0], x_lb=x_lb, x_ub=x_ub)
        smpc.set_initial_guess(x_guess=x0, u_guess=u0)
        smpc.setup(solver_options={'ipopt.print_level': 5, 'ipopt.linear_solver': 'ma27',
                                   'ipopt.max_iter': 5000},
                   options={'integration_method': 'discrete', 'ipopt_debugger': True})

        smpc.optimize(x0=x0_new, cp=theta0)
        smpc.plot_iterations(plot_last=True)
        # Get the sigma
        sigma_pred = np.zeros((model.n_x, model.n_x, smpc.robust_horizon))
        sigma_pred[:, :, 0] = smpc.covariance_states @ smpc.covariance_states.T
        for ii in range(smpc.robust_horizon - 1):
            sigma_pred[:, :, ii + 1] = np.asarray(smpc._nlp_solution['x'][smpc._sigma_ind[ii]]).reshape(
                (model.n_x, model.n_x)) @ np.asarray(smpc._nlp_solution['x'][smpc._sigma_ind[ii]]).reshape(
                (model.n_x, model.n_x)).T

        # Extend sigma with the last sigma until the prediciton horizon
        sigma_pred = np.concatenate((sigma_pred, np.dstack(
            [sigma_pred[:, :, -1] for i in range(smpc.prediction_horizon + 1 - smpc.robust_horizon)])), axis=2)
        x_mean = self.get_mean_sigma_points(model, smpc)

        p_states = []
        color_list = ['blue', 'green', 'magenta', 'red']
        for k, state in enumerate(model.dynamical_state_names):
            p = figure(background_fill_color='#fafafa')
            p.varea(smpc.solution['t'].toarray().squeeze(),
                    x_mean[state] - np.sqrt(sigma_pred[k, k, :].squeeze()),
                    x_mean[state] + np.sqrt(sigma_pred[k, k, :].squeeze()),
                    fill_alpha=0.5, fill_color=color_list[k], legend_label=state)
            p.line(smpc.solution['t'].toarray().squeeze(), x_mean[state],
                   legend_label=state,
                   line_color=color_list[k])

            for j in range((2 * smpc.n_L + 1)):
                p.line(smpc.solution['t'].toarray().squeeze(), smpc.solution[state][j, :].toarray().squeeze(),
                       line_alpha=0.3, line_color=color_list[k])
            p_states.append(p)
        grid = gridplot([[p_states[0], p_states[1]], [p_states[2], p_states[3]]])
        show(grid)

        color_list = ['red']
        p = figure(background_fill_color='#fafafa')
        p.step(smpc.solution['t'].toarray().squeeze()[0:-2], smpc.solution['Fglu'].toarray().squeeze()[0:-1],
               legend_label=state,
               line_color=color_list[0])
        show(p)


if __name__ == '__main__':
    unittest.main()
