from unittest import TestCase

import numpy as np

from hilo_mpc import Model, KF, EKF, UKF


class TestKalmanFilterInitialization(TestCase):
    """"""
    def test_kalman_filter_nonlinear_model(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh')
        model.set_dynamical_states('x')
        model.set_dynamical_equations('x^2')
        with self.assertRaises(ValueError) as context:
            KF(model)
        self.assertTrue(str(context.exception) == "The supplied model is nonlinear. Please use an estimator targeted at"
                                                  " the estimation of nonlinear systems.")

    def test_kalman_filter_model_not_set_up(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh')
        model.set_dynamical_states('x')
        model.set_dynamical_equations('2*x')
        with self.assertRaises(RuntimeError) as context:
            KF(model)
        self.assertTrue(str(context.exception) == "Model is not set up. Run Model.setup() before passing it to the "
                                                  "Kalman filter.")

    def test_kalman_filter_initial_dimensions(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh')
        model.set_dynamical_states('x')
        model.set_dynamical_equations('2*x')
        model.setup(dt=1.)

        kf = KF(model, plot_backend='bokeh')

        self.assertTrue(kf.n_x == 0)
        self.assertTrue(kf.n_y == 0)
        self.assertTrue(kf.n_z == 0)
        self.assertTrue(kf.n_u == 0)
        self.assertTrue(kf.n_p == 0)
        self.assertTrue(kf.n_p_est == 0)

    def test_kalman_filter_initial_matrices(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh')
        model.set_dynamical_states('x')
        model.set_dynamical_equations('2*x')
        model.setup(dt=1.)

        kf = KF(model, plot_backend='bokeh')

        self.assertIsNone(kf.P)
        self.assertIsNone(kf.Q)
        self.assertIsNone(kf.R)


class TestKalmanFilterSetup(TestCase):
    """"""
    def test_kalman_filter_no_measurement_equations(self) -> None:
        """

        :return:
        """
        # FIXME: Right now the function arguments y and R would be empty, so the supplied measurements won't have any
        #  impact on the solution
        # TODO: Migrate to KalmanFilterEstimation
        model = Model(plot_backend='bokeh')
        model.set_dynamical_states('x')
        model.set_dynamical_equations('2*x')
        model.setup(dt=1.)
        kf = KF(model, plot_backend='bokeh')
        with self.assertWarns(UserWarning) as context:
            kf.setup()
        self.assertTrue(len(context.warnings) == 1)
        self.assertTrue(str(context.warning) == "The model has no measurement equations, I am assuming measurements of "
                                                "all states ['x'] are available.")
        # TODO: Also check estimation once bug is fixed

    def test_kalman_filter_is_set_up(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh')
        model.set_dynamical_states('x')
        model.set_dynamical_equations('2*x')
        model.set_measurement_equations('x')
        model.setup(dt=1.)
        kf = KF(model, plot_backend='bokeh')
        kf.setup()
        self.assertTrue(kf.is_setup())

    def test_kalman_filter_dimensions(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh')
        equations = """
        dx_1/dt = -k_1*x_1(t) + u(k)
        dx_2/dt = k_1*x_1(t) - k_2*x_2(t)
        y(k) = x_2(t)
        """
        model.set_equations(equations=equations)
        model.discretize('erk', order=1, inplace=True)
        model.setup(dt=1.)

        kf = KF(model, plot_backend='bokeh')
        kf.setup()

        self.assertTrue(kf.n_x == 2)
        self.assertTrue(kf.n_y == 1)
        self.assertTrue(kf.n_z == 0)
        self.assertTrue(kf.n_u == 1)
        self.assertTrue(kf.n_p == 2)
        self.assertTrue(kf.n_p_est == 0)


class TestKalmanFilterMatrixSetters(TestCase):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh')
        equations = """
        dx_1/dt = -k_1*x_1(t) + u(k)
        dx_2/dt = k_1*x_1(t) - k_2*x_2(t)
        y(k) = x_2(t)
        """
        model.set_equations(equations=equations)
        model.discretize('erk', order=1, inplace=True)
        model.setup(dt=1.)

        kf = KF(model, plot_backend='bokeh')
        kf.setup()

        self.kf = kf

    def test_kalman_filter_initial_matrices(self) -> None:
        """

        :return:
        """
        kf = self.kf

        self.assertIsNone(kf.P)
        np.testing.assert_equal(kf.Q, np.zeros((2, 2)))
        np.testing.assert_equal(kf.R, np.zeros((1, 1)))

    def test_kalman_filter_process_noise_setter_dimension_mismatch(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            self.kf.Q = [0., 0., 0.]
        self.assertTrue(str(context.exception) == "Dimension mismatch. Supplied dimension is 3x3, but required "
                                                  "dimension is 2x2")

    def test_kalman_filter_process_noise_setter(self) -> None:
        """

        :return:
        """
        kf = self.kf

        kf.Q = [.01, .01]
        np.testing.assert_equal(kf.Q, .01 * np.eye(2))

    def test_kalman_filter_measurement_noise_setter_dimension_mismatch(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            self.kf.R = [0., 0.]
        self.assertTrue(str(context.exception) == "Dimension mismatch. Supplied dimension is 2x2, but required "
                                                  "dimension is 1x1")

    def test_kalman_filter_measurement_noise_setter(self) -> None:
        """

        :return:
        """
        kf = self.kf

        kf.R = .064
        np.testing.assert_equal(kf.R, .064 * np.eye(1))

    def test_kalman_filter_error_covariance_setter_dimension_mismatch(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            self.kf.set_initial_guess([.8, 0.], P0=[1., 1., 1.])
        self.assertTrue(str(context.exception) == "Dimension mismatch. Supplied dimension is 3x3, but required "
                                                  "dimension is 2x2")

    def test_kalman_filter_error_covariance_setter_default(self) -> None:
        """

        :return:
        """
        kf = self.kf

        kf.set_initial_guess([.8, 0.])
        # FIXME: Fix bug in error_covariance property
        # np.testing.assert_equal(kf.P, np.eye(2))

    def test_kalman_filter_error_covariance_setter(self) -> None:
        """

        :return:
        """
        kf = self.kf

        kf.set_initial_guess([.8, 0.], P0=[1., 1.])
        # FIXME: Fix bug in error_covariance property
        # np.testing.assert_equal(kf.P, np.eye(2))


class TestKalmanFilterEstimation(TestCase):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh')
        equations = """
        dx_1/dt = -k_1*x_1(t) + u(k)
        dx_2/dt = k_1*x_1(t) - k_2*x_2(t)
        y(k) = x_2(t)
        """
        model.set_equations(equations=equations)
        model.discretize('erk', order=1, inplace=True)
        model.setup(dt=1.)

        kf = KF(model, plot_backend='bokeh')

        self.model = model
        self.kf = kf

    def test_kalman_filter_not_set_up(self) -> None:
        """

        :return:
        """
        with self.assertRaises(RuntimeError) as context:
            self.kf.estimate()
        self.assertTrue(str(context.exception) == "Kalman filter is not set up. Run KalmanFilter.setup() before running"
                                                  " simulations.")

    def test_kalman_filter_no_initial_guess_supplied(self) -> None:
        """

        :return:
        """
        kf = self.kf

        kf.setup()
        with self.assertRaises(RuntimeError) as context:
            kf.estimate()
        self.assertTrue(str(context.exception) == "No initial guess for the states found. Please set initial guess "
                                                  "before running the Kalman filter!")

    def test_kalman_filter_no_measurement_data_supplied(self) -> None:
        """

        :return:
        """
        kf = self.kf

        kf.setup()
        kf.set_initial_guess([.8, 0.])
        with self.assertRaises(RuntimeError) as context:
            kf.estimate()
        self.assertTrue(str(context.exception) == "No measurement data supplied.")

    def test_kalman_filter_one_step(self) -> None:
        """

        :return:
        """
        model = self.model
        kf = self.kf

        kf.setup()
        model.set_initial_conditions([.8, 0.])
        model.set_initial_parameter_values([.5, .4])
        kf.R = .064
        kf.Q = [.01, .01]
        kf.set_initial_guess([.8, 0.])
        kf.set_initial_parameter_values([.5, .4])

        model.simulate(u=.8)
        kf.estimate(y=.3894626, u=.8)
        np.testing.assert_allclose(kf.solution.get_by_id('x:f'), np.array([[1.19614861], [.39044856]]))

    def test_kalman_filter_one_step_p(self) -> None:
        """

        :return:
        """
        model = self.model
        kf = self.kf

        kf.setup()
        model.set_initial_conditions([.8, 0.])
        model.set_initial_parameter_values([.5, .4])
        kf.R = .064
        kf.Q = [.01, .01]
        kf.set_initial_guess([.8, 0.])

        model.simulate(u=.8)
        kf.estimate(y=.3894626, u=.8, p=[.5, .4])
        np.testing.assert_allclose(kf.solution.get_by_id('x:f'), np.array([[1.19614861], [.39044856]]))

    def test_kalman_filter_multi_step_dimension_mismatch_in_y(self) -> None:
        """

        :return:
        """
        kf = self.kf

        kf.setup()
        kf.set_initial_guess([.8, 0.])
        # FIXME: Convert IndexError to ValueError
        # FIXME: The method convert should be able to deal with list of lists
        # FIXME: Improve handling of dimensions
        with self.assertRaises(IndexError) as context:
            kf.estimate(y=np.array([[.3894626, .3894626]]), steps=3)
        self.assertTrue(str(context.exception) == "Dimension mismatch for variable y. Supplied dimension is 2, but "
                                                  "required dimension is 3.")

    def test_kalman_filter_multi_step_dimension_mismatch_in_u(self) -> None:
        """

        :return:
        """
        kf = self.kf

        kf.setup()
        kf.set_initial_guess([.8, 0.])
        # FIXME: See test_kalman_filter_multi_step_dimension_mismatch_in_y
        with self.assertRaises(IndexError) as context:
            kf.estimate(y=np.array([[.3894626, .3894626, .3894626]]), u=np.array([[.8, .8]]), steps=3)
        self.assertTrue(str(context.exception) == "Dimension mismatch for variable u. Supplied dimension is 2, but "
                                                  "required dimension is 3.")

    def test_kalman_filter_multi_step_dimension_mismatch_in_p(self) -> None:
        """

        :return:
        """
        kf = self.kf

        kf.setup()
        kf.set_initial_guess([.8, 0.])
        # FIXME: See test_kalman_filter_multi_step_dimension_mismatch_in_y
        with self.assertRaises(IndexError) as context:
            kf.estimate(y=np.array([[.3894626, .3894626, .3894626]]), p=np.array([[.5, .5], [.4, .4]]), steps=3)
        self.assertTrue(str(context.exception) == "Dimension mismatch for variable p. Supplied dimension is 2, but "
                                                  "required dimension is 3.")

    # def test_kalman_filter_multi_step(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     model = self.model
    #     kf = self.kf
    #
    #     kf.setup()
    #     model.set_initial_conditions([.8, 0.])
    #     model.set_initial_parameter_values([.5, .4])
    #     kf.R = .064
    #     kf.Q = [.01, .01]
    #     kf.set_initial_guess([.8, 0.])
    #     kf.set_initial_parameter_values([.5, .4])
    #
    #     model.simulate(u=.8, steps=3)
    #     # FIXME: Running the following without parameters will not work since there will be a dimension mismatch
    #     kf.estimate(y=np.array([[.322052, 1.07195, 1.44033]]),
    #                 u=np.array([[.8, .8, .8]]),
    #                 steps=3)
    #     # TODO: Finish once bugs are fixed

    # def test_kalman_filter_multi_step_p(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     model = self.model
    #     kf = self.kf
    #
    #     kf.setup()
    #     model.set_initial_conditions([.8, 0.])
    #     model.set_initial_parameter_values([.5, .4])
    #     kf.R = .064
    #     kf.Q = [.01, .01]
    #     kf.set_initial_guess([.8, 0.])
    #
    #     model.simulate(u=.8, steps=3)
    #     kf.estimate(y=np.array([[.322052, 1.07195, 1.44033]]),
    #                 u=np.array([[.8, .8, .8]]),
    #                 p=np.array([[.5, .5, .5], [.4, .4, .4]]),
    #                 steps=3)
    #     # TODO: Finish once bugs are fixed

    # def test_kalman_filter_multi_step_repmat(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     model = self.model
    #     kf = self.kf
    #
    #     kf.setup()
    #     model.set_initial_conditions([.8, 0.])
    #     model.set_initial_parameter_values([.5, .4])
    #     kf.R = .064
    #     kf.Q = [.01, .01]
    #     kf.set_initial_guess([.8, 0.])
    #
    #     model.simulate(u=.8, steps=3)
    #     kf.estimate(y=.322052, u=.8, p=[.5, .4], steps=3)
    #     # TODO: Finish once bugs are fixed

    # def test_kalman_filter_multi_step_from_y(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     model = self.model
    #     kf = self.kf
    #
    #     kf.setup()
    #     model.set_initial_conditions([.8, 0.])
    #     model.set_initial_parameter_values([.5, .4])
    #     kf.R = .064
    #     kf.Q = [.01, .01]
    #     kf.set_initial_guess([.8, 0.])
    #
    #     model.simulate(u=.8, steps=3)
    #     # FIXME: Update according to Model class
    #     kf.estimate(y=np.array([[.322052, 1.07195, 1.44033]]), u=.8, p=[.5, .4])
    #     # TODO: Finish once bugs are fixed

    # def test_kalman_filter_final_time(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     model = self.model
    #     kf = self.kf
    #
    #     kf.setup()
    #     model.set_initial_conditions([.8, 0.])
    #     model.set_initial_parameter_values([.5, .4])
    #     kf.R = .064
    #     kf.Q = [.01, .01]
    #     kf.set_initial_guess([.8, 0.])
    #     kf.set_initial_parameter_values([.5, .4])
    #
    #     model.simulate(u=.8, steps=3)
    #     kf.estimate(y=np.array([[.322052, 1.07195, 1.44033]]), u=np.array([[.8, .8, .8]]), tf=3.)
    #     # TODO: Finish once bugs are fixed

    def test_kalman_filter_predict_only(self) -> None:
        """

        :return:
        """
        x = np.array([[.8], [0.]])
        P = np.eye(2)
        u = .8
        p = [.5, .4]
        Q = .01 * np.eye(2)

        kf = self.kf
        kf.setup()
        prediction = kf.predict(np.append(x, P, axis=1), np.append([u], p), Q)

        np.testing.assert_allclose(prediction, np.array([[1.2, .26, .25], [.4, .25, .62]]))

    def test_kalman_filter_update_only(self) -> None:
        """

        :return:
        """
        prediction = np.array([[1.2, .26, .25], [.4, .25, .62]])
        y = .322052
        u = .8
        p = [.5, .4]
        R = .064

        kf = self.kf
        kf.setup()
        update, y_pred = kf.update(prediction, y, np.append([u], p), R)

        np.testing.assert_allclose(update, np.array([[1.17151023, .16862573, .023391813],
                                                     [.32934538, .023391813, .0580117]]))
        np.testing.assert_allclose(y_pred, np.array([[.4]]))


class ExtendedKalmanFilter(TestCase):
    """"""
    def test_extended_kalman_filter_initialization_linear_model_warning(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh')
        equations = """
        dx_1/dt = -k_1*x_1(t) + u(k)
        dx_2/dt = k_1*x_1(t) - k_2*x_2(t)
        y(k) = x_2(t)
        """
        model.set_equations(equations=equations)
        model.discretize('erk', order=1, inplace=True)
        model.setup(dt=1.)

        with self.assertWarns(UserWarning) as context:
            EKF(model, plot_backend='bokeh')
        self.assertTrue(len(context.warnings) == 1)
        self.assertTrue(str(context.warning) == "The supplied model is linear. For better efficiency use an observer "
                                                "targeted at the estimation of linear systems.")

    def test_extended_kalman_filter_one_step(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh', discrete=True)
        model.set_dynamical_states('x')
        model.set_measurements('y')
        model.set_dynamical_equations('x/2 + 25*dt*x/(1 + x^2)')
        model.set_measurement_equations('x^2/20')
        model.setup(dt=1.)

        ekf = EKF(model, plot_backend='bokeh')
        ekf.setup()
        ekf.Q = 10.
        ekf.R = 1.
        ekf.set_initial_guess(9.)
        ekf.estimate(y=2.59109)
        np.testing.assert_allclose(ekf.solution.get_by_id('x:f'), np.array([[7.206059]]))


class TestUnscentedKalmanFilter(TestCase):
    """"""
    def test_unscented_kalman_filter_initialization_linear_model_warning(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh')
        equations = """
        dx_1/dt = -k_1*x_1(t) + u(k)
        dx_2/dt = k_1*x_1(t) - k_2*x_2(t)
        y(k) = x_2(t)
        """
        model.set_equations(equations=equations)
        model.discretize('erk', order=1, inplace=True)
        model.setup(dt=1.)

        with self.assertWarns(UserWarning) as context:
            UKF(model, plot_backend='bokeh')
        self.assertTrue(len(context.warnings) == 1)
        self.assertTrue(str(context.warning) == "The supplied model is linear. For better efficiency use an observer "
                                                "targeted at the estimation of linear systems.")

    # def test_unscented_kalman_filter_no_measurement_equations(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     model = Model(plot_backend='bokeh', discrete=True)
    #     model.set_dynamical_states('x')
    #     model.set_dynamical_equations('x/2 + 25*dt*x/(1 + x^2)')
    #     model.setup(dt=1.)
    #
    #     ukf = UKF(model, plot_backend='bokeh')
    #     # FIXME: Update setup according to _KalmanFilter class
    #     # TODO: Finish once bugs are fixed

    def test_unscented_kalman_filter_one_step(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh', discrete=True)
        model.set_dynamical_states('x')
        model.set_measurements('y')
        model.set_dynamical_equations('x/2 + 25*dt*x/(1 + x^2)')
        model.set_measurement_equations('x^2/20')
        model.setup(dt=1.)

        ukf = UKF(model, plot_backend='bokeh')
        ukf.setup()
        ukf.Q = 10.
        ukf.R = 1.
        ukf.set_initial_guess(9.)
        ukf.estimate(y=2.59109)
        np.testing.assert_allclose(ukf.solution.get_by_id('x:f'), np.array([[7.2739647]]))


class UnscentedKalmanFilterTuningParameters(TestCase):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh', discrete=True)
        model.set_dynamical_states('x')
        model.set_measurements('y')
        model.set_dynamical_equations('x/2 + 25*dt*x/(1 + x^2)')
        model.set_measurement_equations('x^2/20')
        model.setup(dt=1.)

        ukf = UKF(model, plot_backend='bokeh')
        self.ukf = ukf

    def test_unscented_kalman_filter_initial_tuning_parameters(self) -> None:
        """

        :return:
        """
        ukf = self.ukf

        np.testing.assert_allclose(ukf.alpha, .001)
        np.testing.assert_allclose(ukf.beta, 2.)
        np.testing.assert_allclose(ukf.kappa, 0.)

    def test_unscented_kalman_filter_tuning_parameter_setters(self) -> None:
        """

        :return:
        """
        ukf = self.ukf

        ukf.alpha = .1
        ukf.beta = 0.
        ukf.kappa = 10.

        np.testing.assert_allclose(ukf.alpha, .1)
        np.testing.assert_allclose(ukf.beta, 0.)
        np.testing.assert_allclose(ukf.kappa, 10.)

    def test_unscented_kalman_filter_alpha_out_of_bounds(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            self.ukf.alpha = 2.
        self.assertTrue(str(context.exception) == "The parameter alpha needs to lie in the interval (0, 1]. Supplied "
                                                  "alpha is 2.0.")

    def test_unscented_kalman_filter_kappa_out_of_bounds(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            self.ukf.kappa = -1.
        self.assertTrue(str(context.exception) == "The parameter kappa needs to be greater or equal to 0. Supplied "
                                                  "kappa is -1.0.")


class UnscentedKalmanFilterSigmaPoints(TestCase):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh')
        equations = """
        dT/dt = alpha*(T_amb - T(t))
        dcB/dt = r - D(k)*cB(t)
        dcS/dt = -r/Y - D(k)*cS(t)
        
        y_0(k) = T(t)
        y_1(k) = cB(t)
        
        r = (mu_0 + mu_1*T(t))*cS(t)*cB(t)/(K + cS(t))
        """
        model.set_equations(equations=equations)
        model.setup(dt=.1)

        ukf = UKF(model, alpha=1., plot_backend='bokeh')
        self.ukf = ukf

    def test_unscented_kalman_filter_predict_only(self) -> None:
        """

        :return:
        """
        x = np.array([[299.876], [.217108], [20.]])
        P = np.eye(3)
        u = .01
        p = [.15, 303.15, .13, .00025, 15., .14]
        Q = np.zeros((3, 3))

        ukf = self.ukf
        ukf.setup()
        prediction = ukf.predict(np.append(x, P, axis=1), np.append([u], p), Q)

        np.testing.assert_allclose(prediction, np.array([[299.925, 0.970453, 1.08254e-06, -2.11206e-05, 299.925,
                                                          301.631, 299.925, 299.925, 298.218, 299.925, 299.925],
                                                         [0.219433, 1.08254e-06, 1.02165, -0.0848792, 0.219446,
                                                          0.219451, 1.97011, 0.219537, 0.21944, -1.53128, 0.219345],
                                                         [19.9619, -2.11206e-05, -0.0848792, 1.00427, 19.9618, 19.9617,
                                                          19.8164, 21.6914, 19.9618, 20.1075, 18.2322]]), atol=1e-3)

    def test_unscented_kalman_filter_update_only(self) -> None:
        """

        :return:
        """
