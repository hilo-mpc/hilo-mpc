from unittest import TestCase

import numpy as np

from hilo_mpc import Model, KF


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

    # def test_kalman_filter_multi_step_dimension_mismatch_in_y(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     kf = self.kf
    #
    #     kf.setup()
    #     kf.set_initial_guess([.8, 0.])
    #     # FIXME: Convert IndexError to ValueError
    #     with self.assertRaises(IndexError) as context:
    #         kf.estimate(y=[.3894626, .3894626], steps=3)
    #     self.assertTrue(str(context.exception) == "Dimension mismatch for variable y. Supplied dimension is 2, but "
    #                                               "required dimension is 3.")

    # def test_kalman_filter_multi_step_dimension_mismatch_in_u(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #
    # def test_kalman_filter_multi_step_dimension_mismatch_in_p(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #
    # def test_kalman_filter_multi_step(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #
    # def test_kalman_filter_multi_step_p(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #
    # def test_kalman_filter_multi_step_repmat(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #
    # def test_kalman_filter_final_time(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #
    # def test_kalman_filter_predict_only(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #
    # def test_kalman_filter_update_only(self) -> None:
    #     """
    #
    #     :return:
    #     """
