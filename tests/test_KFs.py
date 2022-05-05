from unittest import TestCase

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
