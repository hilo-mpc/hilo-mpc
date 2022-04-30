from unittest import TestCase

from hilo_mpc import LQR, Model


class TestLQRGeneral(TestCase):
    """"""
    def test_no_model_supplied(self) -> None:
        """

        :return:
        """
        with self.assertRaises(TypeError) as context:
            LQR('model')
        self.assertTrue(str(context.exception) == "The model must be an object of the Model class.")

    def test_model_not_set_up(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh')
        with self.assertRaises(RuntimeError) as context:
            LQR(model)
        self.assertTrue(str(context.exception) == "Model is not set up. Run Model.setup() before passing it to the "
                                                  "controller.")

    def test_model_continuous_lqr_discrete(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh')
        model.set_dynamical_states('x')
        model.set_dynamical_equations('x^2')
        model.setup(dt=1.)
        with self.assertRaises(RuntimeError) as context:
            LQR(model)
        self.assertTrue(str(context.exception) == "The model used for the LQR needs to be discrete. Use "
                                                  "Model.discretize() to obtain a discrete model.")

    def test_model_nonlinear(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh', discrete=True)
        model.set_dynamical_states('x')
        model.set_dynamical_equations('x + dt*x^2')
        model.setup(dt=1.)
        with self.assertRaises(RuntimeError) as context:
            LQR(model)
        self.assertTrue(str(context.exception) == "The model used for the LQR needs to be linear. Use Model.linearize()"
                                                  " to obtain a linearized model.")

    def test_model_autonomous(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh', discrete=True)
        model.set_dynamical_states('x')
        model.set_dynamical_equations('x + dt*2*x')
        model.setup(dt=1.)
        with self.assertRaises(RuntimeError) as context:
            LQR(model)
        self.assertTrue(str(context.exception) == "The model used for the LQR is autonomous.")
