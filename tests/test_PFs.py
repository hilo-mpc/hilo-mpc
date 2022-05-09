from unittest import TestCase

import numpy as np

from hilo_mpc import Model, PF


class TestParticleFilterInitialization(TestCase):
    """"""
    def test_particle_filter_linear_model_warning(self) -> None:
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
            PF(model, plot_backend='bokeh')
        self.assertTrue(len(context.warnings) == 1)
        self.assertTrue(str(context.warning) == "The supplied model is linear. For better efficiency use an observer "
                                                "targeted at the estimation of linear systems.")

    def test_particle_filter_initial_tuning_parameters(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh', discrete=True)
        model.set_dynamical_states('x')
        model.set_measurements('y')
        model.set_dynamical_equations('x/2 + 25*dt*x/(1 + x^2)')
        model.set_measurement_equations('x^2/20')
        model.setup(dt=1.)

        pf = PF(model, plot_backend='bokeh')
        self.assertTrue(callable(pf.probability_density_function))
        # TODO: Any other assertions with respect to the probability density function?
        self.assertIsNone(pf.variant)
        self.assertTrue(pf.sample_size == 15)


class TestParticleFilterPDF(TestCase):
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

        pf = PF(model, plot_backend='bokeh')
        self.pf = pf

    def test_particle_filter_pdf_not_callable(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            self.pf.probability_density_function = 'pdf'
        self.assertTrue(str(context.exception) == "Probability density function of the particle filter needs to be "
                                                  "callable.")

    def test_particle_filter_pdf_annotations(self) -> None:
        """

        :return:
        """
        def pdf(mu: np.ndarray, sigma: np.ndarray, n: int) -> np.ndarray:
            """

            :param mu:
            :param sigma:
            :param n:
            :return:
            """
