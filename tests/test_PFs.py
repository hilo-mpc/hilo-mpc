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
        pf.setup()
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
            np.random.seed(0)
            return np.random.multivariate_normal(mu, sigma, size=n)

        pf = self.pf
        pf.probability_density_function = pdf
        self.assertTrue(callable(pf.probability_density_function))
        np.testing.assert_allclose(pf.probability_density_function(np.array([0., 0., 0.]), np.eye(3), 3),
                                   pdf(np.array([0., 0., 0.]), np.eye(3), 3))

    def test_particle_filter_pdf_wrong_annotation_mu(self) -> None:
        """

        :return:
        """
        def pdf(mu: float, sigma: np.ndarray, n: int) -> np.ndarray:
            """

            :param mu:
            :param sigma:
            :param n:
            :return:
            """
            pass

        with self.assertRaises(TypeError) as context:
            self.pf.probability_density_function = pdf
        self.assertTrue(str(context.exception) == "The 1st argument to the probability density function (pdf) needs to "
                                                  "be the 'mean' with type ndarray.")

    def test_particle_filter_pdf_wrong_annotation_sigma(self) -> None:
        """

        :return:
        """
        def pdf(mu: np.ndarray, sigma: float, n: int) -> np.ndarray:
            """

            :param mu:
            :param sigma:
            :param n:
            :return:
            """
            pass

        with self.assertRaises(TypeError) as context:
            self.pf.probability_density_function = pdf
        self.assertTrue(str(context.exception) == "The 2nd argument to the probability density function (pdf) needs to "
                                                  "be the 'covariance' with type ndarray.")

    def test_particle_filter_pdf_wrong_annotation_n(self) -> None:
        """

        :return:
        """
        def pdf(mu: np.ndarray, sigma: np.ndarray, n: float) -> np.ndarray:
            """

            :param mu:
            :param sigma:
            :param n:
            :return:
            """
            pass

        with self.assertRaises(TypeError) as context:
            self.pf.probability_density_function = pdf
        self.assertTrue(str(context.exception) == "The 3rd argument to the probability density function (pdf) needs to "
                                                  "be the 'sample size' with type int.")

    def test_particle_filter_pdf_wrong_annotation_return(self) -> None:
        """

        :return:
        """
        def pdf(mu: np.ndarray, sigma: np.ndarray, n: int) -> float:
            """

            :param mu:
            :param sigma:
            :param n:
            :return:
            """
            pass

        with self.assertRaises(TypeError) as context:
            self.pf.probability_density_function = pdf
        self.assertTrue(str(context.exception) == "The return value of the probability density function (pdf) needs to "
                                                  "be a 'random sample' with type ndarray.")

    def test_particle_filter_pdf_no_annotation(self) -> None:
        """

        :return:
        """
        def pdf(mu, sigma, n):
            """

            :param mu:
            :param sigma:
            :param n:
            :return:
            """
            np.random.seed(0)
            return np.random.multivariate_normal(mu, sigma, size=n).T

        pf = self.pf
        pf.probability_density_function = pdf
        self.assertTrue(callable(pf.probability_density_function))
        np.testing.assert_allclose(pf.probability_density_function(np.array([0., 0., 0.]), np.eye(3), 3),
                                   pdf(np.array([0., 0., 0.]), np.eye(3), 3))

    def test_particle_filter_pdf_no_annotation_transposed(self) -> None:
        """

        :return:
        """
        def pdf(mu, sigma, n):
            """

            :param mu:
            :param sigma:
            :param n:
            :return:
            """
            np.random.seed(0)
            return np.random.multivariate_normal(mu, sigma, size=n)

        pf = self.pf
        pf.probability_density_function = pdf
        self.assertTrue(callable(pf.probability_density_function))
        np.testing.assert_allclose(pf.probability_density_function(np.array([0., 0., 0.]), np.eye(3), 3),
                                   pdf(np.array([0., 0., 0.]), np.eye(3), 3))

    # def test_particle_filter_pdf_no_annotation_dimension_mismatch(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     def pdf(mu, sigma, n):
    #         """
    #
    #         :param mu:
    #         :param sigma:
    #         :param n:
    #         :return:
    #         """
    #         return np.random.multivariate_normal(np.tile(mu, 2), np.tile(sigma, (2, 2)), size=n)
    #
    #     pf = self.pf
    #     # FIXME: Right now this ValueError will be caught by the following except statement
    #     with self.assertRaises(ValueError) as context:
    #         pf.probability_density_function = pdf
    #     self.assertTrue(str(context.exception) == "Dimension mismatch. Expected dimension 1x15, got 15x2.")

    def test_particle_filter_pdf_no_annotation_exception(self) -> None:
        """

        :return:
        """
        def pdf(mu, sigma):
            """

            :param mu:
            :param sigma:
            :return:
            """
            pass

        pf = self.pf
        with self.assertRaises(RuntimeError) as context:
            pf.probability_density_function = pdf
        self.assertTrue(str(context.exception) == "The following exception was raised\n"
                                                  "   TypeError: 'pdf() takes 2 positional arguments but 3 were given'."
                                                  "\n"
                                                  "Please make sure that the supplied probability density function "
                                                  "(pdf) has the following arguments\n"
                                                  "   mu - mean of the pdf (type: numpy.ndarray),\n"
                                                  "   sigma - covariance of the mean (type: numpy.ndarray),\n"
                                                  "   n - sample size (type: int),\n"
                                                  "and the following return value\n"
                                                  "   X - random sample (type: numpy.ndarray).")


class TestParticleFilterOtherTunableParameterSetters(TestCase):
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
        pf.setup()
        self.pf = pf

    def test_particle_filter_variant_setter(self) -> None:
        """

        :return:
        """
        pf = self.pf
        pf.variant = 'default'
        self.assertTrue(pf.variant == 'default')

    def test_particle_filter_sample_size_setter(self) -> None:
        """

        :return:
        """
        pf = self.pf
        pf.sample_size = 20
        self.assertTrue(pf.sample_size == 20)


class TestParticleFilterEstimation(TestCase):
    """"""
    # TODO: Tests for transpose_pdf equal to True and False + dimension mismatch error (maybe the first 2 can be
    #  integrated into the already existing tests, dimension mismatch will get an extra test)
    def setUp(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh')
        equations = """
        dT/dt = alpha*(T_amb - T(t))
        dcB/dt = r - D(k)*cB(t)
        dcS/dt = -r/Y - D(k)*cS(t)

        r = (mu_0 + mu_1*T(t))*cS(t)*cB(t)/(K + cS(t))
        """
        model.set_equations(equations=equations)

        self.model = model

    # def test_particle_filter_no_measurement_equations(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     model = self.model
    #     model.setup(dt=.1)
    #
    #     pf = PF(model, plot_backend='bokeh')
    #     # FIXME: Update setup according to _KalmanFilter class
    #     # TODO: Finish once bugs are fixed

    # def test_particle_filter_one_step(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     model = self.model
    #     model.set_measurement_equations(['T', 'cB'])
    #     model.setup(dt=.1)
    #
    #     pf = PF(model, plot_backend='bokeh')
    #     # FIXME: Move call to method ParticleFilter._setup_normpdf from ParticleFilter.__init__ to ParticleFilter.setup
    #     #  and add dimensions. Right now the particle filter will only work for one measurement.
    #     pf.setup()
    #
    #     pf.R = np.diag([.25, .01])
    #     pf.set_initial_guess([299.876, .217108, 20.])
    #     pf.set_initial_parameter_values([.15, 303.15, .13, .00025, 15., .14])
    #
    #     pf.estimate(y=[300.941, .245805], u=.01)
    #     # TODO: Finish once bugs are fixed

    # def test_particle_filter_one_step_roughening(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     model = self.model
    #     model.set_measurement_equations(['T', 'cB'])
    #     model.setup(dt=.1)
    #
    #     pf = PF(model, roughening=True, plot_backend='bokeh')
    #     # FIXME: Move call to method ParticleFilter._setup_normpdf from ParticleFilter.__init__ to ParticleFilter.setup
    #     #  and add dimensions. Right now the particle filter will only work for one measurement.
    #     pf.setup()
    #
    #     pf.R = np.diag([.25, .01])
    #     pf.set_initial_guess([299.876, .217108, 20.])
    #     pf.set_initial_parameter_values([.15, 303.15, .13, .00025, 15., .14])
    #
    #     pf.estimate(y=[300.941, .245805], u=.01)
    #     # TODO: Finish once bugs are fixed

    # def test_particle_filter_one_step_prior_editing_K_supplied(self) -> None:
    #     """
    #
    #     :return:
    #     """
    #     model = self.model
    #     model.set_measurement_equations(['T', 'cB'])
    #     model.setup(dt=.1)
    #
    #     pf = PF(model, prior_editing=True, K=.02, plot_backend='bokeh')
    #     # FIXME: Move call to method ParticleFilter._setup_normpdf from ParticleFilter.__init__ to ParticleFilter.setup
    #     #  and add dimensions. Right now the particle filter will only work for one measurement.
    #     pf.setup(n_samples=20)
    #
    #     pf.R = np.diag([.25, .01])
    #     pf.set_initial_guess([299.876, .217108, 20.])
    #     pf.set_initial_parameter_values([.15, 303.15, .13, .00025, 15., .14])
    #
    #     pf.estimate(y=[300.941, .245805], u=.01)
    #     # TODO: Finish once bugs are fixed
