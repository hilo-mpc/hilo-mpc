from unittest import TestCase

import numpy as np

from hilo_mpc import LQR, Model


class TestLQRInitialization(TestCase):
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


class TestLQRSetup(TestCase):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh', discrete=True)
        model.set_dynamical_states('x', 'y', 'z')
        model.set_inputs('u', 'w')
        model.set_parameters('p')
        model.set_dynamical_equations(['x + dt*(2*y + p*u)', 'y - dt*x', 'z + dt*w'])
        model.setup(dt=1.)

        lqr = LQR(model, plot_backend='bokeh')
        lqr.horizon = 5

        self.lqr = lqr

    def test_horizon(self) -> None:
        """

        :return:
        """
        self.assertEqual(self.lqr.horizon, 5)

    def test_dimensions(self) -> None:
        """

        :return:
        """
        lqr = self.lqr

        self.assertEqual(lqr.n_x, 0)
        self.assertEqual(lqr.n_u, 0)
        self.assertEqual(lqr.n_p, 0)

        lqr.setup()

        self.assertEqual(lqr.n_x, 3)
        self.assertEqual(lqr.n_u, 2)
        self.assertEqual(lqr.n_p, 1)

    def test_is_set_up(self) -> None:
        """

        :return:
        """
        lqr = self.lqr

        lqr.setup()
        self.assertTrue(lqr.is_setup())


class TestLQRMatrixSetters(TestCase):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh', discrete=True)
        model.set_dynamical_states('x', 'y', 'z')
        model.set_inputs('u', 'w')
        model.set_dynamical_equations(['x + dt*(2*y + u)', 'y - dt*x', 'z + dt*w'])
        model.setup(dt=1.)

        lqr = LQR(model, plot_backend='bokeh')
        lqr.horizon = 5
        lqr.setup()

        self.lqr = lqr

    def test_q_is_square(self) -> None:
        """

        :return:
        """
        self.lqr.Q = [1, 1, 1]
        np.testing.assert_allclose(self.lqr.Q, np.eye(3))

    def test_q_dimension_mismatch(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            self.lqr.Q = [1, 1]
        self.assertTrue(str(context.exception) == "Dimension mismatch. Supplied dimension is 2x2, but required "
                                                  "dimension is 3x3")

    def test_q_complex(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            self.lqr.Q = np.complex(2, 1) * np.eye(3)
        self.assertTrue(str(context.exception) == "LQR matrix Q needs to be real-valued")

    def test_q_not_symmetric(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            self.lqr.Q = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        self.assertTrue(str(context.exception) == "LQR matrix Q needs to be symmetric")

    def test_q_not_positive_semidefinite(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            self.lqr.Q = -np.eye(3)
        self.assertTrue(str(context.exception) == "LQR matrix Q needs to be positive semidefinite")

    def test_r_is_square(self) -> None:
        """

        :return:
        """
        self.lqr.R = [1, 1]
        np.testing.assert_allclose(self.lqr.R, np.eye(2))

    def test_r_dimension_mismatch(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            self.lqr.R = [1, 1, 1]
        self.assertTrue(str(context.exception) == "Dimension mismatch. Supplied dimension is 3x3, but required "
                                                  "dimension is 2x2")

    def test_r_complex(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            self.lqr.R = np.complex(2, 1) * np.eye(2)
        self.assertTrue(str(context.exception) == "LQR matrix R needs to be real-valued")

    def test_r_not_symmetric(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            self.lqr.R = np.array([[1, 0], [1, 0]])
        self.assertTrue(str(context.exception) == "LQR matrix R needs to be symmetric")

    def test_r_not_positive_definite(self) -> None:
        """

        :return:
        """
        with self.assertRaises(ValueError) as context:
            self.lqr.R = -np.eye(2)
        self.assertTrue(str(context.exception) == "LQR matrix R needs to be positive definite")


class TestLQRCall(TestCase):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        model = Model(plot_backend='bokeh', discrete=True)
        model.set_dynamical_states('x', 'y', 'z')
        model.set_inputs('u', 'w')
        model.set_parameters('p')
        model.set_dynamical_equations(['x + dt*(2*y + p*u)', 'y - dt*x', 'z + dt*w'])
        model.setup(dt=1.)

        lqr = LQR(model, plot_backend='bokeh')
        lqr.horizon = 5

        self.lqr = lqr

    def test_lqr_not_set_up(self) -> None:
        """

        :return:
        """
        with self.assertRaises(RuntimeError) as context:
            self.lqr.call()
        self.assertTrue(str(context.exception) == "LQR is not set up. Run LQR.setup(...) before calling the LQR.")

    def test_lqr_no_q_matrix(self) -> None:
        """

        :return:
        """
        lqr = self.lqr
        lqr.setup()
        with self.assertRaises(RuntimeError) as context:
            lqr.call()
        self.assertTrue(str(context.exception) == "Matrix Q is not set properly. To ensure that a unique solution "
                                                  "exists, the matrix Q needs to be symmetric, real-valued and positive"
                                                  " semidefinite.")

    def test_lqr_no_r_matrix(self) -> None:
        """

        :return:
        """
        lqr = self.lqr
        lqr.setup()
        lqr.Q = [1, 1, 1]
        with self.assertRaises(RuntimeError) as context:
            lqr.call()
        self.assertTrue(str(context.exception) == "Matrix R is not set properly. To ensure that a unique solutions "
                                                  "exists, the matrix R needs to be symmetric, real-valued and positive"
                                                  " definite.")

    def test_lqr_no_x_supplied(self) -> None:
        """

        :return:
        """
        lqr = self.lqr
        lqr.setup()
        lqr.Q = [1, 1, 1]
        lqr.R = [1, 1]

        with self.assertRaises(ValueError) as context:
            lqr.call()
        self.assertTrue(str(context.exception) == "No state information was supplied to the LQR!")

    def test_lqr_k_matrix_no_p_supplied(self) -> None:
        """

        :return:
        """
        lqr = self.lqr
        lqr.setup()
        self.assertIsNone(lqr.K)

        lqr.Q = [1, 1, 1]
        lqr.R = [1, 1]

        _ = lqr.call(x=[1, 0, 1])

        np.testing.assert_allclose(lqr.K, np.array([[0., 0., 0.], [0., 0., .61802575]]))

    def test_lqr_k_matrix_p_supplied(self) -> None:
        """

        :return:
        """
        lqr = self.lqr
        lqr.setup()
        lqr.Q = [1, 1, 1]
        lqr.R = [1, 1]

        _ = lqr.call(x=[1, 0, 1], p=1)

        np.testing.assert_allclose(lqr.K, np.array([[1.39207671, 1.35221712, 0.], [0., 0., .61802575]]))
