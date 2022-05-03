from unittest import TestCase, skip

import casadi as ca
import numpy as np

from hilo_mpc import Model


class TestGeneral(TestCase):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        self.model = Model(plot_backend='bokeh')

    def test_initialization_1(self) -> None:
        """

        :return:
        """
        A = self.model.A
        B = self.model.B
        C = self.model.C
        D = self.model.D
        M = self.model.M

        self.assertIsNone(A)
        self.assertIsNone(B)
        self.assertIsNone(C)
        self.assertIsNone(D)
        self.assertIsNone(M)

    def test_initialization_2(self) -> None:
        """

        :return:
        """
        A = np.random.randn(2, 3)
        with self.assertRaises(ValueError) as context:
            self.model.A = A
        self.assertTrue("The state matrix needs to be a square matrix. Supplied matrix has dimensions 2x3." == str(
            context.exception))

    def test_initialization_3(self) -> None:
        """

        :return:
        """
        M = np.eye(2, 3)
        with self.assertRaises(ValueError) as context:
            self.model.M = M
        self.assertTrue("The mass matrix (M) needs to be a square matrix. Supplied matrix has dimensions 2x3." == str(
            context.exception))


class TestODESystem(TestCase):
    """"""
    def setUp(self) -> None:
        """

        :return:
        """
        self.model = Model(plot_backend='bokeh')
        self.scenario = 0

    def tearDown(self) -> None:
        """

        :return:
        """
        if self.scenario in [1, 2, 5]:
            self.model.set_initial_conditions([5., 0., 0., 0., 0.])
            K = np.array([[775., 48., 19., 8., -21.]])
            u = K @ self.model.solution.get_by_id('x:f')
            if self.scenario == 2:
                self.model.set_initial_parameter_values(u)

            if self.scenario == 2:
                self.model.simulate()
            else:
                self.model.simulate(u=u)

            np.testing.assert_allclose(self.model.solution.get_by_id('x:f'), [[5.016586321846274], [.00627089461146664],
                                                                              [6.160937724223938], [1.2708949868233719],
                                                                              [-86.50514360918245]])
        elif self.scenario in [3, 4]:
            # NOTE: Parameters are initialized before states, since measurements depend on parameters
            self.model.set_initial_parameter_values([.5, .25, .1, .3])
            self.model.set_initial_conditions([1., 1., 1.])

            self.model.simulate()

            np.testing.assert_allclose(self.model.solution.get_by_id('x:f'), [[2.517059337383009], [2.208782390915479],
                                                                              [1.1051709515096617]])
            np.testing.assert_approx_equal(self.model.solution.get_by_id('y:0'), .3)
            np.testing.assert_approx_equal(self.model.solution.get_by_id('y:f'), .6626347172746436)

    def test_initialization_1(self):
        """
        Linear ODE system via corresponding matrices.

        :return:
        """
        # TODO: Set model._state_space_rep entry to None, if corresponding equation is reset
        A = np.array([[0., 0., 1., 0., 0.],
                      [0., 0., 0., 1., 0.],
                      [-50., 50., -5., 5., -20.],
                      [25., -25., 2., -2., 0.],
                      [-600., 0., 20., 0., -3.]])
        B = np.array([[0.], [0.], [0.], [0.], [-1.5]])
        C = np.array([[0., 0., 0., 0., 1.]])

        self.assertIsInstance(A, np.ndarray)
        self.assertIsInstance(B, np.ndarray)
        self.assertIsInstance(C, np.ndarray)

        self.model.A = A
        self.model.B = B
        self.model.C = C

        A = self.model.A
        B = self.model.B
        C = self.model.C

        self.assertIsInstance(A, ca.SX)
        self.assertIsInstance(B, ca.SX)
        self.assertIsInstance(C, ca.SX)

        self.model.setup(dt=.01)
        self.assertTrue(self.model.is_linear())

        self.scenario = 1

    def test_initialization_3(self):
        """
        Linear ODE system via corresponding matrices with unset parameters, i.e. parameters not set by a Model method.

        :return:
        """
        a = ca.SX.sym('a')
        b = ca.SX.sym('b')
        c = ca.SX.sym('c')
        d = ca.SX.sym('d')
        p = ca.vertcat(a, b, c, d)

        A = np.array([[0., 1., 0.],
                      [a, b, 0.],
                      [0., 0., c]])
        C = np.array([[0., d, 0.]])

        self.model.A = A
        self.model.C = C

        self.model.setup(dt=1.)
        self.assertTrue(self.model.is_linear())
        self.assertTrue(self.model.n_p, 4)
        self.assertTrue(ca.depends_on(self.model.ode, p))
        self.assertTrue(ca.depends_on(self.model.meas, p))

        self.scenario = 3

    def test_initialization_4(self):
        """
        Like TestODESystem.test_initialization_3(), but a mixture of set and unset parameters.

        :return:
        """
        p = self.model.set_parameters(['a', 'b'])
        c = ca.SX.sym('c')
        d = ca.SX.sym('d')

        A = np.array([[0., 1., 0.],
                      [p[0], p[1], 0.],
                      [0., 0., c]])
        C = np.array([[0., d, 0.]])

        self.model.A = A
        self.model.C = C

        self.model.setup(dt=1.)
        self.assertTrue(self.model.is_linear())
        self.assertTrue(self.model.n_p, 4)
        self.assertTrue(ca.depends_on(self.model.ode, p))
        self.assertTrue(ca.depends_on(self.model.ode, c))
        self.assertTrue(ca.depends_on(self.model.meas, d))

        self.scenario = 4

    def test_initialization_5(self):
        """
        Generate matrices from ODE system in equation form.

        :return:
        """
        self.model.set_dynamical_states('x', 5)
        self.model.set_inputs('u')

        self.model.set_equations(ode=['x_2', 'x_3', '-50*x_0 + 50*x_1 - 5*x_2 + 5*x_3 - 20*x_4',
                                      '25*x_0 - 25*x_1 + 2*x_2 - 2*x_3', '-600*x_0 + 20*x_2 - 3*x_4 - 1.5*u'],
                                 meas=['x_4'])

        self.model.setup(dt=.01)
        self.assertTrue(self.model.is_linear())

        A = ca.SX([[0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0],
                   [-50, 50, -5, 5, -20],
                   [25, -25, 2, -2, 0],
                   [-600, 0, 20, 0, -3]])
        B = ca.SX([[0], [0], [0], [0], [-1.5]])
        C = ca.SX([[0, 0, 0, 0, 1]])

        A_model = self.model.A
        B_model = self.model.B
        C_model = self.model.C

        self.assertIsInstance(A_model, ca.SX)
        self.assertIsInstance(B_model, ca.SX)
        self.assertIsInstance(C_model, ca.SX)

        self.assertTrue(A_model.shape == (5, 5))
        self.assertTrue(A_model.is_constant())
        self.assertTrue(ca.simplify(A - A_model).is_zero())
        self.assertTrue(B_model.shape == (5, 1))
        self.assertTrue(B_model.is_constant())
        self.assertTrue(ca.simplify(B - B_model).is_zero())
        self.assertTrue(C_model.shape == (1, 5))
        self.assertTrue(C_model.is_constant())
        self.assertTrue(ca.simplify(C - C_model).is_zero())

        self.scenario = 5

    def test_initialization_6(self):
        """
        Raise dimension mismatch error for wrong dimensions in supplied state matrix A.

        :return:
        """
        self.model.set_dynamical_states('x', 5)
        A = np.array([[0., 0., 1., 0.],
                      [0., 0., 0., 1.],
                      [-50., 50., -5., 5.],
                      [25., -25., 2., -2.]])
        B = np.array([[0.], [0.], [0.], [0.], [-1.5]])
        C = np.array([[0., 0., 0., 0., 1.]])

        self.assertIsInstance(A, np.ndarray)
        self.assertIsInstance(B, np.ndarray)
        self.assertIsInstance(C, np.ndarray)

        self.model.A = A
        self.model.B = B
        self.model.C = C

        A = self.model.A
        B = self.model.B
        C = self.model.C

        self.assertIsInstance(A, ca.SX)
        self.assertIsInstance(B, ca.SX)
        self.assertIsInstance(C, ca.SX)

        with self.assertRaises(ValueError) as context:
            self.model.setup(dt=.01)
        self.assertTrue(
            "Dimension mismatch in state matrix (A). Supplied dimension is 4x4, but required dimension is 5x5." == str(
                context.exception))

    def test_initialization_7(self):
        """
        Raise dimension mismatch error for wrong dimensions in supplied input matrix B.

        :return:
        """
        A = np.array([[0., 0., 1., 0., 0.],
                      [0., 0., 0., 1., 0.],
                      [-50., 50., -5., 5., -20.],
                      [25., -25., 2., -2., 0.],
                      [-600., 0., 20., 0., -3.]])
        B = np.array([[0.], [0.], [0.], [-1.5]])
        C = np.array([[0., 0., 0., 0., 1.]])

        self.assertIsInstance(A, np.ndarray)
        self.assertIsInstance(B, np.ndarray)
        self.assertIsInstance(C, np.ndarray)

        self.model.A = A
        self.model.B = B
        self.model.C = C

        A = self.model.A
        B = self.model.B
        C = self.model.C

        self.assertIsInstance(A, ca.SX)
        self.assertIsInstance(B, ca.SX)
        self.assertIsInstance(C, ca.SX)

        with self.assertRaises(ValueError) as context:
            self.model.setup(dt=.01)
        self.assertTrue(
            "Dimension mismatch in input matrix (B). Supplied dimension is 4x1, but required dimension is 5x1." == str(
                context.exception))

    def test_initialization_8(self):
        """
        Raise dimension mismatch error for wrong dimensions in supplied output matrix C.

        :return:
        """
        A = np.array([[0., 0., 1., 0., 0.],
                      [0., 0., 0., 1., 0.],
                      [-50., 50., -5., 5., -20.],
                      [25., -25., 2., -2., 0.],
                      [-600., 0., 20., 0., -3.]])
        B = np.array([[0.], [0.], [0.], [0.], [-1.5]])
        C = np.array([[0., 0., 0., 1.]])

        self.assertIsInstance(A, np.ndarray)
        self.assertIsInstance(B, np.ndarray)
        self.assertIsInstance(C, np.ndarray)

        self.model.A = A
        self.model.B = B
        self.model.C = C

        A = self.model.A
        B = self.model.B
        C = self.model.C

        self.assertIsInstance(A, ca.SX)
        self.assertIsInstance(B, ca.SX)
        self.assertIsInstance(C, ca.SX)

        with self.assertRaises(ValueError) as context:
            self.model.setup(dt=.01)
        self.assertTrue(
            "Dimension mismatch in output matrix (C). Supplied dimension is 1x4, but required dimension is 1x5." == str(
                context.exception))

    def test_initialization_9(self):
        """
        Raise dimension mismatch error for wrong dimensions in supplied feedthrough matrix D.

        :return:
        """
        A = np.array([[0., 0., 1., 0., 0.],
                      [0., 0., 0., 1., 0.],
                      [-50., 50., -5., 5., -20.],
                      [25., -25., 2., -2., 0.],
                      [-600., 0., 20., 0., -3.]])
        B = np.array([[0.], [0.], [0.], [0.], [-1.5]])
        C = np.array([[0., 0., 0., 0., 1.]])
        D = np.array([[0.], [0.]])

        self.assertIsInstance(A, np.ndarray)
        self.assertIsInstance(B, np.ndarray)
        self.assertIsInstance(C, np.ndarray)
        self.assertIsInstance(D, np.ndarray)

        self.model.A = A
        self.model.B = B
        self.model.C = C
        self.model.D = D

        A = self.model.A
        B = self.model.B
        C = self.model.C
        D = self.model.D

        self.assertIsInstance(A, ca.SX)
        self.assertIsInstance(B, ca.SX)
        self.assertIsInstance(C, ca.SX)
        self.assertIsInstance(D, ca.SX)

        with self.assertRaises(ValueError) as context:
            self.model.setup(dt=.01)
        self.assertTrue(
            "Dimension mismatch in feedthrough matrix (D). Supplied dimension is 2x1, but required dimension is 1x1."
            == str(context.exception))

    @skip("Test will be moved to discrete model testing.")
    def test_initialization_12(self):
        """
        Discrete

        :return:
        """
        dt = self.model.dt

        A = np.array([[1., 0., dt, 0., 0.],
                      [0., 1., 0., dt, 0.],
                      [-50. * dt, 50. * dt, 1. - 5. * dt, 5. * dt, -20. * dt],
                      [25. * dt, -25. * dt, 2. * dt, 1. - 2. * dt, 0.],
                      [-600. * dt, 0., 20. * dt, 0., 1. - 3. * dt]])
        B = np.array([[0.], [0.], [0.], [0.], [-1.5 * dt]])
        C = np.array([[0., 0., 0., 0., 1.]])

        self.model.A = A
        self.model.B = B
        self.model.C = C

        self.model.setup(dt=1.)

    @skip("Test not yet implemented. Test will be moved to time-variant model testing.")
    def test_initialization_13(self):
        """

        :return:
        """
        # TODO: Time-variant matrices

    @skip("Test will be moved to discretization tests.")
    def test_initialization_14(self):
        """

        :return:
        """
        A = np.array([[0., 0., 1., 0., 0.],
                      [0., 0., 0., 1., 0.],
                      [-50., 50., -5., 5., -20.],
                      [25., -25., 2., -2., 0.],
                      [-600., 0., 20., 0., -3.]])
        B = np.array([[0.], [0.], [0.], [0.], [-1.5]])
        C = np.array([[0., 0., 0., 0., 1.]])

        self.model.A = A
        self.model.B = B
        self.model.C = C

        self.model.discretize('erk', order=1, inplace=True)

        A = self.model.A
        B = self.model.B
        C = self.model.C

        self.model.setup(dt=1.)


class TestDAESystem(TestCase):
    """"""
    # NOTE: Adapted from https://python.hotexamples.com/de/examples/assimulo.solvers.sundials/IDA/simulate/python-ida
    #  -simulate-method-examples.html#0x689b3e64a65dcad07ba6938357cfc40e851970a72e41ca416382c54b37a7be76-23,,110,
    def setUp(self) -> None:
        """

        :return:
        """
        self.model = Model(solver='idas', plot_backend='bokeh')
        self.scenario = 0

    def tearDown(self) -> None:
        """

        :return:
        """
        if self.scenario in [1, 2, 3, 4, 5]:
            self.model.set_initial_conditions([0, 0], z0=0)
            self.model.set_initial_parameter_values([.0211, .0162, .0111, .0124, .0039, .000035])

            if self.scenario in [1, 3, 4, 5]:
                self.model.simulate(u=.001)
            elif self.scenario == 2:
                self.model.simulate()

            if self.scenario in [3, 5]:
                np.testing.assert_allclose(self.model.M, np.diag([1., 1., 0.]))

            if self.scenario == 4:
                np.testing.assert_allclose(self.model.solution.get_by_id('x:f'),
                                           [[.0019370039212824376], [5.406953838018339e-06]])
                np.testing.assert_approx_equal(self.model.solution.get_by_id('z:f'), .21583757980004303)
            else:
                np.testing.assert_allclose(self.model.solution.get_by_id('x:f'),
                                           [[.0009840937669093723], [5.438849408026158e-06]])
                np.testing.assert_approx_equal(self.model.solution.get_by_id('z:f'), .10965616259847293)

    def test_initialization_1(self):
        """
        Linear DAE via corresponding matrices

        :return:
        """
        p = self.model.set_parameters('p', 6)
        A = np.array([[-(p[0] + p[2] + p[4]), p[3], p[5]],
                      [p[2], -(p[1] + p[3]), 0.],
                      [p[4], 0., -p[5]]])
        B = np.array([[1.], [0.], [0.]])
        m = np.array([1., 1., 0.])

        self.model.A = A
        self.model.B = B
        self.model.M = m

        A = self.model.A
        B = self.model.B
        M = self.model.M

        self.assertIsInstance(A, ca.SX)
        self.assertIsInstance(B, ca.SX)
        self.assertIsInstance(M, np.ndarray)

        self.model.setup(dt=1.)
        self.assertTrue(self.model.is_linear())
        self.assertTrue(ca.depends_on(self.model.ode, p))
        self.assertTrue(ca.depends_on(self.model.alg, p))

        self.scenario = 1

    def test_initialization_3(self):
        """
        Linear DAE, no mass matrix M supplied. Differential and algebraic states are directly defined via the
        corresponding Model methods.

        :return:
        """
        self.model.set_dynamical_states('x', 2)
        self.model.set_algebraic_states('z')
        self.model.set_inputs('u')
        p = self.model.set_parameters('p', 6)

        A = np.array([[-(p[0] + p[2] + p[4]), p[3], p[5]],
                      [p[2], -(p[1] + p[3]), 0.],
                      [p[4], 0., -p[5]]])
        B = np.array([[1.], [0.], [0.]])

        self.model.A = A
        self.model.B = B

        A = self.model.A
        B = self.model.B
        M = self.model.M

        self.assertIsInstance(A, ca.SX)
        self.assertIsInstance(B, ca.SX)
        self.assertIsNone(M)

        self.model.setup(dt=1.)
        self.assertTrue(self.model.is_linear())
        self.assertTrue(ca.depends_on(self.model.ode, p))
        self.assertTrue(ca.depends_on(self.model.alg, p))

        M = self.model.M
        self.assertIsInstance(M, np.ndarray)

        self.scenario = 3

    def test_initialization_4(self):
        """
        Linear DAE, different 'weighting' in the mass matrix M. Don't know if this actually makes sense.

        :return:
        """
        p = self.model.set_parameters('p', 6)
        A = np.array([[-(p[0] + p[2] + p[4]), p[3], p[5]],
                      [p[2], -(p[1] + p[3]), 0.],
                      [p[4], 0., -p[5]]])
        B = np.array([[1.], [0.], [0.]])
        m = np.array([.5, 2., 0.])

        self.model.A = A
        self.model.B = B
        self.model.M = m

        A = self.model.A
        B = self.model.B
        M = self.model.M

        self.assertIsInstance(A, ca.SX)
        self.assertIsInstance(B, ca.SX)
        self.assertIsInstance(M, np.ndarray)

        self.model.setup(dt=1.)
        self.assertTrue(self.model.is_linear())
        self.assertTrue(ca.depends_on(self.model.ode, p))
        self.assertTrue(ca.depends_on(self.model.alg, p))

        self.scenario = 4

    def test_initialization_5(self):
        """
        Generate matrices from DAE system in equation form.

        :return:
        """
        self.model.set_dynamical_states('x', 2)
        self.model.set_algebraic_states('z')
        self.model.set_inputs('u')
        p = self.model.set_parameters('p', 6)

        self.model.set_equations(ode=['-(p_0 + p_2 + p_4)*x_0 + p_3*x_1 + p_5*z + u', 'p_2*x_0 - (p_1 + p_3)*x_1'],
                                 alg=['p_4*x_0 - p_5*z'])

        self.model.setup(dt=1.)
        self.assertTrue(self.model.is_linear())

        A = ca.vertcat(ca.horzcat(*[-(p[0] + p[2] + p[4]), p[3], p[5]]),
                       ca.horzcat(*[p[2], -(p[1] + p[3]), 0.]),
                       ca.horzcat(*[p[4], 0., -p[5]]))
        B = ca.SX([[1.], [0.], [0.]])

        A_model = self.model.A
        B_model = self.model.B
        M_model = self.model.M

        self.assertIsInstance(A_model, ca.SX)
        self.assertIsInstance(B_model, ca.SX)
        self.assertIsInstance(M_model, np.ndarray)

        self.assertTrue(A_model.shape == (3, 3))
        self.assertFalse(A_model.is_constant())
        self.assertTrue(ca.simplify(A - A_model).is_zero())
        self.assertTrue(B_model.shape == (3, 1))
        self.assertTrue(B_model.is_constant())
        self.assertTrue(ca.simplify(B - B_model).is_zero())

        self.scenario = 5

    def test_initialization_6(self):
        """
        Raise dimension mismatch error for wrong number of differential and algebraic states. Differential and
        algebraic states are set directly via Model methods, but entries in mass matrix M don't match. The error is
        thrown for differential states.

        :return:
        """
        self.model.set_dynamical_states('x')
        self.model.set_algebraic_states('z', 2)
        p = self.model.set_parameters('p', 6)
        A = np.array([[-(p[0] + p[2] + p[4]), p[3], p[5]],
                      [p[2], -(p[1] + p[3]), 0.],
                      [p[4], 0., -p[5]]])
        B = np.array([1., 0., 0.])
        m = np.array([1., 1., 0.])

        self.model.A = A
        self.model.B = B  # one-dimensional array will be handled by CasADi as a column vector
        self.model.M = m

        A = self.model.A
        B = self.model.B
        M = self.model.M

        self.assertIsInstance(A, ca.SX)
        self.assertIsInstance(B, ca.SX)
        self.assertIsInstance(M, np.ndarray)

        with self.assertRaises(ValueError) as context:
            self.model.setup(dt=1.)
        self.assertTrue("Dimension mismatch. Supplied mass matrix (M) has 2 non-zero elements on its diagonal "
                        "(i.e. dynamical states), but the number of set dynamical states is 1." ==
                        str(context.exception))

    def test_initialization_7(self):
        """
        Same as TestDAESystem.test_initialization_6(), but error is thrown for algebraic states.

        :return:
        """
        self.model.set_dynamical_states('x', 2)
        self.model.set_algebraic_states('z', 2)
        p = self.model.set_parameters('p', 6)
        A = np.array([[-(p[0] + p[2] + p[4]), p[3], p[5]],
                      [p[2], -(p[1] + p[3]), 0.],
                      [p[4], 0., -p[5]]])
        B = np.array([1., 0., 0.])
        m = np.array([1., 1., 0.])

        self.model.A = A
        self.model.B = B  # one-dimensional array will be handled by CasADi as a column vector
        self.model.M = m

        A = self.model.A
        B = self.model.B
        M = self.model.M

        self.assertIsInstance(A, ca.SX)
        self.assertIsInstance(B, ca.SX)
        self.assertIsInstance(M, np.ndarray)

        with self.assertRaises(ValueError) as context:
            self.model.setup(dt=1.)
        self.assertTrue("Dimension mismatch. Supplied mass matrix (M) has 1 zero elements on its diagonal "
                        "(i.e. algebraic states), but the number of set algebraic states is 2." ==
                        str(context.exception))

    def test_initialization_8(self):
        """
        Raise dimension mismatch error for wrong dimensions in supplied state matrix A.

        :return:
        """
        p = self.model.set_parameters('p', 6)
        A = np.array([[-(p[0] + p[2] + p[4]), p[3]],
                      [p[2], -(p[1] + p[3])]])
        B = np.array([1., 0., 0.])
        m = np.array([1., 1., 0.])

        self.model.A = A
        self.model.B = B  # one-dimensional array will be handled by CasADi as a column vector
        self.model.M = m

        A = self.model.A
        B = self.model.B
        M = self.model.M

        self.assertIsInstance(A, ca.SX)
        self.assertIsInstance(B, ca.SX)
        self.assertIsInstance(M, np.ndarray)

        with self.assertRaises(ValueError) as context:
            self.model.setup(dt=1.)
        self.assertTrue("Dimension mismatch in state matrix (A). Supplied dimension is 2x2, but required dimension is "
                        "3x3." == str(context.exception))

    def test_initialization_9(self):
        """
        Raise error for missing differential equations. Only algebraic equations were supplied as part of the DAE.

        :return:
        """
        p = self.model.set_parameters('p', 6)
        A = np.array([[-(p[0] + p[2] + p[4]), p[3], p[5]],
                      [p[2], -(p[1] + p[3]), 0.],
                      [p[4], 0., -p[5]]])
        B = np.array([1., 0., 0.])
        m = np.array([0., 0., 0.])

        self.model.A = A
        self.model.B = B  # one-dimensional array will be handled by CasADi as a column vector
        self.model.M = m

        A = self.model.A
        B = self.model.B
        M = self.model.M

        self.assertIsInstance(A, ca.SX)
        self.assertIsInstance(B, ca.SX)
        self.assertIsInstance(M, np.ndarray)

        with self.assertRaises(RuntimeError) as context:
            self.model.setup(dt=1.)
        self.assertTrue("Only algebraic equations were supplied for the DAE system. ODE's are still missing." == str(
            context.exception))
