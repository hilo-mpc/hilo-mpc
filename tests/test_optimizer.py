import unittest
import warnings

import casadi as ca
import numpy as np

from hilo_mpc import NLP, LP, QP

# Silent IPOPT output options
_IPOPT_SILENT = {'ipopt.print_level': 0, 'print_time': 0}


class TestNonlinearProgram(unittest.TestCase):
    """Tests for NonlinearProgram (NLP)"""

    def test_nlp_initialization(self):
        """NLP can be instantiated"""
        nlp = NLP()
        self.assertIsNotNone(nlp)

    def test_nlp_default_solver_is_ipopt(self):
        """NLP default solver is ipopt"""
        nlp = NLP()
        self.assertEqual(nlp.solver, 'ipopt')

    def test_nlp_set_decision_variables_by_name(self):
        """NLP accepts decision variable names"""
        nlp = NLP()
        nlp.set_decision_variables('x', 'y')
        self.assertEqual(nlp.n_x, 2)

    def test_nlp_set_single_decision_variable(self):
        """NLP accepts a single decision variable"""
        nlp = NLP()
        nlp.set_decision_variables('x')
        self.assertEqual(nlp.n_x, 1)

    def test_nlp_set_decision_variable_count(self):
        """NLP accepts integer count for decision variables"""
        nlp = NLP()
        nlp.set_decision_variables(3)
        self.assertEqual(nlp.n_x, 3)

    def test_nlp_set_objective(self):
        """NLP objective can be set and is non-empty"""
        nlp = NLP()
        x = nlp.set_decision_variables('x', 'y')
        nlp.objective = x[0] ** 2 + x[1] ** 2
        self.assertFalse(nlp.objective.is_empty())

    def test_nlp_objective_wrong_type_raises(self):
        """NLP raises TypeError when objective is an integer literal"""
        nlp = NLP()
        with self.assertRaises(TypeError):
            nlp.set_objective(5)

    def test_nlp_variable_bounds(self):
        """NLP stores variable bounds correctly"""
        nlp = NLP()
        nlp.set_decision_variables('x', 'y', lower_bound=-10., upper_bound=10.)
        np.testing.assert_allclose(float(nlp.lower_bound[0]), -10.)
        np.testing.assert_allclose(float(nlp.upper_bound[0]), 10.)

    def test_nlp_solve_simple_quadratic(self):
        """NLP minimizes x^2 + y^2 — optimal solution is (0, 0)"""
        nlp = NLP(solver_options=_IPOPT_SILENT)
        x = nlp.set_decision_variables('x', 'y')
        nlp.objective = x[0] ** 2 + x[1] ** 2
        nlp.setup()
        nlp.set_initial_guess(x0=[1., 1.])
        nlp.solve()
        # Solution is stored as (n_x, n_iterations) matrix; last column is optimal
        sol = nlp.solution.get_by_id('x')[:, -1]
        np.testing.assert_allclose(float(sol[0]), 0., atol=1e-4)
        np.testing.assert_allclose(float(sol[1]), 0., atol=1e-4)

    def test_nlp_solve_with_parameter(self):
        """NLP minimizes (x - p)^2 — optimal solution equals the parameter value"""
        nlp = NLP(solver_options=_IPOPT_SILENT)
        x = nlp.set_decision_variables('x')
        p = nlp.set_parameters('p')
        nlp.objective = (x[0] - p[0]) ** 2
        nlp.setup()
        nlp.set_initial_guess(x0=[0.])
        nlp.solve(p=[3.])
        sol = nlp.solution.get_by_id('x')[:, -1]
        np.testing.assert_allclose(float(sol[0]), 3., atol=1e-4)

    def test_nlp_solve_with_lower_bound(self):
        """NLP minimizes x subject to x >= 2 — optimal solution is 2"""
        nlp = NLP(solver_options=_IPOPT_SILENT)
        nlp.set_decision_variables('x', lower_bound=2.)
        nlp.objective = nlp.decision_variables[0]
        nlp.setup()
        nlp.set_initial_guess(x0=[5.])
        nlp.solve()
        sol = nlp.solution.get_by_id('x')[:, -1]
        np.testing.assert_allclose(float(sol[0]), 2., atol=1e-4)

    def test_nlp_n_p_with_parameters(self):
        """NLP correctly reports number of parameters"""
        nlp = NLP()
        nlp.set_parameters('a', 'b', 'c')
        self.assertEqual(nlp.n_p, 3)

    def test_nlp_n_p_no_parameters(self):
        """NLP reports 0 parameters when none set"""
        nlp = NLP()
        self.assertEqual(nlp.n_p, 0)

    def test_nlp_not_setup_raises_on_solve(self):
        """NLP raises RuntimeError when solved without calling setup() first"""
        nlp = NLP()
        x = nlp.set_decision_variables('x')
        nlp.objective = x[0] ** 2
        with self.assertRaises(RuntimeError):
            nlp.solve(p=[])

    def test_nlp_solver_type_error(self):
        """NLP raises TypeError when solver is assigned a non-string"""
        nlp = NLP()
        with self.assertRaises(TypeError):
            nlp.solver = 42

    def test_nlp_warm_start(self):
        """NLP warm-start re-uses previous solution without error"""
        nlp = NLP(solver_options=_IPOPT_SILENT)
        x = nlp.set_decision_variables('x', 'y')
        nlp.objective = x[0] ** 2 + x[1] ** 2
        nlp.setup()
        nlp.set_initial_guess(x0=[1., 1.])
        nlp.solve(p=[])
        # Second solve with warm start
        nlp.solve(warm_start=True, p=[])
        sol = nlp.solution.get_by_id('x')[:, -1]
        np.testing.assert_allclose(float(sol[0]), 0., atol=1e-4)


class TestLinearProgram(unittest.TestCase):
    """Tests for LinearProgram (LP)"""

    def test_lp_initialization(self):
        """LP can be instantiated"""
        lp = LP()
        self.assertIsNotNone(lp)

    def test_lp_default_solver_is_clp(self):
        """LP default solver is clp"""
        lp = LP()
        self.assertEqual(lp.solver, 'clp')

    def test_lp_set_decision_variables(self):
        """LP accepts decision variable names"""
        lp = LP()
        lp.set_decision_variables('x', 'y')
        self.assertEqual(lp.n_x, 2)

    def test_lp_set_objective(self):
        """LP objective can be set"""
        lp = LP()
        x = lp.set_decision_variables('x', 'y')
        lp.objective = x[0] + 2 * x[1]
        self.assertFalse(lp.objective.is_empty())

    def test_lp_solve_simple(self):
        """LP minimizes x + y subject to x, y >= 1"""
        lp = LP()
        x = lp.set_decision_variables('x', 'y', lower_bound=1.)
        lp.objective = x[0] + x[1]
        lp.setup()
        lp.solve()
        # Solution is stored as (n_x, n_iterations) matrix; last column is optimal
        sol = lp.solution.get_by_id('x')[:, -1]
        np.testing.assert_allclose(float(sol[0]), 1., atol=1e-4)
        np.testing.assert_allclose(float(sol[1]), 1., atol=1e-4)

    def test_lp_nonlinear_objective_raises(self):
        """LP raises RuntimeError when objective is nonlinear (x^2)"""
        lp = LP()
        x = lp.set_decision_variables('x')
        lp.objective = x[0] ** 2  # nonlinear
        with self.assertRaises(RuntimeError):
            lp.setup()

    def test_lp_variable_bounds(self):
        """LP stores variable bounds correctly"""
        lp = LP()
        lp.set_decision_variables('x', 'y', lower_bound=-5., upper_bound=5.)
        np.testing.assert_allclose(float(lp.lower_bound[0]), -5.)
        np.testing.assert_allclose(float(lp.upper_bound[0]), 5.)

    def test_lp_solve_minimization(self):
        """LP finds minimum cost direction (minimize -x subject to -1 <= x <= 1)"""
        lp = LP()
        x = lp.set_decision_variables('x', lower_bound=-1., upper_bound=1.)
        lp.objective = -x[0]   # minimize -x → maximize x
        lp.setup()
        lp.solve(p=[])
        sol = lp.solution.get_by_id('x')[:, -1]
        np.testing.assert_allclose(float(sol[0]), 1., atol=1e-4)


class TestQuadraticProgram(unittest.TestCase):
    """Tests for QuadraticProgram (QP)

    Note: QP in HILO-MPC inherits from LinearProgram and uses the same
    linearity check. Quadratic objectives (x^2) cause QP.setup() to raise a
    RuntimeError. This is a known limitation; the tests below verify the
    current behaviour.
    """

    def test_qp_initialization(self):
        """QP can be instantiated"""
        qp = QP()
        self.assertIsNotNone(qp)

    def test_qp_set_decision_variables(self):
        """QP accepts decision variable names"""
        qp = QP()
        qp.set_decision_variables('x', 'y')
        self.assertEqual(qp.n_x, 2)

    def test_qp_linear_objective_setup_succeeds(self):
        """QP with a linear objective sets up without error (same as LP)"""
        qp = QP()
        x = qp.set_decision_variables('x', 'y', lower_bound=1.)
        qp.objective = x[0] + x[1]   # linear
        qp.setup()

    def test_qp_quadratic_objective_raises(self):
        """QP raises RuntimeError for a quadratic objective (known limitation)"""
        qp = QP()
        x = qp.set_decision_variables('x', 'y', lower_bound=1.)
        qp.objective = 0.5 * (x[0] ** 2 + x[1] ** 2)
        with self.assertRaises(RuntimeError):
            qp.setup()

    def test_qp_solve_linear(self):
        """QP solves a linear minimization problem correctly"""
        qp = QP()
        x = qp.set_decision_variables('x', 'y', lower_bound=1.)
        qp.objective = x[0] + x[1]
        qp.setup()
        qp.solve(p=[])
        sol = qp.solution.get_by_id('x')[:, -1]
        np.testing.assert_allclose(float(sol[0]), 1., atol=1e-4)
        np.testing.assert_allclose(float(sol[1]), 1., atol=1e-4)


if __name__ == '__main__':
    unittest.main()
