import unittest

import casadi as ca
import numpy as np

from hilo_mpc.util.probability import (
    Prior, GaussianPrior, LaplacePrior, StudentsTPrior, DeltaPrior,
    Gaussian, Laplace
)


class TestGaussianDistribution(unittest.TestCase):
    """Tests for the Gaussian distribution (likelihood model)"""

    def test_gaussian_can_be_instantiated(self):
        """Gaussian distribution instantiates without error"""
        dist = Gaussian()
        self.assertIsNotNone(dist)

    def test_gaussian_log_pdf_at_mean_is_maximum(self):
        """Log PDF of Gaussian is highest at the mean"""
        dist = Gaussian()
        mu = 0.0
        var = 1.0
        log_at_mean = float(dist(mu, mu, var, log=True))
        log_off_mean = float(dist(mu + 1.0, mu, var, log=True))
        self.assertGreater(log_at_mean, log_off_mean)

    def test_gaussian_log_pdf_numeric(self):
        """Gaussian log PDF returns expected value at y=0, mu=0, var=1"""
        dist = Gaussian()
        result = float(dist(0., 0., 1., log=True))
        expected = -0.5 * np.log(2. * np.pi)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_gaussian_log_pdf_returns_casadi_dm(self):
        """Gaussian log PDF returns a CasADi DM (numeric) value"""
        dist = Gaussian()
        result = dist(1.0, 0.0, 1.0, log=True)
        self.assertIsInstance(result, ca.DM)


class TestLaplaceDistribution(unittest.TestCase):
    """Tests for the Laplace distribution"""

    def test_laplace_can_be_instantiated(self):
        """Laplace distribution instantiates without error"""
        dist = Laplace()
        self.assertIsNotNone(dist)

    def test_laplace_log_pdf_at_mean_is_maximum(self):
        """Log PDF of Laplace is highest at the mean"""
        dist = Laplace()
        mu = 0.0
        var = 1.0
        log_at_mean = float(dist(mu, mu, var, log=True))
        log_off_mean = float(dist(mu + 1.0, mu, var, log=True))
        self.assertGreater(log_at_mean, log_off_mean)


class TestPriorFactory(unittest.TestCase):
    """Tests for the Prior factory methods"""

    def test_prior_gaussian_factory(self):
        """Prior.gaussian() creates a GaussianPrior"""
        prior = Prior.gaussian(mean=0., variance=1.)
        self.assertIsInstance(prior, GaussianPrior)

    def test_prior_laplace_factory(self):
        """Prior.laplace() creates a LaplacePrior"""
        prior = Prior.laplace(mean=0., variance=1.)
        self.assertIsInstance(prior, LaplacePrior)

    def test_prior_delta_factory(self):
        """Prior.delta() creates a DeltaPrior"""
        prior = Prior.delta()
        self.assertIsInstance(prior, DeltaPrior)

    def test_prior_students_t_factory_raises_bug(self):
        """Prior.students_t() raises TypeError due to known scipy/CasADi incompatibility"""
        with self.assertRaises(TypeError):
            Prior.students_t(mean=0., variance=1., nu=3.)


class TestGaussianPrior(unittest.TestCase):
    """Tests for GaussianPrior"""

    def test_gaussian_prior_creation(self):
        """GaussianPrior can be created with mean and variance"""
        prior = GaussianPrior(mean=0., variance=1.)
        self.assertIsNotNone(prior)

    def test_gaussian_prior_name(self):
        """GaussianPrior has correct name"""
        prior = GaussianPrior(mean=0., variance=1.)
        self.assertEqual(prior.name, 'Gaussian')

    def test_gaussian_prior_stores_mean(self):
        """GaussianPrior stores mean correctly"""
        prior = GaussianPrior(mean=2.5, variance=1.)
        self.assertAlmostEqual(prior.mean, 2.5)

    def test_gaussian_prior_stores_variance(self):
        """GaussianPrior stores variance correctly"""
        prior = GaussianPrior(mean=0., variance=3.0)
        self.assertAlmostEqual(prior.variance, 3.0)

    def test_gaussian_prior_log_density(self):
        """GaussianPrior log density at mean is higher than off-mean"""
        prior = GaussianPrior(mean=1., variance=2.)
        log_at_mean = float(prior(1., log=True))
        log_off_mean = float(prior(5., log=True))
        self.assertGreater(log_at_mean, log_off_mean)

    def test_gaussian_prior_wrong_mean_dim_raises(self):
        """GaussianPrior raises ValueError for multi-dimensional mean"""
        with self.assertRaises(ValueError):
            GaussianPrior(mean=[1., 2.], variance=1.)

    def test_gaussian_prior_wrong_variance_dim_raises(self):
        """GaussianPrior raises ValueError for multi-dimensional variance"""
        with self.assertRaises(ValueError):
            GaussianPrior(mean=0., variance=[1., 2.])

    def test_gaussian_prior_mean_setter(self):
        """GaussianPrior mean can be updated via setter"""
        prior = GaussianPrior(mean=0., variance=1.)
        prior.mean = 5.
        self.assertAlmostEqual(prior.mean, 5.)

    def test_gaussian_prior_variance_setter(self):
        """GaussianPrior variance can be updated via setter"""
        prior = GaussianPrior(mean=0., variance=1.)
        prior.variance = 4.
        self.assertAlmostEqual(prior.variance, 4.)


class TestLaplacePrior(unittest.TestCase):
    """Tests for LaplacePrior"""

    def test_laplace_prior_creation(self):
        """LaplacePrior can be created with mean and variance"""
        prior = LaplacePrior(mean=0., variance=1.)
        self.assertIsNotNone(prior)

    def test_laplace_prior_name(self):
        """LaplacePrior has correct name"""
        prior = LaplacePrior(mean=0., variance=1.)
        self.assertEqual(prior.name, 'Laplace')

    def test_laplace_prior_log_density_at_mean(self):
        """LaplacePrior log density at mean is higher than off-mean"""
        prior = LaplacePrior(mean=0., variance=2.)
        log_at_mean = float(prior(0., log=True))
        log_off_mean = float(prior(3., log=True))
        self.assertGreater(log_at_mean, log_off_mean)


class TestDeltaPrior(unittest.TestCase):
    """Tests for DeltaPrior"""

    def test_delta_prior_creation(self):
        """DeltaPrior can be created without arguments"""
        prior = DeltaPrior()
        self.assertIsNotNone(prior)

    def test_delta_prior_name(self):
        """DeltaPrior has correct name"""
        prior = DeltaPrior()
        self.assertEqual(prior.name, 'Delta')


class TestStudentsTPrior(unittest.TestCase):
    """Tests for StudentsTPrior

    NOTE: StudentsTPrior._initialize() calls scipy.special.gammaln on a
    CasADi symbolic variable, which raises a TypeError. This is a known bug
    in the library. Tests below document this behaviour.
    """

    def test_students_t_prior_creation_raises_bug(self):
        """StudentsTPrior creation raises TypeError due to scipy/CasADi incompatibility (known bug)"""
        with self.assertRaises(TypeError):
            StudentsTPrior(mean=0., variance=1., nu=3.)

    def test_prior_students_t_factory_raises_bug(self):
        """Prior.students_t() raises TypeError due to the same known bug"""
        with self.assertRaises(TypeError):
            Prior.students_t(mean=0., variance=1., nu=3.)


if __name__ == '__main__':
    unittest.main()
