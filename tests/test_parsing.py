import platform
from unittest import TestCase, skip

import casadi as ca

from hilo_mpc.util.parsing import Parser, parse_dynamic_equations


class TestParser(TestCase):
    """"""
    @skip('skip')
    def test_parser(self) -> None:
        """

        :return:
        """
        exp = 'x+2'

        subs = {
            'x': ca.SX.sym('x'),
            '2': 2
        }
        res = Parser.eval_expression(exp, subs=subs, free=[])
        print(res)

    def test_double_replace(self) -> None:
        """

        :return:
        """
        self.equations = """
        dx/dt = u(k)
        dy/dt = ar + ar
        
        ar = sin(2*x(t))
        """
        self.discrete = False

    def test_continuous_dae_no_alg_no_meas(self) -> None:
        """

        :return:
        """
        self.equations = """
        # DAE's
        dx(t)/dt = v(t)
        dv(t)/dt = (3/4*m*g*sin(theta(t))*cos(theta(t)) - 1/2*m*l*sin(theta(t))*omega(t)^2 + F(k))/...
        (M + m - 3/4*m*cos(theta(t))^2)
        d/dt(theta(t)) = omega(t)
        d/dt(omega(t)) = 3/(2*l)*(dv(t)/dt*cos(theta(t)) + g*sin(theta(t)))
        y(t) = h + l*cos(theta(t))
        
        # Constants
        g = 9.81
        """
        self.discrete = False

    def test_continuous_dae_no_alg_no_meas_no_input(self) -> None:
        """

        :return:
        """
        self.equations = """
        # DAE's
        d/dt(x(t)) = u(t)
        d/dt(u(t)) = l(t) * x(t)
        0 = x(t)^2 + y(t)^2 - L^2
        0 = u(t) * x(t) + v(t) * y(t)
        0 = u(t)^2 - g * y(t) + v(t)^2 + L^2 * l(t)
        
        # Constants
        g = 9.81
        """
        self.discrete = False

    def test_continuous_ode_no_const_no_input(self) -> None:
        """

        :return:
        """
        self.equations = """
        # ODE's
        d/dt(x_1(t)) = x_2(t)
        d/dt(x_2(t)) = -p*x_1(t)

        # Algebraic equations
        p = r - 2*q*cos(omega*t)

        # Measurement equations
        y(k) = x_1(t) + x_2(t)*sin(omega*t)
        """
        self.discrete = False

    def test_discrete_ode_no_const_no_input(self) -> None:
        """

        :return:
        """
        self.equations = """
        # ODE's
        x_1(k + 1) = x_1(k) + x_2(k)*dt
        x_2(k + 1) = x_2(k) - p*x_1(k)*dt
        
        # Algebraic equations
        p = r - 2*q*cos(omega*t(k))
        
        # Measurement equations
        y(k) = x_1(k) + x_2(k)*sin(omega*t(k))
        """
        self.discrete = True

    def tearDown(self) -> None:
        """

        :return:
        """
        if platform.system() == 'Linux':
            equations = self.equations.split('\n')
        elif platform.system() == 'Windows':
            equations = self.equations.split('\n')
        else:
            raise NotImplementedError(f"Parsing from string not supported for operating system {platform.system()}")
        t = ca.SX.sym('t')
        parse_dynamic_equations(equations, discrete=self.discrete, t=t.elements())
