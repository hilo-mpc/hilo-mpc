"""
Tests for add_dynamical_states and add_algebraic_states return values
"""
from unittest import TestCase

import casadi as ca

from hilo_mpc import Model


class TestAddStatesReturn(TestCase):
    """Test that add_dynamical_states and add_algebraic_states return the added states"""

    def setUp(self) -> None:
        """Initialize empty model"""
        self.model = Model()

    def test_add_dynamical_states_returns_added_states(self) -> None:
        """Test that add_dynamical_states returns the symbolic variables of added states"""
        # Set initial states
        self.model.set_dynamical_states(['x1', 'x2'])
        
        # Add new states
        added_states = self.model.add_dynamical_states(['x3', 'x4'])
        
        # Check that the return value is not None
        self.assertIsNotNone(added_states)
        
        # Check that it's a CasADi symbolic variable
        self.assertIsInstance(added_states, (ca.SX, ca.MX))
        
        # Check that the size is correct (2 states added)
        self.assertEqual(added_states.size1(), 2)
        
        # Check that the names are correct
        names = [added_states[i].name() for i in range(added_states.size1())]
        self.assertEqual(names, ['x3', 'x4'])

    def test_add_dynamical_states_list_comprehension(self) -> None:
        """Test that add_dynamical_states works with list comprehension as in the issue"""
        # Set initial states
        self.model.set_dynamical_states(['x1'])
        
        # Add states using list comprehension (like in the issue)
        N = 3
        added_states = self.model.add_dynamical_states([f"A_{j}" for j in range(N+2)])
        
        # Check that the return value is not None
        self.assertIsNotNone(added_states)
        
        # Check that it's a CasADi symbolic variable
        self.assertIsInstance(added_states, (ca.SX, ca.MX))
        
        # Check that the size is correct (N+2 = 5 states added)
        self.assertEqual(added_states.size1(), 5)
        
        # Check that the names are correct
        names = [added_states[i].name() for i in range(added_states.size1())]
        self.assertEqual(names, ['A_0', 'A_1', 'A_2', 'A_3', 'A_4'])
        
        # Verify they can be used in model construction
        # This demonstrates the use case from the issue
        self.assertEqual(added_states[0].name(), 'A_0')
        self.assertEqual(added_states[-1].name(), 'A_4')

    def test_add_algebraic_states_returns_added_states(self) -> None:
        """Test that add_algebraic_states returns the symbolic variables of added states"""
        # Set initial algebraic states
        self.model.set_algebraic_states(['z1'])
        
        # Add new algebraic states
        added_states = self.model.add_algebraic_states(['z2', 'z3'])
        
        # Check that the return value is not None
        self.assertIsNotNone(added_states)
        
        # Check that it's a CasADi symbolic variable
        self.assertIsInstance(added_states, (ca.SX, ca.MX))
        
        # Check that the size is correct (2 states added)
        self.assertEqual(added_states.size1(), 2)
        
        # Check that the names are correct
        names = [added_states[i].name() for i in range(added_states.size1())]
        self.assertEqual(names, ['z2', 'z3'])

    def test_add_single_dynamical_state(self) -> None:
        """Test that add_dynamical_states returns correctly for a single state"""
        # Set initial states
        self.model.set_dynamical_states(['x1'])
        
        # Add a single new state
        added_state = self.model.add_dynamical_states(['x2'])
        
        # Check that the return value is not None
        self.assertIsNotNone(added_state)
        
        # Check that it's a CasADi symbolic variable
        self.assertIsInstance(added_state, (ca.SX, ca.MX))
        
        # Check that the size is correct (1 state added)
        self.assertEqual(added_state.size1(), 1)
        
        # Check that the name is correct
        self.assertEqual(added_state.name(), 'x2')

    def test_add_states_to_empty_model(self) -> None:
        """Test that add_dynamical_states works when adding to empty model"""
        # Add states to empty model
        added_states = self.model.add_dynamical_states(['x1', 'x2', 'x3'])
        
        # Check that the return value is not None
        self.assertIsNotNone(added_states)
        
        # Check that it's a CasADi symbolic variable
        self.assertIsInstance(added_states, (ca.SX, ca.MX))
        
        # Check that the size is correct (3 states added)
        self.assertEqual(added_states.size1(), 3)
        
        # Check that all model states match the added states
        self.assertEqual(self.model.n_x, 3)

    def test_added_states_can_be_used_in_expressions(self) -> None:
        """Test that returned states can be used in symbolic expressions"""
        # Set initial states
        self.model.set_dynamical_states(['x1'])
        
        # Add new states
        added_states = self.model.add_dynamical_states(['x2', 'x3'])
        
        # Use the added states in an expression
        expr = added_states[0] + added_states[1]
        
        # Verify the expression is valid
        self.assertIsInstance(expr, (ca.SX, ca.MX))
        
        # Check that the expression has the correct size
        self.assertEqual(expr.size1(), 1)
        self.assertEqual(expr.size2(), 1)
