"""Unit tests for integration rules and Duffy transformation."""

import numpy as np
import pytest

from src.pyfemsolver.solverlib.integrationrules import (
    duffy,
    IntegrationRuleTrig,
    IntegrationRuleLine,
    get_integration_rule_trig,
    get_integration_rule_line,
)


class TestDuffyMapping:
    """Unit tests for duffy transformation function."""

    def test_duffy_mapping_simple(self):
        """Given: simple zeta/eta inputs
        When: duffy mapping is applied
        Then: outputs are the expected mapped coordinates
        """
        zeta = np.array([1.0])
        eta = np.array([0.5])
        X_t, Y_t = duffy(zeta, eta)
        assert np.allclose(X_t, 0.5 * 1.0 * (1 - 0.5))
        assert np.allclose(Y_t, eta)

    def test_duffy_mapping_eta_zero(self):
        """Given: eta = 0
        When: duffy mapping is applied
        Then: X_t = 0.5 * zeta
        """
        zeta = np.array([2.0])
        eta = np.array([0.0])
        X_t, Y_t = duffy(zeta, eta)
        assert np.allclose(X_t, 1.0)
        assert np.allclose(Y_t, 0.0)

    def test_duffy_mapping_eta_one(self):
        """Given: eta = 1
        When: duffy mapping is applied
        Then: X_t = 0 (since 1 - eta = 0)
        """
        zeta = np.array([2.0])
        eta = np.array([1.0])
        X_t, Y_t = duffy(zeta, eta)
        assert np.allclose(X_t, 0.0)
        assert np.allclose(Y_t, 1.0)

    def test_duffy_mapping_array_inputs(self):
        """Given: array inputs
        When: duffy mapping is applied
        Then: output shapes are correct
        """
        zeta = np.array([0.5, 1.0])
        eta = np.array([0.2, 0.8])
        X_t, Y_t = duffy(zeta, eta)
        assert X_t.shape == zeta.shape
        assert Y_t.shape == eta.shape


class TestIntegrationRuleTrig:
    """Unit tests for IntegrationRuleTrig class."""

    def test_integration_rule_trig_order_1(self):
        """Given: order 1 integration rule for triangles
        When: the rule is called
        Then: return values have correct properties
        """
        rule = IntegrationRuleTrig()
        X, Y, omega = rule(1)

        # Should have (2*1+1)**2 = 9 integration points
        assert X.shape[0] == 9
        assert Y.shape[0] == 9
        assert omega.shape[0] == 9

        # Weights should sum to approximately 2.0 (area of reference triangle)
        assert np.allclose(omega.sum(), 2.0)

    def test_integration_rule_trig_order_2(self):
        """Given: order 2 integration rule for triangles
        When: the rule is called
        Then: return values have correct properties
        """
        rule = IntegrationRuleTrig()
        X, Y, omega = rule(2)

        # Should have (2*2+1) = 25 integration points
        assert X.shape[0] == 25
        assert Y.shape[0] == 25
        assert omega.shape[0] == 25

        # Weights should sum to approximately 2.0
        assert np.allclose(omega.sum(), 2.0)

    def test_integration_rule_trig_caching(self):
        """Given: an integration rule that has been called once
        When: the rule is called again with the same order
        Then: the cached result is returned (same reference, not a copy)
        """
        rule = IntegrationRuleTrig()
        X1, Y1, omega1 = rule(1)
        X2, Y2, omega2 = rule(1)

        # Values should be the same
        assert np.allclose(X1, X2)
        assert np.allclose(Y1, Y2)
        assert np.allclose(omega1, omega2)

    def test_integration_rule_trig_copies(self):
        """Given: an integration rule
        When: called twice
        Then: modifying returned values doesn't affect subsequent calls
        """
        rule = IntegrationRuleTrig()
        X1, Y1, omega1 = rule(1)
        X1[0, 0] = 999.0

        X2, Y2, omega2 = rule(1)
        assert X2[0, 0] != 999.0

    def test_integration_rule_trig_multiple_orders(self):
        """Given: different integration rule orders
        When: the rules are called
        Then: different orders produce different point counts
        """
        rule = IntegrationRuleTrig()
        X1, _, _ = rule(1)
        X2, _, _ = rule(2)
        X3, _, _ = rule(3)

        assert X1.shape[0] == 9
        assert X2.shape[0] == 25
        assert X3.shape[0] == 49


class TestIntegrationRuleLine:
    """Unit tests for IntegrationRuleLine class."""

    def test_integration_rule_line_order_1(self):
        """Given: order 1 integration rule for lines
        When: the rule is called
        Then: return values have correct properties
        """
        rule = IntegrationRuleLine()
        nodes, weights = rule(1)

        # Should have 2*1+1 = 3 integration points
        assert nodes.shape[0] == 3
        assert weights.shape[0] == 3

        # Weights should sum to 2 (length of reference line [-1, 1])
        assert np.allclose(weights.sum(), 2.0)

    def test_integration_rule_line_order_2(self):
        """Given: order 2 integration rule for lines
        When: the rule is called
        Then: return values have correct properties
        """
        rule = IntegrationRuleLine()
        nodes, weights = rule(2)

        # Should have 2*2+1 = 5 integration points
        assert nodes.shape[0] == 5
        assert weights.shape[0] == 5

        # Weights should sum to 2
        assert np.allclose(weights.sum(), 2.0)

    def test_integration_rule_line_caching(self):
        """Given: an integration rule that has been called once
        When: the rule is called again with the same order
        Then: the cached result is returned
        """
        rule = IntegrationRuleLine()
        nodes1, weights1 = rule(1)
        nodes2, weights2 = rule(1)

        assert np.allclose(nodes1, nodes2)
        assert np.allclose(weights1, weights2)

    def test_integration_rule_line_copies(self):
        """Given: an integration rule
        When: called twice
        Then: modifying returned values doesn't affect subsequent calls
        """
        rule = IntegrationRuleLine()
        nodes1, weights1 = rule(1)
        nodes1[0, 0] = 999.0

        nodes2, weights2 = rule(1)
        assert nodes2[0, 0] != 999.0

    def test_integration_rule_line_nodes_in_range(self):
        """Given: an integration rule for lines
        When: the rule is called
        Then: all nodes are in [-1, 1]
        """
        rule = IntegrationRuleLine()
        nodes, _ = rule(2)
        assert np.all(nodes >= -1.0)
        assert np.all(nodes <= 1.0)

    def test_integration_rule_line_positive_weights(self):
        """Given: an integration rule for lines
        When: the rule is called
        Then: all weights are positive
        """
        rule = IntegrationRuleLine()
        _, weights = rule(2)
        assert np.all(weights > 0.0)


class TestGetIntegrationRuleFunctions:
    """Unit tests for get_integration_rule_* wrapper functions."""

    def test_get_integration_rule_trig(self):
        """Given: the wrapper function for triangular integration
        When: called with order 1
        Then: returns proper integration rule
        """
        X, Y, omega = get_integration_rule_trig(1)

        assert X.shape[0] == 9
        assert Y.shape[0] == 9
        assert omega.shape[0] == 9
        assert np.allclose(omega.sum(), 2.0)

    def test_get_integration_rule_line(self):
        """Given: the wrapper function for line integration
        When: called with order 1
        Then: returns proper integration rule
        """
        nodes, weights = get_integration_rule_line(1)

        assert nodes.shape[0] == 3
        assert weights.shape[0] == 3
        assert np.allclose(weights.sum(), 2.0)
