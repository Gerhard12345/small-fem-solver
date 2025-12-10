"""
Integration rules for triangular and line elements with caching.

This module provides classes to compute and cache Gauss-Legendre integration rules
for triangular and line elements using Duffy transformation for triangles.
"""

from typing import Tuple, Dict

import numpy as np
from numpy.typing import NDArray


def duffy(zeta: NDArray[np.floating], eta: NDArray[np.floating]) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Apply Duffy transformation to map Gauss-Legendre quadrature to reference triangle.

    :param zeta: First coordinate in reference domain
    :type zeta: NDArray[np.floating]
    :param eta: Second coordinate in reference domain
    :type eta: NDArray[np.floating]
    :return: Transformed coordinates (X_t, Y_t)
    :rtype: Tuple[NDArray[np.floating], NDArray[np.floating]]
    """
    return 0.5 * zeta * (1 - eta), eta


class IntegrationRuleTrig:
    """Compute and cache Gauss-Legendre integration rules for triangular elements.

    Integration rules are computed on demand and cached for reuse in subsequent calls.
    """

    def __init__(self):
        """Initialize the integration rule cache for triangles."""
        self._cache: Dict[int, Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]] = {}

    def __call__(self, p: int) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Get integration rule for triangles of order p.

        Computes the rule on first call and returns a copy of the cached result on subsequent calls.
        Returns copies to prevent external modifications of cached values.

        :param p: Order of the integration rule (number of Gauss points = 2p+1)
        :type p: int
        :return: Tuple of (X coordinates, Y coordinates, weights)
        :rtype: Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]
        """
        if p not in self._cache:
            self._cache[p] = self._compute_integration_rule_trig(p)
        # Return copies to prevent external modifications of cached values
        X, Y, omega = self._cache[p]
        return X.copy(), Y.copy(), omega.copy()

    @staticmethod
    def _compute_integration_rule_trig(p: int) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Compute integration rule for triangles using Duffy transformation.

        :param p: Order of the integration rule
        :type p: int
        :return: Tuple of (X coordinates, Y coordinates, weights)
        :rtype: Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]
        """
        nodes, weights = np.polynomial.legendre.leggauss(2 * p + 1)
        X, Y = np.meshgrid(nodes, nodes)
        X_t, Y_t = duffy(X, Y)
        X_t = X_t.reshape(X_t.size, 1)
        Y_t = Y_t.reshape(X_t.size, 1)
        omega = np.outer(weights * 1.0 / 2.0 * (1.0 - nodes), weights).flatten()
        return X_t, Y_t, omega


class IntegrationRuleLine:
    """Compute and cache Gauss-Legendre integration rules for line elements.

    Integration rules are computed on demand and cached for reuse in subsequent calls.
    """

    def __init__(self):
        """Initialize the integration rule cache for lines."""
        self._cache: Dict[int, Tuple[NDArray[np.floating], NDArray[np.floating]]] = {}

    def __call__(self, p: int) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Get integration rule for lines of order p.

        Computes the rule on first call and returns a copy of the cached result on subsequent calls.
        Returns copies to prevent external modifications of cached values.

        :param p: Order of the integration rule (number of Gauss points = 2p+1)
        :type p: int
        :return: Tuple of (nodes, weights)
        :rtype: Tuple[NDArray[np.floating], NDArray[np.floating]]
        """
        if p not in self._cache:
            self._cache[p] = self._compute_integration_rule_line(p)
        # Return copies to prevent external modifications of cached values
        nodes, weights = self._cache[p]
        return nodes.copy(), weights.copy()

    @staticmethod
    def _compute_integration_rule_line(p: int) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Compute integration rule for lines using Gauss-Legendre quadrature.

        :param p: Order of the integration rule
        :type p: int
        :return: Tuple of (nodes, weights)
        :rtype: Tuple[NDArray[np.floating], NDArray[np.floating]]
        """
        nodes, weights = np.polynomial.legendre.leggauss(2 * p + 1)
        nodes = nodes.reshape(nodes.size, 1)
        return nodes, weights


def get_integration_rule_trig(
    p: int,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Get integration rule for triangles of order p.

    :param p: Order of the integration rule (number of Gauss points = 2p+1)
    :type p: int
    :return: Tuple of (X coordinates, Y coordinates, weights)
    :rtype: Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]
    """
    return _integration_rule_trig(p)


def get_integration_rule_line(p: int) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Get integration rule for lines of order p.

    :param p: Order of the integration rule (number of Gauss points = 2p+1)
    :type p: int
    :return: Tuple of (nodes, weights)
    :rtype: Tuple[NDArray[np.floating], NDArray[np.floating]]
    """
    return _integration_rule_line(p)


# Instantiate global integration rule objects for reuse
_integration_rule_trig = IntegrationRuleTrig()
_integration_rule_line = IntegrationRuleLine()
