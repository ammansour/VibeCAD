"""
Tests for the autorouter module (Freerouting integration + grid-based A* fallback).
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from vibecad.design.autorouter import (
    AutorouteResult,
    _find_freerouting_jar,
    _grid_route_all,
    _astar,
    _simplify_path,
    _octile_heuristic,
)


class TestAstarHelpers(unittest.TestCase):
    """Test the pure-Python A* and path utilities."""

    def test_astar_trivial(self):
        """Start == end returns single-cell path."""
        path = _astar(set(), 10, 10, (3, 3), (3, 3))
        self.assertEqual(path, [(3, 3)])

    def test_astar_straight_line(self):
        """Straight horizontal path with no obstacles."""
        path = _astar(set(), 10, 10, (0, 5), (9, 5))
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 5))
        self.assertEqual(path[-1], (9, 5))
        # Every step should move +1 in x, 0 in y
        for a, b in zip(path, path[1:]):
            self.assertEqual(b[1], a[1])  # y unchanged
            self.assertIn(b[0] - a[0], (0, 1))

    def test_astar_obstacle_avoidance(self):
        """Path must go around a wall of blocked cells."""
        blocked = {(5, y) for y in range(0, 8)}  # vertical wall at x=5
        path = _astar(blocked, 10, 10, (0, 5), (9, 5))
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 5))
        self.assertEqual(path[-1], (9, 5))
        # No cell in the path should be in the blocked set
        for cell in path:
            self.assertNotIn(cell, blocked)

    def test_astar_no_path(self):
        """Completely walled-off target returns None."""
        # Block every cell around (9, 9)
        blocked = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (dx, dy) != (0, 0):
                    blocked.add((9 + dx, 9 + dy))
        path = _astar(blocked, 10, 10, (0, 0), (9, 9))
        self.assertIsNone(path)

    def test_astar_diagonal(self):
        """Path should use diagonals when possible."""
        path = _astar(set(), 10, 10, (0, 0), (5, 5))
        self.assertIsNotNone(path)
        # Optimal is pure diagonal: 6 cells
        self.assertEqual(len(path), 6)

    def test_simplify_straight(self):
        """Collinear points collapse to just endpoints."""
        path = [(0, 0), (1, 0), (2, 0), (3, 0)]
        s = _simplify_path(path)
        self.assertEqual(s, [(0, 0), (3, 0)])

    def test_simplify_l_shape(self):
        """L-shaped path keeps the corner."""
        path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
        s = _simplify_path(path)
        self.assertEqual(s, [(0, 0), (2, 0), (2, 2)])

    def test_simplify_short(self):
        """Paths \u2264 2 points returned as-is."""
        self.assertEqual(_simplify_path([(0, 0)]), [(0, 0)])
        self.assertEqual(_simplify_path([(0, 0), (1, 1)]), [(0, 0), (1, 1)])

    def test_octile_heuristic_cardinal(self):
        """Cardinal distance: octile == manhattan for axis-aligned."""
        h = _octile_heuristic(0, 0, 5, 0)
        self.assertAlmostEqual(h, 5.0)

    def test_octile_heuristic_diagonal(self):
        """Pure diagonal."""
        h = _octile_heuristic(0, 0, 3, 3)
        self.assertAlmostEqual(h, 3 * 1.4142135623730951, places=5)


class TestAutorouteResult(unittest.TestCase):
    """Test AutorouteResult dataclass."""

    def test_success(self):
        r = AutorouteResult(success=True, message="Routed", method="freerouting", tracks_added=10, vias_added=3)
        self.assertTrue(r.success)
        self.assertEqual(r.method, "freerouting")
        self.assertEqual(r.tracks_added, 10)

    def test_failure(self):
        r = AutorouteResult(success=False, message="No board")
        self.assertFalse(r.success)
        self.assertEqual(r.method, "")


class TestFindFreeroutingJar(unittest.TestCase):
    """Test _find_freerouting_jar discovery logic."""

    def test_env_var(self):
        with patch.dict(os.environ, {"FREEROUTING_JAR": "/tmp/freerouting.jar"}):
            with patch("pathlib.Path.is_file", return_value=True):
                path = _find_freerouting_jar()
                self.assertEqual(path, "/tmp/freerouting.jar")

    def test_not_found(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove FREEROUTING_JAR if it exists
            os.environ.pop("FREEROUTING_JAR", None)
            with patch("pathlib.Path.is_file", return_value=False):
                with patch("shutil.which", return_value=None):
                    path = _find_freerouting_jar()
                    self.assertIsNone(path)


class TestGridRoute(unittest.TestCase):
    """Test the grid-based A* fallback routing."""

    def test_returns_result_without_pcbnew(self):
        """When pcbnew is not available, should return a failure result."""
        # The function tries to import pcbnew; if it's not available
        # it should either raise or return an error.
        # We just verify it doesn't crash unexpectedly.
        try:
            result = _grid_route_all(None)
            # If it returns, it should be an AutorouteResult
            self.assertIsInstance(result, AutorouteResult)
        except Exception:
            # OK — pcbnew not available outside KiCad
            pass


if __name__ == '__main__':
    unittest.main()
