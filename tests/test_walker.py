import pytest
import numpy as np
from biblioteca.graph import ToroidalGrid
from biblioteca.walker import QuantumWalker

def test_initial_state():
    grid = ToroidalGrid(L=2, n=1)
    walker = QuantumWalker(grid)
    assert np.isclose(np.sum(np.abs(walker.state)**2), 1.0)

def test_step_normalization():
    grid = ToroidalGrid(L=2, n=1)
    walker = QuantumWalker(grid)
    walker.step()
    assert np.isclose(np.sum(np.abs(walker.state)**2), 1.0)
