import pytest
import numpy as np
from biblioteca.graph import ToroidalGrid

def test_neighbors_2d():
    grid = ToroidalGrid(L=3, n=2)
    # Vértice 0 em 3x3: vizinhos devem ser 1, 2, 3, 6
    neighbors = set(grid.get_neighbors(0))
    assert neighbors == set([1, 2, 3, 6])

def test_selfloop():
    grid = ToroidalGrid(L=2, n=1, num_selfloop=2)
    assert grid.adj_matrix[0,0] == 2
