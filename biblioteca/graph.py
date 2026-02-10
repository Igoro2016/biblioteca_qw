"""
Módulo para estrutura de grafos toroidais.
"""
import numpy as np

class ToroidalGrid:
    def __init__(self, L, n, num_selfloop=0, weight_value=1.0):
        self.L = L  # dimensão da grade
        self.n = n  # número de dimensões
        self.num_selfloop = num_selfloop
        self.weight_value = weight_value
        self.size = L ** n
        self.adj_matrix = self._build_adj_matrix()

    def _build_adj_matrix(self):
        adj = np.zeros((self.size, self.size))
        for v in range(self.size):
            neighbors = self.get_neighbors(v)
            for u in neighbors:
                adj[v, u] = self.weight_value
            for _ in range(self.num_selfloop):
                adj[v, v] += self.weight_value
        return adj

    def get_neighbors(self, v):
        idx = self._index_to_coords(v)
        neighbors = []
        for d in range(self.n):
            for delta in [-1, 1]:
                neighbor_idx = list(idx)
                neighbor_idx[d] = (neighbor_idx[d] + delta) % self.L
                neighbors.append(self._coords_to_index(tuple(neighbor_idx)))
        return neighbors

    def _index_to_coords(self, idx):
        coords = []
        for _ in range(self.n):
            coords.append(idx % self.L)
            idx //= self.L
        return tuple(reversed(coords))

    def _coords_to_index(self, coords):
        idx = 0
        for c in coords:
            idx = idx * self.L + c
        return idx
