"""
Módulo para o estado e evolução do caminhante quântico.
"""
import numpy as np

class QuantumWalker:
    def __init__(self, graph, marked_vertices=None):
        self.graph = graph
        self.size = graph.size
        self.state = np.zeros(self.size, dtype=complex)
        self.state[0] = 1.0  # inicializa no vértice 0
        self.marked_vertices = marked_vertices or []

    def step(self):
        # Operador de evolução simples: matriz de adjacência
        self.state = self.graph.adj_matrix @ self.state
        self.state /= np.linalg.norm(self.state)

    def measure_probabilities(self):
        return np.abs(self.state) ** 2

    def detection_probability(self):
        probs = self.measure_probabilities()
        return sum(probs[v] for v in self.marked_vertices)
