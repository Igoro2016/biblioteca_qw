"""
Módulo para execução da simulação de caminhadas quânticas.
Armazena resultados em db.json via JSON Server.
"""
import numpy as np
import requests
from .graph import ToroidalGrid
from .walker import QuantumWalker

class QuantumWalkSimulation:
    def __init__(self, L, n, num_selfloop, t_f, weight_value, marked_vertices, db_url='http://localhost:3000/results'):
        self.L = L
        self.n = n
        self.num_selfloop = num_selfloop
        self.t_f = t_f
        self.weight_value = weight_value
        self.marked_vertices = marked_vertices
        self.db_url = db_url
        self.graph = ToroidalGrid(L, n, num_selfloop, weight_value)
        self.walker = QuantumWalker(self.graph, marked_vertices)
        self.results = {
            'parameters': {
                'L': L, 'n': n, 'num_selfloop': num_selfloop, 't_f': t_f,
                'weight_value': weight_value, 'marked_vertices': marked_vertices
            },
            'probabilities': [],
            'detection_times': []
        }

    def run(self):
        detection_probs = []
        for t in range(self.t_f):
            self.walker.step()
            prob = self.walker.detection_probability()
            detection_probs.append(prob)
            self.results['probabilities'].append(self.walker.measure_probabilities().tolist())
            self.results['detection_times'].append(prob)
        self._save_results()
        return np.array(self.results['probabilities']), np.array(self.results['detection_times'])

    def _save_results(self):
        try:
            requests.post(self.db_url, json=self.results)
        except Exception as e:
            print(f'Erro ao salvar no JSON Server: {e}')
