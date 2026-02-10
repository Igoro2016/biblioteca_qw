import pytest
from biblioteca.simulation import QuantumWalkSimulation

def test_simulation_runs():
    sim = QuantumWalkSimulation(L=2, n=1, num_selfloop=0, t_f=3, weight_value=1.0, marked_vertices=[1], db_url=None)
    probs, det_times = sim.run()
    assert probs.shape[0] == 3
    assert det_times.shape[0] == 3
