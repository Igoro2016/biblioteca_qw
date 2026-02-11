from biblioteca.simulation import QuantumWalkSimulation

# Parâmetros da simulação
sim = QuantumWalkSimulation(
    L=3,
    n=2,
    num_selfloop=1,
    t_f=10,
    weight_value=1.0,
    marked_vertices=[0, 1]
)

probs, det_times = sim.run()
print('Probabilidades shape:', probs.shape)
print('Detecção shape:', det_times.shape)
print('Última probabilidade de detecção:', det_times[-1])
