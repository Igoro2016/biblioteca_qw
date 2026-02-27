import matplotlib.pyplot as plt
from biblioteca.simulation import QuantumWalkSimulation
import numpy as np

# Parâmetros da simulação
sim = QuantumWalkSimulation(
    L=3,
    n=2,
    num_selfloop=1,
    t_f=3500,  # 3500 passos
    weight_value=1.0,
    marked_vertices=[0, 1]
)

probs, det_times = sim.run()

# Salvar resultados em CSV
np.savetxt('probs.csv', probs, delimiter=',')
np.savetxt('det_times.csv', det_times, delimiter=',')

print('Probabilidades shape:', probs.shape)
print('Detecção shape:', det_times.shape)
print('Última probabilidade de detecção:', det_times[-1])

# Resumo estatístico das probabilidades de detecção
print('\nResumo estatístico das probabilidades de detecção:')
print('Média:', det_times.mean())
print('Máximo:', det_times.max())
print('Mínimo:', det_times.min())

# Gerar gráfico da evolução da detecção
plt.figure(figsize=(10, 5))
plt.plot(det_times, label='Probabilidade de Detecção')
plt.xlabel('Passos')
plt.ylabel('Probabilidade')
plt.title('Evolução da Detecção ao Longo dos Passos')
plt.legend()
plt.show()
