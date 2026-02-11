import numpy as np
import matplotlib.pyplot as plt

# Carrega os resultados salvos pelo exemplo.py
probs = np.loadtxt('probs.csv', delimiter=',') if False else None  # Placeholder caso queira salvar em CSV depois

def main():
    # Para este exemplo, vamos rodar a simulação novamente e analisar o resultado em memória
    from biblioteca.simulation import QuantumWalkSimulation
    sim = QuantumWalkSimulation(L=3, n=2, num_selfloop=1, t_f=10, weight_value=1.0, marked_vertices=[0, 1], db_url=None)
    probs, det_times = sim.run()

    # Salva resultados em CSV
    np.savetxt('probs.csv', probs, delimiter=',')
    np.savetxt('det_times.csv', det_times, delimiter=',')

    # Salva resultados em JSON
    import json
    with open('probs.json', 'w') as f:
        json.dump(probs.tolist(), f)
    with open('det_times.json', 'w') as f:
        json.dump(det_times.tolist(), f)

    print('Resultados salvos em probs.csv, det_times.csv, probs.json e det_times.json')

    # Plotando a evolução da probabilidade de detecção
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(det_times, marker='o')
    plt.title('Probabilidade de Detecção')
    plt.xlabel('Passo')
    plt.ylabel('Probabilidade')

    # Plotando a evolução das probabilidades em cada vértice
    plt.subplot(1, 2, 2)
    plt.imshow(probs.T, aspect='auto', cmap='viridis')
    plt.title('Evolução das Probabilidades')
    plt.xlabel('Passo')
    plt.ylabel('Vértice')
    plt.colorbar(label='Probabilidade')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
