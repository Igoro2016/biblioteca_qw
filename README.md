# Biblioteca de Caminhadas Quânticas em Grafos Toroidais

Esta biblioteca permite configurar e executar simulações de caminhadas quânticas em grafos toroidais, com armazenamento de resultados em JSON Server (`db.json`).

## Instalação

1. Clone o repositório.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Instale o JSON Server (Node.js):
   ```bash
   npm install -g json-server
   ```
4. Inicie o JSON Server:
   ```bash
   json-server --watch db.json
   ```

## Uso

```python
from biblioteca.simulation import QuantumWalkSimulation
sim = QuantumWalkSimulation(L=3, n=2, num_selfloop=1, t_f=10, weight_value=1.0, marked_vertices=[0,1])
probs, det_times = sim.run()
```

## Parâmetros
- `L`: dimensão da grade
- `n`: número de dimensões
- `num_selfloop`: self-loops por vértice
- `t_f`: passos de simulação
- `weight_value`: peso das arestas
- `marked_vertices`: lista de vértices marcados

## Resultados
Os resultados são armazenados em arrays NumPy e enviados ao `db.json` via JSON Server.

Além disso, dois arquivos CSV são gerados para facilitar a análise:
- **probs.csv**: contém, em cada linha, as probabilidades de encontrar a partícula em cada vértice do grafo para cada passo da simulação.
- **det_times.csv**: contém, em cada linha, o tempo de detecção (em que passo a partícula foi detectada) para cada simulação.

## Testes
Execute os testes com:
```bash
pytest tests/
```

## Exemplo em Jupyter Notebook
Veja o notebook de exemplo para visualizações e animações.
