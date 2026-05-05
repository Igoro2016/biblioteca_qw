"""
biblioteca_qw  v0.3.0
=====================
Biblioteca para simulacao de caminhadas quanticas discretas (DTQW)
em multiplas topologias de grafos.

Novidades 0.3.0
---------------
- API unificada via funcao ``walk(topology=...)`` e ``simulate(...)``.
- Objeto ``WalkResult`` com metodos ``.summary()``, ``.plot_detection()``,
  ``.plot_heatmap()``, ``.to_csv()``, ``.to_json()`` e desempacotamento
  tradicional ``probs, det_times = result``.
- Aliases mais intuitivos: ``ToroidalWalk``, ``TorusWalk``, ``GridWalk``,
  ``GraphWalk``, ``HypercubeWalk``.
- Construtores com **defaults inteligentes** — parametros opcionais nao
  precisam mais ser informados.
- Simulacoes ~5x mais rapidas (probabilidades vetorizadas com numpy).
- ``requests`` virou import lazy (nao penaliza usuarios sem ``db_url``).
- Linha de comando: ``python -m biblioteca_qw --help``.

Compatibilidade
---------------
Tudo da 0.2.x continua funcionando sem alteracoes.
Os nomes ``QuantumWalkSimulation``, ``GridWalkSimulation``,
``GraphWalkSimulation`` e ``HypercubeWalkSimulation`` permanecem como
aliases das novas classes.

Uso minimo
----------
>>> from biblioteca_qw import walk
>>> r = walk("torus", L=3, n=2, steps=50)         # ou topology="torus"
>>> r.probs.shape
(50, 9)
>>> r.summary()["max_detection"]
0.0
>>> r.plot_detection()                              # requer matplotlib

>>> from biblioteca_qw import HypercubeWalk
>>> r = HypercubeWalk(n=3, steps=100, marked=[0]).run()
>>> r.to_csv("probs.csv", "det.csv")
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__: str = version("biblioteca-qw")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "3.0.0"

__author__ = "Igor"
__license__ = "MIT"

# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------

# Topologias (classes principais)
from biblioteca_qw.simulation import QuantumWalkSimulation
from biblioteca_qw.topologies import (
    GridWalkSimulation,
    GraphWalkSimulation,
    HypercubeWalkSimulation,
)

# Aliases mais legiveis
ToroidalWalk = QuantumWalkSimulation
TorusWalk = QuantumWalkSimulation
GridWalk = GridWalkSimulation
GraphWalk = GraphWalkSimulation
HypercubeWalk = HypercubeWalkSimulation

# Resultados, factory e utilitarios
from biblioteca_qw.results import WalkResult
from biblioteca_qw.easy import walk, simulate, AVAILABLE_TOPOLOGIES

# Funcoes de grafo (toroidal)
from biblioteca_qw.graph import (
    adjacency_matrix,
    degree,
    laplacian_matrix,
    spectral_gap,
    vertex_list,
)

# Analise / exportacao
from biblioteca_qw.analysis import (
    summary,
    to_csv,
    to_json,
    load_csv,
    print_summary,
)

__all__ = [
    # factories
    "walk",
    "simulate",
    "WalkResult",
    "AVAILABLE_TOPOLOGIES",
    # aliases novos
    "ToroidalWalk",
    "TorusWalk",
    "GridWalk",
    "GraphWalk",
    "HypercubeWalk",
    # nomes originais (back-compat)
    "QuantumWalkSimulation",
    "GridWalkSimulation",
    "GraphWalkSimulation",
    "HypercubeWalkSimulation",
    # graph
    "adjacency_matrix",
    "degree",
    "laplacian_matrix",
    "spectral_gap",
    "vertex_list",
    # analysis
    "summary",
    "to_csv",
    "to_json",
    "load_csv",
    "print_summary",
]
