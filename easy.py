"""
easy.py
=======
Camada de conveniencia: funcao ``walk()`` unificada que cobre todas as
topologias com uma assinatura unica + apelidos amigaveis em portugues
(``"toroidal"``, ``"toro"``, ``"grade"``, ``"grafo"``, ``"hipercubo"``).

>>> from biblioteca_qw import walk
>>> r = walk("torus", L=3, n=2, steps=50)
>>> r.summary()
{...}

>>> r = walk("hypercube", n=3, steps=100, marked=[0])

>>> r = walk("graph", N=5,
...          edges=[(0,1),(1,2),(2,3),(3,4),(4,0)],
...          steps=30, marked=[0])

>>> r = walk("grid", dims=[4, 4], steps=20, marked=[0])
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np

from biblioteca_qw.results import WalkResult
from biblioteca_qw.simulation import QuantumWalkSimulation
from biblioteca_qw.topologies import (
    GraphWalkSimulation,
    GridWalkSimulation,
    HypercubeWalkSimulation,
)

# ---------------------------------------------------------------------------
# Mapeamento de aliases -> topologia canonica
# ---------------------------------------------------------------------------

_TOROIDAL = {"torus", "toroidal", "toroid", "toro", "qw"}
_GRID = {"grid", "grade", "lattice"}
_GRAPH = {"graph", "grafo", "general", "szegedy"}
_HYPERCUBE = {"hypercube", "hipercubo", "qn", "cube", "cubo"}

AVAILABLE_TOPOLOGIES: Tuple[str, ...] = (
    "torus", "grid", "graph", "hypercube",
)


def _normalize(topology: str) -> str:
    t = topology.strip().lower()
    if t in _TOROIDAL:
        return "torus"
    if t in _GRID:
        return "grid"
    if t in _GRAPH:
        return "graph"
    if t in _HYPERCUBE:
        return "hypercube"
    raise ValueError(
        f"Topologia desconhecida: {topology!r}. "
        f"Use uma de {AVAILABLE_TOPOLOGIES} (alias em PT-BR tambem aceitos: "
        f"toroidal, toro, grade, grafo, hipercubo)."
    )


# ---------------------------------------------------------------------------
# walk()
# ---------------------------------------------------------------------------

def walk(
    topology: str,
    *,
    # parametros comuns
    steps: Optional[int] = None,
    t_f: Optional[int] = None,                 # alias 0.2.x
    marked: Optional[Iterable[int]] = None,
    marked_vertices: Optional[Iterable[int]] = None,  # alias 0.2.x
    weight_value: float = 1.0,
    db_url: Optional[str] = None,
    # toroidal / hipercubo
    L: Optional[int] = None,
    n: Optional[int] = None,
    num_selfloop: int = 0,
    # grade
    dims: Optional[List[int]] = None,
    # grafo
    N: Optional[int] = None,
    edges: Optional[List[Tuple[int, int]]] = None,
    adjacency: Optional[np.ndarray] = None,
    # opcoes do walker
    run: bool = True,
) -> Union[WalkResult, Any]:
    """Cria e executa uma simulacao DTQW na topologia indicada.

    Parametros comuns
    -----------------
    topology : str
        Nome da topologia. Aceita apelidos: "torus"/"toroidal"/"toro",
        "grid"/"grade", "graph"/"grafo", "hypercube"/"hipercubo".
    steps : int
        Numero de passos. Alias: ``t_f`` (compatibilidade 0.2.x).
    marked : list of int, optional
        Vertices marcados (oraculo). Alias: ``marked_vertices``.
        Se omitido, usa ``[0]``.
    weight_value : float
        Peso das arestas (padrao 1.0).
    db_url : str, optional
        URL do JSON Server para persistencia REST.

    Toroidal / Hipercubo
    --------------------
    L, n : int
        Toroidal: ``L`` vertices por dimensao, ``n`` dimensoes (N = L^n).
        Hipercubo: apenas ``n`` (N = 2^n).
    num_selfloop : int
        Self-loops por vertice (toroidal e hipercubo).

    Grade
    -----
    dims : list of int
        Tamanho de cada dimensao da grade.

    Grafo
    -----
    N : int
        Numero de vertices.
    edges : list of (int, int)
        Lista de arestas nao-orientadas. Mutuamente exclusivo com ``adjacency``.
    adjacency : np.ndarray
        Matriz de adjacencia (NxN). Mutuamente exclusivo com ``edges``.

    run : bool
        Se True (padrao) executa e retorna ``WalkResult``.
        Se False, retorna a instancia da classe sem executar.

    Returns
    -------
    WalkResult ou instancia da classe (se ``run=False``).

    Examples
    --------
    >>> r = walk("torus", L=3, n=2, steps=20)
    >>> r.probs.shape
    (20, 9)

    >>> r = walk("hypercube", n=4, steps=50, marked=[0, 15])
    >>> r.summary()["max_detection"] > 0
    True
    """
    name = _normalize(topology)

    # Aceitar tanto steps quanto t_f
    if steps is None and t_f is not None:
        steps = t_f
    if steps is None:
        raise TypeError("walk() requer 'steps' (ou alias 't_f').")

    # Aceitar tanto marked quanto marked_vertices
    if marked is None and marked_vertices is not None:
        marked = marked_vertices
    if marked is None:
        marked = [0]
    marked = list(marked)

    # ------------------------------------------------------------------
    # Construir simulador
    # ------------------------------------------------------------------
    if name == "torus":
        if L is None or n is None:
            raise TypeError("walk('torus') requer L e n.")
        sim = QuantumWalkSimulation(
            L=L, n=n,
            num_selfloop=num_selfloop,
            t_f=steps,
            weight_value=weight_value,
            marked_vertices=marked,
            db_url=db_url,
        )

    elif name == "grid":
        if dims is None:
            raise TypeError("walk('grid') requer dims=[...].")
        sim = GridWalkSimulation(
            dims=list(dims),
            t_f=steps,
            marked_vertices=marked,
            weight_value=weight_value,
            db_url=db_url,
        )

    elif name == "graph":
        if adjacency is not None:
            if edges is not None:
                raise TypeError(
                    "walk('graph') aceita 'edges' OU 'adjacency', nao ambos."
                )
            sim = GraphWalkSimulation.from_adjacency(
                adjacency=np.asarray(adjacency),
                t_f=steps,
                marked_vertices=marked,
                weight_value=weight_value,
                db_url=db_url,
            )
        else:
            if N is None or edges is None:
                raise TypeError(
                    "walk('graph') requer N e edges (ou matriz adjacency)."
                )
            sim = GraphWalkSimulation(
                N=N, edges=edges,
                t_f=steps,
                marked_vertices=marked,
                weight_value=weight_value,
                db_url=db_url,
            )

    elif name == "hypercube":
        if n is None:
            raise TypeError("walk('hypercube') requer n.")
        sim = HypercubeWalkSimulation(
            n=n,
            t_f=steps,
            marked_vertices=marked,
            num_selfloop=num_selfloop,
            weight_value=weight_value,
            db_url=db_url,
        )
    else:  # pragma: no cover
        raise AssertionError(f"Topologia normalizada inesperada: {name}")

    if not run:
        return sim

    probs, det_times = sim.run()
    return WalkResult(
        probs=probs,
        det_times=det_times,
        topology=name,
        params=_collect_params(sim, name),
    )


# Alias mais explicito
def simulate(*args, **kwargs) -> WalkResult:
    """Alias mais legivel de :func:`walk` para uso de script."""
    return walk(*args, **kwargs)


def _collect_params(sim, topology: str) -> dict:
    base = {
        "topology": topology,
        "t_f": sim.t_f,
        "marked_vertices": list(sim.marked_vertices),
        "weight_value": sim.weight_value,
    }
    if topology == "torus":
        base.update(L=sim.L, n=sim.n, num_selfloop=sim.num_selfloop)
    elif topology == "grid":
        base.update(dims=sim.dims)
    elif topology == "graph":
        base.update(N=sim.N, num_edges=sim.num_edges)
    elif topology == "hypercube":
        base.update(n=sim.n, num_selfloop=sim.num_selfloop)
    return base
