"""
topologies/grafo.py
===================
DTQW em grafo arbitrario via formulacao de Szegedy sobre arestas orientadas.

Versao 0.3.0
- Aliases ``steps``/``marked``.
- ``marked_vertices`` agora e opcional (padrao [0]).
- Construcao de arestas orientadas O(N^2) substituida por scan linear
  sobre os vizinhos -> mais rapido em grafos esparsos.
- Codigo morto em ``_build_reflection_operator`` removido.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from biblioteca_qw.simulation import _persist_to_json_server


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------

def _build_edge_index(
    N: int, adjacency: np.ndarray
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """Lista e dicionario de arestas orientadas a partir da matriz."""
    edges: List[Tuple[int, int]] = []
    edge_to_idx: Dict[Tuple[int, int], int] = {}
    nz = np.argwhere(adjacency != 0)
    for u, v in nz:
        edge_to_idx[(int(u), int(v))] = len(edges)
        edges.append((int(u), int(v)))
    return edges, edge_to_idx


def _build_reflection_operator(
    N: int,
    edges: List[Tuple[int, int]],
    edge_to_idx: Dict[Tuple[int, int], int],
    marked_vertices: List[int],
) -> np.ndarray:
    """R = diag(R_u) onde R_u = 2|s_u><s_u| - I, ou -I se *u* esta marcado."""
    dim = len(edges)
    R = np.zeros((dim, dim), dtype=complex)
    marked_set = set(marked_vertices)

    # agrupa indices das arestas saindo de cada vertice
    out_by_u: Dict[int, List[int]] = {u: [] for u in range(N)}
    for (u, v), idx in edge_to_idx.items():
        out_by_u[u].append(idx)

    for u in range(N):
        out_idx = out_by_u[u]
        if not out_idx:
            continue
        if u in marked_set:
            for i in out_idx:
                R[i, i] = -1.0
            continue
        k = len(out_idx)
        # 2/k * J - I sobre o bloco
        idx_arr = np.array(out_idx)
        R[np.ix_(idx_arr, idx_arr)] += 2.0 / k
        for i in out_idx:
            R[i, i] -= 1.0
    return R


def _build_swap_operator(
    edges: List[Tuple[int, int]],
    edge_to_idx: Dict[Tuple[int, int], int],
) -> np.ndarray:
    """S: |u,v> -> |v,u>."""
    dim = len(edges)
    S = np.zeros((dim, dim), dtype=complex)
    for (u, v), i in edge_to_idx.items():
        if (v, u) in edge_to_idx:
            S[edge_to_idx[(v, u)], i] = 1.0
    return S


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class GraphWalkSimulation:
    """DTQW em grafo arbitrario nao-direcionado (formulacao de Szegedy).

    Parameters
    ----------
    N : int
        Numero de vertices (>= 2).
    edges : list[tuple[int, int]]
        Arestas nao-orientadas. A reversa e adicionada automaticamente.
    t_f : int
        Passos de simulacao. Alias: ``steps``.
    marked_vertices : list[int], optional
        Vertices marcados. Padrao: ``[0]``. Alias: ``marked``.
    weight_value : float
        Peso uniforme das arestas (padrao 1.0).
    db_url : str, optional
        URL do JSON Server.

    Examples
    --------
    >>> sim = GraphWalkSimulation(N=5,
    ...     edges=[(0,1),(1,2),(2,3),(3,4),(4,0)], steps=20)
    >>> probs, det = sim.run()
    >>> probs.shape
    (20, 5)
    """

    def __init__(
        self,
        N: int,
        edges: List[Tuple[int, int]],
        t_f: Optional[int] = None,
        *,
        marked_vertices: Optional[List[int]] = None,
        weight_value: float = 1.0,
        db_url: Optional[str] = None,
        steps: Optional[int] = None,
        marked: Optional[List[int]] = None,
    ) -> None:
        if t_f is None:
            t_f = steps
        if t_f is None:
            raise TypeError("Informe t_f (ou steps).")
        if marked_vertices is None:
            marked_vertices = marked
        if marked_vertices is None:
            marked_vertices = [0]

        if N < 2:
            raise ValueError("N deve ser >= 2.")
        if t_f < 1:
            raise ValueError("t_f deve ser >= 1.")

        self.N = N
        self.t_f = t_f
        self.marked_vertices = list(marked_vertices)
        self.weight_value = float(weight_value)
        self.db_url = db_url

        self._adj = np.zeros((N, N), dtype=float)
        for (u, v) in edges:
            self._adj[u, v] = weight_value
            self._adj[v, u] = weight_value

        self._edges, self._edge_to_idx = _build_edge_index(N, self._adj)
        self.total_dim = len(self._edges)
        self._U: Optional[np.ndarray] = None

    @classmethod
    def from_adjacency(
        cls,
        adjacency: np.ndarray,
        t_f: Optional[int] = None,
        *,
        marked_vertices: Optional[List[int]] = None,
        weight_value: float = 1.0,
        db_url: Optional[str] = None,
        steps: Optional[int] = None,
        marked: Optional[List[int]] = None,
    ) -> "GraphWalkSimulation":
        """Cria simulacao a partir de matriz de adjacencia simetrica."""
        adjacency = np.asarray(adjacency)
        N = adjacency.shape[0]
        edges = []
        for u in range(N):
            for v in range(u + 1, N):
                if adjacency[u, v] != 0:
                    edges.append((u, v))
        return cls(
            N=N, edges=edges, t_f=t_f,
            marked_vertices=marked_vertices,
            weight_value=weight_value, db_url=db_url,
            steps=steps, marked=marked,
        )

    def _build_evolution_operator(self) -> np.ndarray:
        R = _build_reflection_operator(
            self.N, self._edges, self._edge_to_idx, self.marked_vertices
        )
        S = _build_swap_operator(self._edges, self._edge_to_idx)
        return S @ R

    def _initial_state(self) -> np.ndarray:
        psi = np.ones(self.total_dim, dtype=complex)
        psi /= np.linalg.norm(psi)
        return psi

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        """Executa a simulacao e retorna ``(probs, det_times)``.

        ``probs[t, v]`` e a probabilidade de encontrar a particula no
        vertice ``v`` no passo ``t`` (somando sobre arestas saindo de v).
        """
        if self._U is None:
            self._U = self._build_evolution_operator()

        # Matriz de assignment (E x N): linha i tem 1 na coluna u se a
        # aresta i sai de u — assim probabilidade por vertice e |psi|^2 @ M.
        M = np.zeros((self.total_dim, self.N), dtype=float)
        for (u, _), idx in self._edge_to_idx.items():
            M[idx, u] = 1.0

        psi = self._initial_state()
        marked = np.fromiter(self.marked_vertices, dtype=int)

        probs = np.zeros((self.t_f, self.N), dtype=float)
        det_times = np.zeros(self.t_f, dtype=float)

        for t in range(self.t_f):
            psi = self._U @ psi
            norm = np.linalg.norm(psi)
            if norm > 0:
                psi /= norm
            row = (np.abs(psi) ** 2) @ M
            total = row.sum()
            if total > 0:
                row /= total
            probs[t] = row
            if marked.size:
                det_times[t] = float(row[marked].sum())

        if self.db_url is not None:
            _persist_to_json_server(
                self.db_url,
                payload={
                    "topology": "graph",
                    "params": {
                        "N": self.N, "t_f": self.t_f,
                        "marked_vertices": self.marked_vertices,
                        "num_edges": len(self._edges) // 2,
                    },
                    "probs": probs.tolist(),
                    "det_times": det_times.tolist(),
                },
            )
        return probs, det_times

    @property
    def num_vertices(self) -> int:
        return self.N

    @property
    def hilbert_dim(self) -> int:
        return self.total_dim

    @property
    def num_edges(self) -> int:
        """Numero de arestas nao-orientadas."""
        return len(self._edges) // 2

    def adjacency_matrix(self) -> np.ndarray:
        """Matriz de adjacencia do grafo."""
        return self._adj.copy()

    def __repr__(self) -> str:
        return (
            f"GraphWalkSimulation(N={self.N}, edges={self.num_edges}, "
            f"t_f={self.t_f}, hilbert_dim={self.total_dim})"
        )
