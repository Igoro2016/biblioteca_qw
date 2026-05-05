"""
topologies/grade.py
===================
DTQW em Grade (Grid) finita n-dimensional, sem periodicidade.
Vertices de borda recebem moeda de reflexao (boundary coin).

Versao 0.3.0 — alias ``steps``/``marked``, defaults sensatos, vetorizacao.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np

from biblioteca_qw.simulation import (
    _grover_coin,
    _oracle_coin,
    _persist_to_json_server,
    _run_evolution,
)


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------

def _grid_neighbors(
    vertex: int, dims: List[int]
) -> List[Tuple[Optional[int], int]]:
    n = len(dims)
    strides = [1] * n
    for i in range(n - 2, -1, -1):
        strides[i] = strides[i + 1] * dims[i + 1]

    coords = []
    tmp = vertex
    for i in range(n):
        coords.append(tmp // strides[i])
        tmp = tmp % strides[i]

    result = []
    coin_dir = 0
    for dim in range(n):
        for delta in (+1, -1):
            nc = coords.copy()
            nc[dim] += delta
            if 0 <= nc[dim] < dims[dim]:
                idx = sum(c * s for c, s in zip(nc, strides))
                result.append((idx, coin_dir))
            else:
                result.append((None, coin_dir))
            coin_dir += 1
    return result


def _build_grid_shift(N: int, coin_dim: int, dims: List[int]) -> np.ndarray:
    dim = N * coin_dim
    S = np.zeros((dim, dim), dtype=complex)
    for v in range(N):
        for (u, d) in _grid_neighbors(v, dims):
            if u is not None:
                d_opp = d ^ 1
                S[u * coin_dim + d_opp, v * coin_dim + d] = 1.0
            else:
                # reflexao de borda (permanece no proprio vertice)
                S[v * coin_dim + d, v * coin_dim + d] = 1.0
    return S


def _build_grid_coin(
    N: int, coin_dim: int, dims: List[int], marked_vertices: List[int]
) -> np.ndarray:
    marked_set = set(marked_vertices)
    oracle = _oracle_coin(coin_dim)
    C = np.zeros((N * coin_dim, N * coin_dim), dtype=complex)
    for v in range(N):
        r = v * coin_dim
        if v in marked_set:
            C[r:r + coin_dim, r:r + coin_dim] = oracle
            continue
        active = [d for (u, d) in _grid_neighbors(v, dims) if u is not None]
        block = -np.eye(coin_dim, dtype=complex)
        if active:
            k = len(active)
            for i in active:
                for j in active:
                    block[i, j] += 2.0 / k
        C[r:r + coin_dim, r:r + coin_dim] = block
    return C


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class GridWalkSimulation:
    """DTQW em grade finita n-dimensional.

    Parameters
    ----------
    dims : list[int]
        Tamanho de cada dimensao da grade. Cada dimensao >= 2.
    t_f : int
        Passos de simulacao. Pode ser passado como ``steps``.
    marked_vertices : list[int], optional
        Indices dos vertices marcados. Padrao: ``[0]``. Alias: ``marked``.
    weight_value : float
        Peso das arestas. Padrao: 1.0.
    db_url : str, optional
        URL do JSON Server.

    Examples
    --------
    >>> sim = GridWalkSimulation(dims=[4, 4], steps=20)   # 1 marcado por padrao
    >>> probs, det = sim.run()
    >>> probs.shape
    (20, 16)
    """

    def __init__(
        self,
        dims: List[int],
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

        if any(d < 2 for d in dims):
            raise ValueError("Todas as dimensoes devem ser >= 2.")
        if t_f < 1:
            raise ValueError("t_f deve ser >= 1.")

        self.dims = list(dims)
        self.n = len(dims)
        self.t_f = t_f
        self.marked_vertices = list(marked_vertices)
        self.weight_value = float(weight_value)
        self.db_url = db_url

        self.N: int = 1
        for d in self.dims:
            self.N *= d
        self.coin_dim: int = 2 * self.n
        self.total_dim: int = self.N * self.coin_dim
        self._U: Optional[np.ndarray] = None

    def _build_evolution_operator(self) -> np.ndarray:
        S = _build_grid_shift(self.N, self.coin_dim, self.dims)
        C = _build_grid_coin(self.N, self.coin_dim, self.dims, self.marked_vertices)
        return self.weight_value * (S @ C)

    def _initial_state(self) -> np.ndarray:
        psi = np.ones(self.total_dim, dtype=complex)
        psi /= np.linalg.norm(psi)
        return psi

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._U is None:
            self._U = self._build_evolution_operator()
        probs, det_times = _run_evolution(
            U=self._U, psi0=self._initial_state(),
            steps=self.t_f, coin_dim=self.coin_dim, N=self.N,
            marked_vertices=self.marked_vertices,
        )
        if self.db_url is not None:
            _persist_to_json_server(
                self.db_url,
                payload={
                    "topology": "grid",
                    "params": {
                        "dims": self.dims, "t_f": self.t_f,
                        "marked_vertices": self.marked_vertices,
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

    def __repr__(self) -> str:
        return (
            f"GridWalkSimulation(dims={self.dims}, t_f={self.t_f}, "
            f"N={self.N}, coin_dim={self.coin_dim})"
        )
