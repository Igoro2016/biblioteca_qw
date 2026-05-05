"""
topologies/hipercubo.py
=======================
DTQW no hipercubo n-dimensional Q_n.

Versao 0.3.0 — alias ``steps``/``marked``, defaults sensatos, vetorizacao.
"""

from __future__ import annotations

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

def _build_hypercube_shift(N: int, coin_dim: int, n: int) -> np.ndarray:
    """S|v, d> = |v XOR e_d, d>."""
    dim = N * coin_dim
    S = np.zeros((dim, dim), dtype=complex)
    for v in range(N):
        for d in range(n):
            u = v ^ (1 << d)
            S[u * coin_dim + d, v * coin_dim + d] = 1.0
        for sl in range(n, coin_dim):
            S[v * coin_dim + sl, v * coin_dim + sl] = 1.0
    return S


def _build_hypercube_coin(
    N: int, coin_dim: int, marked_vertices: List[int]
) -> np.ndarray:
    marked_set = set(marked_vertices)
    grover = _grover_coin(coin_dim)
    oracle = _oracle_coin(coin_dim)
    C = np.zeros((N * coin_dim, N * coin_dim), dtype=complex)
    for v in range(N):
        r = v * coin_dim
        C[r:r + coin_dim, r:r + coin_dim] = oracle if v in marked_set else grover
    return C


# ---------------------------------------------------------------------------
# Classe principal
# ---------------------------------------------------------------------------

class HypercubeWalkSimulation:
    """DTQW no hipercubo Q_n (N = 2^n vertices, grau n).

    Parameters
    ----------
    n : int
        Dimensao (>= 1). N = 2^n.
    t_f : int
        Passos. Alias: ``steps``.
    marked_vertices : list[int], optional
        Vertices marcados em [0, 2^n - 1]. Padrao: ``[0]``. Alias: ``marked``.
    num_selfloop : int, optional
        Self-loops (padrao 0).
    weight_value : float
        Padrao 1.0.
    db_url : str, optional
        URL JSON Server.

    Examples
    --------
    >>> sim = HypercubeWalkSimulation(n=3, steps=20)        # marcado=[0] por default
    >>> probs, det = sim.run()
    >>> probs.shape
    (20, 8)
    """

    def __init__(
        self,
        n: int,
        t_f: Optional[int] = None,
        *,
        marked_vertices: Optional[List[int]] = None,
        num_selfloop: int = 0,
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

        if n < 1:
            raise ValueError("n deve ser >= 1.")
        if t_f < 1:
            raise ValueError("t_f deve ser >= 1.")
        if num_selfloop < 0:
            raise ValueError("num_selfloop deve ser >= 0.")

        self.n = n
        self.t_f = t_f
        self.marked_vertices = list(marked_vertices)
        self.num_selfloop = num_selfloop
        self.weight_value = float(weight_value)
        self.db_url = db_url

        self.N: int = 2 ** n
        self.coin_dim: int = n + num_selfloop
        self.total_dim: int = self.N * self.coin_dim
        self._U: Optional[np.ndarray] = None

    def _build_evolution_operator(self) -> np.ndarray:
        S = _build_hypercube_shift(self.N, self.coin_dim, self.n)
        C = _build_hypercube_coin(self.N, self.coin_dim, self.marked_vertices)
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
                    "topology": "hypercube",
                    "params": {
                        "n": self.n, "N": self.N, "t_f": self.t_f,
                        "marked_vertices": self.marked_vertices,
                        "num_selfloop": self.num_selfloop,
                    },
                    "probs": probs.tolist(),
                    "det_times": det_times.tolist(),
                },
            )
        return probs, det_times

    # ------------------------------------------------------------------
    # Utilitarios
    # ------------------------------------------------------------------

    def vertex_to_binary(self, v: int) -> str:
        """Representacao binaria de *v* como string de n bits."""
        return format(v, f"0{self.n}b")

    def hamming_distance(self, u: int, v: int) -> int:
        """Distancia de Hamming entre *u* e *v*."""
        return bin(u ^ v).count("1")

    def antipodal(self, v: int) -> int:
        """Vertice antipodal de *v* (todos os bits invertidos)."""
        return v ^ ((1 << self.n) - 1)

    @property
    def num_vertices(self) -> int:
        return self.N

    @property
    def hilbert_dim(self) -> int:
        return self.total_dim

    @property
    def diameter(self) -> int:
        """Diametro do hipercubo (= n)."""
        return self.n

    def __repr__(self) -> str:
        return (
            f"HypercubeWalkSimulation(n={self.n}, N={self.N}, "
            f"t_f={self.t_f}, coin_dim={self.coin_dim})"
        )
