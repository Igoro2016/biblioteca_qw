"""
simulation.py
=============
DTQW em grafo toroidal n-dimensional T(L, n) com vertices marcados.

v0.3.0 — defaults inteligentes, calculo vetorizado de probabilidades,
import lazy de ``requests`` (so e carregado quando ``db_url`` e usado).

Modelo
------
H = H_P ⊗ H_C com ``dim H_P = L^n`` e ``dim H_C = 2n + num_selfloop``.
U = S · (I ⊗ C_oracle); estado inicial uniforme.
"""

from __future__ import annotations

import warnings
from typing import Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Auxiliares de construcao do grafo toroidal
# ---------------------------------------------------------------------------

def _toroidal_neighbors(vertex: int, L: int, n: int) -> List[Tuple[int, int]]:
    """Vizinhos de *vertex* num grafo toroidal T(L,n)."""
    coords = []
    tmp = vertex
    for _ in range(n):
        coords.append(tmp % L)
        tmp //= L
    coords = list(reversed(coords))

    neighbors = []
    coin_dir = 0
    for dim in range(n):
        for delta in (+1, -1):
            nc = coords.copy()
            nc[dim] = (nc[dim] + delta) % L
            idx = 0
            for c in nc:
                idx = idx * L + c
            neighbors.append((idx, coin_dir))
            coin_dir += 1
    return neighbors


def _build_shift_operator(
    N: int, coin_dim: int, L: int, n: int, num_selfloop: int
) -> np.ndarray:
    dim = N * coin_dim
    S = np.zeros((dim, dim), dtype=complex)
    for v in range(N):
        for u, d in _toroidal_neighbors(v, L, n):
            d_opp = d ^ 1
            S[u * coin_dim + d_opp, v * coin_dim + d] = 1.0
        for sl in range(num_selfloop):
            coin_idx = 2 * n + sl
            S[v * coin_dim + coin_idx, v * coin_dim + coin_idx] = 1.0
    return S


def _grover_coin(dim: int) -> np.ndarray:
    """Moeda de Grover (2/dim·J − I) de dimensao *dim*."""
    return (2.0 / dim) * np.ones((dim, dim), dtype=complex) - np.eye(dim, dtype=complex)


def _oracle_coin(dim: int) -> np.ndarray:
    """Moeda oraculo (-I) de dimensao *dim*."""
    return -np.eye(dim, dtype=complex)


def _build_coin_operator(
    N: int, coin_dim: int, marked_vertices: Iterable[int]
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
# Helper de "rodar a evolucao" — usado por todas as classes
# ---------------------------------------------------------------------------

def _run_evolution(
    U: np.ndarray,
    psi0: np.ndarray,
    steps: int,
    coin_dim: int,
    N: int,
    marked_vertices: Iterable[int],
    re_normalize_probs: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aplica U *steps* vezes e devolve (probs, det_times) — vetorizado."""
    psi = psi0.astype(complex, copy=True)
    marked = np.fromiter(marked_vertices, dtype=int)

    probs = np.zeros((steps, N), dtype=float)
    det_times = np.zeros(steps, dtype=float)

    for t in range(steps):
        psi = U @ psi
        norm = np.linalg.norm(psi)
        if norm > 0:
            psi /= norm

        # |psi|^2 reagrupado em (N, coin_dim) e somado pela dimensao da moeda
        row = (np.abs(psi) ** 2).reshape(N, coin_dim).sum(axis=1)
        if re_normalize_probs:
            total = row.sum()
            if total > 0:
                row = row / total
        probs[t] = row

        if marked.size:
            det_times[t] = float(row[marked].sum())

    return probs, det_times


# ---------------------------------------------------------------------------
# Classe principal — DTQW toroidal
# ---------------------------------------------------------------------------

class QuantumWalkSimulation:
    """DTQW em grafo toroidal n-dimensional T(L, n).

    Parameters
    ----------
    L : int
        Vertices por dimensao (>= 2).
    n : int
        Numero de dimensoes (>= 1).
    t_f : int
        Passos de simulacao (>= 1). **Pode ser passado como ``steps``**.
    num_selfloop : int, optional
        Self-loops por vertice. Padrao: 0.
    weight_value : float, optional
        Peso das arestas. Padrao: 1.0.
    marked_vertices : list[int], optional
        Vertices marcados (oraculo). Padrao: ``[0]``.
        **Pode ser passado como ``marked``**.
    db_url : str, optional
        URL do JSON Server para persistencia REST.

    Examples
    --------
    >>> sim = QuantumWalkSimulation(L=3, n=2, t_f=20)         # defaults: 1 marcado, 0 self-loop
    >>> probs, det_times = sim.run()
    >>> probs.shape
    (20, 9)

    Tambem aceita os apelidos novos:

    >>> sim = QuantumWalkSimulation(L=3, n=2, steps=20, marked=[0, 1])
    """

    def __init__(
        self,
        L: int,
        n: int,
        t_f: Optional[int] = None,
        *,
        num_selfloop: int = 0,
        weight_value: float = 1.0,
        marked_vertices: Optional[List[int]] = None,
        db_url: Optional[str] = None,
        # aliases novos
        steps: Optional[int] = None,
        marked: Optional[List[int]] = None,
    ) -> None:
        # Aceitar steps como alias de t_f
        if t_f is None:
            t_f = steps
        if t_f is None:
            raise TypeError("Informe t_f (ou steps).")
        # Aceitar marked como alias de marked_vertices
        if marked_vertices is None:
            marked_vertices = marked
        if marked_vertices is None:
            marked_vertices = [0]

        if L < 2:
            raise ValueError("L deve ser >= 2.")
        if n < 1:
            raise ValueError("n deve ser >= 1.")
        if num_selfloop < 0:
            raise ValueError("num_selfloop deve ser >= 0.")
        if t_f < 1:
            raise ValueError("t_f deve ser >= 1.")

        self.L = L
        self.n = n
        self.num_selfloop = num_selfloop
        self.t_f = t_f
        self.weight_value = float(weight_value)
        self.marked_vertices = list(marked_vertices)
        self.db_url = db_url

        self.N: int = L ** n
        self.coin_dim: int = 2 * n + num_selfloop
        self.total_dim: int = self.N * self.coin_dim
        self._U: Optional[np.ndarray] = None

    def _build_evolution_operator(self) -> np.ndarray:
        S = _build_shift_operator(self.N, self.coin_dim, self.L, self.n, self.num_selfloop)
        C = _build_coin_operator(self.N, self.coin_dim, self.marked_vertices)
        return self.weight_value * (S @ C)

    def _initial_state(self) -> np.ndarray:
        psi = np.ones(self.total_dim, dtype=complex)
        psi /= np.linalg.norm(psi)
        return psi

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        """Executa a simulacao por ``t_f`` passos.

        Returns
        -------
        probs : np.ndarray, shape (t_f, N)
        det_times : np.ndarray, shape (t_f,)
        """
        if self._U is None:
            self._U = self._build_evolution_operator()

        probs, det_times = _run_evolution(
            U=self._U,
            psi0=self._initial_state(),
            steps=self.t_f,
            coin_dim=self.coin_dim,
            N=self.N,
            marked_vertices=self.marked_vertices,
        )

        if self.db_url is not None:
            _persist_to_json_server(
                self.db_url,
                payload={
                    "topology": "torus",
                    "params": {
                        "L": self.L, "n": self.n,
                        "num_selfloop": self.num_selfloop,
                        "t_f": self.t_f, "weight_value": self.weight_value,
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
            f"QuantumWalkSimulation(L={self.L}, n={self.n}, "
            f"num_selfloop={self.num_selfloop}, t_f={self.t_f}, "
            f"N={self.N}, coin_dim={self.coin_dim})"
        )


# ---------------------------------------------------------------------------
# Persistencia REST (lazy import de requests)
# ---------------------------------------------------------------------------

def _persist_to_json_server(db_url: str, payload: dict) -> None:
    """Envia payload para JSON Server. Falha silenciosa com warning."""
    try:
        import requests  # lazy
    except ImportError:  # pragma: no cover
        warnings.warn(
            "db_url foi informado mas 'requests' nao esta instalado. "
            "Instale com: pip install requests"
        )
        return
    try:
        resp = requests.post(
            f"{db_url.rstrip('/')}/resultados",
            json=payload,
            timeout=5,
        )
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Falha ao persistir no JSON Server: {exc}")
