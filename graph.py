"""
graph.py
========
Utilitarios para construcao e analise do grafo toroidal T(L, n).
(Versao 0.3.0 — sem mudanca de API; pequenas limpezas.)
"""

from __future__ import annotations

import itertools
from typing import List, Tuple

import numpy as np


def adjacency_matrix(L: int, n: int, num_selfloop: int = 0) -> np.ndarray:
    """Matriz de adjacencia do grafo toroidal T(L,n)."""
    N = L ** n
    A = np.zeros((N, N), dtype=float)
    for v in range(N):
        coords = _index_to_coords(v, L, n)
        for dim in range(n):
            for delta in (+1, -1):
                nc = coords.copy()
                nc[dim] = (nc[dim] + delta) % L
                A[v, _coords_to_index(nc, L)] += 1.0
        A[v, v] += float(num_selfloop)
    return A


def degree(L: int, n: int, num_selfloop: int = 0) -> int:
    """Grau uniforme dos vertices de T(L, n)."""
    return 2 * n + num_selfloop


def laplacian_matrix(L: int, n: int, num_selfloop: int = 0) -> np.ndarray:
    """Laplaciano L = D - A do grafo toroidal."""
    A = adjacency_matrix(L, n, num_selfloop)
    return np.diag(A.sum(axis=1)) - A


def spectral_gap(L: int, n: int, num_selfloop: int = 0) -> float:
    """Gap espectral (lambda_2 - lambda_1) do laplaciano."""
    lap = laplacian_matrix(L, n, num_selfloop)
    eigs = np.sort(np.linalg.eigvalsh(lap))
    return float(eigs[1] - eigs[0])


def vertex_list(L: int, n: int) -> List[Tuple[int, ...]]:
    """Lista das coordenadas (em [0, L-1]^n) de todos os vertices."""
    return list(itertools.product(range(L), repeat=n))


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _index_to_coords(idx: int, L: int, n: int) -> List[int]:
    coords: List[int] = []
    for _ in range(n):
        coords.append(idx % L)
        idx //= L
    return list(reversed(coords))


def _coords_to_index(coords: List[int], L: int) -> int:
    idx = 0
    for c in coords:
        idx = idx * L + c
    return idx
