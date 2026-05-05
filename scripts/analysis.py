"""
analysis.py
===========
Analise / exportacao de resultados de simulacao.

Versao 0.3.0
- ``summary`` agora inclui ``argmax_vertex`` (vertice mais provavel no pico)
  e ``time_to_threshold(p)``.
- ``to_csv`` aceita parametro ``fmt`` para precisao customizada.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np


def summary(probs: np.ndarray, det_times: np.ndarray) -> Dict[str, Any]:
    """Estatisticas-chave da simulacao."""
    peak_step = int(np.argmax(det_times))
    argmax_vertex = (
        int(np.argmax(probs[peak_step])) if probs.ndim == 2 else None
    )
    return {
        "mean_detection": float(det_times.mean()),
        "max_detection": float(det_times.max()),
        "min_detection": float(det_times.min()),
        "peak_step": peak_step,
        "final_detection": float(det_times[-1]),
        "num_steps": int(len(det_times)),
        "num_vertices": int(probs.shape[1]) if probs.ndim == 2 else None,
        "argmax_vertex_at_peak": argmax_vertex,
    }


def to_csv(
    probs: np.ndarray,
    det_times: np.ndarray,
    probs_path: Union[str, Path] = "probs.csv",
    det_times_path: Union[str, Path] = "det_times.csv",
    fmt: str = "%.10f",
) -> None:
    """Exporta probs e det_times para CSV (precisao em ``fmt``)."""
    np.savetxt(str(probs_path), probs, delimiter=",", fmt=fmt)
    np.savetxt(str(det_times_path), det_times, delimiter=",", fmt=fmt)


def to_json(
    probs: np.ndarray,
    det_times: np.ndarray,
    probs_path: Union[str, Path] = "probs.json",
    det_times_path: Union[str, Path] = "det_times.json",
) -> None:
    """Exporta probs e det_times para JSON."""
    with open(str(probs_path), "w", encoding="utf-8") as f:
        json.dump(probs.tolist(), f)
    with open(str(det_times_path), "w", encoding="utf-8") as f:
        json.dump(det_times.tolist(), f)


def load_csv(
    probs_path: Union[str, Path] = "probs.csv",
    det_times_path: Union[str, Path] = "det_times.csv",
):
    """Carrega ``probs`` e ``det_times`` previamente exportados em CSV."""
    probs = np.loadtxt(str(probs_path), delimiter=",")
    det_times = np.loadtxt(str(det_times_path), delimiter=",")
    return probs, det_times


def print_summary(probs: np.ndarray, det_times: np.ndarray) -> None:
    """Imprime um resumo formatado da simulacao."""
    s = summary(probs, det_times)
    print("=" * 50)
    print("  Resumo da Simulacao - biblioteca_qw")
    print("=" * 50)
    print(f"  Passos totais        : {s['num_steps']}")
    print(f"  Vertices             : {s['num_vertices']}")
    print(f"  Det. media           : {s['mean_detection']:.6f}")
    print(f"  Det. maxima          : {s['max_detection']:.6f}  (passo {s['peak_step']})")
    print(f"  Det. minima          : {s['min_detection']:.6f}")
    print(f"  Det. final           : {s['final_detection']:.6f}")
    if s.get("argmax_vertex_at_peak") is not None:
        print(f"  Vertice top no pico  : {s['argmax_vertex_at_peak']}")
    print("=" * 50)


def time_to_threshold(det_times: np.ndarray, threshold: float) -> Optional[int]:
    """Primeiro passo *t* tal que ``det_times[t] >= threshold``.

    Retorna ``None`` se o limiar nao foi atingido.
    """
    hits = np.where(det_times >= threshold)[0]
    return int(hits[0]) if hits.size else None
