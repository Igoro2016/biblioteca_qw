"""
results.py
==========
Objeto ``WalkResult`` que agrega ``probs`` e ``det_times`` numa unica estrutura
com metodos uteis. Substitui o uso direto da tupla retornada por ``.run()`` —
mas continua se comportando como ela.

>>> from biblioteca_qw import walk
>>> r = walk("torus", L=3, n=2, steps=20)
>>> probs, det_times = r        # tupla unpacking continua funcionando
>>> r.summary()["max_detection"]
0.123...
>>> r.to_csv("p.csv", "d.csv")
>>> r.plot_detection()           # requer matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import numpy as np


@dataclass
class WalkResult:
    """Resultado completo de uma simulacao DTQW.

    Atributos
    ---------
    probs : np.ndarray, shape (steps, N)
        Probabilidade de presenca por vertice em cada passo.
    det_times : np.ndarray, shape (steps,)
        Probabilidade total nos vertices marcados em cada passo.
    topology : str
        Topologia usada (``"torus"``, ``"grid"``, ``"graph"``, ``"hypercube"``).
    params : dict
        Parametros usados na simulacao (para reproducibilidade).
    """

    probs: np.ndarray
    det_times: np.ndarray
    topology: str = "unknown"
    params: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Compatibilidade com retorno antigo "probs, det_times = sim.run()"
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.probs
        yield self.det_times

    def __len__(self) -> int:
        return 2

    def __getitem__(self, i: int) -> np.ndarray:
        if i in (0, -2):
            return self.probs
        if i in (1, -1):
            return self.det_times
        raise IndexError(f"WalkResult tem 2 itens, indice {i} fora do limite.")

    # ------------------------------------------------------------------
    # Acessores derivados
    # ------------------------------------------------------------------

    @property
    def steps(self) -> int:
        """Numero de passos simulados."""
        return self.probs.shape[0]

    @property
    def num_vertices(self) -> int:
        """Numero de vertices do grafo simulado."""
        return self.probs.shape[1]

    @property
    def peak_step(self) -> int:
        """Passo em que a deteccao foi maxima."""
        return int(np.argmax(self.det_times))

    @property
    def peak_detection(self) -> float:
        """Maior probabilidade de deteccao alcancada."""
        return float(self.det_times.max())

    # ------------------------------------------------------------------
    # Estatisticas / IO
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Dicionario com estatisticas-chave da simulacao."""
        from biblioteca_qw.analysis import summary as _summary
        s = _summary(self.probs, self.det_times)
        s["topology"] = self.topology
        return s

    def print_summary(self) -> None:
        """Imprime um resumo formatado da simulacao."""
        from biblioteca_qw.analysis import print_summary as _print
        print(f"  Topologia        : {self.topology}")
        _print(self.probs, self.det_times)

    def to_csv(
        self,
        probs_path: Union[str, Path] = "probs.csv",
        det_times_path: Union[str, Path] = "det_times.csv",
    ) -> None:
        """Exporta probs e det_times para arquivos CSV."""
        from biblioteca_qw.analysis import to_csv as _to_csv
        _to_csv(self.probs, self.det_times, probs_path, det_times_path)

    def to_json(
        self,
        probs_path: Union[str, Path] = "probs.json",
        det_times_path: Union[str, Path] = "det_times.json",
    ) -> None:
        """Exporta probs e det_times para arquivos JSON."""
        from biblioteca_qw.analysis import to_json as _to_json
        _to_json(self.probs, self.det_times, probs_path, det_times_path)

    # ------------------------------------------------------------------
    # Visualizacao (matplotlib opcional)
    # ------------------------------------------------------------------

    def plot_detection(
        self,
        ax=None,
        show: bool = True,
        title: Optional[str] = None,
    ):
        """Plota probabilidade de deteccao acumulada vs. passo.

        Requer ``matplotlib`` (extra ``[viz]``).
        """
        plt = _require_matplotlib()
        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(np.arange(self.steps), self.det_times, lw=1.5)
        ax.axvline(self.peak_step, color="red", linestyle="--", alpha=0.4,
                   label=f"pico no passo {self.peak_step}")
        ax.set_xlabel("Passo t")
        ax.set_ylabel("P(deteccao)")
        ax.set_title(title or f"Deteccao - {self.topology}")
        ax.grid(alpha=0.3)
        ax.legend()
        if created_fig and show:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_heatmap(self, ax=None, show: bool = True, cmap: str = "viridis"):
        """Plota um heatmap das probabilidades por vertice ao longo do tempo.

        Requer ``matplotlib`` (extra ``[viz]``).
        """
        plt = _require_matplotlib()
        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=(10, 4))

        im = ax.imshow(
            self.probs.T,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            interpolation="nearest",
        )
        ax.set_xlabel("Passo t")
        ax.set_ylabel("Vertice")
        ax.set_title(f"Distribuicao de probabilidade - {self.topology}")
        if created_fig:
            plt.colorbar(im, ax=ax, label="P(v, t)")
            plt.tight_layout()
            if show:
                plt.show()
        return ax

    # ------------------------------------------------------------------
    # repr amigavel
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"WalkResult(topology={self.topology!r}, "
            f"steps={self.steps}, num_vertices={self.num_vertices}, "
            f"peak_detection={self.peak_detection:.4f} @ step={self.peak_step})"
        )


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib nao esta instalado. Instale com:\n"
            "    pip install 'biblioteca-qw[viz]'\n"
            "ou\n"
            "    pip install matplotlib"
        ) from exc
    return plt
