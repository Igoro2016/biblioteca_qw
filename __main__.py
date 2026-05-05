"""
biblioteca_qw  CLI
==================
Roda uma simulacao DTQW direto da linha de comando.

Exemplos
--------
$ python -m biblioteca_qw torus --L 3 --n 2 --steps 100
$ python -m biblioteca_qw grid --dims 4 4 --steps 50 --marked 0
$ python -m biblioteca_qw hypercube --n 4 --steps 200 --marked 0 15 \\
      --probs probs.csv --det det.csv
$ python -m biblioteca_qw graph --N 5 --edges 0 1 1 2 2 3 3 4 4 0 --steps 50
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from biblioteca_qw import walk
from biblioteca_qw.results import WalkResult


def _parse_edge_pairs(values: List[str]) -> List[tuple]:
    if len(values) % 2 != 0:
        raise SystemExit("--edges precisa de um numero PAR de inteiros.")
    return [(int(values[i]), int(values[i + 1])) for i in range(0, len(values), 2)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m biblioteca_qw",
        description="Simulacao de caminhada quantica discreta (DTQW).",
    )
    sub = p.add_subparsers(dest="topology", required=True)

    # ---------------------- torus ----------------------
    pt = sub.add_parser("torus", help="Toroidal T(L, n).",
                        aliases=["toroidal", "toro"])
    pt.add_argument("--L", type=int, required=True, help="vertices por dimensao")
    pt.add_argument("--n", type=int, required=True, help="numero de dimensoes")
    pt.add_argument("--num-selfloop", type=int, default=0)

    # ---------------------- grid -----------------------
    pg = sub.add_parser("grid", help="Grade finita.", aliases=["grade"])
    pg.add_argument("--dims", nargs="+", type=int, required=True)

    # ---------------------- graph ----------------------
    pgr = sub.add_parser("graph", help="Grafo arbitrario.", aliases=["grafo"])
    pgr.add_argument("--N", type=int, required=True)
    pgr.add_argument("--edges", nargs="+", required=True,
                     help="lista plana de inteiros: u0 v0 u1 v1 ...")

    # ---------------------- hypercube ------------------
    ph = sub.add_parser("hypercube", help="Hipercubo Q_n.", aliases=["hipercubo"])
    ph.add_argument("--n", type=int, required=True)
    ph.add_argument("--num-selfloop", type=int, default=0)

    # opcoes comuns para todos os subcomandos
    for s in (pt, pg, pgr, ph):
        s.add_argument("--steps", type=int, required=True)
        s.add_argument("--marked", nargs="+", type=int, default=None)
        s.add_argument("--weight", type=float, default=1.0)
        s.add_argument("--db-url", default=None)
        s.add_argument("--probs", default=None, help="caminho para salvar probs CSV")
        s.add_argument("--det", default=None, help="caminho para salvar det_times CSV")
        s.add_argument("--plot", action="store_true",
                       help="exibe grafico de deteccao (precisa matplotlib)")
        s.add_argument("--summary", action="store_true",
                       help="imprime resumo da simulacao")

    args = p.parse_args(argv)

    kwargs = dict(
        steps=args.steps,
        marked=args.marked,
        weight_value=args.weight,
        db_url=args.db_url,
    )

    name = args.topology
    if name in {"torus", "toroidal", "toro"}:
        result = walk("torus", L=args.L, n=args.n,
                      num_selfloop=args.num_selfloop, **kwargs)
    elif name in {"grid", "grade"}:
        result = walk("grid", dims=args.dims, **kwargs)
    elif name in {"graph", "grafo"}:
        edges = _parse_edge_pairs(args.edges)
        result = walk("graph", N=args.N, edges=edges, **kwargs)
    elif name in {"hypercube", "hipercubo"}:
        result = walk("hypercube", n=args.n,
                      num_selfloop=args.num_selfloop, **kwargs)
    else:  # pragma: no cover
        p.error(f"topologia desconhecida: {name}")
        return 2

    assert isinstance(result, WalkResult)

    if args.summary or (not args.probs and not args.det and not args.plot):
        result.print_summary()
    if args.probs and args.det:
        result.to_csv(args.probs, args.det)
        print(f"[ok] CSV salvo em {args.probs} e {args.det}")
    if args.plot:
        result.plot_detection()

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
