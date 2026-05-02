#!/usr/bin/env python3
"""Create Python visualizations for the screened 3-fold star BVP test."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def level_key(path: Path) -> int:
    match = re.search(r"N(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def load_grid(path: Path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        raise RuntimeError(f"empty grid CSV: {path}")
    data = np.atleast_1d(data)

    ii = data["i"].astype(int)
    jj = data["j"].astype(int)
    xs = np.unique(data["x"])
    ys = np.unique(data["y"])
    isamp = np.unique(ii)
    jsamp = np.unique(jj)
    imap = {int(v): k for k, v in enumerate(isamp)}
    jmap = {int(v): k for k, v in enumerate(jsamp)}
    shape = (len(jsamp), len(isamp))

    def field(name: str) -> np.ndarray:
        values = np.full(shape, np.nan, dtype=float)
        for row, i, j in zip(data[name], ii, jj):
            values[jmap[int(j)], imap[int(i)]] = row
        return values

    return {
        "extent": [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())],
        "u_bulk": field("u_bulk"),
        "u_exact": field("u_exact"),
        "abs_error": field("abs_error"),
        "label": field("label"),
    }


def plot_convergence(out_dir: Path) -> None:
    csv_path = out_dir / "convergence.csv"
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    data = np.atleast_1d(data)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.loglog(data["N"], data["max_err"], marker="o")
    ax.set_xlabel("N")
    ax.set_ylabel("max-norm error")
    ax.set_title("Screened interior Dirichlet convergence")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(out_dir / "convergence.png", dpi=180)
    plt.close(fig)


def plot_grid(csv_path: Path) -> None:
    grid = load_grid(csv_path)
    n_label = csv_path.stem.split("_")[-1]

    u_bulk = np.where(grid["label"] == 1, grid["u_bulk"], np.nan)
    abs_error = np.where(grid["label"] == 1, grid["abs_error"], np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
    panels = [
        ("u_bulk", u_bulk, "viridis"),
        ("abs_error", abs_error, "magma"),
        ("domain_label", grid["label"], "tab10"),
    ]

    for ax, (title, values, cmap) in zip(axes, panels):
        image = ax.imshow(
            values,
            origin="lower",
            extent=grid["extent"],
            interpolation="nearest",
            cmap=cmap,
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, shrink=0.82)

    fig.suptitle(f"Screened 3-fold star BVP, {n_label}")
    fig.savefig(csv_path.with_suffix(".png"), dpi=180)
    plt.close(fig)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: visualize_screened_star3.py OUTPUT_DIR", file=sys.stderr)
        return 2

    out_dir = Path(argv[1])
    if not out_dir.is_dir():
        print(f"output directory not found: {out_dir}", file=sys.stderr)
        return 2

    plot_convergence(out_dir)
    for csv_path in sorted(out_dir.glob("screened_star3_N*.csv"), key=level_key):
        plot_grid(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
