#!/usr/bin/env python3
"""Create visualizations for the 3D torus transmission diagnostic output."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.collections import LineCollection
import numpy as np


TORUS_CX = 0.07
TORUS_CY = -0.04
TORUS_CZ = 0.03
TORUS_MAJOR_RADIUS = 0.42
TORUS_MINOR_RADIUS = 0.18


def level_key(path: Path) -> int:
    match = re.search(r"N(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def load_csv(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        raise RuntimeError(f"empty CSV: {path}")
    return np.atleast_1d(data)


def latest_level(out_dir: Path) -> int:
    candidates = sorted(out_dir.glob("torus_solution_zslice_N*.csv"), key=level_key)
    if not candidates:
        raise RuntimeError(f"no torus z-slice CSVs found in {out_dir}")
    return level_key(candidates[-1])


def set_equal_2d(ax) -> None:
    ax.set_aspect("equal", adjustable="box")


def projected_edges(coords: np.ndarray, tri: np.ndarray) -> np.ndarray:
    edges = np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return np.stack((coords[edges[:, 0]], coords[edges[:, 1]]), axis=1)


def plot_geometry(out_dir: Path, level: int) -> Path:
    tag = f"N{level:04d}"
    points = load_csv(out_dir / f"torus_surface_points_{tag}.csv")
    panels = load_csv(out_dir / f"torus_surface_panels_{tag}.csv")

    xyz = np.column_stack((points["x"], points["y"], points["z"]))
    tri = np.column_stack((
        panels["v0"].astype(int),
        panels["v1"].astype(int),
        panels["v2"].astype(int),
    ))
    value = points["u_avg"]
    cmap = plt.get_cmap("viridis")
    norm = colors.Normalize(vmin=float(value.min()), vmax=float(value.max()))

    projections = [
        ("isometric", np.column_stack((xyz[:, 0] - 0.55 * xyz[:, 1],
                                       xyz[:, 2] + 0.35 * xyz[:, 1])), "x - 0.55 y", "z + 0.35 y"),
        ("xy projection", xyz[:, [0, 1]], "x", "y"),
        ("xz projection", xyz[:, [0, 2]], "x", "z"),
        ("yz projection", xyz[:, [1, 2]], "y", "z"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.4), constrained_layout=True)
    for ax, (title, coords, xlabel, ylabel) in zip(axes.flat, projections):
        line_collection = LineCollection(
            projected_edges(coords, tri),
            colors=(0.0, 0.0, 0.0, 0.14),
            linewidths=0.08,
        )
        ax.add_collection(line_collection)
        scat = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=value,
            s=2.0,
            cmap=cmap,
            norm=norm,
            linewidths=0,
            rasterized=True,
        )
        ax.autoscale()
        set_equal_2d(ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    fig.suptitle(f"P2 torus interface geometry, {tag}")
    fig.colorbar(scat, ax=axes.ravel().tolist(), shrink=0.80, label="interface u_avg")

    out_path = out_dir / f"torus_geometry_{tag}.png"
    fig.savefig(out_path, dpi=190)
    plt.close(fig)
    return out_path


def load_slice(path: Path) -> dict[str, np.ndarray | list[float] | float]:
    data = load_csv(path)
    ii = data["i"].astype(int)
    jj = data["j"].astype(int)
    isamp = np.unique(ii)
    jsamp = np.unique(jj)
    imap = {int(v): k for k, v in enumerate(isamp)}
    jmap = {int(v): k for k, v in enumerate(jsamp)}
    shape = (len(jsamp), len(isamp))

    def field(name: str) -> np.ndarray:
        values = np.full(shape, np.nan, dtype=float)
        for value, i, j in zip(data[name], ii, jj):
            values[jmap[int(j)], imap[int(i)]] = value
        return values

    return {
        "extent": [
            float(data["x"].min()),
            float(data["x"].max()),
            float(data["y"].min()),
            float(data["y"].max()),
        ],
        "z": float(data["z"][0]),
        "u_bulk": field("u_bulk"),
        "u_exact": field("u_exact"),
        "abs_error": field("abs_error"),
        "label": field("label"),
    }


def overlay_torus_cross_section(ax, z: float) -> None:
    dz = z - TORUS_CZ
    if abs(dz) >= TORUS_MINOR_RADIUS:
        return
    minor_slice = np.sqrt(TORUS_MINOR_RADIUS**2 - dz**2)
    theta = np.linspace(0.0, 2.0 * np.pi, 512)
    for radius in (TORUS_MAJOR_RADIUS - minor_slice,
                   TORUS_MAJOR_RADIUS + minor_slice):
        if radius <= 0.0:
            continue
        ax.plot(
            TORUS_CX + radius * np.cos(theta),
            TORUS_CY + radius * np.sin(theta),
            color="white",
            linewidth=1.5,
        )
        ax.plot(
            TORUS_CX + radius * np.cos(theta),
            TORUS_CY + radius * np.sin(theta),
            color="black",
            linewidth=0.55,
            alpha=0.85,
        )


def plot_solution_slice(out_dir: Path, level: int) -> Path:
    tag = f"N{level:04d}"
    grid = load_slice(out_dir / f"torus_solution_zslice_{tag}.csv")
    extent = grid["extent"]
    z = float(grid["z"])
    err = np.maximum(grid["abs_error"], 1.0e-15)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), constrained_layout=True)
    panels = [
        ("numerical solution u", grid["u_bulk"], "viridis", "u_num"),
        ("log10 absolute error", np.log10(err), "magma", "log10 |error|"),
        ("domain label", grid["label"], "tab10", "label"),
    ]

    for ax, (title, values, cmap, cbar_label) in zip(axes, panels):
        image = ax.imshow(
            values,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=cmap,
            aspect="equal",
        )
        overlay_torus_cross_section(ax, z)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, shrink=0.82, label=cbar_label)

    fig.suptitle(f"Torus transmission z-slice, {tag}, z={z:.5f}")
    out_path = out_dir / f"torus_solution_zslice_{tag}.png"
    fig.savefig(out_path, dpi=190)
    plt.close(fig)
    return out_path


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(
            "usage: visualize_transmission_torus_3d.py OUTPUT_DIR [N]",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(argv[1])
    if not out_dir.is_dir():
        print(f"output directory not found: {out_dir}", file=sys.stderr)
        return 2

    level = int(argv[2]) if len(argv) == 3 else latest_level(out_dir)
    geom_path = plot_geometry(out_dir, level)
    solution_path = plot_solution_slice(out_dir, level)
    print(geom_path)
    print(solution_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
