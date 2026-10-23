#!/usr/bin/env python3
"""Dam break: surge-front position Z = x/a against T = t sqrt(2g/a), compared with the
experiment of Martin & Moyce (1952) for a column of width a and height 2a (n^2 = 2).

The front is the furthest point of the PLIC interface (interface_*.vtp) below y = a/2.
The simulation leads the experiment by a nearly constant time, usually attributed to the
release of the gate in the experiment; the comparison is printed both unshifted and with
the simulated T shifted by --shift (the best-fitting shift is reported too). Writes
<outdir>/front.dat for every run.

Examples
  dam_break_front.py output/dam_break
  dam_break_front.py out/l8 out/l9 --labels "levels 4-8" "levels 4-9" --out front.png

Run vof2d with a dense output interval (e.g. outputEvery=0.002) for a smooth curve.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_field import interface_segments, read_vtk  # noqa: E402

# Martin & Moyce (1952), Phil. Trans. R. Soc. A 244, 312: n^2 = 2, a = 2.25 in, surge front
# (T, Z) as tabulated in the dam-break example of the Lethe CFD code
# (examples/multiphysics/dam-break/dam-break-2d.py).
T_MM = np.array([0, 0.41, 0.84, 1.19, 1.43, 1.63, 1.82, 1.97, 2.2, 2.32, 2.5, 2.64, 2.82, 2.96])
Z_MM = np.array([1, 1.11, 1.23, 1.44, 1.67, 1.89, 2.11, 2.33, 2.56, 2.78, 3, 3.22, 3.44, 3.67])


def read_params(outdir):
    out = {}
    try:
        with open(os.path.join(outdir, "params.log")) as fh:
            for line in fh:
                k, _, v = line.partition("=")
                if v:
                    out[k.strip()] = v.split("#")[0].strip()
    except OSError:
        pass
    return out


def front(outdir):
    """(T, Z) of every frame of a dam_break run."""
    prm = read_params(outdir)
    a = float(prm.get("L0", 5 * 0.05715)) / 5.0   # the case box is 5a wide
    g = -float(prm.get("gy", -9.81))
    T, Z = [], []
    for p in sorted(glob.glob(os.path.join(outdir, "interface_*.vtp"))):
        v = read_vtk(p)
        seg = interface_segments(v).reshape(-1, 2)
        near = seg[seg[:, 1] < 0.5 * a]
        if len(near) == 0:
            continue
        T.append(v["time"] * np.sqrt(2 * g / a))
        Z.append(near[:, 0].max() / a)
    return np.array(T), np.array(Z)


def mismatch(T, Z, shift):
    """Simulated minus measured Z at the measured times, simulation shifted by `shift`."""
    m = T_MM > 0
    return np.interp(T_MM[m] - shift, T, Z) - Z_MM[m]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdirs", nargs="+", help="vof2d output directories of dam_break runs")
    ap.add_argument("--labels", nargs="+", default=None, help="legend labels (default: directory names)")
    ap.add_argument("--shift", type=float, default=0.175, help="shift of the simulated T (gate release)")
    ap.add_argument("--out", default=None, help="output PNG (default: front.png in the first directory)")
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args()
    labels = a.labels or [os.path.basename(os.path.normpath(d)) for d in a.outdirs]

    runs = []
    for d, lab in zip(a.outdirs, labels):
        T, Z = front(d)
        np.savetxt(os.path.join(d, "front.dat"), np.c_[T, Z], header="T = t sqrt(2g/a)   Z = x_front / a")
        shifts = np.arange(-0.1, 0.5, 0.005)
        best = shifts[int(np.argmin([np.sqrt(np.mean(mismatch(T, Z, s) ** 2)) for s in shifts]))]
        print(f"{lab}: {len(T)} frames, T up to {T[-1]:.2f}")
        for s, name in ((0.0, "unshifted"), (a.shift, f"shift {a.shift}"), (best, f"best shift {best:.3f}")):
            e = mismatch(T, Z, s)
            print(f"  {name:18s} rms dZ = {np.sqrt(np.mean(e ** 2)):.3f}  max |dZ| = {np.abs(e).max():.3f}  "
                  f"mean dZ = {e.mean():+.3f}")
        late = (T > 1.5) & (Z < 4.8)
        if late.sum() > 2:
            print(f"  front speed dZ/dT for T > 1.5: {np.polyfit(T[late], Z[late], 1)[0]:.3f} "
                  f"(experiment {np.polyfit(T_MM[T_MM >= 1.5], Z_MM[T_MM >= 1.5], 1)[0]:.3f})")
        runs.append((lab, T, Z))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots in fixed order
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=a.dpi)
    fig.patch.set_facecolor(surf)
    ax.set_facecolor(surf)
    lab, T, Z = runs[-1]
    ax.plot(T, Z, color=colors[(len(runs) - 1) % 4], lw=1.2, ls=(0, (4, 3)), label=f"{lab}, unshifted")
    ax.annotate("unshifted", (1.9, np.interp(1.9, T, Z)), xytext=(-56, 6), textcoords="offset points",
                color=ink2, fontsize=8)
    for k, (lab, T, Z) in enumerate(runs):
        ax.plot(T + a.shift, Z, color=colors[k % 4], lw=2, label=f"{lab}, T + {a.shift}")
    ax.plot(T_MM, Z_MM, ls="none", marker="o", ms=6, mfc=ink, mec=surf, mew=1.5,
            label="Martin & Moyce (1952), n² = 2", zorder=5)
    ax.annotate("experiment", (T_MM[-1], Z_MM[-1]), xytext=(8, -3), textcoords="offset points", color=ink2, fontsize=8)
    ax.set_xlim(0, 3.6)
    ax.set_ylim(0.9, 4.6)
    ax.set_xlabel("T = t √(2g/a)", color=ink2)
    ax.set_ylabel("surge front Z = x / a", color=ink2)
    ax.set_title("Dam break: surge front vs Martin & Moyce (1952)", color=ink, fontsize=11, loc="left")
    ax.grid(color=grid, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(grid)
    ax.tick_params(colors=ink2, labelsize=8)
    leg = ax.legend(loc="upper left", fontsize=8, frameon=False)
    for t in leg.get_texts():
        t.set_color(ink)
    ax.text(0.99, 0.03, f"simulated T shifted by {a.shift} for the gate release", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color=ink2)
    fig.tight_layout()
    out = a.out or os.path.join(a.outdirs[0], "front.png")
    fig.savefig(out, facecolor=surf)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
