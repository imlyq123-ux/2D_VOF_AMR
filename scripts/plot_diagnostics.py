#!/usr/bin/env python3
"""Plot vof2d diagnostics (outDir/diagnostics.dat) against time: rise velocity and centroid
of phase 2, circularity, volume error, kinetic energy and max velocity. For the Hysing et
al. (2009) rising-bubble cases the benchmark reference values are overlaid and a table
of the benchmark quantities is printed.

Examples
  plot_diagnostics.py output/rising_bubble/diagnostics.dat
  plot_diagnostics.py l6/diagnostics.dat l7/diagnostics.dat --labels h=1/64 h=1/128 --out conv.png
  plot_diagnostics.py run/diagnostics.dat --case rising_bubble2

The case is read from params.log next to the first diagnostics file unless --case is given.
"""
import argparse
import os
import re
import sys

import numpy as np

# Hysing et al., Int. J. Numer. Meth. Fluids 60 (2009) 1259, tables 4 and 9 (TP2D; for TC2
# also MooNMD, since the groups disagree on the circularity after break-up).
# Entries: (t, value) points, or (None, value) for "value at t = 3".
REFERENCE = {
    "rising_bubble": {
        "TP2D": {"circ_min": (1.9041, 0.9013), "vel_max": [(0.9214, 0.2417)], "yc_end": (3.0, 1.0813)},
        "MooNMD": {"circ_min": (1.9046, 0.9013), "vel_max": [(0.9201, 0.2417)], "yc_end": (3.0, 1.0817)},
    },
    "rising_bubble2": {
        "TP2D": {"circ_min": (2.4004, 0.5869), "vel_max": [(0.7332, 0.2524), (2.0705, 0.2434)],
                 "yc_end": (3.0, 1.1380)},
        "MooNMD": {"circ_min": (3.0000, 0.5144), "vel_max": [(0.7317, 0.2502), (2.0600, 0.2393)],
                   "yc_end": (3.0, 1.1376)},
    },
}

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
REF_STYLE = {"TP2D": dict(marker="D", ms=7, mfc="none", mec=INK, mew=1.5),
             "MooNMD": dict(marker="s", ms=7, mfc="none", mec=INK2, mew=1.2)}


def read_diagnostics(path):
    """Return a dict column name -> numpy array (columns from the '#' header line)."""
    names = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                names = line[1:].split()
                break
    if names is None:
        raise ValueError(f"{path}: no '#' header line")
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[0] == 0:
        raise ValueError(f"{path}: no data")
    if data.shape[1] != len(names):
        raise ValueError(f"{path}: {data.shape[1]} columns but {len(names)} names in the header")
    return {n: data[:, k] for k, n in enumerate(names)}


def case_from_params(diag_path):
    p = os.path.join(os.path.dirname(os.path.abspath(diag_path)), "params.log")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        for line in fh:
            m = re.match(r"\s*caseName\s*=\s*(\S+)", line)
            if m:
                return m.group(1)
    return None


def local_maxima(t, v, split):
    """(t, v) of the maximum of v before and after the time `split`."""
    out = []
    for sel in (t <= split, t > split):
        if sel.any():
            k = np.argmax(np.where(sel, v, -np.inf))
            out.append((t[k], v[k]))
    return out


def benchmark_table(label, d, case):
    t, vc, yc, c = d["t"], d["vc2"], d["yc2"], d["circularity"]
    kmin = np.argmin(np.where(c > 0, c, np.inf))
    rows = [f"{label}:",
            f"  min circularity  {c[kmin]:.4f} at t = {t[kmin]:.4f}"]
    if case == "rising_bubble2":
        for k, (tm, vm) in enumerate(local_maxima(t, vc, 1.5)):
            rows.append(f"  max rise vel. {k + 1}  {vm:.4f} at t = {tm:.4f}")
    else:
        k = np.argmax(vc)
        rows.append(f"  max rise velocity {vc[k]:.4f} at t = {t[k]:.4f}")
    k = np.argmin(np.abs(t - 3.0))
    rows.append(f"  centroid y        {yc[k]:.4f} at t = {t[k]:.4f}")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="diagnostics.dat file(s)")
    ap.add_argument("--labels", nargs="+", default=None, help="legend labels (default: directory names)")
    ap.add_argument("--case", default="auto",
                    help="case for reference values: auto (from params.log), rising_bubble, rising_bubble2, none")
    ap.add_argument("--out", default=None, help="output PNG (default: diagnostics.png next to the first file)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--show", action="store_true", help="also open a window")
    args = ap.parse_args(argv)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [read_diagnostics(f) for f in args.files]
    labels = args.labels or [os.path.basename(os.path.dirname(os.path.abspath(f))) or f for f in args.files]
    if len(labels) != len(runs):
        sys.exit("--labels needs one label per file")
    if len(runs) > len(SERIES):
        sys.exit(f"at most {len(SERIES)} files per plot")
    case = case_from_params(args.files[0]) if args.case == "auto" else args.case
    refs = REFERENCE.get(case or "", {})

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    ax = dict(zip(["vel", "yc", "circ", "vol", "ekin", "umax"], axes.ravel()))
    for k, (d, lab) in enumerate(zip(runs, labels)):
        t = d["t"]
        kw = dict(color=SERIES[k], lw=2, label=lab)
        ax["vel"].plot(t, d["vc2"], **kw)
        ax["yc"].plot(t, d["yc2"], **kw)
        circ = np.where(d["circularity"] > 0, d["circularity"], np.nan)
        ax["circ"].plot(t, circ, **kw)
        small = "vol2" if d["vol2"][0] <= d["vol1"][0] else "vol1"
        v0 = d[small][0]
        ax["vol"].plot(t, (d[small] - v0) / v0 if v0 > 0 else d[small] * 0, **kw)
        ax["ekin"].plot(t, d["ekin"], **kw)
        ax["umax"].plot(t, d["umax"], **kw)

    # reference values (Hysing et al. 2009)
    for name, r in refs.items():
        st = REF_STYLE[name]
        tm, cm = r["circ_min"]
        ax["circ"].plot([tm], [cm], ls="none", label=f"{name} min", **st)
        tv = [p[0] for p in r["vel_max"]]
        vv = [p[1] for p in r["vel_max"]]
        ax["vel"].plot(tv, vv, ls="none", label=f"{name} max", **st)
        te, ye = r["yc_end"]
        ax["yc"].plot([te], [ye], ls="none", label=f"{name} t=3", **st)

    titles = {"vel": "rise velocity vc2 (mean v of phase 2)", "yc": "centroid yc2 of phase 2",
              "circ": "circularity", "vol": "relative volume error of the smaller phase",
              "ekin": "kinetic energy", "umax": "max |u|"}
    for key, a in ax.items():
        a.set_title(titles[key], fontsize=10, color=INK, loc="left")
        a.set_xlabel("t", color=INK2)
        a.grid(True, color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(INK2)
        a.tick_params(colors=INK2, labelsize=8)
    if all((d["umax"] > 0).all() for d in runs):
        ax["umax"].set_yscale("log")
    ax["vol"].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    for key in ("vel", "yc", "circ"):
        h, _ = ax[key].get_legend_handles_labels()
        if len(h) > 1:
            ax[key].legend(fontsize=8, frameon=False)
    if len(runs) > 1:
        for key in ("vol", "ekin", "umax"):
            ax[key].legend(fontsize=8, frameon=False)
    fig.suptitle(f"vof2d diagnostics" + (f" - {case}" if case else ""), color=INK, fontsize=12)
    fig.tight_layout()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.files[0])), "diagnostics.png")
    fig.savefig(out, dpi=args.dpi)
    print(f"wrote {out}")

    if case in REFERENCE:
        for d, lab in zip(runs, labels):
            print(benchmark_table(lab, d, case))
        for name, r in refs.items():
            vm = ", ".join(f"{v:.4f} at t = {t:.4f}" for t, v in r["vel_max"])
            print(f"reference {name}: min circularity {r['circ_min'][1]:.4f} at t = {r['circ_min'][0]:.4f}; "
                  f"max rise velocity {vm}; centroid y(3) = {r['yc_end'][1]:.4f}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
