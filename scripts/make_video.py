#!/usr/bin/env python3
"""Render a vof2d output directory into an MP4 showing the AMR grid and the flow.

Each frame has three map panels and two time-series panels:
  1. AMR grid: leaf cells coloured by refinement level, cell edges, PLIC interface
  2. volume fraction f with velocity arrows (sampled on a regular grid)
  3. a third field (default: vorticity; e.g. --field3 speed --log for spurious currents)
  4. a diagnostic from diagnostics.dat vs time (default: rise velocity vc2)
  5. number of leaf cells vs time
Needs numpy, matplotlib and the ffmpeg executable.

Examples
  make_video.py output/video_rising_bubble
  make_video.py output/video_static_drop --field3 speed --log --diag umax --diag-log
  make_video.py output/breaking_wave_deike2015 --wave-energy

--wave-energy (periodic wave, box length = one wavelength): the top time-series panel
shows the total, kinetic, gravitational and surface energies E/E0 against t/T, computed
from the frames and written to <outdir>/energy.dat, with literature values for the
plunging breaker eps = 0.55, Bo = 1000, Re = 40000 of Deike, Popinet & Melville,
JFM 769 (2015): ~70% of the energy dissipated after 4 periods, breaking-stage decay
E = E(T) exp(-0.22 t) for t/T in [1, 2] (t in sqrt(lambda/g)); Chen et al. (1999) ~80%
and Iafrati (2011) ~60% after 4 periods for similar plunging breakers (as quoted there);
and the linear viscous decay exp(-4 nu k^2 t).
"""
import argparse
import glob
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_field import default_interface_file, interface_segments, leaf_quads, rasterize, read_vtk  # noqa: E402

# Hysing et al. (2009) TP2D reference points shown on the diagnostic panel
REFERENCE = {
    ("rising_bubble", "vc2"): (0.9214, 0.2417, "TP2D max"),
    ("rising_bubble2", "vc2"): (0.7317, 0.2502, "max (ref.)"),
}


def load_frame(path):
    vtu = read_vtk(path)
    cd = vtu["sections"]["CellData"]
    itf = default_interface_file(path)
    segs = interface_segments(read_vtk(itf)) if itf else np.zeros((0, 2, 2))
    return vtu["time"], leaf_quads(vtu), cd, segs


def field3_values(cd, name):
    if name == "speed":
        return np.hypot(cd["u"], cd["v"])
    return np.asarray(cd[name], dtype=float)


def load_diagnostics(path):
    with open(path) as fh:
        hdr = fh.readline().lstrip("#").split()
    d = np.loadtxt(path, ndmin=2)
    return {k: d[:, i] for i, k in enumerate(hdr)}


def read_params(outdir):
    """All 'key = value' entries of params.log as strings."""
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


def wave_energies(path, prm):
    """(t, Ek, Eg, Es) of one frame: kinetic, gravitational (zero for a flat surface at
    the same water volume) and surface energy sigma (L - Lx), both phases."""
    t, quads, cd, segs = load_frame(path)
    lo, hi = quads.min(axis=1), quads.max(axis=1)
    area = (hi[:, 0] - lo[:, 0]) * (hi[:, 1] - lo[:, 1])
    yc = 0.5 * (lo[:, 1] + hi[:, 1])
    f = np.asarray(cd["f"], float)
    rho1, rho2 = float(prm["rho1"]), float(prm["rho2"])
    g = -float(prm.get("gy", "-1"))
    rho = rho1 * f + (1 - f) * rho2
    ek = 0.5 * np.sum(rho * (np.asarray(cd["u"], float) ** 2 + np.asarray(cd["v"], float) ** 2) * area)
    x0, y0 = quads[..., 0].min(), quads[..., 1].min()
    lx, y1 = quads[..., 0].max() - x0, quads[..., 1].max()
    ys = y0 + np.sum(f * area) / lx                 # flat surface with the same water volume
    flat = 0.5 * lx * g * (rho1 * (ys ** 2 - y0 ** 2) + rho2 * (y1 ** 2 - ys ** 2))
    eg = g * np.sum(rho * yc * area) - flat
    length = np.sum(np.hypot(*(segs[:, 1] - segs[:, 0]).T)) if len(segs) else lx
    es = float(prm.get("sigma", "0")) * (length - lx)
    return t, ek, eg, es


def read_case(outdir):
    try:
        with open(os.path.join(outdir, "params.log")) as fh:
            for line in fh:
                k, _, v = line.partition("=")
                if k.strip() == "caseName":
                    return v.split("#")[0].strip()
    except OSError:
        pass
    return ""


def energy_panel(ax, en, t):
    """E/E0 components vs t/T with the Deike et al. (2015) references."""
    T, tt = en["T"], en["t"] / en["T"]
    now = tt <= t / T + 1e-9
    for key, col, lab in (("E", "k", "total E"), ("Ek", "C0", "kinetic"), ("Eg", "C2", "gravitational"),
                          ("Es", "C3", "surface")):
        y = en[key] / en["E0"]
        ax.plot(tt, y, color=col, lw=0.8, alpha=0.25)
        ax.plot(tt[now], y[now], color=col, lw=1.6 if key == "E" else 1.1, label=lab)
    tl = np.linspace(0, 4, 50)
    ax.plot(tl, np.exp(-4 * en["nu"] * en["k"] ** 2 * tl * T), ":", color="0.4", lw=1, label="linear viscous decay")
    if en.get("E1") is not None:
        tb = np.linspace(1, 2, 20)
        ax.plot(tb, en["E1"] * np.exp(-0.22 * np.sqrt(2 * np.pi) * (tb - 1)), "--", color="C1", lw=1.5,
                label="Deike+15 breaking fit")
    ax.errorbar([4], [0.30], yerr=[[0.05], [0.05]], fmt="D", color="C1", ms=5, capsize=2, label="Deike+15: ~70% lost")
    ax.plot([4], [0.20], "v", color="C4", ms=5, label="Chen+99: ~80% lost")
    ax.plot([4], [0.40], "^", color="C5", ms=5, label="Iafrati 11: ~60% lost")
    ax.set_xlim(0, 4.15)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("t / T", fontsize=8)
    ax.set_title("wave energy E/E0 vs Deike et al. 2015", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=5.5, loc="upper right", ncol=2)


def render(job):
    (k, path, out, cfg) = job
    t, quads, cd, segs = load_frame(path)
    lo, hi = quads.min(axis=(0, 1)), quads.max(axis=(0, 1))
    if cfg["xlim"]:
        lo[0], hi[0] = cfg["xlim"]
    if cfg["ylim"]:
        lo[1], hi[1] = cfg["ylim"]
    aspect = (hi[1] - lo[1]) / (hi[0] - lo[0])
    H = cfg["height"]
    pw = H / aspect * 0.82  # panel width in inches (leave room for titles/colorbars)
    fig = plt.figure(figsize=(3 * pw + 4.6, H), dpi=cfg["dpi"])
    gs = fig.add_gridspec(2, 4, width_ratios=[pw, pw, pw, 4.0], left=0.03, right=0.985,
                          bottom=0.13, top=0.88, wspace=0.28, hspace=0.45)
    axes = [fig.add_subplot(gs[:, i]) for i in range(3)]
    ad1, ad2 = fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[1, 3])
    levels = cd["level"].astype(int)
    nleaf = len(levels)

    def frame_axes(ax, title):
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=7)

    def interface(ax, color="k", lw=1.2):
        if len(segs):
            ax.add_collection(LineCollection(segs, colors=color, linewidths=lw))

    def colorbar(im, ax, label):
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045, pad=0.06)
        cb.ax.tick_params(labelsize=7)
        cb.set_label(label, fontsize=8)
        return cb

    # 1. AMR grid coloured by level
    lmin, lmax = cfg["lmin"], cfg["lmax"]
    cmap = plt.get_cmap("viridis", lmax - lmin + 1)
    norm = BoundaryNorm(np.arange(lmin - 0.5, lmax + 1.5), cmap.N)
    ax = axes[0]
    pc = PolyCollection(quads, array=levels, cmap=cmap, norm=norm, edgecolors="k", linewidths=0.15)
    ax.add_collection(pc)
    interface(ax, "r", 1.4)
    frame_axes(ax, f"AMR grid ({nleaf} leaves)")
    cb = colorbar(pc, ax, "refinement level")
    cb.set_ticks(range(lmin, lmax + 1))

    # 2. volume fraction + velocity arrows
    ax = axes[1]
    img, ext = rasterize(quads, np.asarray(cd["f"], float))
    # light blues so the dark velocity arrows stay visible inside phase 1
    fmap = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.0, 0.5, 256)))
    im = ax.imshow(img, origin="lower", extent=ext, cmap=fmap, vmin=0, vmax=1, interpolation="nearest")
    u_img, _ = rasterize(quads, np.asarray(cd["u"], float))
    v_img, _ = rasterize(quads, np.asarray(cd["v"], float))
    ny, nx = img.shape
    step = max(1, int(round(nx / cfg["arrows"])))
    jj, ii = np.mgrid[step // 2:ny:step, step // 2:nx:step]
    X = ext[0] + (ii + 0.5) * (ext[1] - ext[0]) / nx
    Y = ext[2] + (jj + 0.5) * (ext[3] - ext[2]) / ny
    U, V = u_img[jj, ii], v_img[jj, ii]
    if cfg["normalize_arrows"]:
        mag = np.hypot(U, V)
        mag[mag == 0] = 1
        U, V = U / mag, V / mag
        ax.quiver(X, Y, U, V, color="0.15", scale=cfg["arrows"] * 1.4, width=0.004)
    else:
        ax.quiver(X, Y, U, V, color="0.15", scale=cfg["arrow_scale"], scale_units="width", width=0.004)
    interface(ax, "r", 1.2)
    frame_axes(ax, "volume fraction f, velocity" + (" (direction)" if cfg["normalize_arrows"] else ""))
    colorbar(im, ax, "f")

    # 3. third field
    ax = axes[2]
    vals = field3_values(cd, cfg["field3"])
    img, ext = rasterize(quads, vals)
    if cfg["log"]:
        img = np.where(img > 0, img, np.nan)
        nrm = LogNorm(*cfg["clim3"])
        cm3 = "magma"
    else:
        nrm = Normalize(*cfg["clim3"])
        cm3 = "RdBu_r" if cfg["clim3"][0] < 0 else "magma"
    im = ax.imshow(img, origin="lower", extent=ext, cmap=cm3, norm=nrm, interpolation="nearest")
    interface(ax, "k" if cm3 == "RdBu_r" else "w", 1.0)
    frame_axes(ax, cfg["field3_label"])
    colorbar(im, ax, cfg["field3_label"])

    # 4./5. time series with a moving marker
    dg = cfg["diag"]
    td = dg["t"]
    sel = td <= t + 1e-12
    panels = ((ad1, cfg["diag_key"], cfg["diag_log"], cfg["diag_label"]), (ad2, "nLeaves", False, "leaf cells"))
    if cfg.get("energy"):
        energy_panel(ad1, cfg["energy"], t)
        panels = panels[1:]
    tscale = cfg["energy"]["T"] if cfg.get("energy") else 1.0   # energy mode: t/T axis
    for axd, key, logy, label in panels:
        y = dg[key]
        axd.plot(td / tscale, y, color="0.8", lw=1)
        axd.plot(td[sel] / tscale, y[sel], color="C0", lw=1.5)
        if sel.any():
            axd.plot(td[sel][-1] / tscale, y[sel][-1], "o", color="C3", ms=4)
        ref = REFERENCE.get((cfg["case"], key))
        if ref:
            axd.plot(ref[0], ref[1], "kD", ms=5, mfc="none", label=ref[2])
            axd.legend(fontsize=7, loc="lower right")
        if logy:
            axd.set_yscale("log")
        axd.set_xlim(td[0] / tscale, td[-1] / tscale)
        if cfg.get("energy"):
            axd.set_xlim(0, 4.15)
        axd.set_title(label, fontsize=9)
        axd.set_xlabel("t / T" if cfg.get("energy") else "t", fontsize=8)
        axd.tick_params(labelsize=7)
        axd.grid(alpha=0.3)

    fig.suptitle(f"{cfg['title']}    t = {t:.3f}", fontsize=13)
    fig.savefig(out)
    plt.close(fig)
    return k


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", help="vof2d output directory (fields_*.vtu, interface_*.vtp, diagnostics.dat)")
    ap.add_argument("--out", help="output .mp4 (default: <outdir>/<case>.mp4)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--height", type=float, default=7.0, help="figure height in inches")
    ap.add_argument("--field3", default="vorticity", help="vorticity|speed|p|kappa|...")
    ap.add_argument("--log", action="store_true", help="log colour scale for field 3")
    ap.add_argument("--clim3", type=float, nargs=2, help="colour limits for field 3 (default: from data)")
    ap.add_argument("--diag", default="vc2", help="diagnostics.dat column for the time-series panel")
    ap.add_argument("--diag-log", action="store_true")
    ap.add_argument("--arrows", type=int, default=18, help="arrows across the domain width")
    ap.add_argument("--normalize-arrows", action="store_true", help="show velocity direction only")
    ap.add_argument("--arrow-umax", type=float, help="speed of a full-length arrow (default: max |u| of the run)")
    ap.add_argument("--xlim", type=float, nargs=2)
    ap.add_argument("--ylim", type=float, nargs=2)
    ap.add_argument("--title", help="figure title (default: case name)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--wave-energy", action="store_true", help="energy budget panel vs literature (breaking wave)")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.outdir, "fields_*.vtu")))
    if not files:
        sys.exit(f"no fields_*.vtu in {a.outdir}")
    case = read_case(a.outdir)
    diag = load_diagnostics(os.path.join(a.outdir, "diagnostics.dat"))

    # global colour limits / level range so every frame uses the same scales
    lmin, lmax, vmax, vpos = 99, 0, 0.0, []
    umax = 0.0
    for p in files:
        _, _, cd, _ = load_frame(p)
        lv = cd["level"]
        lmin, lmax = min(lmin, int(lv.min())), max(lmax, int(lv.max()))
        vals = field3_values(cd, a.field3)
        vmax = max(vmax, np.percentile(np.abs(vals), 99.5))
        if a.log:
            pos = vals[vals > 0]
            if pos.size:
                vpos.append(np.percentile(pos, 5))
        umax = max(umax, float(np.hypot(cd["u"], cd["v"]).max()))
    if a.clim3:
        clim3 = tuple(a.clim3)
    elif a.log:
        hi = vmax if vmax > 0 else 1.0
        clim3 = (max(min(vpos) if vpos else hi * 1e-4, hi * 1e-4), hi)
    elif a.field3 in ("vorticity", "u", "v", "p", "kappa"):
        clim3 = (-vmax, vmax)
    else:
        clim3 = (0.0, vmax)
    labels = {"vorticity": "vorticity", "speed": "|u|", "p": "pressure", "kappa": "curvature"}
    diag_labels = {"vc2": "rise velocity of phase 2", "umax": "max |u|", "yc2": "centroid y of phase 2",
                   "circularity": "circularity", "ekin": "kinetic energy"}
    cfg = dict(case=case, title=a.title or case, dpi=a.dpi, height=a.height, lmin=lmin, lmax=lmax,
               field3=a.field3, field3_label=labels.get(a.field3, a.field3) + (" (log)" if a.log else ""),
               log=a.log, clim3=clim3, diag=diag, diag_key=a.diag, diag_log=a.diag_log,
               diag_label=diag_labels.get(a.diag, a.diag), arrows=a.arrows,
               normalize_arrows=a.normalize_arrows,
               arrow_scale=max(a.arrow_umax or umax, 1e-300) * a.arrows * 0.5, xlim=a.xlim, ylim=a.ylim)

    if a.wave_energy:
        prm = read_params(a.outdir)
        rows = np.array([wave_energies(p, prm) for p in files])
        lx = float(prm["L0"]) * int(prm.get("nx0", "1"))
        k = 2 * np.pi / lx
        g = -float(prm.get("gy", "-1"))
        T = 2 * np.pi / np.sqrt(g * k)
        en = dict(t=rows[:, 0], Ek=rows[:, 1], Eg=rows[:, 2], Es=rows[:, 3], E=rows[:, 1:].sum(axis=1), T=T, k=k,
                  nu=float(prm["mu1"]) / float(prm["rho1"]))
        en["E0"] = en["E"][0]
        i1 = np.argmin(np.abs(en["t"] / T - 1.0))
        en["E1"] = en["E"][i1] / en["E0"] if en["t"][-1] / T >= 1.0 else None
        cfg["energy"] = en
        with open(os.path.join(a.outdir, "energy.dat"), "w") as fh:
            fh.write("# t t/T Ek Eg Es E E/E0\n")
            for r, e in zip(rows, en["E"]):
                fh.write("%.6g %.6g %.8g %.8g %.8g %.8g %.6g\n" % (r[0], r[0] / T, r[1], r[2], r[3], e, e / en["E0"]))
        last = en["E"][-1] / en["E0"]
        print("wave energy: E0 = %.5g, E/E0 = %.3f at t/T = %.2f (%.0f%% dissipated)" %
              (en["E0"], last, en["t"][-1] / T, 100 * (1 - last)))

    fdir = os.path.join(a.outdir, "frames")
    os.makedirs(fdir, exist_ok=True)
    jobs = [(k, p, os.path.join(fdir, f"frame_{k:05d}.png"), cfg) for k, p in enumerate(files)]
    with Pool(a.jobs) as pool:
        for n, _ in enumerate(pool.imap_unordered(render, jobs), 1):
            print(f"\rrendered {n}/{len(jobs)} frames", end="", flush=True)
    print()

    out = a.out or os.path.join(a.outdir, f"{case or 'vof2d'}.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i",
           os.path.join(fdir, "frame_%05d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out]
    subprocess.run(cmd, check=True)
    if not a.keep_frames:
        for _, _, f, _ in jobs:
            os.remove(f)
        os.rmdir(fdir)
    print(f"wrote {out} ({len(files)} frames, {len(files) / a.fps:.1f} s at {a.fps} fps)")


if __name__ == "__main__":
    main()
