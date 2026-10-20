#!/usr/bin/env python3
"""Plot a cell field of a vof2d .vtu file (leaf quads), optionally with the AMR cell edges
and the PLIC interface from the matching .vtp file. Needs only numpy and matplotlib.

Examples
  plot_field.py output/rising_bubble/fields_00030.vtu                 # f, interface auto-detected
  plot_field.py fields_00010.vtu --field vorticity --edges --out w.png
  plot_field.py fields_00010.vtu --field p --xlim 0.2 0.8 --ylim 0.6 1.4 --mode poly

The reader understands the VTK XML files written by vof2d (appended raw binary,
header_type UInt32/UInt64) and plain ascii DataArrays.
"""
import argparse
import os
import re
import sys

import numpy as np

VTK_TYPES = {
    "Int8": np.int8, "UInt8": np.uint8, "Int16": np.int16, "UInt16": np.uint16,
    "Int32": np.int32, "UInt32": np.uint32, "Int64": np.int64, "UInt64": np.uint64,
    "Float32": np.float32, "Float64": np.float64,
}

_ATTR = re.compile(rb'(\w+)="([^"]*)"')


def _attrs(tag):
    return {k.decode(): v.decode() for k, v in _ATTR.findall(tag)}


def read_vtk(path):
    """Read a VTK XML file (.vtu/.vtp) written by vof2d.

    Returns a dict with 'piece' (attributes of the Piece element), 'time' (or None) and
    'sections': {section name: {array name: numpy array}}, where section is one of
    Points, Cells, Lines, Polys, CellData, PointData, FieldData. Arrays with several
    components are returned with shape (n, ncomp)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    app = raw.find(b"<AppendedData")
    xml = raw if app < 0 else raw[:app]
    head = re.search(rb"<VTKFile[^>]*>", xml)
    if head is None:
        raise ValueError(f"{path}: not a VTK XML file")
    vattr = _attrs(head.group(0))
    order = "<" if vattr.get("byte_order", "LittleEndian") == "LittleEndian" else ">"
    htype = np.dtype(VTK_TYPES[vattr.get("header_type", "UInt32")]).newbyteorder(order)
    base = None
    if app >= 0:
        tag_end = raw.find(b">", app)
        tag = raw[app:tag_end + 1]
        if _attrs(tag).get("encoding", "raw") != "raw":
            raise ValueError(f"{path}: only raw appended data is supported")
        base = raw.find(b"_", tag_end) + 1

    piece_m = re.search(rb"<Piece[^>]*>", xml)
    out = {"piece": _attrs(piece_m.group(0)) if piece_m else {}, "time": None, "sections": {}}
    # walk the XML part: track the enclosing section of each DataArray
    token = re.compile(rb"<(/?)(\w+)([^>]*?)(/?)>")
    section = None
    pos = 0
    while True:
        m = token.search(xml, pos)
        if m is None:
            break
        closing, name, rest, selfclose = m.group(1), m.group(2).decode(), m.group(3), m.group(4)
        pos = m.end()
        if name in ("Points", "Cells", "Lines", "Polys", "Verts", "CellData", "PointData", "FieldData"):
            section = None if closing else name
            continue
        if name != "DataArray" or closing:
            continue
        a = _attrs(rest)
        dtype = np.dtype(VTK_TYPES[a["type"]]).newbyteorder(order)
        ncomp = int(a.get("NumberOfComponents", "1"))
        fmt = a.get("format", "ascii")
        if fmt == "appended":
            if base is None:
                raise ValueError(f"{path}: appended array without AppendedData")
            off = base + int(a["offset"])
            nbytes = int(np.frombuffer(raw, dtype=htype, count=1, offset=off)[0])
            arr = np.frombuffer(raw, dtype=dtype, count=nbytes // dtype.itemsize, offset=off + htype.itemsize)
        elif fmt == "ascii":
            if selfclose:
                arr = np.zeros(0, dtype=dtype)
            else:
                end = xml.find(b"</DataArray>", pos)
                arr = np.array(xml[pos:end].split(), dtype=dtype)
                pos = end + len(b"</DataArray>")
        else:
            raise ValueError(f"{path}: DataArray format '{fmt}' not supported")
        arr = arr.astype(dtype.newbyteorder("="))
        if ncomp > 1:
            arr = arr.reshape(-1, ncomp)
        out["sections"].setdefault(section or "", {})[a.get("Name", "")] = arr
        if section == "FieldData" and a.get("Name") == "TimeValue" and arr.size:
            out["time"] = float(arr[0])
    return out


def leaf_quads(vtu):
    """(ncells, 4, 2) array of quad corner coordinates."""
    s = vtu["sections"]
    pts = s["Points"]["Points"][:, :2]
    conn, offs = s["Cells"]["connectivity"], s["Cells"]["offsets"]
    if not np.all(np.diff(np.concatenate(([0], offs))) == 4):
        raise ValueError("only quad cells are supported")
    return pts[conn.reshape(-1, 4)]


def interface_segments(vtp):
    """(nseg, 2, 2) array of segment end points."""
    s = vtp["sections"]
    pts = s["Points"]["Points"]
    if pts.size == 0:
        return np.zeros((0, 2, 2))
    conn = s["Lines"]["connectivity"]
    return pts[:, :2][conn.reshape(-1, 2)]


def rasterize(quads, values):
    """Paint each quad into an image on the finest lattice. Returns (img, extent)."""
    lo = quads.min(axis=1)
    size = quads[:, 2, 0] - quads[:, 0, 0]
    h = size.min()
    x0, y0 = lo[:, 0].min(), lo[:, 1].min()
    I = np.rint((lo[:, 0] - x0) / h).astype(np.int64)
    J = np.rint((lo[:, 1] - y0) / h).astype(np.int64)
    n = np.rint(size / h).astype(np.int64)
    nx, ny = int((I + n).max()), int((J + n).max())
    if nx * ny > 64e6:
        raise ValueError(f"image of {nx}x{ny} pixels too large; use --mode poly or zoom with --xlim/--ylim")
    img = np.full((ny, nx), np.nan)
    for s in np.unique(n):
        sel = n == s
        di, dj = np.meshgrid(np.arange(s), np.arange(s))
        jj = (J[sel][:, None, None] + dj[None]).ravel()
        ii = (I[sel][:, None, None] + di[None]).ravel()
        img[jj, ii] = np.repeat(values[sel], s * s)
    return img, (x0, x0 + nx * h, y0, y0 + ny * h)


def default_interface_file(vtu_path):
    d, b = os.path.split(vtu_path)
    cand = re.sub(r"^fields_", "interface_", b)
    cand = os.path.join(d, os.path.splitext(cand)[0] + ".vtp")
    return cand if cand != vtu_path and os.path.exists(cand) else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vtu", help="fields_XXXXX.vtu written by vof2d")
    ap.add_argument("--field", default="f", help="cell field to plot (default f); 'list' lists the fields")
    ap.add_argument("--vtp", default="auto",
                    help="interface .vtp file ('auto': matching interface_XXXXX.vtp if present, 'none': off)")
    ap.add_argument("--edges", action="store_true", help="draw the AMR cell edges")
    ap.add_argument("--mode", choices=["auto", "poly", "image"], default="auto",
                    help="poly: one polygon per cell; image: rasterize on the finest lattice "
                         "(fast for large meshes); auto: image above 40000 cells")
    ap.add_argument("--cmap", default=None, help="matplotlib colormap (default depends on the field)")
    ap.add_argument("--clim", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    ap.add_argument("--xlim", nargs=2, type=float, default=None, metavar=("X0", "X1"))
    ap.add_argument("--ylim", nargs=2, type=float, default=None, metavar=("Y0", "Y1"))
    ap.add_argument("--title", default=None)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--width", type=float, default=6.0, help="figure width in inches")
    ap.add_argument("--out", default=None, help="output PNG (default: <vtu>_<field>.png)")
    ap.add_argument("--show", action="store_true", help="also open a window")
    args = ap.parse_args(argv)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    vtu = read_vtk(args.vtu)
    cdata = vtu["sections"].get("CellData", {})
    if args.field == "list":
        print(" ".join(cdata))
        return 0
    if args.field not in cdata:
        sys.exit(f"field '{args.field}' not in {args.vtu}; available: {' '.join(cdata)}")
    vals = np.asarray(cdata[args.field], dtype=float)
    quads = leaf_quads(vtu)

    # zoom: keep only cells overlapping the window
    if args.xlim or args.ylim:
        lo, hi = quads.min(axis=1), quads.max(axis=1)
        keep = np.ones(len(quads), bool)
        if args.xlim:
            keep &= (hi[:, 0] >= args.xlim[0]) & (lo[:, 0] <= args.xlim[1])
        if args.ylim:
            keep &= (hi[:, 1] >= args.ylim[0]) & (lo[:, 1] <= args.ylim[1])
        quads, vals = quads[keep], vals[keep]

    diverging = args.field in ("vorticity", "u", "v")
    cmap = args.cmap or {"f": "Blues", "level": "Greys", "kappa": "RdBu_r"}.get(
        args.field, "RdBu_r" if diverging else "viridis")
    if args.clim:
        vmin, vmax = args.clim
    elif args.field == "f":
        vmin, vmax = 0.0, 1.0
    elif diverging or args.field == "kappa":
        a = np.nanpercentile(np.abs(vals), 99.5) if vals.size else 1.0
        vmin, vmax = -a or -1.0, a or 1.0     # symmetric: neutral midpoint at 0
    else:
        vmin, vmax = (float(vals.min()), float(vals.max())) if vals.size else (0.0, 1.0)
        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5

    lo, hi = quads.min(axis=(0, 1)), quads.max(axis=(0, 1))
    xl = args.xlim or (lo[0], hi[0])
    yl = args.ylim or (lo[1], hi[1])
    aspect = (yl[1] - yl[0]) / (xl[1] - xl[0])
    fig, ax = plt.subplots(figsize=(args.width, min(4 * args.width, (args.width - 1.3) * aspect + 1.0)))
    mode = args.mode
    if mode == "auto":
        mode = "image" if len(quads) > 40000 else "poly"
    if mode == "image":
        img, ext = rasterize(quads, vals)
        art = ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax,
                        interpolation="nearest")
    else:
        art = PolyCollection(quads, array=vals, cmap=cmap, edgecolors="face", linewidths=0.0)
        art.set_clim(vmin, vmax)
        ax.add_collection(art)
    if args.edges:
        # each quad outline (shared edges drawn twice; fine for visualization)
        edges = np.concatenate([quads[:, [0, 1]], quads[:, [1, 2]], quads[:, [2, 3]], quads[:, [3, 0]]])
        ax.add_collection(LineCollection(edges, colors="#8a8986", linewidths=0.3, alpha=0.8))

    vtp_path = args.vtp
    if vtp_path == "auto":
        vtp_path = default_interface_file(args.vtu)
    elif vtp_path == "none":
        vtp_path = None
    if vtp_path:
        segs = interface_segments(read_vtk(vtp_path))
        ax.add_collection(LineCollection(segs, colors="#e34948", linewidths=1.2))

    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cb = fig.colorbar(art, ax=ax, shrink=0.85)
    cb.set_label(args.field)
    t = vtu["time"]
    title = args.title or (os.path.basename(args.vtu) + (f"   t = {t:.4g}" if t is not None else ""))
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out = args.out or (os.path.splitext(args.vtu)[0] + f"_{args.field}.png")
    fig.savefig(out, dpi=args.dpi)
    print(f"wrote {out} ({len(quads)} cells, {args.field} in [{np.nanmin(vals):.4g}, {np.nanmax(vals):.4g}])")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
