#!/usr/bin/env python3
"""Detect structural elements on a vector (CAD-exported) foundation plan PDF.

Pipeline
    1. load      - read vector paths (geometry, colour, layer) and AutoCAD "SHX Text" annotations
    2. calibrate - fit the drawing scale from dimension numbers vs. their dimension lines
    3. learn     - learn each element family's drawing style from the shapes next to its labels
    4. detect    - build element shapes per class from geometry + learned style
    5. label     - attach plan labels (P1/124, V3/120, S7-120, ...) to shapes, one-to-one;
                   infer the type of an unlabelled shape when exactly one known type fits it
    6. check     - flag elements whose size disagrees with other elements of the same type
    7. export    - annotated PDF (one toggleable layer per class), PNG render, JSON

Classes currently detected
    column / column
    footing / isolated_footing, combined_footing, crane_footing
    wall / shear_wall            (stand-alone straight walls)
    wall / retaining_wall        (stand-alone wall labelled VPnn)
    wall / wall_network          (connected perimeter/core walls, NOT yet split into segments)
    other / mass_concrete_pad    (unlabelled squares under grade beams; style given in the config)
    other / crane_foundation     (group box around a crane's footings)

Plan conventions (label formats) and tolerances live in plan_config.toml; no colour or
layer name is written in this file.

Usage
    python detect_structures.py "GHF_EXE_PLN_STR (1)-FONDATIONS (1).pdf" --out out --dpi 150

Requires: pymupdf, shapely, numpy (Python 3.11+ for tomllib)
"""
import argparse
import collections
import json
import math
import re
import statistics
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pymupdf
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree

# Output vocabulary: the classes this tool reports and how the overlay draws them.
# These describe the deliverable, not the input plan.
ELEMENT_TYPE = {
    "column": "column",
    "isolated_footing": "footing", "combined_footing": "footing", "crane_footing": "footing",
    "shear_wall": "wall", "retaining_wall": "wall", "wall_network": "wall",
    "mass_concrete_pad": "other", "crane_foundation": "other",
}
DET_COLOURS = {
    "column": (0.0, 0.65, 0.0),
    "isolated_footing": (1.0, 0.45, 0.0),
    "combined_footing": (0.55, 0.25, 0.0),
    "crane_footing": (0.0, 0.6, 0.6),
    "shear_wall": (0.85, 0.0, 0.0),
    "retaining_wall": (0.6, 0.0, 0.3),
    "wall_network": (0.85, 0.0, 0.0),
    "mass_concrete_pad": (0.25, 0.25, 1.0),
    "crane_foundation": (0.0, 0.45, 0.45),
}


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def _rgb(c):
    return tuple(round(v, 2) for v in c) if c else None


def load_page(pdf_path, page_no=0):
    doc = pymupdf.open(pdf_path)
    page = doc[page_no]
    paths = []
    for d in page.get_drawings():
        items = []
        for it in d["items"]:
            kind = it[0]
            if kind == "l":
                items.append(("l", [(it[1].x, it[1].y), (it[2].x, it[2].y)]))
            elif kind == "re":
                r = it[1]
                items.append(("re", [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]))
            elif kind == "qu":
                q = it[1]
                items.append(("qu", [(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y)]))
            elif kind == "c":
                items.append(("c", [(p.x, p.y) for p in it[1:5]]))
        paths.append({
            "layer": d.get("layer") or "",
            "type": d["type"],
            "color": _rgb(d.get("color")),
            "fill": _rgb(d.get("fill")),
            "width": d.get("width") or 0.0,
            "rect": tuple(d["rect"]),
            "items": items,
        })
    texts, seen = [], set()
    for a in page.annots():
        t = a.info.get("content", "").strip()
        key = (t, tuple(round(v) for v in a.rect))
        if t and key not in seen:  # the same text is sometimes stored twice at the same spot
            seen.add(key)
            texts.append({"text": t, "rect": tuple(a.rect)})
    return doc, page, paths, texts


def glyph_size(texts, cfg):
    """Stroked text comes as one path per glyph; derive the glyph size from the text height."""
    heights = [min(t["rect"][2] - t["rect"][0], t["rect"][3] - t["rect"][1]) for t in texts]
    if not heights:
        raise RuntimeError("no text annotations: labels and dimensions cannot be read")
    return cfg["text"]["glyph_share_of_text_height"] * statistics.median(heights)


def is_glyph(path, ctx):
    r = path["rect"]
    return max(r[2] - r[0], r[3] - r[1]) < ctx.glyph


def subpolygons(path):
    """Split one drawing path into its closed sub-polygons."""
    out, chain = [], []

    def add(pts):
        p = Polygon(pts).buffer(0)  # may split a self-touching ring into several parts
        out.extend(g for g in getattr(p, "geoms", [p]) if g.area > 0)

    def flush():
        if len(chain) >= 3:
            add(chain)

    for kind, pts in path["items"]:
        if kind in ("re", "qu"):
            flush()
            chain = []
            add(pts)
            continue
        if chain and math.dist(chain[-1], pts[0]) < 1e-3:
            chain += pts[1:]
        else:
            flush()
            chain = list(pts)
    flush()
    return out


def path_lines(path):
    for kind, pts in path["items"]:
        if kind == "l":
            yield LineString(pts)
        elif kind in ("re", "qu"):
            for i in range(4):
                yield LineString([pts[i], pts[(i + 1) % 4]])


def centre(rect):
    return Point((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)


# ---------------------------------------------------------------------------
# 2. Calibrate: dimension numbers vs. the length of the line they sit on
# ---------------------------------------------------------------------------
NUMBER = re.compile(r"^\d+[.,]\d+$")


def calibrate(paths, texts, glyph, cfg):
    c = cfg["calibration"]
    segs = [ls for p in paths if p["type"] in ("s", "fs")
            for ls in path_lines(p) if ls.length > glyph]
    tree = STRtree(segs)
    samples = []
    for t in texts:
        if not NUMBER.match(t["text"]):
            continue
        value = float(t["text"].replace(",", "."))
        if value <= 0:
            continue
        r = t["rect"]
        mid = centre(r)
        size = max(r[2] - r[0], r[3] - r[1])
        best = None
        for i in tree.query(mid.buffer(size)):
            s = segs[i]
            (ax, ay), (bx, by) = s.coords[0], s.coords[-1]
            dx, dy = bx - ax, by - ay
            length = math.hypot(dx, dy)
            along = ((mid.x - ax) * dx + (mid.y - ay) * dy) / length
            perp = abs((mid.x - ax) * dy - (mid.y - ay) * dx) / length
            # the number sits over its line: centred along it, closer than its own size
            if 0 < along < length and perp < size and (best is None or perp < best[0]):
                best = (perp, length)
        if best:
            samples.append((value, best[1]))
    if len(samples) < c["min_samples"]:
        raise RuntimeError(f"calibration: only {len(samples)} dimension samples found")

    v = np.array([s[0] for s in samples])
    L = np.array([s[1] for s in samples])
    # robust slope (Theil-Sen on random pairs with clearly different values), then intercept
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(v), 20000), rng.integers(0, len(v), 20000)
    ok = np.abs(v[i] - v[j]) > c["min_value_spread"]
    k = float(np.median((L[i][ok] - L[j][ok]) / (v[i][ok] - v[j][ok])))
    gap = float(np.median(k * v - L))
    # refine by least squares on inliers (dimension line length = k * value - gap)
    for _ in range(3):
        inl = np.abs((L + gap) / k - v) < c["tolerance"]
        A = np.c_[v[inl], -np.ones(inl.sum())]
        (k, gap), *_ = np.linalg.lstsq(A, L[inl], rcond=None)
    err = np.abs((L + gap) / k - v)
    inl = err < c["tolerance"]
    calib = {
        "pt_per_m": float(k),
        "scale_denominator": float(72 / 25.4 * 1000 / k),
        "dimension_line_gap_pt": float(gap),
        "samples": int(len(v)),
        "inliers": int(inl.sum()),
        "median_error_cm": float(np.median(err[inl]) * 100),
        "p95_error_cm": float(np.percentile(err[inl], 95) * 100),
    }
    if inl.mean() < c["min_inlier_share"]:
        raise RuntimeError(f"calibration unreliable: {calib}")
    return calib


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def oriented(geom, k):
    """Minimum rotated rectangle -> size in metres, angle, corners and rectangularity."""
    hull = MultiPoint(list(geom.exterior.coords)).convex_hull
    mrr = hull.minimum_rotated_rectangle
    c = list(mrr.exterior.coords)[:4]
    a, b = math.dist(c[0], c[1]), math.dist(c[1], c[2])
    p, q = (c[0], c[1]) if a >= b else (c[1], c[2])
    return {
        "length_m": max(a, b) / k,
        "width_m": min(a, b) / k,
        "angle_deg": math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])) % 180,
        "obb": c,
        "rectness": geom.area / mrr.area if mrr.area else 0,
    }


def components(polys, eps):
    u = unary_union([p.buffer(eps) for p in polys]).buffer(-eps)
    return [g for g in (u.geoms if hasattr(u, "geoms") else [u]) if not g.is_empty]


def dedupe(geoms, iou):
    kept = []
    for g in sorted(geoms, key=lambda g: -g.area):
        if all(g.intersection(h).area / g.union(h).area < iou for h in kept):
            kept.append(g)
    return kept


def make_det(subclass, geom, k, source, **extra):
    d = {"subclass": subclass, "element_type": ELEMENT_TYPE[subclass], "geom": geom, "o": oriented(geom, k),
         "source": source, "label": None, "flags": []}
    d.update(extra)
    return d


def size_of(d):
    return d["o"]["length_m"], d["o"]["width_m"]


# ---------------------------------------------------------------------------
# 3. Labels and style learning
# ---------------------------------------------------------------------------
def parse_labels(texts, cfg):
    rules = [(kind, re.compile(r["pattern"]), r) for kind, r in cfg["labels"].items()]
    labels = []
    for t in texts:
        for kind, rx, r in rules:
            m = rx.match(t["text"])
            if not m:
                continue
            g = {name: (v or "") for name, v in m.groupdict().items()}
            dims = None
            if g.get("a") and "sized_key" in r:
                u = r["size_unit_m"]
                dims = tuple(sorted((int(g["a"]) * u, int(g["b"]) * u), reverse=True))
                key = r["sized_key"].format(**g)
            else:
                key = r["key"].format(**g)
            labels.append({"kind": kind, "raw": t["text"], "c": centre(t["rect"]), "key": key,
                           "groups": g, "building": g.get("bld") or None, "dims": dims})
            break
    return labels


def dominant(votes, share):
    if not votes:
        return set()
    top = votes.most_common(1)[0][1]
    return {v for v, n in votes.items() if n >= share * top}


def learn_fill(paths, labels, ctx):
    """Fill colour of the filled shape nearest to each label, by majority vote."""
    fills = [(p["fill"], g) for p in paths if p["type"] in ("f", "fs") and p["fill"] for g in subpolygons(p)]
    tree = STRtree([g for _, g in fills])
    votes = collections.Counter()
    for l in labels:
        idx = tree.query(l["c"].buffer(ctx.radius))
        if len(idx):
            votes[fills[min(idx, key=lambda i: fills[i][1].distance(l["c"]))][0]] += 1
    return dominant(votes, ctx.share), votes


def learn_outline(paths, labels, ctx):
    """Stroke colour and layer of closed outlines around/next to each label, by majority vote."""
    colours, layers = collections.Counter(), collections.Counter()
    shapes = [(p, g) for p in paths if p["type"] == "s" and not is_glyph(p, ctx)
              for g in subpolygons(p) if is_footing_shape(g, ctx)]
    tree = STRtree([g for _, g in shapes])
    for l in labels:
        # the element a label names is the tightest outline around or next to it (not, say, the sheet frame)
        near = [shapes[i] for i in tree.query(l["c"].buffer(ctx.radius))
                if shapes[i][1].contains(l["c"]) or shapes[i][1].exterior.distance(l["c"]) < ctx.radius]
        if near:
            p, _ = min(near, key=lambda s: s[1].area)
            colours[p["color"]] += 1
            layers[p["layer"]] += 1
    return dominant(colours, ctx.share), dominant(layers, ctx.share), colours


# ---------------------------------------------------------------------------
# 4. Detect shapes
# ---------------------------------------------------------------------------
def detect_columns(paths, fills, ctx):
    c = ctx.cfg["column"]
    polys = [sp for p in paths if p["type"] in ("f", "fs") and p["fill"] in fills for sp in subpolygons(p)]
    dets = []
    for g in components(polys, eps=c["merge_gap"] * ctx.k):
        o = oriented(g, ctx.k)
        if o["rectness"] > c["min_rectness"] and c["min_width"] <= o["width_m"] and o["length_m"] <= c["max_length"]:
            dets.append(make_det("column", g, ctx.k, "filled rectangle in the columns' learned colour"))
    return dets


def detect_walls(paths, fills, ctx):
    c = ctx.cfg["wall"]
    polys = [sp for p in paths if p["type"] in ("f", "fs") and p["fill"] in fills for sp in subpolygons(p)]
    dets = []
    for g in components(polys, eps=c["merge_gap"] * ctx.k):
        if g.area / ctx.k ** 2 < c["min_area"]:
            continue
        if oriented(g, ctx.k)["rectness"] > c["min_rectness"]:
            dets.append(make_det("shear_wall", g, ctx.k, "filled rectangle in the walls' learned colour"))
        else:
            dets.append(make_det("wall_network", g, ctx.k, "connected walls in the walls' learned colour",
                                 flags=["not_segmented"]))
    return dets


def is_footing_shape(g, ctx):
    c = ctx.cfg["footing"]
    o = oriented(g, ctx.k)
    return (g.area / ctx.k ** 2 > c["min_area"] and o["rectness"] > c["min_rectness"]
            and o["length_m"] / o["width_m"] <= c["max_aspect"])


def detect_footings(paths, colours, layers, ctx):
    """Footing outlines drawn as one closed path are taken as they are; outlines drawn as
    loose lines are rebuilt as faces between lines, which crossing lines can cut short,
    so faces only fill in where no closed outline exists."""
    lines, closed = [], []
    for p in paths:
        if p["type"] != "s" or is_glyph(p, ctx):
            continue
        if p["color"] in colours or p["layer"] in layers:
            lines.extend(path_lines(p))
            closed.extend(g for g in subpolygons(p) if is_footing_shape(g, ctx))
    iou = ctx.cfg["matching"]["dedupe_iou"]
    kept = dedupe(closed, iou)
    for f in polygonize(unary_union(lines)):
        if is_footing_shape(f, ctx) and all(f.intersection(g).area / f.area < ctx.cfg["matching"]["same_shape_overlap"] for g in kept):
            kept.append(f)
    return [make_det("isolated_footing", g, ctx.k, "outline in the footings' learned colour/layer")
            for g in dedupe(kept, iou)]


def detect_pads(paths, ctx):
    c = ctx.cfg.get("mass_concrete_pad")
    if not c:
        return []
    stroke = tuple(c["stroke"])
    squares = []
    for p in paths:
        if p["type"] != "s" or p["color"] != stroke or p["width"] < c["min_line_width_pt"]:
            continue
        for g in subpolygons(p):
            o = oriented(g, ctx.k)
            if (o["rectness"] > c["min_rectness"] and c["min_side"] <= o["width_m"]
                    and o["length_m"] <= c["max_side"] and o["length_m"] / o["width_m"] < c["max_aspect"]):
                squares.append(g)
    return [make_det("mass_concrete_pad", g, ctx.k, "square in the configured pad style")
            for g in dedupe(squares, ctx.cfg["matching"]["pad_dedupe_iou"])]


# ---------------------------------------------------------------------------
# 5. Attach labels
# ---------------------------------------------------------------------------
def assign(dets, labels, max_dist, dist):
    """Greedy one-to-one assignment: every (shape, label) pair is scored by distance, pairs are
    taken shortest first, and a shape or label already used is skipped. Returns unused labels."""
    pairs = sorted((dist(d, l), i, j) for i, d in enumerate(dets) for j, l in enumerate(labels))
    used_d, used_l = set(), set()
    for dd, i, j in pairs:
        if dd > max_dist or i in used_d or j in used_l:
            continue
        used_d.add(i)
        used_l.add(j)
        dets[i]["label"] = labels[j]
    return [l for j, l in enumerate(labels) if j not in used_l]


def near_geom(d, l):
    return d["geom"].distance(l["c"])


def near_outline(d, l):
    return 0.0 if d["geom"].contains(l["c"]) else d["geom"].exterior.distance(l["c"])


def catalogue(dets, labels, min_count):
    """Expected size per type key: stated in the label (S0 (240x240x60)) or, failing that,
    the median of the shapes already carrying that label (at least `min_count` of them)."""
    sizes = collections.defaultdict(list)
    for d in dets:
        if d["label"]:
            sizes[d["label"]["key"]].append(size_of(d))
    cat = {key: (statistics.median(s[0] for s in v), statistics.median(s[1] for s in v))
           for key, v in sizes.items() if len(v) >= min_count}
    for l in labels:
        if l["dims"]:
            cat[l["key"]] = l["dims"]
    return cat


def fits(size, key, cat, tol, match="both"):
    if key not in cat:
        return True
    width_ok = abs(size[1] - cat[key][1]) <= tol
    return width_ok if match == "width" else width_ok and abs(size[0] - cat[key][0]) <= tol


def assign_by_catalogue(dets, labels, ctx, dist):
    """Pass 1: plain nearest assignment, which yields the size of each type. Pass 2: redo it,
    allowing a label only on a shape of its type's size, first within the near radius, then,
    for shapes still unlabelled, within the far radius."""
    m = ctx.cfg["matching"]
    assign(dets, labels, m["footing_near"] * ctx.k, dist)
    cat = catalogue(dets, labels, min_count=2)
    for d in dets:
        d["label"] = None

    def sized(d, l):
        return dist(d, l) if fits(size_of(d), l["key"], cat, m["size_tolerance"]) else math.inf

    left = assign(dets, labels, m["footing_near"] * ctx.k, sized)

    def sized_known(d, l):
        return sized(d, l) if l["key"] in cat else math.inf

    return assign([d for d in dets if not d["label"]], left, m["footing_far"] * ctx.k, sized_known)


def footings_from_labels(paths, labels, existing, cat, ctx):
    """A footing label with no footing nearby: accept a closed rectangle of any colour or
    layer next to it, provided its size is the one expected for that type."""
    m = ctx.cfg["matching"]
    rects = [g for p in paths if p["type"] == "s" and not is_glyph(p, ctx)
             for g in subpolygons(p) if oriented(g, ctx.k)["rectness"] > ctx.cfg["footing"]["min_rectness"]]
    found, left = [], []
    for l in labels:
        cands = [g for g in rects if l["key"] in cat and g.distance(l["c"]) < m["footing_fallback_radius"] * ctx.k
                 and fits(size_of({"o": oriented(g, ctx.k)}), l["key"], cat, m["size_tolerance"])
                 and all(g.intersection(e["geom"]).area / g.area < m["same_shape_overlap"] for e in existing + found)]
        if cands:
            g = min(cands, key=lambda g: 0 if g.contains(l["c"]) else g.distance(l["c"]))
            found.append(make_det("isolated_footing", g, ctx.k, "rectangle of the label's catalogue size",
                                  label=l, flags=["outline_in_unexpected_style"]))
        else:
            left.append(l)
    return found, left


def infer_missing_labels(dets, all_labels, rule, ctx):
    """An unlabelled shape gets a type when exactly one known type has its size; for types that
    vary per building, only the types of the building around it (majority of nearby labels).
    Returns the shapes that received a type."""
    m = ctx.cfg["matching"]
    per_building, match = rule["per_building"], rule.get("size_match", "both")
    done = []
    labelled = [d for d in dets if d["label"]]
    cat = catalogue(labelled, [], min_count=1)
    rep = {d["label"]["key"]: d["label"] for d in labelled}
    for d in dets:
        if d["label"]:
            continue
        cands = [key for key in cat if fits(size_of(d), key, cat, m["size_tolerance"], match)]
        if per_building:
            votes = collections.Counter(l["building"] for l in all_labels if l["building"]
                                        and d["geom"].distance(l["c"]) < m["building_radius"] * ctx.k)
            if not votes:
                continue
            bld = votes.most_common(1)[0][0]
            cands = [key for key in cands if rep[key]["building"] == bld]
        if len(cands) == 1:
            d["label"] = dict(rep[cands[0]], raw=None, inferred=True)
            d["flags"].append("label_inferred_from_size")
            done.append(d)
    return done


def group_cranes(footings, texts, ctx):
    """Footings whose label states its size and that sit together near a crane note."""
    c = ctx.cfg["crane"]
    rx = re.compile(c["note_pattern"], re.I)
    groups = []
    notes = [centre(t["rect"]) for t in texts if rx.search(t["text"])]
    sized = [f for f in footings if f["label"] and f["label"]["dims"]]
    for n in notes:
        members = [f for f in sized if f["geom"].distance(n) < c["group_radius_m"] * ctx.k]
        if len(members) >= c["min_members"]:
            hull = unary_union([f["geom"] for f in members]).convex_hull
            if hull.buffer(c["note_margin_m"] * ctx.k).contains(n):
                for f in members:
                    f["subclass"] = "crane_footing"
                    f["element_type"] = ELEMENT_TYPE["crane_footing"]
                    f["flags"].append("temporary_works")
                groups.append(make_det("crane_foundation", hull, ctx.k, "group of sized footings next to a crane note",
                                       flags=["temporary_works"], members=len(members)))
    return groups


# ---------------------------------------------------------------------------
# 6. Consistency checks
# ---------------------------------------------------------------------------
def check_types(dets, rules, tol):
    """Elements sharing a type key should share a size (or thickness, for width-only types)."""
    groups = collections.defaultdict(list)
    for d in dets:
        if d["label"] and d["subclass"] not in ("wall_network", "crane_foundation"):
            groups[d["label"]["key"]].append(d)  # P1/124 per building, S7 plan-wide: the key says which
    for key, ds in groups.items():
        if len(ds) < 2:
            continue
        match = rules[ds[0]["label"]["kind"]].get("size_match", "both")
        cat = {key: (statistics.median(d["o"]["length_m"] for d in ds),
                     statistics.median(d["o"]["width_m"] for d in ds))}
        for d in ds:
            if not fits(size_of(d), key, cat, tol, match):
                d["flags"].append(f"size_differs_from_type_{key}_median_{cat[key][0]:.2f}x{cat[key][1]:.2f}m")


# ---------------------------------------------------------------------------
# 7. Export
# ---------------------------------------------------------------------------
def to_json(dets, calib, learned, pdf_path, page, dpi, warnings):
    s = dpi / 72.0

    def px(pts):
        return [[round(x * s, 1), round(y * s, 1)] for x, y in pts]

    out = []
    for n, d in enumerate(dets, 1):
        g, o, lab = d["geom"], d["o"], d["label"]
        x0, y0, x1, y1 = g.bounds
        item = {
            "id": n,
            "class": d["element_type"],
            "subclass": d["subclass"],
            "label": lab["raw"] if lab else None,
            "type_key": lab["key"] if lab else None,
            "bbox_px": [round(x0 * s, 1), round(y0 * s, 1), round(x1 * s, 1), round(y1 * s, 1)],
            "obb_px": px(o["obb"]),
            "size_m": {"length": round(o["length_m"], 3), "width": round(o["width_m"], 3)},
            "angle_deg": round(o["angle_deg"], 2),
            "area_m2": round(g.area / calib["pt_per_m"] ** 2, 3),
            "source": d["source"],
            "flags": d["flags"] + ([] if lab or d.get("labels") else ["unlabelled"]),
        }
        if d["subclass"] == "wall_network":
            item["polygon_px"] = px(list(g.exterior.coords))
        if d.get("labels"):
            item["other_labels"] = [l["raw"] for l in d["labels"]]
        if d["subclass"] == "crane_foundation":
            item["members"] = d["members"]
        out.append(item)
    return {
        "source_pdf": str(pdf_path),
        "image": {"dpi": dpi, "width_px": round(page.rect.width * s), "height_px": round(page.rect.height * s),
                  "note": "pixel = PDF point * dpi / 72, origin top-left"},
        "calibration": calib,
        "learned_style": learned,
        "counts": dict(collections.Counter(d["subclass"] for d in out)),
        "detections": out,
        "warnings": warnings,
    }


def annotate(doc, page, dets):
    layers = {}
    for d in dets:
        sub = d["subclass"]
        if sub not in layers:
            layers[sub] = doc.add_ocg(f"DET {sub}", on=True)
        col, oc = DET_COLOURS[sub], layers[sub]
        x0, y0, x1, y1 = d["geom"].bounds
        if sub == "wall_network":
            page.draw_polyline([pymupdf.Point(x, y) for x, y in d["geom"].exterior.coords],
                               color=col, width=0.8, dashes="[3 2] 0", oc=oc)
            continue
        # tight rotated rectangle; the axis-aligned box (what the JSON's bbox_px holds) is drawn
        # faint and dashed when the element is not aligned with the page
        obb = [pymupdf.Point(x, y) for x, y in d["o"]["obb"]]
        page.draw_polyline(obb + obb[:1], color=col, width=0.6, oc=oc)
        a = d["o"]["angle_deg"] % 90
        if min(a, 90 - a) > 0.5:
            page.draw_rect(pymupdf.Rect(x0, y0, x1, y1), color=col, width=0.25, dashes="[1.5 1.5] 0",
                           stroke_opacity=0.6, oc=oc)
        lab = d["label"]
        tag = "?" if not lab else ("~" + lab["key"] if lab.get("inferred") else lab["key"])
        if sub != "crane_foundation":
            tag += f" {d['o']['length_m'] * 100:.0f}x{d['o']['width_m'] * 100:.0f}"
        page.insert_text(pymupdf.Point(x0, y0 - 0.8), tag, fontsize=3.2, color=col, oc=oc)


def style_str(v):
    return list(v) if isinstance(v, tuple) else v


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--config", default=str(Path(__file__).with_name("plan_config.toml")))
    ap.add_argument("--out", default="out")
    ap.add_argument("--dpi", type=int, default=150, help="resolution of the PNG and of JSON pixel coordinates")
    ap.add_argument("--page", type=int, default=0)
    args = ap.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("loading ...", flush=True)
    doc, page, paths, texts = load_page(args.pdf, args.page)
    glyph = glyph_size(texts, cfg)
    print(f"  {len(paths)} vector paths, {len(texts)} text annotations, glyph size < {glyph:.1f} pt")

    calib = calibrate(paths, texts, glyph, cfg)
    k = calib["pt_per_m"]
    print(f"scale 1/{calib['scale_denominator']:.1f} ({k:.3f} pt/m) from {calib['inliers']}/{calib['samples']} "
          f"dimensions, median error {calib['median_error_cm']:.2f} cm")
    ctx = SimpleNamespace(k=k, cfg=cfg, glyph=glyph, radius=cfg["style_learning"]["search_radius_m"] * k,
                          share=cfg["style_learning"]["keep_share_of_top_vote"])
    m = cfg["matching"]

    labels = parse_labels(texts, cfg)
    by_kind = collections.defaultdict(list)
    for l in labels:
        by_kind[l["kind"]].append(l)

    col_fill, col_votes = learn_fill(paths, by_kind["column"], ctx)
    wall_fill, wall_votes = learn_fill(paths, by_kind["wall"], ctx)
    foot_col, foot_layers, foot_votes = learn_outline(paths, by_kind["footing"], ctx)
    learned = {"column_fill": [style_str(c) for c in col_fill], "wall_fill": [style_str(c) for c in wall_fill],
               "footing_stroke": [style_str(c) for c in foot_col], "footing_layers": sorted(foot_layers)}
    print("learned style:", learned)
    warnings = []

    columns = detect_columns(paths, col_fill, ctx)
    left = assign(columns, by_kind["column"], m["column_radius"] * k, near_geom)
    warnings += [f"column label without shape: {l['raw']}" for l in left]
    infer_missing_labels(columns, labels, cfg["labels"]["column"], ctx)

    walls = detect_walls(paths, wall_fill, ctx)
    solo = [w for w in walls if w["subclass"] == "shear_wall"]
    left = assign(solo, by_kind["wall"], m["wall_radius"] * k, near_geom)
    # retaining-wall labels: an unlabelled stand-alone wall carrying one becomes a retaining wall
    rw_left = assign([w for w in solo if not w["label"]], by_kind["retaining_wall"], m["wall_radius"] * k, near_geom)
    for w in solo:
        if w["label"] and w["label"]["kind"] == "retaining_wall":
            w["subclass"] = "retaining_wall"
    infer_missing_labels([w for w in solo if w["subclass"] == "shear_wall"], labels, cfg["labels"]["wall"], ctx)
    # a wall still unlabelled may be a retaining wall whose thickness matches exactly one VP type
    pool = [w for w in solo if w["subclass"] == "retaining_wall" or not w["label"]]
    for w in infer_missing_labels(pool, labels, cfg["labels"]["retaining_wall"], ctx):
        w["subclass"] = "retaining_wall"
    # everything else is recorded on the nearest wall of any kind (mostly the unsplit networks)
    for l in left + rw_left:
        w = min(walls, key=lambda w: w["geom"].distance(l["c"]))
        if w["geom"].distance(l["c"]) < m["wall_radius"] * k:
            w.setdefault("labels", []).append(l)
        else:
            warnings.append(f"wall label without shape: {l['raw']}")

    footings = detect_footings(paths, foot_col, foot_layers, ctx)
    flabels = by_kind["footing"]
    left = assign_by_catalogue(footings, flabels, ctx, near_outline)
    extra, left = footings_from_labels(paths, left, footings, catalogue(footings, flabels, min_count=1), ctx)
    footings += extra
    warnings += [f"footing label without shape: {l['raw']}" for l in left]
    infer_missing_labels(footings, labels, cfg["labels"]["footing"], ctx)
    series = cfg["labels"]["footing"]["series_subclass"]
    for f in footings:
        if f["label"]:
            f["subclass"] = series.get(f["label"]["groups"].get("series"), "isolated_footing")
        else:
            f["flags"].append("subclass_assumed")
    cranes = group_cranes(footings, texts, ctx)

    pads = detect_pads(paths, ctx)

    dets = columns + walls + footings + pads + cranes
    check_types(dets, cfg["labels"], m["size_tolerance"])

    report = to_json(dets, calib, learned, args.pdf, page, args.dpi, warnings)
    (out / "detections.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    annotate(doc, page, dets)
    doc.save(out / "annotated.pdf", garbage=3, deflate=True)
    page.get_pixmap(dpi=args.dpi).save(out / "annotated.png")

    print("detections:", report["counts"])
    inferred = [d for d in report["detections"] if "label_inferred_from_size" in d["flags"]]
    print(f"{len(inferred)} labels inferred from size:", [(d["subclass"], d["type_key"]) for d in inferred])
    flagged = sum(1 for d in report["detections"] if d["flags"])
    print(f"{flagged} detections carry flags, {len(warnings)} warnings -> {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
