"""Reading a vector foundation plan and the geometric primitives the detector works on.

- load_page: the page's vector paths (type, filled or not, pen width, items) and its texts
  (AutoCAD SHX text is stored as PDF annotations). No colour and no CAD layer is read.
- calibrate: the drawing scale, from the dimension numbers and the lines they sit on.
- primitives: straight segments, dashed lines and the grid pen, fills grouped by paint order,
  hatched areas, closed outlines, runs of parallel line pairs.
- labels: the plan's label grammar (plan_config.toml) applied to the texts.

All tolerances are in T, in metres (or as shares), so they do not depend on the paper scale.
"""
import collections
import math
import re
import statistics
from types import SimpleNamespace

import numpy as np
import pymupdf
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

T = SimpleNamespace(
    # filled shapes
    touch=0.02,                 # fill pieces closer than this (m) touch
    paint_gap=3,                # fill pieces at most this many paths apart in paint order can be one object
    min_side=0.15,              # smallest structural section side (m)
    rectness=0.95,              # a fill is a rectangle if its area is at least this share of its rotated box
    dup_iou=0.9,                # rectangles overlapping more than this share are one object drawn twice
    fuse=0.01,                  # pieces this close (m) are fused into one shape
    stretch_thickness=(0.1, 0.6),  # a bent fill is split into straight stretches this thick (m) ...
    stretch_min_len=0.3,        # ... and at least this long (m)
    ec2_ratio=4.0,              # EN 1992-1-1 5.3.1(7): a section up to 4 x its width is a column, longer a wall
    # dashed lines
    dash_max=2.0,               # dash pieces are at most this long (m)
    collinear=0.03,             # pieces this close (m) across their direction lie on one line
    grid_min_pieces=8,          # a dashed line has at least this many pieces
    # hatching
    hatch_len=(0.05, 1.0),      # hatch stroke length range (m)
    hatch_search=0.5,           # a hatch stroke's parallel twins lie within this (m) ...
    hatch_neighbours=3,         # ... and it has at least this many of them
    hatch_gap=0.06,             # gaps up to twice this (m) between hatch strokes are closed
    hatch_min_strokes=20,       # a hatched area has at least this many strokes
    hatch_local=0.8,            # a hatch's thickness is measured from stroke ends within this (m)
    # closed outlines
    outline_min_area=0.1,       # m2; smaller closed outlines are ignored
    outline_rectness=0.97,      # closed outlines are kept if at least this rectangular
    # runs of parallel line pairs
    band_offset=0.05,           # band pieces this close (m) across their direction are one run ...
    band_run_gap=0.6,           # ... when the gap between them along it is at most this (m)
    run_dup=0.02,               # a run inside another one grown by this (m) is a duplicate
    # buildings
    building_k=5,               # a point's building: the majority of its nearest labels with a building number ...
    building_radius=25.0,       # ... within this distance (m)
)


NUMBER = re.compile(r"^\d+[.,]\d+$")


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
        paths.append({                     # no colour and no CAD layer is kept
            "type": d["type"],
            "filled": d.get("fill") is not None,
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


def centre(rect):
    return Point((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)


def path_lines(path):
    for kind, pts in path["items"]:
        if kind == "l":
            yield LineString(pts)
        elif kind in ("re", "qu"):
            for i in range(4):
                yield LineString([pts[i], pts[(i + 1) % 4]])


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


def axis(ang):
    return math.cos(ang), math.sin(ang)


def offset(pt, ang):
    ux, uy = axis(ang)
    return -pt[0] * uy + pt[1] * ux


def along(pt, ang):
    ux, uy = axis(ang)
    return pt[0] * ux + pt[1] * uy


def valid_polygon(g):
    """A shape made valid (a self-touching outline, a bow-tie from buffering) with its largest part
    kept, so later unions and intersections cannot fail on it."""
    if g is None or g.is_empty or g.is_valid:
        return g
    from shapely.validation import make_valid
    fixed = make_valid(g)
    parts = [q for q in getattr(fixed, "geoms", [fixed]) if q.geom_type == "Polygon" and not q.is_empty]
    if not parts:                   # only lines or points left: fall back to the zero-width buffer trick
        return g.buffer(0)
    return max(parts, key=lambda q: q.area)


def text_index(texts):
    boxes = [box(*t["rect"]).buffer(1.0) for t in texts]
    return boxes, STRtree(boxes)


def in_text(geom, tboxes):
    """A stroke is part of a text if it lies inside a text's box and is no longer than the text
    is tall (a label written between two lines must not swallow the lines)."""
    boxes, tree = tboxes
    for i in tree.query(geom):
        b = boxes[i]
        if b.contains(geom):
            x0, y0, x1, y1 = b.bounds
            if max(geom.bounds[2] - geom.bounds[0], geom.bounds[3] - geom.bounds[1]) <= 1.2 * min(x1 - x0, y1 - y0):
                return True
    return False


def is_closed(path):
    """True if the path draws a closed shape: a rectangle/quad item, or a chain of segments that
    ends where it starts (an open polyline is not closed, however many segments it has)."""
    chain = []
    for kind, pts in path["items"]:
        if kind in ("re", "qu"):
            return True
        if chain and math.dist(chain[-1], pts[0]) < 1e-3:
            chain += pts[1:]
        else:
            if len(chain) >= 4 and math.dist(chain[0], chain[-1]) < 1e-3:
                return True
            chain = list(pts)
    return len(chain) >= 4 and math.dist(chain[0], chain[-1]) < 1e-3


def straight_segments(paths, k, tboxes, min_m=0.0, max_m=math.inf):
    """Every straight stroke segment, with its paint index; text strokes are skipped."""
    out = []
    for i, p in enumerate(paths):
        if p["type"] not in ("s", "fs"):
            continue
        closed = is_closed(p)                   # the stroke is a side of a closed shape
        for ls in path_lines(p):
            L = ls.length / k
            if min_m <= L <= max_m and not in_text(ls, tboxes):
                (ax, ay), (bx, by) = ls.coords[0], ls.coords[-1]
                out.append({"g": ls, "a": (ax, ay), "b": (bx, by), "len": L, "idx": i, "stroke": p["type"] == "s",
                            "closed": closed,
                            "ang": math.atan2(by - ay, bx - ax) % math.pi, "w": round(p["width"], 2)})
    return out


def mark_dashed(segs, k):
    """Flag segments that are pieces of a dashed line (many collinear pieces with gaps). Dashed
    lines are axes or hidden lines, never the visible edge of an element."""
    groups = collections.defaultdict(list)
    for i, s_ in enumerate(segs):
        if s_["len"] > T.dash_max:
            continue
        a = round(math.degrees(s_["ang"]) * 5) / 5 % 180
        mid = ((s_["a"][0] + s_["b"][0]) / 2, (s_["a"][1] + s_["b"][1]) / 2)
        groups[(a, round(offset(mid, s_["ang"]) / (T.collinear * k)))].append(i)
    n = 0
    for ids in groups.values():
        if len(ids) < T.grid_min_pieces:
            continue
        ang = segs[ids[0]]["ang"]
        iv = sorted((min(along(segs[i]["a"], ang), along(segs[i]["b"], ang)),
                     max(along(segs[i]["a"], ang), along(segs[i]["b"], ang))) for i in ids)
        lo, hi = iv[0][0], max(v[1] for v in iv)
        cover = sum(v[1] - v[0] for v in iv) / (hi - lo) if hi > lo else 1
        if cover <= 0.97:
            for i in ids:
                segs[i]["dashed"] = True
                n += 1
    # the pen the dashed axes are drawn with; continuous lines of that pen are axes too
    pen = collections.Counter(s_["w"] for s_ in segs if s_.get("dashed") and s_["w"] > 0).most_common(1)
    pen = pen[0][0] if pen else None
    for s_ in segs:
        if pen is not None and s_["w"] == pen:
            s_["axis_pen"] = True
    return n, pen


def filled_objects(paths, k):
    objs, cur, last = [], None, -10
    for i, p in enumerate(paths):
        if p["type"] not in ("f", "fs") or not p["filled"]:
            continue
        parts = subpolygons(p)
        if not parts:
            continue
        g = unary_union(parts)
        if cur is not None and i - last <= T.paint_gap and cur["g"].distance(g) < T.touch * k:
            cur["g"] = cur["g"].union(g)
            cur["paths"].append(i)
        else:
            cur = {"g": g, "paths": [i]}
            objs.append(cur)
        last = i
    rects, others = [], []
    for o in objs:
        g = o["g"]
        if g.geom_type != "Polygon":
            g = g.buffer(T.fuse * k).buffer(-T.fuse * k)
        if g.geom_type != "Polygon" or g.is_empty:
            if not g.is_empty:
                others.append(g)
            continue
        oo = oriented(g, k)
        if oo["rectness"] > T.rectness and oo["width_m"] >= T.min_side:
            rects.append({"g": g, "o": oo, "paths": o["paths"]})
        else:
            others.append(g)
    kept, dups = [], 0
    for r in sorted(rects, key=lambda r: -r["g"].area):
        if any(r["g"].intersection(q["g"]).area / r["g"].union(q["g"]).area > T.dup_iou for q in kept):
            dups += 1
            continue
        kept.append(r)
    return kept, dups, others


def straight_stretches(g, k, t_range=None, min_len=None):
    """Split a thin filled polygon with bends (a wall drawn as one object around corners) into
    straight stretches: pairs of opposite, parallel edges one wall-thickness apart."""
    t_range = t_range or T.stretch_thickness
    min_len = T.stretch_min_len if min_len is None else min_len
    ring = list(g.simplify(T.fuse * k).exterior.coords)
    edges = []
    for a, b in zip(ring, ring[1:]):
        L = math.dist(a, b)
        if L / k >= min_len:
            edges.append((a, b, math.atan2(b[1] - a[1], b[0] - a[0]) % math.pi))
    out = []
    for i, (a, b, ang) in enumerate(edges):
        for c, d, ang2 in edges[i + 1:]:
            if abs(math.sin(ang2 - ang)) > 0.02:
                continue
            t = abs(offset(((c[0] + d[0]) / 2, (c[1] + d[1]) / 2), ang) - offset(a, ang)) / k
            if not t_range[0] <= t <= t_range[1]:
                continue
            lo = max(min(along(a, ang), along(b, ang)), min(along(c, ang), along(d, ang)))
            hi = min(max(along(a, ang), along(b, ang)), max(along(c, ang), along(d, ang)))
            if (hi - lo) / k < min_len:
                continue
            ux, uy = axis(ang)
            o1, o2 = offset(a, ang), offset(((c[0] + d[0]) / 2, (c[1] + d[1]) / 2), ang)
            P = lambda u, v: (u * ux - v * uy, u * uy + v * ux)
            piece = Polygon([P(lo, o1), P(hi, o1), P(hi, o2), P(lo, o2)])
            if piece.is_valid and g.buffer(T.fuse * k).contains(piece):   # both faces of the same stretch
                out.append(piece)
    kept = []
    for p_ in sorted(out, key=lambda p_: -p_.area):
        if all(p_.intersection(q).area < 0.5 * p_.area for q in kept):
            kept.append(p_)
    return kept


def hatched_regions(segs, k):
    cand = [s for s in segs if T.hatch_len[0] <= s["len"] <= T.hatch_len[1]]
    tree = STRtree([s["g"] for s in cand])
    hatch = []
    for s in cand:
        twins = 0
        for j in tree.query(s["g"].buffer(T.hatch_search * k)):
            t = cand[j]
            if t is s:
                continue
            if abs(math.sin(t["ang"] - s["ang"])) < 0.02 and abs(t["len"] - s["len"]) < 0.15 * s["len"]:
                twins += 1
        if twins >= T.hatch_neighbours:
            hatch.append(s)
    if not hatch:
        return []
    # close the gaps between strokes to recover the hatched area
    blobs = unary_union([s["g"].buffer(T.hatch_gap * k) for s in hatch])
    blobs = [b for b in getattr(blobs, "geoms", [blobs])]
    htree = STRtree([s["g"] for s in hatch])
    out = []
    for b in blobs:
        members = [hatch[i] for i in htree.query(b) if b.contains(hatch[i]["g"])]
        if len(members) < T.hatch_min_strokes:
            continue
        fams = collections.Counter(round(math.degrees(s["ang"]) / 5) * 5 % 180 for s in members)
        top = fams.most_common(2)
        cross = (len(top) == 2 and 80 <= abs(top[0][0] - top[1][0]) <= 100
                 and top[1][1] >= 0.3 * top[0][1])
        # thickness: hatch strokes run face to face, so the local spread of their end points
        # across the wall is its thickness
        ends = [pt for s in members for pt in (s["a"], s["b"])]
        etree = STRtree([Point(p) for p in ends])
        samples = []
        for s in members[::max(1, len(members) // 60)]:
            c = s["g"].centroid
            local = [ends[i] for i in etree.query(c.buffer(T.hatch_local * k))]
            if len(local) >= 6:
                mrr = MultiPoint(local).minimum_rotated_rectangle
                if mrr.geom_type == "Polygon":
                    q = list(mrr.exterior.coords)
                    a, bb = math.dist(q[0], q[1]), math.dist(q[1], q[2])
                    if max(a, bb) > 2.5 * min(a, bb):  # a straight stretch, not a corner
                        samples.append(min(a, bb) / k)
        thickness = statistics.median(samples) if samples else None
        region = valid_polygon(b.buffer(-T.hatch_gap * k).buffer(0))
        region = max(getattr(region, "geoms", [region]), key=lambda g: g.area) if not region.is_empty else b.buffer(0)
        area = region.area / k ** 2
        out.append({"g": region, "strokes": len(members), "cross": cross,
                    "stroke_len": statistics.median(s["len"] for s in members),
                    "family_deg": sorted(a for a, _ in top), "thickness": thickness,
                    "length": area / thickness if thickness else None, "area": area})
    return out


def closed_outlines(paths, k, tboxes):
    out = []
    for i, p in enumerate(paths):
        if p["type"] != "s":
            continue
        for g in subpolygons(p):
            if g.area / k ** 2 < T.outline_min_area or in_text(g, tboxes):
                continue
            o = oriented(g, k)
            if o["rectness"] > T.outline_rectness:
                out.append({"g": g, "o": o, "idx": i, "width_pt": p["width"]})
    kept = []
    for r in sorted(out, key=lambda r: -r["g"].area):
        if not any(abs(r["g"].area - q["g"].area) / q["g"].area < 0.02 and r["g"].intersection(q["g"]).area / r["g"].area > 0.98
                   for q in kept):
            kept.append(r)
    return kept


def band_runs(pieces, k):
    """Merge collinear band pieces into continuous runs."""
    groups = collections.defaultdict(list)
    for p in pieces:
        a = round(math.degrees(p["ang"]) * 2) / 2 % 180
        groups[(a, round(p["off"] / (T.band_offset * k)), p["width"])].append(p)
    runs = []
    for (a, _, w), ps in groups.items():
        ps.sort(key=lambda p: p["lo"])
        cur = None
        for p in ps:
            if cur and p["lo"] <= cur["hi"] + T.band_run_gap * k:
                cur["hi"] = max(cur["hi"], p["hi"])
                cur["pieces"].append(p)
            else:
                cur = {"ang": p["ang"], "off": p["off"], "lo": p["lo"], "hi": p["hi"], "width": w, "pieces": [p]}
                runs.append(cur)
    for r in runs:
        ux, uy = axis(r["ang"])
        nx, ny = -uy, ux
        P = lambda u, v: (u * ux + v * nx, u * uy + v * ny)
        half = r["width"] * k / 2
        r["poly"] = Polygon([P(r["lo"], r["off"] - half), P(r["hi"], r["off"] - half),
                             P(r["hi"], r["off"] + half), P(r["lo"], r["off"] + half)])
        r["axis"] = LineString([P(r["lo"], r["off"]), P(r["hi"], r["off"])])
        r["length"] = (r["hi"] - r["lo"]) / k
    # a run shorter than a typical band piece and fully inside another run is a duplicate
    runs.sort(key=lambda r: -r["length"])
    kept = []
    for r in runs:
        if any(q["poly"].buffer(T.run_dup * k).contains(r["poly"]) for q in kept):
            continue
        kept.append(r)
    return kept


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


def all_labels(texts, cfg):
    labels = parse_labels(texts, cfg)          # column, wall, retaining_wall, footing
    for kind, r in cfg["bands"].items():
        rx = re.compile(r["label"])
        for t in texts:
            m = rx.search(t["text"])
            if m:
                g = {n: v or "" for n, v in m.groupdict().items()}
                labels.append({"kind": kind, "raw": t["text"], "c": centre(t["rect"]), "key": r["key"].format(**g),
                               "groups": g, "building": None, "dims": None,
                               "numbers": [int(g["a"]) / 100, int(g["b"]) / 100]})
    return labels


class Buildings:
    """Building of a point: majority of the nearest labels that carry a building number."""

    def __init__(self, labels, k):
        self.ls = [l for l in labels if l.get("building")]
        self.k = k

    def of(self, pt):
        near = sorted(self.ls, key=lambda l: l["c"].distance(pt))[:T.building_k]
        near = [l for l in near if l["c"].distance(pt) < T.building_radius * self.k]
        if not near:
            return None, 0.0
        b, n = collections.Counter(l["building"] for l in near).most_common(1)[0]
        return b, n / len(near)
