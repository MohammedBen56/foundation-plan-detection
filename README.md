# Structural element detection on a foundation plan

Detects the structural elements on a vector PDF foundation plan exported from AutoCAD. Every element found gets:
- a class (column, wall, footing, raft, beam, …);
- its type as printed on the plan (`P1/128`, `V3/123`, `S7`, `LG40x15`, …);
- its measured size;
- a bounding box: axis-aligned, rotated, and the true outline when the element isn't a rectangle.

Elements are found from the drawing's geometry. **No colour and no CAD layer is read.** The printed labels only vote on what a group of similar shapes is, and give each element its type.

## Deliverables

| File | Content |
|---|---|
| `out/annotated.pdf` | **The annotated drawing**, written by the run (not in the repository, see below). Each element is boxed and labelled with its class, type and size, e.g. `Column P1/128 1.00 × 0.25 m` or `Retaining wall VP20 18.13 × 0.20 m`. There is one PDF layer per class, so classes can be switched on and off. Objects no label could name are in one hidden layer per group. |
| `out/annotated.png` | The same page rendered at 150 dpi, also written by the run. The JSON pixel coordinates refer to this image. |
| `out/detections.json` | **The detections**, machine-readable (format below). Included. |
| `detect_elements.py` | The detector: objects, clusters, labels, classes, outputs. Its tolerances are in `G`. |
| `drawing.py` | Reading the PDF, the scale, geometric primitives and the label grammar. Its tolerances are in `T`. |
| `plan_config.toml` | The plan's label wording (P, V, VP, S/SC, SF, LG, Radier, Fosse) and the scale-fit settings. |

## Run

The plan is confidential, so neither it nor anything rendered from it is in this repository. `*.pdf` and `*.png` are git-ignored.

1. Copy the plan into this folder, next to `detect_elements.py`: `GHF_EXE_PLN_STR (1)-FONDATIONS (1).pdf`.
2. Run:

```bash
pip install -r requirements.txt        # Python >= 3.11
python detect_elements.py "GHF_EXE_PLN_STR (1)-FONDATIONS (1).pdf" --out out
```

This writes `out/annotated.pdf`, `out/annotated.png` and `out/detections.json`. The plan can also be anywhere else; pass its path instead.

It takes about 2 minutes. `--dpi` sets the PNG's resolution and therefore the JSON's pixel scale (default 150). `--config` points to another label grammar.

## Results on this plan

The scale found is 1/200: 222 of 266 dimension texts agree, with a median error of 0.24 cm. 573 of 590 labels are matched to an element.

| Class | Count | | Class | Count |
|---|---|---|---|---|
| column | 35 | | isolated footing | 78 |
| shear wall | 136 | | combined footing | 30 |
| retaining wall | 83 | | strip footing | 51 |
| core wall | 138 | | grade beam | 118 |
| raft | 14 | | sump pit | 5 |

`element_type` groups these into the brief's categories: column, wall, footing, slab_raft, beam, other.

## `detections.json`

- `image`: the dpi and size of `annotated.png`. Pixels have their origin top-left, with y pointing down.
- `calibration`: the scale found.
- `counts`: elements per class.
- `detections`: one entry per element:
  - `id`, `element_type`, `class`, `display_label`;
  - `type_key`: as printed, or `~key` when inferred, or `~A|B` when several types fit;
  - `type_source`, `label` (the matched text), `class_named_by` (why it has its class), `building`;
  - `bbox_px`: axis-aligned `[x0, y0, x1, y1]`;
  - `obb_px`: the rotated box's 4 corners;
  - `polygon_px`: the true outline, for non-rectangular elements such as rafts;
  - `size_m` (`length`, `width`), `angle_deg`, `area_m2`;
  - `net_length_m`: a beam's length outside the footings it crosses;
  - `nested_boxes_m`: the inner boxes of an element drawn as nested boxes, e.g. a pit.
- `unidentified_groups`: groups of similar objects that no label named, each with a description generated from their geometry (e.g. "13 closed outlines, 0.60 x 0.60 m; with a pattern inside") and their members' boxes. They are reported, not claimed as elements.

## Approach

1. **Page and scale.** PyMuPDF gives every vector path (fill or stroke, pen width, paint order) and the SHX texts, which AutoCAD stores as PDF annotations. Every decimal number (`1.35`) is paired with the line under it, and a robust fit gives the scale. Every tolerance after that is in metres.
2. **Objects, from geometry only.** Each way of drawing an element is detected on its own:
   - **filled shapes:** touching fills painted one after another form one object; bent ones are split into straight stretches;
   - **hatched areas:** families of short parallel strokes; a bent area is split into straight legs using its border lines;
   - **closed outlines:** an outline around a wall already found is merged into it; nested boxes around one centre are one element;
   - **bands:** two parallel lines drawn with the same pen. The widths in use are found as peaks of all pair spacings, and each band is empty, holds a wall, or is hatched;
   - **raft stamps:** a circle with rays from the raft's corners; the outline is traced from the lines around it.

   Grid axes (dashed lines, whose pen is learned) and text strokes are recognised and ignored.
3. **Clusters.** Each object gets geometric features:
   - size and shape;
   - the EN 1992-1-1 §5.3.1(7) column/wall test (section ≤ 4× width = column);
   - what it touches or stands in;
   - whether it lies on the building's edge, which is found from the footprint of all walls;
   - its outline pen;
   - what it contains: filled shapes, a dot pattern, nested boxes.

   Objects of each drawing kind are clustered (Ward). The number of clusters is chosen by silhouette.
4. **Labels vote.** A first global assignment (Hungarian algorithm; cost = distance + misalignment with the text) pairs labels with objects. A cluster takes the class most of its labels name, and needs a clear majority to do so.
   - A label type votes only for the drawing kind where its types come out size-consistent.
   - A cluster's name is not extended to members far outside the sizes its labels cover.
5. **Types.** A second assignment, restricted to each label's class, gives every element its type. It penalises sizes that differ from the type's size elsewhere, and elements in another building. Buildings are found from expansion joints (pairs of long, close lines) and label votes.
   - Unlabelled elements take the type of their size, marked `~`.
   - A printed label whose element has the wrong size is re-typed when other elements agree, otherwise flagged.
6. **Rules that need no label:**
   - an unlabelled wall along the building edge, or on a strip footing, is a retaining wall and takes the VP type of its thickness;
   - any other unlabelled wall is a core wall;
   - a column-shaped piece drawn like a wall is a wall piece;
   - on a plan without P labels, the EN 1992 test decides columns.

## Assumptions

- **Vector PDF.** A scan would need a raster front end: line and fill extraction plus OCR. At 1/200, 1 cm on site is under 1 pixel at 300 dpi.
- **Dimension texts** sit on their dimension lines (for the scale).
- **Elements are drawn** as fills, hatches, closed outlines or same-pen line pairs; **grid axes** are dashed.
- **Labels** are written next to their element, with the wording in `plan_config.toml`. Text is stored as PDF text or annotations; pure stroked text would need OCR.
- **This office's conventions:**
  - a raft is marked by a stamp circle with corner rays;
  - pits are drawn as nested boxes;
  - hatch strokes are about 10 cm apart (`T.hatch_gap`).

## Limitations

- **Core walls and unlabelled retaining walls** come from rules, since no label names them. The edge rule treats walls at expansion joints as interior.
- **Walls drawn twice** (about 30 places, e.g. a short V wall over a long retaining wall) are both kept, so those stretches would be counted twice in a quantity take-off.
- **Footings and beams need their labels:** without them they stay in unidentified groups.
- **Labels of stacked elements** (a wall inside a beam band) can overlap in the PDF. Switching layers separates them.
- **One plan tested.** The real test of generality is a plan from another office.

**How robust it is on this plan**, measured during development:
- Against a hand-verified set of elements it found 35/35 columns, 88/88 shear walls, 5/5 retaining walls, 72/73 isolated footings and 29/29 combined footings.
- Moving each of the 97 tolerances by ±30% left 152 of 205 runs unchanged and 183 under 1% change.
- Removing 25–50% of the labels at random kept 96–99.7% of the classes.

## External models and APIs

None. There is no machine-learning model, no pretrained weights and no online API. The code uses open-source libraries only:
- PyMuPDF (reading the PDF, writing the annotated PDF and PNG);
- Shapely (geometry);
- NumPy and SciPy (Hungarian assignment);
- scikit-learn (Ward clustering, silhouette score).
