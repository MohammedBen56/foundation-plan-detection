# Foundation-plan element detection

Finds every structural element on a vector PDF foundation plan exported from AutoCAD. It gives each one:
- a class (column, wall, footing, raft, beam, …);
- its type as printed on the plan (`P1/128`, `V3/123`, `S7`, `LG40x15`, …);
- its measured size;
- a bounding box and, where the shape is not a rectangle, its outline.

Colours and CAD layers are never used: they are assumed to be wrong or unknown. Elements are found from the geometry of the drawing. The printed labels only vote on what a group of similar shapes is, and give each element its type.

## Run

```bash
pip install -r requirements.txt            # Python >= 3.11
python geometry_classes.py "GHF_EXE_PLN_STR (1)-FONDATIONS (1).pdf" --out out
```

Each run writes the next free version, `out/geometry_classes_vN.*`, so earlier results stay for comparison.

| File | Content |
|---|---|
| `.pdf` | The plan with one PDF layer per class. The unclassified layer is off by default. Each box is labelled with its class, type and size, e.g. `Retaining wall VP20 18.13 × 0.20 m`. |
| `.json` | Every classified element, in pixels of the page rendered at 150 dpi (origin top-left): class, type, `display_label`, `bbox_px`, `obb_px` (the rotated box), `polygon_px` (non-rectangular shapes), size in m, area, net length of beams. |
| `.xlsx` | Summary by building (counts, sizes, total lengths; live formulas), every object with its features and the reason for its class, the labels and how each was matched, the conflicts, and the validation. |
| `.png` | The annotated page rendered at 150 dpi. |

`--reference out/detections.json` (made by the older `detect_structures.py`) is only used to fill the validation columns.

## Files

| File | Role |
|---|---|
| `geometry_classes.py` | The pipeline: objects → clusters → labels → classes → outputs. All its tolerances are in `G` at the top. |
| `element_stats.py` | Geometry helpers: straight segments, dashed lines, fills grouped by paint order, hatching, closed outlines, band runs. Tolerances are in `T`. |
| `detect_structures.py` | PDF loading, scale calibration, oriented boxes, the label parser. It also holds the first (colour-based) detector, kept only as the validation reference. |
| `plan_config.toml` | The plan's wording: label grammar (P, V, VP, S/SC, SF, LG), the raft stamp word, the pit note, the crane note. Change this for an office that labels differently. |

## How it works

### 0. Reading the page and its scale

- **Drawing:** PyMuPDF gives every vector path: fills, strokes, their pen width and paint order.
- **Text:** the AutoCAD SHX text is read from the PDF's text annotations.
- **Scale:** every dimension text (`1.35`) is paired with its dimension line, and a robust line fit gives points per metre. Here that is 1/200, 14.17 pt/m, with under 0.5 cm error at every angle.

From then on every tolerance is in metres, so the code does not depend on the paper scale.

### 1. Objects, from geometry only

Structural elements are drawn in a handful of ways, and each is detected on its own:

| Drawn as | Detected by | Typical elements |
|---|---|---|
| **Solid fill** | Fill paths painted one after the other and touching form one object (hatches come in pieces). A fill bent around corners is split into straight stretches. A wall drawn twice keeps one copy, plus any part that sticks out. | columns, shear and retaining walls |
| **Hatched area** | A family of short parallel strokes (both diagonals = cross-hatch). A bent hatched area (L, U, T of walls) is split into legs, using its two border lines. A wall that is both filled and hatched is merged into one. | core walls |
| **Closed outline** | Closed rectangles of lines. An outline around a wall already found is that wall's border and is merged into it. A thin outline with a pattern inside (dots or strokes) is a wall. | footings, pits, walls drawn as outlines |
| **Band** (two parallel lines) | Two parallel lines drawn with the same pen, not dashed, not the grid pen. The band widths used on the plan are found as peaks of all pair spacings. A band is **empty** (beam), **holding** a wall (strip footing), or **hatched** (a wall between its borders). Only a line of the same pen between the edges splits a band, and only the held wall's own border may lie inside a strip. | grade beams, strip footings |
| **Stamp** | A circle with rays from the raft's corners. The raft's outline is traced from the lines joined end to end around it. | rafts |

The grid is recognised as dash chains, and its pen is learned from them. Text is recognised as strokes inside a text box that are no longer than the text is tall. Both are then ignored.

### 2. Clusters: similar shapes together, still without text

Every object gets geometric features:
- length, width, aspect;
- the **EN 1992-1-1 §5.3.1(7) test** (long side ≤ 4× short side = column section, else wall);
- what it touches and what it stands in: footing outline, strip band, raft;
- whether it lies on the building's edge;
- the pen of its own outline;
- for outlines, how many solids and dots are inside.

Objects of each drawing kind are clustered (Ward). The number of clusters is chosen by silhouette. There are deliberately more clusters than classes; naming merges them back.

### 3. Labels vote

Labels are parsed with the grammar in `plan_config.toml`. A first, neutral assignment (Hungarian algorithm) pairs every label with one object. The cost is distance plus misalignment with the text. A label may stay unmatched.

Each cluster takes the class most of its labels name. It needs at least 5 votes and 1.5× the runner-up, or a smaller clear majority with enough members labelled. So a few wrong or misplaced labels are outvoted.

Two safeguards:
- **Home kind:** a label type only votes for the drawing kind where its matches come out size-consistent per type (a type names one size). V labels therefore vote for solids, not for the beams that pass through the walls.
- **Unnamed clusters:** a cluster without enough votes takes the name of the nearest named cluster only if it is close in feature space.

### 4. Types and sizes

A second assignment matches labels only to objects of their own class. It penalises objects whose size differs from the one their type has elsewhere on the plan, and objects in another building's region.

Buildings are found from the drawing: expansion joints are pairs of long lines 3–16 cm apart, the footprint is cut along them, and each region takes its building from a label vote.

Unlabelled members get the type of their size in their building, marked `~` (inferred). If several types fit they get `~A|B`; if none fits they get no type. A printed label whose element doesn't have its type's size is re-typed when at least 2 other elements agree, otherwise flagged as a conflict.

### 5. Rules that need no label

- A column-shaped piece drawn like a wall is a wall piece, not a column. "Drawn like a wall" means its outline pen is none of the P-labelled columns' pens, or it continues a wall of the same thickness.
- If a plan has no P labels at all, an unlabelled solid of column shape (L ≤ 4W) is a column.
- An unlabelled wall is:
  - a **retaining wall** if it lies along the building's edge or stands on a strip footing; it takes the VP type of its thickness, e.g. `~VP20`;
  - otherwise a **core wall**.

## What it assumes about a foundation plan, and what it learns from each plan

**Assumed** (drafting conventions; if one doesn't hold, the listed part fails):

| Convention | Used for | If it doesn't hold |
|---|---|---|
| Vector PDF, not a scan | everything | nothing is found. A scan needs a raster detector first. |
| Dimension texts next to dimension lines | scale | the run stops (the scale fit reports how many dimensions agree) |
| Elements drawn as fills, hatches, closed outlines or same-pen line pairs | objects | an element drawn another way (e.g. a wall as two lines of different pens) is missed |
| Grid axes dashed | ignoring the grid | grid lines could pair up as bands |
| Labels written next to their element | votes and types | votes go to neighbours. Majority voting absorbs some of this, not systematic offsets. |
| Label wording as in `plan_config.toml` | reading labels | edit the config for another office's wording |
| Raft = stamp circle with rays from its corners | rafts | this is this office's convention. Another office's rafts would need another detector. |
| Text available as PDF text or annotations | labels | pure stroked text would need OCR |

**Learned from each plan** (nothing of this plan is written into the code):
- the scale;
- the grid pen;
- the band widths in use;
- the pens columns are drawn with;
- the size of every type in every building;
- which drawing kind each label type belongs to;
- the clusters;
- the buildings and joints;
- the VP thicknesses.

**Tolerances:** all in `G` / `T`, in metres or as shares, e.g. same size within 3 cm, parallel within 15°, footprint grown 1 m and closed 3 m.

## How sure are we that it isn't tuned to this plan?

- **Validation:** against the elements verified earlier, it finds 35/35 columns, 88/88 shear walls, 5/5 retaining walls, 72/73 isolated footings and 29/29 combined footings.
- **Every tolerance moved −30% and +30%, one at a time (117 runs, on v4):**
  - 84 runs change nothing and 105 change under 1% of classes.
  - The one real judgement knob is how big a notch in the footprint still counts as the building's edge (3.6% at −30%).
- **Labels removed (on v4):**
  - 25–50% removed at random keeps 96–99.7% of classes.
  - With all P labels removed, 31/35 columns are still found (the EC2 rule).
  - With all LG labels removed, beams are left unclassified, not misnamed.
- **What removing labels does not survive:**
  - Without V labels, every interior wall is a core wall, by the rule above.
  - Without S/SC labels, footings are left unclassified: no rule names them yet.
  - At 75% loss, one random draw in three can't name the shear walls.

Details: `reports/v4 audit and decision tree study.md`.

## Known limitations

- **One plan tested.** The real test of generality is a second plan from another office.
- **Double-drawn walls:** at about 30 places a wall is drawn twice, e.g. a short V wall on top of a long retaining-wall fill. Both are kept, so that stretch is counted twice in a quantity take-off.
- **Pits** are drawn as three nested boxes. Each box is its own object, and two innermost boxes are classed isolated footings.
- **Walls along an expansion joint** are core walls. Whether they count as "the building's edge" is an open question.
- **Label clutter:** labels of stacked elements (a wall inside a beam band) can overlap in the PDF. Turn class layers on and off to read them.
- **Footings and beams need labels:** without labels they stay unclassified. Rules like "an outline holding one column/wall is an isolated footing, two or more a combined footing" and "an empty band joining footings is a grade beam" would remove that dependency. They are not written yet.
