# v4: hard-coding audit, robustness tests and decision-tree study

Outputs: `out/geometry_classes_v4.pdf` (one layer per class; the unclassified layer is off by default), `.xlsx`, `.json`, `.png`.
Code: `geometry_classes.py`, `element_stats.py`, `plan_config.toml`.

## 1. Your spots, fixed and checked on the v4 PDF

| Spot | v3 | v4 |
|---|---|---|
| Retaining wall at (2407,821) | lost | retaining, continuous 295 + 92 + 418 cm. The 92 cm piece was drawn twice with different lengths; the part that sticks out is kept now. |
| Wall (2085,1717) → (2408,1657) | lost | retaining (11.45, 17.00, 11.70, 8.92 m pieces) |
| Wall at (361,780) | lost | retaining (VP20) |
| Core walls at (342,675), (660,1300) | one misaligned box | one aligned box per leg. The diagonal wall was drawn both filled and hatched and counted twice; it's one wall now. |
| S11-122 | V2/122 + a false `~V5/122` | V2/122 only |
| (2195,994) | unlabelled shear box | nothing wrong. The two walls on the SF4 strip are expansion-joint walls, classed core_wall (see open question). |
| S5-123 | V4/123 + a wider box | V4/123 only |
| SC3 on the 123/124 border | wider box | P3/122, V3/123, V3/124 only |
| Rafts | oriented rectangles | traced outlines drawn as polygons. All 14 outlines lie 100% on drawn lines. 8 are non-rectangular; 6 are rectangles in the drawing itself (area within 1% of their box). |

Validation against the verified reference is unchanged: columns 35/35, shear walls 88/88, retaining 5/5, isolated footings 72/73, combined footings 29/29. Workbook formulas evaluate with 0 errors.

## 2. Hard-coding audit

**Never used to detect or classify:**
- colours;
- CAD layers;
- coordinates;
- building numbers;
- element counts;
- sizes of this plan's types. Type sizes are read from the labels on each plan.

**Plan wording, all in `plan_config.toml`:**
- the label grammar (P, V, VP, S/SC, SF, LG);
- the raft stamp and pit note words (moved there this round from the code);
- the crane note.

**Fixed this round:**
- the raft and pit words were in the code;
- two distances were in PDF points, so they depended on the page scale: the stamp number's distance (15 pt) and the glyph length (12 pt). They're now relative to the text size or in metres;
- a dozen unnamed numbers were given names in `G`: size and width tolerances, parallel angle, touch distance, footprint growth and closing, edge margin, matching distance unit, column pen share.

**What remains are tolerances, all in metres or as shares, so independent of scale:**

| Kind | Examples | Risk on another plan |
|---|---|---|
| Drafting precision | touch 2 cm, size ±3 cm, width ±2 cm, band width ±1.5 cm, parallel 15°, rectness 0.95 | low: CAD precision is the same everywhere |
| Construction knowledge | EN 1992 column L ≤ 4W, band ≤ 1.5 m, joint gap 3–16 cm, stamp 0.8–3 m, footing ≥ 0.5 m², footprint grown 1 m and closed 3 m | low to medium: typical RC foundations |
| Voting / matching knobs, set while looking at this plan | votes ≥ 5, margin 1.5×, similarity ≤ 1.6, k 6–14, feature weights, cost weights | the ones to watch, see §3 |

## 3. Robustness tests

### Every knob moved −30% and +30%, one at a time (117 runs)

- 84 runs change nothing; 105 change under 1% of the 672 classified objects.
- The verified classes are unchanged in all runs but two.
- Remaining sensitive spots:
  - `affinity_min > 1`: this switches the home-kind mechanism off entirely; it's not a real setting. 21% of classes change, which shows the mechanism is needed.
  - `footprint_close` −30%: 3.6%. Notches up to 4 m then count as "outside", so interior walls there become retaining. This is a genuine judgement: how big a notch counts as the building's edge.
  - `k_range` −30%: 2.2%. Parallel angle −30%: 1.5%.

The sweep also found four real defects, now fixed:
- a crash in `band_families` (division by zero);
- band widths off by one histogram bin (the "0.41" beams were 0.400; the width is now the median spacing of the pairs);
- the strip-band fill ceiling sitting on a knife edge (0.7, now 0.9; any value from 0.63 to 0.99 gives identical results);
- the edge distance silently tied to the footprint growth (now expressed as a margin beyond it).

### Label dropout (your "other plans may miss labels" concern)

| Labels removed | Before the fixes | After |
|---|---|---|
| All P labels | all 35 columns turned into walls | 31/35 verified columns kept; 4 core-wall pieces become columns |
| All LG labels | 114 beams became shear walls, all shear walls became core | beams stay unclassified, nothing wrong |
| 25% at random (3 draws) | one draw: every shear wall became core | 97–99.7% of classes kept |
| 50% at random | one draw: every shear wall became core | 96–97.5% kept |
| 75% at random | 77–84% kept | 77–89% kept. One draw still loses the shear-wall naming: no wall cluster gets more than 3 votes. |
| All V labels | all walls become core/retaining | same. Your rule makes an unlabelled interior wall a core wall. |
| All S/SC labels | footings unclassified, 17 named sump pit | same (limitation: pits and footings look alike) |

Fixes:
1. **Column fallback:** on a plan with no P-labelled columns, an unlabelled solid no longer than 4× its width (EN 1992-1-1 §5.3.1(7)) is a column, unless it continues a same-thickness wall.
2. **Home kind:** a label type's home drawing kind is where its matches come out size-consistent per type key (a type names one size), not merely where it lands most. V labels had been landing on the beams running through the walls almost as often (55 vs 71).

**Columns and labels:** columns never needed a label each. A column cluster is named by the P labels voting on it. Unlabelled members stay columns. A member is demoted only if its outline pen matches none of the P-labelled columns' pens, or if it continues a same-thickness wall. With no P labels at all, the EC2 shape rule now takes over.

## 4. Decision tree: feasible, and is it recommended?

Data: the 469 objects whose own printed label confirmed their class (11 buildings), with 23 geometric features (the same ones the clustering uses, plus rectness and band width).

| Test | Result |
|---|---|
| Trained on 10 buildings, tested on the 11th | 97.4% accuracy, macro-F1 0.97 (per building 92–100%) |
| Same with class weighting 'balanced' | 83.8%. The 3 labelled pits get the weight of 135 walls: fragile with rare classes. |
| Applied to the objects the pipeline classed by rule | it can't produce core_wall at all: no label ever names one, so no training example exists. 86 of the 120 become retaining, 22 column. |
| Reproducing the full pipeline, including rule classes | 94.4%; misses where shear vs core depends on the V label, not the shape |
| Labels needed | 10% of the labels (46) gives 83%, 20% (93) gives 93%, 40% gives 95% |

What the tree learns:
- **General rules it rediscovers:** L ≤ 4.18W → column (the EC2 rule, found from data); two or more solids inside → combined footing; outline pen; on the edge or on a strip → retaining.
- **This plan's sizes:** width ≤ 0.30 m → retaining; area ≤ 0.307 m² → column; aspect > 13.2 → retaining; band ≤ 0.60 m → beam. These are exactly the hard-coded numbers you want to avoid, learned instead of typed.

**Recommendation:**
- **Don't** freeze a tree trained on this plan and run it on other plans. It would carry this plan's wall thicknesses and section sizes, and one plan can't tell which thresholds transfer. The literature on engineering-drawing recognition names generalisation across drawing styles as the open problem.
- **Do** consider it as a per-plan self-training step. On each new plan, its own confidently matched labels (label on the element, size consistent with its type) train a shallow tree, which then classifies the unlabelled objects. The thresholds are then re-learned per plan, never typed. This is standard self-training / weak supervision: rules and labels as noisy supervisors (Snorkel's labelling functions; self-training of decision trees).
  - It would replace the similarity-naming step (Ward clusters + distance ≤ 1.6).
  - It needs roughly 50–100 confident labels on the plan.
  - Classes that no label ever names (core walls) stay rule-based.
  - Low-purity leaves should answer `?`.
- **A cross-plan model** needs 10–20 corrected plans from different offices, with plan-relative features (thickness ÷ the plan's typical wall, pen ÷ the plan's thickest pen). It would be judged by leaving one plan out.

Sources:
- [Semi-supervised self-training for decision tree classifiers (Tanha et al., IJMLC)](https://link.springer.com/article/10.1007/s13042-015-0328-7)
- [Revisiting self-training with regularized pseudo-labeling for tabular data](https://arxiv.org/abs/2302.14013)
- [Snorkel: rapid training data creation with weak supervision](https://arxiv.org/pdf/1711.10160)
- [Deep learning for symbols detection and classification in engineering drawings](https://www.sciencedirect.com/science/article/abs/pii/S0893608020301957)

## 5. Open question for you

Walls on both sides of an expansion joint (e.g. the two 6.6 m walls on the SF4 strip at (2195,994)) are on the border of their own building but inside the complex. They are core walls now. Your rule "long line in the border of the building → retaining" could mean either.
