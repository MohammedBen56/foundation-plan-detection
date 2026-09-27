# Text-to-Entity Label Association in Technical Drawings

Scope: linking text labels (tags, marks, dimensions, room names) to the geometric entities they describe in P&IDs, engineering, electrical, mechanical, floor and structural plans. The focus is on training-free or low-data methods.

Note on method: this search covered about 22 tool calls. Several primary PDFs (Digitize-PID full text, the USPTO patent 12288411, Dori's dimensioning-text paper, Habed & Boufama) could not be parsed or reached from this environment. Claims about their internals are therefore limited to abstracts and snippets, and this is flagged below.

## 1. Published methods for linking labels to entities (P&ID, engineering, mechanical, floor plans) and the features they use

### Takeaway
Published P&ID and drawing pipelines almost all use hand-crafted geometric rules, not learned association:
1. First, test containment or overlap of the text inside a symbol.
2. Then test whether a line passes through the text with the same direction.
3. Only then fall back to nearest-entity rules: Euclidean distance, or axis-aligned ray casting from the text box edges.

Reported accuracy is modest (about 64–97% in early work, and about 10% error in crowded regions). Many recent deep-learning P&ID papers do not address text association at all. This is an open gap, not a solved component.

### Cited Findings
- **Kim et al. (JCDE 2022), end-to-end P&ID digitisation, rule cascade.**
  - "Clear connection" cases come first. (a) Containment: text is inside a symbol when ≥99% of the text box area overlaps the symbol box. (b) Line penetration: a line fully passes through the text, the text direction matches the line direction, and the text width fits within the line. (c) Substantial overlap: >50% area overlap, and the text width or height is fully contained in the symbol width or height.
  - "Uncertain" cases: "the nearest symbol or line is the one that is reached first when a ray is shot along each coordinate axis from the centre of each edge of the text bounding box."
  - Tag IDs are recognised by regex patterns for plant numbering schemes. — [Kim et al., JCDE 9(4):1298, 2022](https://academic.oup.com/jcde/article/9/4/1298/6611631)
- **Rahul et al. (2019), "Automatic Information Extraction from P&IDs".**
  - Pipeline codes go to "the nearest pipeline based on the minimum euclidean distance from any vertex of the bounding box … to the nearest point on the line."
  - Symbols go to the closest pipeline "provided it is not separated from the pipeline". This is an explicit occlusion/separation condition.
  - Inlet/outlet tags use direction: the system finds "the line emerging direction from the orientation of inlet and outlet" and picks the closest pipeline in that direction.
  - Reported association accuracy: pipeline code 41/64 (64%), outlet 14/21 (66.5%), inlet 31/32 (96.8%). Failures were attributed mostly to upstream pipeline-detection errors (65.2%). — [Rahul et al., arXiv 1901.11383](https://ar5iv.labs.arxiv.org/html/1901.11383)
- **Digitize-PID (TCS, 2021)** describes an end-to-end pipeline that detects pipes, symbols and text, then does "their association with each other and eventually, the validation and correction of output data based on inherent domain knowledge". It released a synthetic 500-P&ID dataset. I could not parse the full text to extract the association rules. — [Paliwal et al., arXiv 2109.03794](https://arxiv.org/abs/2109.03794)
- **Microsoft ISE P&ID digitisation.**
  - Text is associated with symbols by spatial proximity using Shapely distance computations.
  - Text outside the main drawing area is removed first to reduce confusion.
  - They report "an average error rate of 10%" when symbols are crowded. — [Microsoft ISE Dev Blog](https://devblogs.microsoft.com/ise/engineering-document-pid-digitization/)
- **Relationformer-based P&ID-to-graph work (PID2Graph)** uses text detection (CRAFT/EasyOCR) only "for filtering purposes". It does not associate text with symbols. Hungarian matching appears there only for evaluation (gIoU cost between predicted and ground-truth nodes), not for text linking. — [arXiv 2411.13929](https://arxiv.org/html/2411.13929v1); [PID2Graph dataset, Zenodo](https://zenodo.org/records/14803338)
- A granted US patent is titled "Techniques for extracting associations between text labels and symbols and links in schematic diagrams". It shows industrial interest, but I could not parse its claims. — [USPTO 12288411](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12288411)
- **Floor plans (Ahmed, Liwicki, Weber, Dengel, DAS 2012).**
  - Room detection plus OCR on a separated text layer, with detected room-type text used to label room regions.
  - When several semantic labels fall in one physical room, the room is split into sub-regions.
  - The method fails on plans without text. — [Semantic Scholar entry](https://www.semanticscholar.org/paper/Automatic-Room-Detection-and-Room-Labeling-from-Ahmed-Liwicki/2b65af651a0c72eb854c27c5d60b9310560b7ef9); [ResearchGate](https://www.researchgate.net/publication/230675743_Automatic_Room_Detection_and_Room_Labeling_from_Architectural_Floor_Plans)
- **Floor-plan OCR text is noisy** (abbreviations, typos, merged words like "kitchendining"). Practical systems split compound names against a room-type vocabulary, and otherwise snap to the nearest known type by Levenshtein distance. — [Kaliada, Medium (practitioner blog, lower authority)](https://andrey-koleda.medium.com/how-i-built-a-computer-vision-system-that-understands-doors-on-floor-plans-ec3dbf0e29c7)
- **Mechanical dimensions.**
  - Classic work (Dori 1989, "syntactic/geometric approach to recognition of dimensions"; Dori & Velkovitch on dimensioning text) groups text with arrowhead pairs and leader/witness lines into "dimension sets", following ANSI drafting standards.
  - Dimension-set components include "textbox guide sets that are single bars", guides that coincide with pairs of arrowhead tails, and bounding-point sets that are pairs of leader endpoints. — [Dori & Velkovitch, CVIU (abstract)](https://www.sciencedirect.com/science/article/abs/pii/S1077314297905853); [Lai & Kasturi, "Detection of dimension sets in engineering drawings"](https://www.semanticscholar.org/paper/Detection-of-dimension-sets-in-engineering-drawings-Lai-Kasturi/0b376498f6063c1d1130280b301d00407185b8b6)
- A leader line is a thin solid line with an arrow at an angle, indicating the feature a dimension or note is associated with. It is the explicit link when present. — [EngineeringTechnology.org](https://engineeringtechnology.org/engineering-graphics/line-conventions-and-lettering/dimension-extension-and-leader-lines/)
- **eDOCr (2023, mechanical-drawing OCR)** explicitly does not link dimensions to features. It only clusters nearby words, and it breaks when text orientation deviates strongly from the baseline. Its recognisers were trained on synthetic data (10k–50k samples). — [Frontiers in Manufacturing Technology 2023](https://www.frontiersin.org/journals/manufacturing-technology/articles/10.3389/fmtec.2023.1154132/full)

### Inferences
- The features used in the literature are: containment/overlap ratio; distance from the text box to the nearest point on the entity (not centroid to centroid); direction (axis rays from the text edges, or the line's emerging direction); co-linearity of text direction with the line axis; a "not separated" (visibility) condition; and regex/type compatibility of the tag string with the symbol class.
- Every cited method is greedy and per-label. None enforces global one-to-one or capacity constraints. That fits the reported failure mode (crowded regions, ~10% error), and a global assignment step is the obvious improvement.
- Build a staged cascade:
  1. Explicit links first: leader lines, multileaders, and a line penetrating the text.
  2. Then containment.
  3. Then a global cost-based assignment for the remainder.

  This mirrors Kim et al. and removes easy cases before any optimisation.

### Gaps
- I could not extract the exact Digitize-PID association and correction rules, or the USPTO 12288411 claims (PDF parsing unavailable).
- I found no public benchmark numbers for text-to-symbol association specifically. PID2Graph labels symbols and edges, not text links.
- I did not find a structural-plan paper (beam/column mark to member) with association details.

## 2. Global assignment formulations (Hungarian, min-cost flow, ILP, CRF), one-to-many, unlabelled elements, duplicate labels

### Takeaway
The closest published, training-free analogue is label–marker matching on historical maps (Budig, van Dijk, Wolff). They solve global minimum-cost matching in polynomial time over thousands of elements and report about 99% correct matches. They add a sensitivity analysis that flags only the ambiguous assignments for review. This transfers directly to drawings.

### Cited Findings
- **Budig, van Dijk & Wolff, "Matching Labels and Markers in Historical Maps: An Algorithm with Interactive Postprocessing"** (ACM TSAS 2(4):13, 2016; earlier at MapInteract 2014).
  - Inputs are bounding boxes of markers and labels.
  - Assignment is by "a minimum cost matching approach" using distance and the surrounding elements, rather than nearest-neighbour.
  - It runs in polynomial time: "a couple of seconds" for thousands of elements.
  - A sensitivity classifier surfaces uncertain matches for human review. — [Project page](http://historical-maps.github.io/); [BibSonomy record](https://www.bibsonomy.org/bibtex/552fa4282c58da9438e47c6997c866c2)
- On maps with over 4,000 markers and labels, the algorithm correctly matches 99% of labels and is robust to noisy input (reported in search snippets of the paper abstract). — [ACM DL entry](https://dl.acm.org/doi/10.1145/2677068.2677073)
- Hungarian matching (bipartite, minimising total cost) is used in P&ID transformer work, though only to match predictions to ground truth by gIoU cost. — [arXiv 2411.13929](https://arxiv.org/html/2411.13929v1)
- Ahmed et al. handle "several labels in one room" by splitting the room into sub-regions. This is a structural answer to many labels per entity. — [Semantic Scholar](https://www.semanticscholar.org/paper/Automatic-Room-Detection-and-Room-Labeling-from-Ahmed-Liwicki/2b65af651a0c72eb854c27c5d60b9310560b7ef9)

### Inferences
These are implementation designs derived from standard OR. They are not taken from a drawing-specific paper.

- **Rectangular LSA with dummies.** Build a cost matrix C[label, entity]. Pad it with "unassigned" dummy columns (cost τ_label) and "unlabelled" dummy rows (cost τ_entity), so that abstaining is allowed. Solve with `scipy.optimize.linear_sum_assignment`. Set infeasible pairs (wrong type, blocked line of sight) to a large constant rather than inf.
- **One label, many entities (e.g. "B1 TYP." or "4× W12x26").** Use min-cost flow or b-matching instead of LSA:
  - Edges run source → label (capacity = multiplicity k parsed from the text: "TYP", "×4", "(4)", "EQ"; otherwise 1) → entity (capacity 1) → sink.
  - Each entity node also has an optional edge to the sink via an "unlabelled" arc with penalty.
  - Solvers: networkx `min_cost_flow`, or OR-Tools `SimpleMinCostFlow`.
  - Integrality of min-cost flow gives integer solutions without an ILP.
- **Duplicate labels** (the same string placed twice, e.g. at both ends of a beam or on two views): let two label nodes map to the same entity by giving the entity capacity >1 when the strings are identical. Alternatively, merge identical-string labels whose best entities coincide before assignment.
- **ILP (PuLP/OR-Tools CP-SAT)** is needed only when constraints are non-flow. Examples: "all entities receiving label X must have the same size" (see §4), mutual exclusion between crossing leader interpretations, or "a label may not be assigned across a gridline". CP-SAT handles thousands of binaries fast.
- **Ambiguity/confidence.** Following Budig et al., compute a margin per assignment. Either use (second-best cost − best cost), or re-solve with that pair forbidden and measure the global cost increase. Flag low-margin pairs for review.
- **Probabilistic/CRF view.** Costs can be read as −log p. A pairwise CRF over labels, with a penalty when two labels claim the same entity, becomes the same matching when solved by MAP. Loopy BP or learned weights would need data. In a training-free setting, hand-set weights plus LSA/flow are simpler and exact.

### Gaps
- I could not read the full Budig et al. cost function; the paper is behind the ACM paywall. Their TSAS paper and Budig's 2018 Würzburg PhD thesis are the key primary references.
- I found no drawing-domain paper using min-cost flow or ILP specifically for text association. The recommendations above are transfers from OR and cartography.

## 3. Visibility / line-of-sight / Voronoi ownership and CAD/cartographic text placement conventions

### Takeaway
"Not separated from the entity" (Rahul et al.) and "first entity hit by an axis ray" (Kim et al.) are the published forms of a visibility test. Cartographic label placement gives a strong prior on where a label sits relative to its feature: an 8-position candidate model with a top-right preference (Imhof). That prior can be inverted into a direction-dependent cost.

### Cited Findings
- Visibility conditions in practice:
  - A symbol is linked to the closest pipeline only "provided it is not separated from the pipeline". — [Rahul et al.](https://ar5iv.labs.arxiv.org/html/1901.11383)
  - Axis-aligned ray casting from each text edge centre, taking the first symbol or line hit. — [Kim et al. 2022](https://academic.oup.com/jcde/article/9/4/1298/6611631)
- Point-feature label placement (PFLP): the 8-position model is the most prevalent. Imhof (1962) proposed a 5-position model for left-to-right scripts, with top-right as the most favoured position. The reasoning is that Latin text has more ascenders than descenders, so a label above looks closer to its point. — [arXiv 2407.11996, "From Top-Right to User-Right"](https://arxiv.org/html/2407.11996v1); [Christensen, Marks, Shieber, ACM TOG 1995](https://www.eecs.harvard.edu/~shieber/Biblio/Papers/tog-final.pdf)
- A leader indicates the associated feature. It is drawn "under an angle", so its arrow/terminal end lies on or points at the feature. — [EngineeringTechnology.org](https://engineeringtechnology.org/engineering-graphics/line-conventions-and-lettering/dimension-extension-and-leader-lines/)
- In the line-penetration case, text direction is compared with line direction (text co-linear with the line it annotates). — [Kim et al. 2022](https://academic.oup.com/jcde/article/9/4/1298/6611631)

### Inferences
These are designs, not published results.

- **Visibility cost.** For a label L and candidate entity E, take the segment from the nearest point on L's box to the nearest point on E. Count crossings with other entities' geometry (Shapely `intersects` against an STRtree of other entities, excluding E and excluding thin annotation layers such as dimension lines and hatches). Add λ·crossings, or make it a hard block when crossings are ≥1.
  - Casting several rays (to E's nearest point, centroid, and axis projections) and taking the minimum crossing count is more robust to small gaps.
- **Voronoi / "zone of influence" ownership.**
  - For linear members (beams, pipes, walls), compute a generalised (segment) Voronoi diagram, or approximate it by rasterising entities and running a distance transform with nearest-label propagation (`scipy.ndimage.distance_transform_edt(return_indices=True)`).
  - A label's owning cell is a strong candidate.
  - Use this as a candidate generator (top-k entities per label: those whose cells the label box touches, plus their neighbours), not as the final decision. The assignment step then resolves conflicts.
  - This matches the room case: room labels live inside the room polygon, so for floor plans point-in-polygon is the primary test, with the distance transform as fallback for labels placed on walls.
- **Orientation compatibility.** For elongated entities, add the term w·(1 − |cos(θ_text − θ_entity)|). Drafting convention writes member marks parallel to the member, and vertical members get text rotated 90°. For a PDF, text rotation comes from the text matrix; for vector-only (SHX) text, estimate it from the stroke geometry's minimum rotated rectangle.
- **Side/offset prior.**
  - Measure the perpendicular offset of the label from the entity axis, and which side it is on.
  - Estimate the drawing's dominant convention from high-confidence matches: the mode of (side, offset/text-height). Then penalise deviation. This is an EM-like self-calibration (§4).
  - For point symbols, apply an 8-position prior with top-right lowest cost, re-estimated per drawing.
- **Distance normalisation.** Normalise distances by text height (cap height). Label offsets scale with text size, not with drawing scale.

### Gaps
- I found no published technical-drawing paper that uses Voronoi ownership explicitly for text association. The idea comes by analogy from floor-plan distance-transform room segmentation and cartography.
- I found no citable source for company CAD standards (e.g. "beam mark centred above the member, parallel"). These are office-specific, so learn them per drawing set rather than hard-coding.

## 4. Consistency constraints (same type ⇒ same size, schedules) and iterative/EM refinement

### Takeaway
Published P&ID work mentions "validation and correction … based on inherent domain knowledge" (Digitize-PID) and regex tag grammars (Kim et al.). I found no paper that does explicit EM over label assignment in drawings. The self-calibrating loop below is therefore a design proposal built from standard components.

### Cited Findings
- Digitize-PID includes a final "validation and correction of output data based on inherent domain knowledge" stage. — [arXiv 2109.03794](https://arxiv.org/abs/2109.03794)
- Tag IDs are recognised by matching predefined numbering patterns via regular expressions. — [Kim et al. 2022](https://academic.oup.com/jcde/article/9/4/1298/6611631)
- Room-label OCR is normalised against a known vocabulary with Levenshtein distance. — [Kaliada, Medium](https://andrey-koleda.medium.com/how-i-built-a-computer-vision-system-that-understands-doors-on-floor-plans-ec3dbf0e29c7)
- Budig et al. use interaction plus sensitivity to find the few uncertain matches. This is a human-in-the-loop form of refinement. — [historical-maps.github.io](http://historical-maps.github.io/)

### Inferences
These are proposed designs.

- **Type-compatibility term.**
  - Parse the label with grammars, e.g. `^B\d+` for beam, `^C\d+` for column, `W\d+x\d+` for a steel section, "Ø/DN" for pipe size.
  - Mismatched class → infeasible edge. Unknown → neutral.
- **Schedule/catalogue anchoring.**
  - If a beam/column schedule table is parsed (mark → size/section), each mark implies expected geometry (e.g. width in plan = section width × scale).
  - Add a cost |measured width − expected width| / tolerance. This lets geometry resolve labels that are ambiguous by position.
- **"Same label ⇒ same geometry" constraint.**
  - After a first assignment, cluster the entities assigned to each label string. Measure within-cluster variance of size, length and symbol class, and flag outliers.
  - In an ILP, enforce this with indicator constraints. Iteratively, a simpler route is: raise the cost of outlier pairs and re-solve.
- **EM-style loop.**
  - E-step: solve the assignment (LSA or flow) with the current cost weights and priors.
  - M-step: re-estimate per-drawing priors from confident matches. These priors are the offset distribution per entity class, the preferred side/position, the orientation agreement rate, and the typical text height per label type.
  - Iterate 2–5 times.
  - This is training-free, because it fits a handful of parameters per drawing from the drawing itself. Stop when assignments do not change.
- **Error handling for drawing mistakes.**
  - Allow "unassigned" dummies (§2) so that a missing or extra label does not force a wrong match.
  - Report every label left unassigned, every entity left unlabelled with a nearby unassigned label, and every low-margin match as QA issues. Do not silently resolve them.

### Gaps
- I found no published EM or self-calibration approach for drawing text association. This is a design proposal, and its effectiveness is unverified.
- I found no source quantifying how often real drawings have duplicated or missing labels.

## 5. AutoCAD specifics: multileaders, block attributes, and what survives PDF export

### Takeaway
In DWG/DXF, associations are partly explicit. MULTILEADER stores leader vertices and an arrowhead endpoint (the arrow tip lies on the target). Block attributes are stored on the block reference itself, so the attribute-to-block link is exact. MULTILEADER does not, however, store a reference to the annotated object. After PDF export, all of this collapses to vector strokes plus (for SHX fonts) text as geometry. Searchable text survives only as PDF comments or hidden text, depending on PDFSHX. Prefer DWG/DXF input when it is available.

### Cited Findings
- **MULTILEADER structure.**
  - Content is either MTEXT or a BLOCK with attributes. One entity can hold one or more leaders; each is a list of leader lines with WCS vertices, plus an arrowhead at the leader end. Optional landing/dogleg segments connect to the content.
  - The content base point is in `MLeaderContext.base_point`.
  - The documentation describes no stored reference to the annotated object; MULTILEADER is a standalone annotation.
  - ezdxf can `virtual_entities()` or `explode()` it into primitives. — [ezdxf MULTILEADER docs](https://ezdxf.readthedocs.io/en/stable/dxfentities/mleader.html)
- **SHX text in PDF export.**
  - When plotting to PDF, SHX-font text is converted to geometry (vector strokes), not text.
  - Since AutoCAD 2016, each SHX text is also added as a PDF comment so it stays searchable.
  - The PDFSHX / EPDFSHX system variable controls whether that text is stored as comments or hidden text; 0 disables it. — [Autodesk support article](https://www.autodesk.com/support/technical/article/caas/sfdcarticles/sfdcarticles/Drawing-text-appears-as-Comments-in-a-PDF-created-by-AutoCAD.html); [cadplotting.com on PDFSHX](https://cadplotting.com/pdf-shx-text-pdfshx-autocad/); [Autodesk forum](https://forums.autodesk.com/t5/autocad-forum/autocad-2016-shx-text-as-comment-in-exported-pdf/td-p/5555583)
- Block attributes using SHX fonts are likewise turned into non-editable geometry in exported PDFs. — [CADTutor forum](https://www.cadtutor.net/forum/topic/99000-shx-text-not-editable-in-pdf/)
- PDF viewers then show "sticky-note bubbles" all over AutoCAD-generated PDFs. These are the SHX comment annotations. — [Qoppa KB](https://kbpdfstudio.qoppa.com/pdf-created-by-autocad-shows-sticky-note-bubbles-on-document/)

### Inferences
- **If DWG/DXF is available:**
  - Block attribute values (e.g. a MARK attribute on a column block) are exact associations. Use them directly.
  - For MULTILEADER, take the arrow-tip vertex of each leader and snap it to the nearest entity within tolerance. This gives near-certain links. Ordinary LEADER entities and plain LINE+TEXT "fake leaders" need the same tip-snapping.
- **If only PDF is available:**
  - Read the PDF annotation layer (comments/hidden text) for SHX text. Each comment's rect gives the location, and the content gives the string, without needing OCR.
  - TrueType text survives as real text with a transform matrix (rotation is available).
  - Leader lines become ordinary thin paths. Detect them as short polylines that start near a text box and end at an arrowhead or dot. The arrowhead is typically a small filled triangle path.
  - PDF layers (Optional Content Groups) may survive if exported with layer information. That can separate annotation from geometry. This is unverified; I found no source in this search.

### Gaps
- I did not verify whether AutoCAD PDF export preserves layer (OCG) names by default, or whether multileader/leader lines are distinguishable from geometry by line weight or colour in the output.
- I did not find the exact wording of the PDFSHX values (the Autodesk help page did not render).
