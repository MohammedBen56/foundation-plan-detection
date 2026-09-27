# Vector/CAD-native methods for recognising structural elements on AutoCAD plans

Scope: panoptic symbol spotting (PSS) and primitive-level learning on vector drawings; whether any public model + weights can detect columns, walls, footings, beams, slabs on AutoCAD-exported single-page vector PDFs (French labels) with no in-house labelled data. Research date: Sept 2026.

## 1. Main methods and datasets: classes, licences, sizes, reported PQ

### Takeaway
The benchmark everyone reports on (FloorPlanCAD) is architectural and has NO column, beam, slab or footing class: its only structural-ish classes are `wall` and `curtain wall` (both "stuff"). The only dataset with structural classes (columns, beams, slabs, foundations, holes) is ArchCAD-400K (NeurIPS 2025). Its code is released under an academic licence. Only a 40K-chunk subset is available, through a gated Hugging Face request, and no official trained DPSS checkpoint has been published. FloorPlanCAD PQ has saturated at about 88–91. It says nothing about structural plans.

### Cited findings
**FloorPlanCAD (ICCV 2021)**
- Over 15,000 floor plans (latest version 15,663 CAD drawings, Nov 2021), residential to commercial. Each drawing ships as SVG with annotation fields, plus a PNG and COCO visualisation. The annotations are **CC BY-NC 4.0**. The authors do not own the copyright of the drawings. The project was discontinued in early 2022. — [floorplancad.github.io](https://floorplancad.github.io/)
- The paper abstract says "over 10,000 floor plans" and 30 object categories. The counts differ between the paper and the site. — [arXiv 2105.07147](https://arxiv.org/abs/2105.07147)
- The full 35-class list, verified from the CADTransformer config:
  - Doors: single, double, sliding, folding, revolving, rolling.
  - Windows: window, bay window, blind window, opening symbol.
  - Furniture and appliances: sofa, bed, chair, table, TV cabinet, wardrobe, cabinet, gas stove, sink, refrigerator, air conditioner, bath, bath tub, washing machine, squat toilet, urinal, toilet.
  - Other things: stairs, elevator, escalator.
  - Stuff: row chairs, parking spot, **wall (33), curtain wall (34)**, railing.
  - There is **no column, beam, slab, footing or foundation class.** — [CADTransformer config/anno_config.py](https://github.com/VITA-Group/CADTransformer/blob/main/config/anno_config.py)
- Wall is one of the hardest classes. SymPoint reports wall PQ 53.5 and railing 53.0 in its per-class table. — [SymPoint arXiv 2401.10556](https://arxiv.org/html/2401.10556)

**Reported FloorPlanCAD PQ (panoptic quality)**

| Method | PQ |
|---|---|
| PanCADNet | 55.3 |
| CADTransformer (CVPR 2022) | 68.9 |
| GAT-CADNet (CVPR 2022) | 73.7 |
| SymPoint (ICLR 2024) | 83.3 |

Source: [SymPoint](https://arxiv.org/html/2401.10556)

| Method | PQ without priors | PQ with priors |
|---|---|---|
| SymPoint-V2 | 83.2 | 90.1 |
| CADSpotting | — | 88.9 |
| DPSS | 86.2 | 89.5 |
| VecFormer (NeurIPS 2025) | 88.4 | 91.1 |

VecFormer also reports Stuff-PQ 85.9 without priors and 90.4 with priors. Source: [VecFormer arXiv 2505.23395](https://arxiv.org/html/2505.23395)

- SymPoint-V2 adds a Layer Feature-Enhanced module that uses the CAD **layer** of each primitive, plus Position-Guided Training. It claims 90.1 PQ. — [SymPoint-V2 arXiv 2407.01928](https://arxiv.org/pdf/2407.01928)
- The "with prior" numbers depend on layer information. That matters because AutoCAD PDFs can carry layers as OCGs (see section 5).
- SymPoint treats each primitive (line, arc, circle, ellipse) as an 8-D point: centre, angle, length, and a one-hot type. — [SymPoint](https://arxiv.org/html/2401.10556)
- **CADSpotting** densely samples points along primitives, using coordinates only, and adds Sliding Window Aggregation for large drawings. It reports PQ 87.4 on FloorPlanCAD in the v4 text and 75.5 on its new LS-CAD set. LS-CAD has 45 floorplans of about 1,000 m² each, labelled in the FloorPlanCAD scheme; its release was only promised ("will be publicly released"). The paper links no code or weights. — [CADSpotting arXiv 2412.07377](https://arxiv.org/html/2412.07377v4)
- **ArchCAD-400K / DPSS (NeurIPS 2025)**:
  - Data: 413,062 chunks from 5,538 drawings; 86% non-residential; average area 11,000 m².
  - Classes: 27 in the paper (the HF card says 30). They include **structural components (columns, beams, holes, slabs, foundations)**, non-structural elements (doors, windows, stairs, furniture, railings) and drawing notations (axis lines, labels, markers).
  - Input: 14 m × 14 m chunks rendered at 700×700.
  - Results: DPSS gets PQ 70.6 on ArchCAD (Things 65.6, Stuff 77.6).
  - Sources: [ArchCAD arXiv 2503.22346](https://arxiv.org/html/2503.22346), [ArchCAD GitHub](https://github.com/ArchiAI-LAB/ArchCAD)
- DPSS fuses an image branch (HRNetV2) with a point branch (Point Transformer). — [mohansshf/dpss-archcad HF card](https://huggingface.co/mohansshf/dpss-archcad)
- **VectorGraphNET** is a GAT (graph attention) segmenter for technical drawings. It was evaluated on a proprietary TUM dataset and on FloorPlanCAD. — [Semantic Scholar](https://www.semanticscholar.org/paper/VectorGraphNET:-Graph-Attention-Networks-for-of-Carrara-Nousias/621ddf30f494450f6975f30d4e6a02e83f15ead8), [ResearchGate](https://www.researchgate.net/publication/397145642_VectorGraphNET_Graph_Attention_Networks_for_Accurate_Segmentation_of_Complex_Technical_Drawings)
- **Other 2025–2026 successors**:
  - "Text-Enhanced Panoptic Symbol Spotting" jointly models geometric and text primitives. — [arXiv 2510.11091](https://arxiv.org/abs/2510.11091)
  - "Text-Aided Multi-Modal PSS for CAD Floor Plans" exists but I did not read it. — [arXiv 2607.12678](https://arxiv.org/pdf/2607.12678)
  - VectorFloorSeg (CVPR 2023) is a two-stream GAT that segments rooms on vectorised roughcast floorplans. — [GitHub DrZiji/VecFloorSeg](https://github.com/DrZiji/VecFloorSeg)

### Inferences
- A model trained on FloorPlanCAD can at best give `wall` / `curtain wall`. Columns, footings, beams and slabs are simply not in its label space.
- ArchCAD-400K is the only public source of learned structural classes. Even so, it is Chinese architectural drafting, not French structural (béton armé) coffrage or fondation plans.

### Gaps
- I did not get per-class ArchCAD PQ for column, beam or foundation. The paper's supplementary tables were not retrieved.
- VectorGraphNET's exact numbers were not retrieved (a 429 on the review page). Its arXiv entry did not surface.
- I did not get CADSpotting's per-class results on wall.

## 2. Code, pretrained weights, input formats, PDF conversion effort

### Takeaway
Pretrained FloorPlanCAD weights exist for SymPoint (OneDrive) and SymPoint-V2 (Google Drive). CADTransformer, VecFormer, CADSpotting and official DPSS ship code but no trained checkpoints. All the repos expect FloorPlanCAD-style SVG, parsed to JSON or npy. None ships a "run on my DXF/PDF" inference script. Converting a PDF vector page into their input is feasible, and a public fork already does it. The main work is:
- Bézier handling. PDFs have no native arcs, only cubic Béziers.
- Coordinate normalisation and scale.
- Layer (OCG) mapping.
- Chunking large sheets.

### Cited findings
| Method | Code | Weights | Licence | Notes |
|---|---|---|---|---|
| CADTransformer | [VITA-Group/CADTransformer](https://github.com/VITA-Group/CADTransformer) | README lists only the ImageNet HRNet backbone. I saw no trained CAD checkpoint. | MIT | Out of memory above about 12k primitives on a 24 GB GPU, so it needs `max_prim` |
| GAT-CADNet | Only an unofficial repo, [Liberation-happy/GAT-CADNet](https://github.com/Liberation-happy/GAT-CADNet) | Unknown | — | Paper: [arXiv 2201.00625](https://arxiv.org/abs/2201.00625) |
| SymPoint | [nicehuster/SymPoint](https://github.com/nicehuster/SymPoint) | Yes, OneDrive link in README | No SPDX licence (NOASSERTION) | Old stack: torch 1.10 / cu111, detectron2, compiled pointops. The ACM/CCL modules were dropped in the release. |
| SymPoint-V2 | [nicehuster/SymPointV2](https://github.com/nicehuster/SymPointV2) | Yes, Google Drive ([link](https://drive.google.com/file/d/1ZeWtgZJKD_yWmFNWwBOMN9_4-x-ZXUuS/view?usp=drive_link)) | Unclear | Uses the CAD layer id per primitive |
| VecFormer | [WesKwong/VecFormer](https://github.com/WesKwong/VecFormer) | No weights; train yourself | Apache-2.0 | Line-based representation; preprocessing outputs JSON |
| CADSpotting | None found | None | — | [arXiv](https://arxiv.org/abs/2412.07377) |
| DPSS (ArchCAD) | [ArchiAI-LAB/ArchCAD](https://github.com/ArchiAI-LAB/ArchCAD), released 2025-10-16 | No official checkpoint (see caveat below) | Academic-only | "CADParser" tool for CAD processing is still on the TODO list |

- **Caveat on "DPSS weights on Hugging Face".** Search snippets claim `mohansshf/dpss-archcad` hosts `dpss_weights.pth`. The HF API file list shows only `README.md`, `train_portable.py` and `upstream_code.tar.gz`: **no weight file** (checked Sept 2026). — [HF API](https://huggingface.co/api/models/mohansshf/dpss-archcad)
- A fork's README also says: "No trained DPSS checkpoint has been published. Without one the demo runs with randomly initialised weights … PQ ≈ 0". It also notes that the release ships PointTransformer v1 with no ScanNet-pretrained point branch, so reproductions should expect to land below the paper's PQ. — [Manavbangotra/ArchCAD-gpu](https://github.com/Manavbangotra/ArchCAD-gpu)
- The ArchCAD dataset on HF (`jackluoluo/ArchCAD`) is gated. Only a first 40K-sample subset was released. — [ArchCAD GitHub](https://github.com/ArchiAI-LAB/ArchCAD), [HF dataset](https://huggingface.co/datasets/jackluoluo/ArchCAD)
- An access-request issue exists for the full 413K set. — [issue #11](https://github.com/ArchiAI-LAB/ArchCAD/issues/11)
- A search snippet says access is "restricted to non-commercial use", with approval taking up to 3 business days.
- **PDF-to-model conversion already exists publicly.** `dataset/parse_pdf_plans.py` in the ArchCAD-gpu fork (Aug–Sept 2026, 0 stars, a solo project) does the following:
  - walks each PDF content stream with pikepdf;
  - tracks the CTM and the marked-content stack (`/OC /ocN BDC … EMC`);
  - assigns every line or curve segment its OCG (CAD layer) name;
  - samples cubic Béziers at t = 0, 1/3, 2/3, 1 into the "four-point form the loader wants";
  - emits the SymPoint/DPSS JSON with a per-primitive layer id.

  The author deliberately avoided PyMuPDF because of its AGPL licence. The same repo has a `takeoff.py --labels layers` baseline that labels primitives **purely from PDF layer names, with no model**. — [ArchCAD-gpu parse_pdf_plans.py](https://github.com/Manavbangotra/ArchCAD-gpu/blob/master/dataset/parse_pdf_plans.py)
- PyMuPDF `Page.get_drawings()` returns paths with items (`l`, `c`, `re`, `qu`), plus `width`, `dashes`, `closePath` and `seqno`. Since v1.22.0 each path also has a **`layer` key (OCG name)**. `extended=True` adds clip/group hierarchy via `level`. PyMuPDF is dual-licensed AGPL/commercial. — [PyMuPDF page.rst](https://github.com/pymupdf/PyMuPDF/blob/main/docs/page.rst), [PyMuPDF docs](https://pymupdf.readthedocs.io/en/latest/page.html)
- CADSpotting needs only points sampled at fixed intervals along primitives (d = 0.14). That is the representation least sensitive to the primitive type, so PDF Béziers are no problem for it. — [CADSpotting](https://arxiv.org/html/2412.07377v4)

### Inferences
- Minimum viable path: PyMuPDF `get_drawings()` (or pikepdf), then segments and Bézier samples with layer names, then SymPoint(-V2) JSON with coordinates normalised to FloorPlanCAD's scale and chunked at a similar metric size. That is on the order of days of engineering.
- Scale normalisation matters. The models learned absolute sizes from FloorPlanCAD, and ArchCAD chunks are 14 m × 14 m. So the drawing scale must be recovered from the PDF (for example from the cartouche "1/50", or from dimension text).
- Licences: every usable pretrained model or dataset here is non-commercial or academic (FloorPlanCAD CC BY-NC, ArchCAD academic, SymPoint unlicensed). Commercial use would need legal clearance or a retrain on your own data.

### Gaps
- I did not verify whether the SymPoint OneDrive and SymPoint-V2 Drive links still resolve in 2026.
- I did not check the exact licence text of the SymPoint repos.

## 3. Transfer to unseen drafting styles, countries, structural drawings and foundation plans

### Takeaway
Almost no published domain-shift evidence exists. The only cross-dataset result I found is CADSpotting's joint-training comparison. I found no paper that evaluates a FloorPlanCAD- or ArchCAD-trained model zero-shot on European (French) plans, structural (coffrage, ferraillage) sheets or foundation plans. Expect poor zero-shot results and treat these models as needing at least fine-tuning.

### Cited findings
- VecFormer reports no cross-dataset experiments; it evaluates on FloorPlanCAD only. — [VecFormer](https://arxiv.org/html/2505.23395)
- CADSpotting shows SymPoint degrading on large drawings. With joint training on FloorPlanCAD + LS-CAD, CADSpotting generalises better than SymPoint. LS-CAD uses the same labelling standard, so this is not a true style shift. — [CADSpotting](https://arxiv.org/html/2412.07377v4)
- ArchCAD-400K motivates itself by saying earlier datasets were small and mostly residential. Its drawings are "highly standardized". — [ArchCAD](https://arxiv.org/abs/2503.22346)
- Practitioner evidence from the ArchCAD-gpu fork:
  - For US PDFs they had to retrain in two stages: FloorPlanCAD first, then joint training with US plan-set tiles that are auto-labelled from CAD layer names.
  - They collapsed the taxonomy to door, window and wall.
  - They measured that "door and window geometry is nearly identical" between Chinese and US drawings. "The China/US mismatch lives in sheet context, not in the symbols": title blocks, dimensions, hatching, notes.
  - This is a README claim, not a peer-reviewed result.
  - Source: [ArchCAD-gpu taxonomy.py](https://github.com/Manavbangotra/ArchCAD-gpu/blob/master/dataset/taxonomy.py)
- A 2026 training-free structural paper (arXiv 2608.17237) targets exactly structural framing plans: columns, beams, walls, braces, openings.
  - Method: a deterministic PDF-operator parser tracks CTM, line width, dash pattern and colours; scale is estimated by dimension-ratio consensus; an explicit drafting grammar does recognition.
  - Grammar examples: columns are closed glyphs of 60–2000 mm, classified as wide-flange vs rectangle by width sampled at 5 stations; beams are closed polygons at true width or chains of collinear medium-weight strokes.
  - A VLM may only propose typed edits from a 12-operation vocabulary, each checked by admission tests.
  - Held-out results: columns recall 0.922 / precision 0.997; beams 0.886 / 0.990; walls, braces and openings about 1.0.
  - **But** the 100-plan benchmark is procedurally generated by the authors' own generator, shared by the development and held-out halves. The authors state it "does not address independently drafted plans". No code is linked, and there is no numeric comparison to SymPoint and similar models.
  - Source: [arXiv 2608.17237](https://arxiv.org/html/2608.17237)

### Inferences
- Structural sheets differ sharply from FloorPlanCAD's data:
  - Structural sheets contain hatched or filled column sections, dashed hidden beam outlines, footing outlines under the slab line (semelles), axis grids, reinforcement notation and a heavy load of dimensions.
  - FloorPlanCAD is dominated by furniture, doors and windows.
  - So zero-shot learned PSS will mostly produce background or wrong-class noise on French structural plans, with the possible exception of walls.
- The most transferable signals are drafting conventions (hatch or solid fill, line weight, dash pattern, closed-polygon size, grid alignment) and CAD layer names. Rule-based or grammar pipelines exploit exactly these, which argues for rules plus layers first, then learned models fine-tuned on weak labels derived from those rules or layers.

### Gaps
- I found no study on French or European drawings. None evaluates foundation plans, and none evaluates ArchCAD-trained models on independent data.

## 4. Rule/grammar-based parsing, primitive-graph GNNs and classic symbol recognition

### Takeaway
Before deep learning there was a mature symbol-spotting literature for architectural drawings:
- vector-primitive indexing;
- error-tolerant subgraph matching (Lladós);
- deformable line models (Valveny);
- Tombre's surveys;
- the SESYD synthetic benchmark (Delalandre, Valveny).

It suits repeated, highly regular symbols such as identical column sections or footing types. Modern GNN variants (GAT-CADNet, VectorGraphNET, VectorFloorSeg) build graphs over line and arc primitives, but all need labelled training data.

### Cited findings
- A literature survey covers symbol spotting in architectural drawings: state of the art and industry-driven developments. It cites Tombre et al. (2005) and Rusiñol & Lladós (2010). Lladós did symbol recognition by error-tolerant subgraph matching of region adjacency graphs; Valveny by deformation of lineal shapes. Isolated symbol recognition is mature, but recognition "in context" is not. — [Symbol spotting for architectural drawings (ResearchGate)](https://www.researchgate.net/publication/333013076_Symbol_spotting_for_architectural_drawings_state-of-the-art_and_new_industry-driven_developments)
- SESYD has 1,000 synthetic vectorised documents with ground truth, including architectural floor plans and symbol models. — [SESYD contributions (Semantic Scholar)](https://www.semanticscholar.org/paper/Recent-contributions-on-the-SESYD-dataset-for-of-Delalandre-Valveny/42a3d89544393fe80acb6d6c4eae0239c9c96b99)
- There is also a performance-evaluation protocol for symbol spotting in terms of recognition and location indices. — [ResearchGate](https://www.researchgate.net/publication/220163448_A_performance_evaluation_protocol_for_symbol_spotting_systems_in_terms_of_recognition_and_location_indices)
- Hierarchical graph representation for symbol spotting (Springer). — [Springer](https://link.springer.com/chapter/10.1007/978-3-642-34166-3_58)
- GAT-CADNet treats each drawing as a graph of primitives. Vertex features give semantics, and cascaded attention scores give instances. — [arXiv 2201.00625](https://arxiv.org/abs/2201.00625)
- A GNN parser for line segments of floor-plan images. — [arXiv 2303.03851](https://arxiv.org/pdf/2303.03851)
- Structural grammar example: see arXiv 2608.17237 in section 3, which gives column and beam rules from line weight, closure, size and dash. — [arXiv 2608.17237](https://arxiv.org/html/2608.17237)

### Inferences
- In AutoCAD structural plans, columns and footings are usually block instances or repeated identical closed polylines, often hatched. On DXF, block INSERT counts and names give near-free detection. On PDF, blocks are flattened, but exact-geometry repetition can be recovered by canonicalising each closed path (translation- and rotation-normalised) and hashing it. That is essentially classic query-by-example symbol spotting and needs no training data.
- Rule set to prototype, entirely unsupervised:
  - Columns: small closed polygons, rectangular or circular, often solid-filled or hatched, snapped to grid-axis intersections.
  - Walls: pairs of parallel lines at constant thickness.
  - Beams: dashed parallel pairs on structural sheets.
  - Footings: larger rectangles concentric with columns on foundation plans.
  - Labels: French text nearby, such as "P1", "POT", "S1", "SF", "SI", "L 20x50", "PH".

### Gaps
- I did not retrieve specific Tombre or Valveny paper URLs beyond the survey's citations.
- I found no public code for classic vector graph-matching symbol spotters that is maintained in 2025–2026.

## 5. DWG/DXF vs PDF parsing practicalities; what AutoCAD PDF export preserves

### Takeaway
If DWG or DXF is obtainable, prefer it:
- ezdxf reads DXF R12–R2018 natively, and DWG through the free ODA File Converter (`ezdxf.addons.odafc`).
- LibreDWG (GPLv3) reads DWG directly but is unreliable on 2018+ files.
- DXF keeps block definitions and INSERTs, layers, true arcs and circles, hatch entities and real text.

AutoCAD PDFs lose blocks, arcs and hatch semantics: everything becomes paths and Béziers. They can keep layers as OCGs if "Include layer information" was ticked. SHX text is turned into geometry and duplicated as "AutoCAD SHX Text" comments or hidden text.

### Cited findings
- ezdxf reads and writes DXF from R12 (AC1009) to R2018 (AC1032). — [ezdxf docs](https://ezdxf.readthedocs.io/en/stable/introduction.html)
- The `odafc` add-on converts DWG to a temporary DXF through the separately installed ODA File Converter, then loads it. — [ezdxf odafc docs](https://ezdxf.readthedocs.io/en/stable/addons/odafc.html)
- LibreDWG is GPLv3 and reads DWG without conversion, but "support for newer DWG versions (2018+) can be inconsistent". There is no reliable pure-Python DWG reader, so ODA is the production route. This comes from a secondary blog. — [Fastio](https://fast.io/resources/metadata-extraction-from-cad-dwg-dxf-files/)
- AutoCAD's Export to PDF has an "Include layer information" option, which lets layers be toggled in the PDF. — [Autodesk Export to PDF Options](https://help.autodesk.com/cloudhelp/2016/ENU/AutoCAD-Core/files/GUID-36D8C81E-754F-4C7A-844E-3A5A131801D9.htm)
- One practitioner recommends keeping layers on for review documents and off for the smallest files. So many PDFs in the wild may lack layers. — [Novedge tip](https://novedge.com/blogs/design-news/autocad-tip-autocad-pdf-and-dwf-export-best-practices)
- SHX-font text is always converted to geometry in the PDF. The `PDFSHX` system variable stores the original text as comments or hidden text labelled "AutoCAD SHX Text". — [PDFSHX 2024](https://help.autodesk.com/view/ACD/2024/ENU/?guid=GUID-56EA988C-A1DA-4E85-8765-B3F31A01AB02), [PDFSHX 2025](https://help.autodesk.com/cloudhelp/2025/ENU/AutoCAD-Core/files/GUID-56EA988C-A1DA-4E85-8765-B3F31A01AB02.htm), [Autodesk forum](https://forums.autodesk.com/t5/autocad-electrical-forum/pdf-s-are-jerky-and-contain-quot-autocad-shx-text-quot/td-p/6835299)
- PDF layers are written as OCGs and marked content (`/OC /ocN BDC … EMC`). Layer names survive, including XREF prefixes such as "XREF - 1st Floor|A-DOOR", and can be mapped by regex to classes. — [ArchCAD-gpu parse_pdf_plans.py / taxonomy.py](https://github.com/Manavbangotra/ArchCAD-gpu/blob/master/dataset/taxonomy.py)
- PyMuPDF ≥ 1.22 exposes each path's OCG name as `layer` in `get_drawings()`. — [PyMuPDF page.rst](https://github.com/pymupdf/PyMuPDF/blob/main/docs/page.rst)

### Inferences
- First check on every PDF: does it have OCProperties (`doc.layer_ui_configs()` or `get_ocgs()` in PyMuPDF)? If it does, French structural layer names often map directly to classes, which is a strong zero-training baseline. Examples: "POTEAUX", "VOILES", "POUTRES", "SEMELLES", "DALLE", "BA", "S-COLS".
- Text: SHX labels (P1, S1, 50x50) are unusable as glyph strokes. They can be recovered from the "AutoCAD SHX Text" annotations with `page.annots()` when PDFSHX was on. TrueType text stays as real extractable text.
- Arcs and circles come through as Bézier chains. Sampling them (as CADSpotting and the fork do) or fitting circles back is needed before SymPoint-style typed primitives.
- Blocks are gone in PDF, but repeated identical geometry, and the ordering of `seqno` and `level` groups, can partially reconstruct instances.

### Gaps
- I did not verify whether AutoCAD PDF export preserves hatch patterns as distinct groups or tags, or whether block names survive in any PDF structure (for example tagged PDF). I believe they do not, but have no source.
- I did not confirm the default state of PDFSHX (believed to be 1, meaning comments) or of "Include layer information" across AutoCAD versions.
