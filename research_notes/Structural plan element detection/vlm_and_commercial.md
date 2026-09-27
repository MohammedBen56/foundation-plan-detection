# VLMs for Engineering/Construction Drawings and Commercial Takeoff / 2D-to-BIM Tools

Research date: 2026-09-26. Note: several arXiv IDs below are from 2026 (e.g. 2601.x to 2609.x); they were retrieved from arXiv abstract pages but most were read at abstract level only (full-text tables not verified).

## 1. Benchmarks and studies: how well do VLMs read, classify and localise elements on technical drawings?

### Takeaway
Frontier VLMs are strong at reading text on drawings (~95% OCR-style QA) but weak at counting and localising repeated small symbols: on AECV-Bench the best model (Gemini 3 Pro) counts doors correctly only 39% of the time and windows 34%, and zero-shot MLLM beam extraction from structural blueprints scored F1 0.30 versus 0.99 for a constraint-driven agentic pipeline. Every strong result on structural/mechanical drawings in 2025-26 comes from hybrid or fine-tuned pipelines, not from zero-shot prompting.

### Cited Findings
**AECV-Bench (architectural floor plans + drawing QA) — independent academic/industry benchmark, arXiv 2601.04819**
- Models tested (2026 version): Gemini 3 Pro, GPT-5.2, Claude Opus 4.5, Grok 4.1 Fast, Amazon Nova 2 Lite, Cohere Command A Vision, GLM-4.6V, Qwen3-VL-8B, Nemotron Nano 12B v2 VL, Mistral Large 3 — [AECFoundry blog](https://www.aecfoundry.com/blog/can-ai-really-read-your-building-plans-aecv-bench-gets-a-major-upgrade)
- Counting on 120 floor plans (CubiCasa5K, CVC-FP, public sources): bedrooms 91% best (GPT-5.2 / Claude Opus 4.5), toilets 82% (Gemini 3 Pro / GLM-4.6V), doors 39% (Gemini 3 Pro), windows 34% (Gemini 3 Pro). Mean accuracy: Gemini 3 Pro 51%, GPT-5.2 49%, Claude Opus 4.5 42%, GLM-4.6V 39%. Mean MAPE: Gemini 3 Pro 16%, Claude Opus 4.5 24.9% — [AECFoundry blog](https://www.aecfoundry.com/blog/can-ai-really-read-your-building-plans-aecv-bench-gets-a-major-upgrade)
- Earlier version: Gemini 2.5 Pro 41% mean counting accuracy, GPT-5 37%, Claude 3.7 Sonnet 35% — [AECFoundry intro blog](https://www.aecfoundry.com/blog/can-ai-really-read-your-building-plans-introducing-aecv-bench) (as summarised by search results)
- Drawing QA (192 questions over 21 drawings): text extraction ~95%, comparative reasoning ~80%, spatial reasoning ~70%, instance counting ~40-55%. Overall Gemini 3 Pro 0.854, Grok 4.1 Fast 0.312. LLM-as-judge with human adjudication — [AECFoundry blog](https://www.aecfoundry.com/blog/can-ai-really-read-your-building-plans-aecv-bench-gets-a-major-upgrade); code: github.com/AECFoundry/AECV-Bench, paper: [arXiv 2601.04819](https://arxiv.org/abs/2601.04819)

**Structural plans specifically**
- BlueprintAgent (arXiv 2609.07362): scanned reinforced-concrete blueprints converted to simulation-ready (FEM) structural models; 300 real scanned sheets from 20 anonymised RC projects. Beam F1 (macro) 0.994 for the agent vs **0.301 for single-MLLM zero-shot** and 0.820 for a fixed pipeline. Failure mode of direct MLLM prompting: outputs break engineering constraints (beam-column support, span counts, 3D continuity across floors) — [arXiv 2609.07362](https://arxiv.org/abs/2609.07362)
- "Training-Free Agentic Computer Vision for Structural Component Detection in 2D Structural Framing Plans" (arXiv 2608.17237): on 50 held-out test drawings, columns recall 0.922 / precision 0.997; beams 0.886 / 0.990; walls, braces, openings 0.964-1.000 precision; scale estimated within 0.1%. Caveat stated by the authors: both benchmark halves come from one generator, with no independently drafted plans and no raster inputs — [arXiv 2608.17237](https://arxiv.org/abs/2608.17237)

**Mechanical / other engineering drawings**
- MechVQA (arXiv 2605.30794): 3.3k high-density mechanical drawings, 21K QA pairs, 10 tasks at recognition/reasoning/judging levels. The fine-tuned MechVL beats the strongest closed-source baseline by 7.57 points. Named challenges: high annotation density, weak domain knowledge, unreliable spatial reasoning under projection rules — [arXiv 2605.30794](https://arxiv.org/abs/2605.30794)
- BlueprintSymVL (Results in Engineering, 2025): discriminative benchmark for VLM symbol recognition and counting on P&ID blueprints; 100 clean + 100 occluded 860x860 regions, with visual in-context examples of symbols (e.g. gate valves) — [Zenodo](https://zenodo.org/records/17250377) (the page does not list per-model results)
- AEC drawing layout detection benchmark (arXiv 2607.18997): RF-DETR mAP50 0.949; Qwen3-VL F1 0.911 (competitive). Models pre-trained on generic document layouts drop in accuracy on AEC drawings ("domain interference") — [arXiv 2607.18997](https://arxiv.org/abs/2607.18997)
- BLUEPRINT retrieval system (arXiv 2602.13345): task-specific pipelines (YOLO region detection + VLM OCR restricted to regions) beat general VLMs (LLaVA, Pixtral, PaliGemma). +10.1 pts Success@3 over the strongest VLM baseline on 5k files / 350 queries — [arXiv 2602.13345](https://arxiv.org/abs/2602.13345)
- Other benchmarks surfaced but not read in detail: MMArch (architectural evidence reasoning, arXiv 2608.09281), MechReason (arXiv 2609.16012), PCB drawing parsing with compact VLMs, "Learning to Ground Before Reading" (arXiv 2608.29268), MeasureBench (arXiv 2510.26865), DesignQA, CADBench, TechMB — [search results](https://arxiv.org/pdf/2608.09281), [MechReason](https://arxiv.org/pdf/2609.16012), [PCB](https://arxiv.org/pdf/2608.29268), [MeasureBench](https://arxiv.org/pdf/2510.26865)

**Open-model grounding**
- Qwen2.5-VL (Jan 2025) outputs absolute-coordinate bounding boxes as JSON. It is widely fine-tuned for grounding (LLaMA-Factory etc.), but box scaling and dataset-format details are a known pitfall — [LearnOpenCV](https://learnopencv.com/object-detection-with-vlms-ft-qwen2-5-vl/); [QwenLM GitHub issue #1616](https://github.com/QwenLM/Qwen3-VL/issues/1616)
- Florence-2 performed worse than Donut when fine-tuned to parse annotation patches from mechanical drawings (Donut F1 93.5%) — [arXiv 2506.17374](https://arxiv.org/abs/2506.17374)

### Inferences
- Failure modes line up across benchmarks: (1) counting many small repeated symbols (doors, windows, valves); (2) localising and keeping topology consistent (beam spans, support conditions); (3) spatial reasoning under drafting conventions. Reading text and titles/labels is close to solved. This suggests using VLMs to read labels, schedules and section marks, and to classify candidates, but not to produce geometry or counts.
- Doors and windows on floor plans are a good proxy for columns and beam lines on structural plans (thin, repeated, dense). Expect zero-shot counting of columns on a full sheet to fail in the same way.
- Many 2026 results report very high F1 (0.99). They come from narrow or synthetic data (the generator-based benchmark in 2608.17237) or were built and evaluated by the same authors. Treat them as upper bounds.

### Gaps
- I found no independent, published bounding-box IoU/mAP numbers for zero-shot GPT-4o/4.1/5, Claude or Gemini on structural plans. AECV-Bench measures counts, not boxes.
- I found no InternVL- or PaliGemma-specific results on floor plans or structural drawings, apart from PaliGemma appearing as a weaker retrieval baseline in BLUEPRINT.
- The MechVQA per-model numbers (e.g. which closed-source model was strongest) were not visible from the abstract.

## 2. Techniques that improve results (tiling, set-of-marks, extracted text/geometry, structured output, verification loops)

### Takeaway
The approaches that work share three features: they limit the VLM to deterministic candidates (from vector geometry or a detector), they feed it local crops rather than whole sheets, and they check its outputs against engineering or geometric constraints, re-querying the specific entity that fails.

### Cited Findings
- Constraint-triggered revisits: engineering constraints work best "as triggers for entity-level targeted revisits rather than as post-hoc output filters". Removing MLLM-led axis/grid adjudication hurt complex multi-sheet projects — [BlueprintAgent, arXiv 2609.07362](https://arxiv.org/abs/2609.07362)
- Deterministic candidates plus "fail-closed transactions": the VLM agent can only correct within candidates the deterministic geometry stage produced. Scale comes from a dimension-ratio consensus, and 5 entity classes come from an explicit drafting grammar — [arXiv 2608.17237](https://arxiv.org/abs/2608.17237)
- Set-of-Mark (SoM) prompting: overlay numbered marks/boxes/masks on candidate regions (originally from SAM/SEEM segmentation) so the model answers by ID instead of by coordinates. Regular marker shapes (boxes, ovals, arrows) work better than irregular ones (scribbles, points) — [Emergent Mind overview](https://www.emergentmind.com/topics/mark-based-visual-prompting); [practitioner experiment](https://djajafer.medium.com/experimenting-with-set-of-mark-prompting-with-gpt4-vision-65c7f03fb491)
- Region-restricted OCR: detect regions first (title block, notes, views), then run VLM OCR only inside them. This beats whole-page VLMs — [BLUEPRINT, arXiv 2602.13345](https://arxiv.org/abs/2602.13345)
- Detect-then-parse on patches: YOLOv11-obb localises oriented annotation patches, then a small fine-tuned VLM (Donut) parses each patch into structured output. P 88.5%, R 99.2%, F1 93.5%, hallucination rate 11.5% on 1,367 drawings — [arXiv 2506.17374](https://arxiv.org/abs/2506.17374)
- Visual in-context examples (giving the model a crop of the target symbol) is the evaluation setting BlueprintSymVL was built around — [Zenodo BlueprintSymVL](https://zenodo.org/records/17250377)
- Practitioner pattern: run deterministic PyMuPDF extraction first and send the PDF to GPT-4 Vision only if the first stage is not confident — [Towards Data Science](https://towardsdatascience.com/from-4-weeks-to-45-minutes-designing-a-document-extraction-system-for-4700-pdfs/) (from search snippet; a general document pipeline, not drawings)

### Inferences
- For structural plans, a natural SoM variant is: extract closed polylines, hatched regions and rectangles from the vector PDF (column candidates) and parallel line pairs (beam/wall candidates), render a crop with numbered overlays, and ask the VLM to label each ID as column / beam / wall / other plus its reference text (e.g. "P1 30x30"). Geometry stays exact because it comes from vectors. The VLM only supplies semantics.
- Useful verification checks for a structural plan: every beam end rests on a column or wall; column labels match the column schedule; grid-axis consistency; counts per label type agree with schedule tables.

### Gaps
- I found no published ablation that quantifies SoM against raw-coordinate prompting specifically on construction drawings.
- I found no quantified study of tile size or overlap for A0/A1 sheets with frontier VLMs.

## 3. Hybrid pipelines (vector geometry / detector for candidates, VLM for classification)

### Takeaway
Hybrid designs dominate the credible 2025-26 literature: deterministic geometry or a trained detector (YOLO / RF-DETR) handles localisation, and a VLM (general or fine-tuned) handles semantics and adjudication. Pure end-to-end VLM approaches are consistently the weakest baseline.

### Cited Findings
- Structural framing plans: a deterministic primitive extraction + scale + drafting-grammar stage, then a VLM agent refines within those candidates. Columns P 0.997 / R 0.922, beams P 0.990 / R 0.886 — [arXiv 2608.17237](https://arxiv.org/abs/2608.17237)
- Scanned RC blueprints: the MLLM is the primary reader, with OCR and CV providing local evidence and constraint validators as callable tools. Beam F1 0.994 vs 0.820 fixed pipeline vs 0.301 zero-shot MLLM — [arXiv 2609.07362](https://arxiv.org/abs/2609.07362)
- Multi-view mechanical drawings: YOLOv11-det (layout) → YOLOv11-obb (annotations) → two Donut VLMs (alphabetical F1 0.672; numerical F1 0.963), with unified JSON output — [arXiv 2510.21862](https://arxiv.org/abs/2510.21862)
- eDOCr2-style pipelines segment drawings into zones and use Qwen2-VL-7B / GPT-4 for semantic analysis — [arXiv 2506.17374 (related work)](https://arxiv.org/pdf/2506.17374)
- Layout detection for AEC sheets: RF-DETR (mAP50 0.949) and Qwen3-VL (F1 0.911) are the leading options for the sheet-segmentation stage — [arXiv 2607.18997](https://arxiv.org/abs/2607.18997)
- AutoCAD Smart Blocks Object Detection (2025 tech preview, "Detect and Convert" in 2026): the drawing's *vector geometry* is sent to Autodesk's ML service, which groups similar geometry into sets to convert into blocks. Optimised for plan views and intended for drawings drafted without blocks or imported from PDF — [Autodesk help 2025](https://help.autodesk.com/cloudhelp/2025/ENU/AutoCAD-WhatsNew/files/GUID-25BD5FB7-119A-42A8-B1C1-62BB812A3F4F.htm); [Autodesk help 2026](https://help.autodesk.com/view/ACD/2026/ENU/?guid=GUID-7D3E7065-AD01-4720-B5DF-95971BAEFA9A)

### Inferences
- Autodesk's Smart Blocks confirms that a major vendor groups similar vector geometry as the candidate-generation step on PDF-imported CAD. This is close to the "vector candidates + classifier" design.
- For vector PDFs, localisation precision can be exact (from coordinates). The VLM's error then only affects labels, and constraints and schedules can check those labels.

### Gaps
- Few practitioner blog posts give full implementations of vector-candidate + VLM-classifier takeoff. Most material is academic or undisclosed vendor internals.

## 4. Commercial tools: what they detect, structural coverage, methods, accuracy, pricing/API

### Takeaway
Mainstream AI takeoff tools (Togal, Kreo, Bluebeam Max, Civils.ai) focus on rooms/areas, walls, doors/windows and example-based symbol counting ("find all like this"). Structural element detection (columns, beams, slabs, footings) is claimed explicitly mainly by niche players (Planaliz in France, Beam AI as an AI + human service) and steel-focused tools. None disclose model architectures, and none publish independent accuracy audits. Accuracy figures (95-98%) are vendor marketing.

### Cited Findings
**Togal.AI**
- Claims detect/measure/compare "with up to 98% accuracy" (elsewhere 97%), using AIA measurement standards for spaces/areas — [Togal](https://www.togal.ai/); [Togal blog](https://www.togal.ai/blog/construction-takeoffs-with-ai-speed-and-accuracy-combined)
- Auto-count combines AI image search, text search and pattern search: draw a box around a symbol and find every instance across the set — [Velocity AI review](https://insights.velocityaipartners.co/tools/togal-ai) (third-party summary)
- Markets a "peer-reviewed" comparison with On-Screen Takeoff showing better area-detection accuracy and time savings. I could not fetch the page to check authors, venue or sample size — [Togal case study](https://www.togal.ai/case-study/peer-reviewed-study-togal-ai-vs-on-screen-takeoff)
- Also embedded in eTakeoff as "SnapAI with Togal.AI" — [eTakeoff](https://etakeoff.com/etakeoff-dimension/ai/)
- Price ~$1,999-2,999/user/year — [Aginera comparison, Sep 2026](https://aginera.ai/blog/ai-takeoff-software-comparison-2026) (a competitor's blog)

**Kreo**
- Public Auto Measure / AI Search API: upload PDF/DWG/DXF/DWF/PNG/JPG/TIFF, runs asynchronously, returns JSON with pixel coordinates, lengths, thicknesses, areas, perimeters and associated text labels. Detects walls (int/ext), doors, windows, rooms/areas (GEA/GIA/NIA). No structural members are listed in the API. No accuracy or price published (sales contact) — [Kreo API](https://www.kreo.net/features/api); [Kreo API docs](https://help-takeoff.kreo.net/en/articles/11688710-getting-started-with-the-kreo-api)
- Framing and steel trade pages offer (manual/assisted) measurement of beams, columns and joists, not claimed automatic detection — [Kreo framing](https://www.kreo.net/trades/framing-estimating-software); [Kreo steel](https://www.kreo.net/trades/steel-estimating-software)

**Beam AI (ibeam.ai)**
- Claims PDF takeoffs across architectural, structural, civil and MEP scopes. Runs as a service model: "AI + human review", 24-72 h turnaround, roughly $500-1,500 per project — [Beam AI](https://www.ibeam.ai/general-contractor-software); [Aginera comparison](https://aginera.ai/blog/ai-takeoff-software-comparison-2026)
- Vendor claim that users save 15-20 h/week and bid 3x more — [Beam AI](https://www.ibeam.ai/)

**Bluebeam (Nemetschek) — Bluebeam Max**
- Announced at Unbound Oct 2025, launched globally May 2026. VisualSearch (show an image and it finds/counts all instances), MagicWand markup tools, AI-REVIEW / AI-MATCH (from the Firmus AI acquisition) for design review and drawing comparison, Smart Overlay, sheet stitching, and a Revu + Anthropic Claude integration for natural-language automation — [Bluebeam press Oct 2025](https://press.bluebeam.com/2025/10/bluebeam-unveils-bluebeam-max-next-generation-ai-powered-innovations-at-unbound-2025/); [Bluebeam press May 2026](https://press.bluebeam.com/2026/05/bluebeam-max-launches-globally-bringing-ai-powered-productivity-to-aec-teams-everywhere/)
- Revu price $260-590/user/year (base tiers) — [Aginera comparison](https://aginera.ai/blog/ai-takeoff-software-comparison-2026)

**PlanSwift / STACK / On-Screen Takeoff**
- Mainly manual digital takeoff. PlanSwift ~$1,749-2,000/yr or perpetual ~$1,595-1,749 + $250/yr; STACK $2,988-3,588/yr, with AI features only on higher tiers; OST ~$995-1,500/yr — [Aginera comparison](https://aginera.ai/blog/ai-takeoff-software-comparison-2026)

**Autodesk**
- Autodesk Takeoff (ACC): the main feature is 3D quantification from Revit models, with 2D PDF measurement as secondary — [SteelFlo comparison](https://www.steelfloai.com/blog/steelflo-vs-autodesk-takeoff) (competitor source)
- AutoCAD Smart Blocks: ML object detection on vector geometry, and Search & Convert (select geometry, find all matches, convert to blocks). Still a tech preview — [Autodesk 2025 help](https://help.autodesk.com/cloudhelp/2025/ENU/AutoCAD-WhatsNew/files/GUID-CC745193-9397-49A5-B50B-236D04EE0845.htm); [Engineering.com](https://www.engineering.com/autocad-2025-adds-ai-features/)

**Swapp (SWAPP.AI)**
- Not a takeoff tool. It automates construction documentation inside Revit/ArchiCAD (dimensions, tags, views, sheets; claims "up to 80%" of CD tasks), trained on firm standards. Israeli startup, raised $11.5M ($18.5M total). Users include Stantec, Page, HGA — [Archinect](https://archinect.com/news/article/150350727/swapp-raises-11-5-million-for-tool-that-offers-ai-powered-construction-documents-in-minutes); [swapp.ai](https://swapp.ai/)

**Civils.ai**
- Pattern recognition + object detection + OCR on 2D CAD PDFs: walls, doors/windows, pipework, ductwork, cable tray, fixtures, areas. Vendor claim ">95% precision". Recommends human review — [Civils.ai blog](https://civils.ai/blog/ai-for-pdf-cad-quantity-takeoffs/)

**French market (métré)**
- Planaliz: automated gros-œuvre métrés from PDF (vector) / DWG. Lists walls (murs), columns (poteaux), beams (poutres), slabs (dalles), openings, lintels, reservations, longrines, piles and mass foundations. Exports Excel, PDF repérage and IFC (Allplan/Revit). No accuracy figures; method stated only as AI analysis of vector/DWG geometry; pricing at app.planaliz.fr/pricing — [Planaliz](https://planaliz.fr/logiciel-metres-BTP-automatiques)
- Metr (metr-plan.com): AI-assisted métré from DWG/DXF/DWF/PDF with automatic scaling and Excel export — [Metr](https://www.metr-plan.com/logiciel-de-metre)
- ATTIC+ (Easy-KUTCH): expert 2D/3D métré from PDF/DWG/IFC, BIM-compatible, structured by lot. Mainly assisted, not AI detection — [ATTIC+ comparatif 2026](https://www.attic-plus.fr/toutes/blog/comparatif-logiciels-de-metre-2026-attic/)
- Quantiplan (ELA Software): computes concrete, block and brick volumes from structural plans — [ELA Software](https://www.elasoftware.fr/logiciel-de-metre-pour-professionnels-du-batiment)
- Others listed: JustBIM (PDF + BIM métré), Quoter Plan (PDF métré) — [socinformatique](https://socinformatique.fr/logiciel-metre/); [Quoter Plan](https://www.quoterplan.com/fr/); overview [LeBonLogiciel 2026](https://lebonlogiciel.com/blog/organisation-facturation-et-planification-gestion-commerciale-erp-gpao/logiciel-metre-comparatif-2026-des-meilleurs-logiciels-de-metre-batiment-plan-pdf-bim-dpgf)

### Inferences
- The common commercial primitive is example-based symbol search (Togal auto-count, Bluebeam VisualSearch, AutoCAD Search & Convert). It sidesteps semantic classification by having the user supply the exemplar. This is essentially template/similarity matching, which works well for repeated column symbols on a single sheet.
- Structural takeoff is under-served by the big AI vendors. Beam AI covers it with humans in the loop, Kreo's public API excludes structural members, and Planaliz is the most explicit structural-detection claim (French gros œuvre, IFC export), though unaudited.
- Kreo is the only vendor found with a documented public detection API. It could serve as a baseline or benchmark for walls/openings, not for columns/beams.

### Gaps
- No independent accuracy audits of any commercial tool. The Togal "peer-reviewed" study page could not be fetched.
- Not covered for lack of time or budget: Kalyzée, Batappli, Trimble (e.g. Trimble takeoff/Tekla tools), Buildots (a site progress-capture tool, not drawing takeoff), Archistar (compliance checking), SteelFlo details, and the Kreo/Planaliz API pricing.

## 5. Cost and latency of VLM APIs on A0 sheets at high resolution

### Takeaway
Every major API silently downscales a full A0 sheet to roughly 55-80 dpi, which is too coarse for dense structural drawings. Tiling is therefore required. Tiled at 150 dpi, one A0 sheet costs roughly $0.04 (Gemini) to $0.25 (Claude) in image input tokens. At 300 dpi, roughly $0.13-0.95. Output tokens and multiple passes come on top.

### Cited Findings
- OpenAI GPT-5.5: 32x32-px patches. "High" detail caps at 2,048 px / 2,500 patches, "original" at 6,000 px / 10,000 patches. $5/M input — [Roboflow, May 2026](https://blog.roboflow.com/image-token-cost-vlm/)
- Claude Opus 4.7: tokens ≈ (w × h)/750, long edge capped at 2,576 px (auto-resized). $5/M input — [Roboflow](https://blog.roboflow.com/image-token-cost-vlm/)
- Gemini 3.1 Pro: 258 tokens per 768x768 tile (258 flat if ≤384 px). $2/M input — [Roboflow](https://blog.roboflow.com/image-token-cost-vlm/)
- The same 1024x1024 image costs 258 tokens on Gemini vs ~1,334 on Claude — [Spoold / search summary](https://spoold.com/tools/vision-tokens); a 4032x3024 photo costs $0.012 (GPT-5.5), $0.033 (Claude), $0.012 (Gemini) — [Roboflow](https://blog.roboflow.com/image-token-cost-vlm/)

### Inferences (my calculations from the formulas above; A0 = 841 × 1189 mm = 33.1 × 46.8 in)
- One A0 image, no tiling: Claude caps at 2,576 px on the long edge, about 55 dpi and ~6.3k tokens (~$0.03). GPT-5.5 "original" mode's 10k-patch budget allows ~10 MP (≈2,700 × 3,800 px), about 81 dpi and ~10k tokens (~$0.05). Line weights and small text on structural plans become unreadable at these resolutions.
- Tiled at 150 dpi (~4,967 × 7,022 px ≈ 34.9 MP): Claude ≈ 46.5k tokens ≈ $0.23; GPT-5.5 ≈ 34k patches ≈ $0.17; Gemini ≈ 70 tiles × 258 ≈ 18k tokens ≈ $0.036.
- At 300 dpi (≈139.5 MP): Claude ≈ 186k tokens ≈ $0.93; GPT-5.5 ≈ 136k ≈ $0.68; Gemini ≈ 247 tiles ≈ 64k tokens ≈ $0.13.
- A set-of-marks design that sends only candidate crops (e.g. 50 columns × one 512 px crop each ≈ 350 tokens on Claude) costs about as much as the whole-sheet tiles, but each crop is at full resolution, and batching several crops per request amortises prompt overhead.
- Gemini's per-tile token count is resolution-independent inside a 768 tile. Whether Gemini downsamples one very large image instead of tiling it depends on the media_resolution setting, so send pre-cut tiles for predictable results (to verify).

### Gaps
- I found no published latency measurements for high-resolution drawing tiles. Latency depends on output length and provider load and needs to be benchmarked in-house.
- Prices and resolution caps change often, so re-check them against provider docs (Anthropic, OpenAI, Google) before budgeting. The figures above come from a May 2026 third-party article, not the vendor docs.
