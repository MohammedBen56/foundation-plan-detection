# Raster (image-based) and zero-shot approaches for structural element detection on plans

Scope: image-only models and datasets for walls, columns, beams, footings and slabs on floor plans and structural drawings. Constraints assumed: no colours or CAD layers and no labelled data of our own. Research date: 2026-09. Budget-limited pass (about 20 tool calls). Some items rely on search snippets, and these are flagged.

## 1. Floor-plan models and datasets (classes, licences, weights, metrics)

### Takeaway
Public floor-plan datasets are almost all residential and architectural. They label walls well (and sometimes columns), but nearly none label beams, footings or slabs. The main exception is ArchCAD-400K (2025), which labels columns and beams on mostly commercial CAD drawings. Its licence is academic, and it is really a vector/CAD dataset. Wall segmentation reaches IoU of about 0.8–0.9 inside a dataset, but drops sharply on other drawing styles.

### Cited Findings
- **CubiCasa5K**: 5,000 floor plans annotated with more than 80 object categories. Licence is CC BY-NC 4.0 (non-commercial). Pretrained weights are downloadable (`model_best_val_loss_var.pkl`). The default eval setup uses 44 classes (rooms and icons, with wall among the room classes). — [CubiCasa5k GitHub](https://github.com/CubiCasa/CubiCasa5k); [LICENSE](https://github.com/CubiCasa/CubiCasa5k/blob/master/LICENSE)
- A community Hugging Face model (UNet with ResNet-34 encoder) trained on CubiCasa5K segments floor, wall, door and window. — [Yytsi/floorplan-to-3d-walls](https://huggingface.co/Yytsi/floorplan-to-3d-walls)
- **MLSTRUCT-FP** (Automation in Construction, 2023): 954 high-resolution multi-unit plan rasters from 165 Chilean residential projects by 52 architectural offices (2004–2018), covering many drawing styles. Images are 6,500–9,500 px. Labels are wall polygons ("rect"), slab and floor. Walls are the only annotated element class. A U-Net wall-segmentation benchmark is in a separate repo, and the data is available as a Python package (`mlstructfp`). — [GitHub MLStructFP](https://github.com/MLSTRUCT/MLStructFP); [benchmarks](https://github.com/MLSTRUCT/MLStructFP_benchmarks); [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0926580523003928); [PyPI](https://pypi.org/project/mlstructfp/0.6.1/)
- A multi-unit floor plan recognition paper (arXiv 2408.01526, 2024) reports wall segmentation IoU/F1 for its CAB models (EfficientNetB2 backbone):
  - CubiCasa (560 refined plans): 0.83 / 0.90. The DFPR baseline scores 0.72 / 0.83.
  - R3D: 0.90 / 0.94.
  - CVC-FP (122 scanned plans): 0.88–0.91 / 0.94–0.95.
  - MLSTRUCT-FP: 0.81–0.83 / 0.89–0.90.
  - Each result appears to come from training and testing within the same dataset.
  - Code is linked on an anonymous repo, and the paper is CC BY 4.0.
  — [arXiv 2408.01526](https://arxiv.org/html/2408.01526v1)
- **R2V (Raster-to-Vector, Liu et al. 2017)**: the repo provides pretrained models through `pretrained/download_links.txt`. — [FloorplanTransformation GitHub](https://github.com/art-programmer/FloorplanTransformation)
- **DeepFloorplan** (Zeng et al., ICCV 2019, room-boundary-guided attention) has pretrained models, plus TF2 and PyTorch ports. I could not confirm the licences in this pass. — [zlzeng/DeepFloorplan](https://github.com/zlzeng/DeepFloorplan); [TF2 port](https://github.com/RogerLiang0725/TF2DeepFloorplan); [arXiv 1908.11025](https://arxiv.org/pdf/1908.11025)
- Survey-level numbers, from a search snippet only: about 80.5% mIoU on R2V and 89.1% mIoU on CubiCasa5K for recent methods. Another snippet gives wall IoU of 95.99% on CubiCasa, 70.39% on rotated CubiCasa and 20.47% on CVC, apparently from a rotation or generalisation study. I could not open the PDF to confirm the setup, because it was over the size limit. — [ACM survey 2025](https://dl.acm.org/doi/10.1145/3747227.3747250); [arXiv 2504.03241](https://arxiv.org/pdf/2504.03241); [arXiv 2512.02413](https://arxiv.org/pdf/2512.02413)
- **ResPlan** (arXiv Aug 2025): 17,000 residential vector-graph plans with a 17-class taxonomy (walls, doors, windows and room types). Polygons are given in pixel and metre coordinates. Data and code are public. It aims to fix the idealised layouts of RPLAN and MSD. — [arXiv 2508.14006](https://arxiv.org/abs/2508.14006)
- **ArchCAD-400K** (arXiv 2503.22346, 2025):
  - Size: 5,538 drawings, 413,062 chunks of 14 m × 14 m; 26 times larger than FloorPlanCAD; 86% commercial or public buildings.
  - Classes: 27, including structural classes (column, beam, hole) and notation classes (axis lines, labels).
  - Annotation: panoptic, in raster and vector forms.
  - Baseline: DPSS reaches PQ 70.6 (stuff 77.6, thing 65.6).
  — [arXiv HTML](https://arxiv.org/html/2503.22346)
- ArchCAD-400K licensing and release: the paper is CC BY 4.0, but the repo shows an "ACADEMIC" licence badge. The initial release is a 40K-sample subset on Hugging Face. The repo references pretrained weights (`svgnet/pretrained`), but I could not confirm that the DPSS weights are included. — [ArchiAI-LAB/ArchCAD](https://github.com/ArchiAI-LAB/ArchCAD)
- **CVC-FP**: about 122–123 real scanned floor plans with SVG polygon labels (walls, doors, windows, rooms). — [arXiv 2408.01526](https://arxiv.org/html/2408.01526v1); [FloorYOLO](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1911589/full)
- **SESYD**: 1,000 synthetic floor plans with 16 symbol classes, mostly furniture, doors and windows. — [FloorYOLO](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1911589/full)

### Inferences
- Walls: models trained on CubiCasa5K, MLSTRUCT-FP or CVC-FP are the most reusable off the shelf. Expect IoU of about 0.8–0.9 on plans in a similar style and much lower on unfamiliar styles.
- The CubiCasa5K weights are non-commercial, so they are fine for a proof of concept but need a licence check before production.
- Columns and beams: ArchCAD-400K is the only large public source found. Its labels are CAD primitives, so it would need rasterising to train a raster detector. The academic licence needs checking before commercial use.
- Footings, rafts and slabs: I found no public raster dataset. The only related label is MLSTRUCT-FP's "slab", which marks the floor-slab outline on architectural plans, not structural slabs.

### Gaps
- Structured3D, ROBIN and RPLAN details (classes, licences) were not verified in this pass. RPLAN is residential layouts only, with no columns (general knowledge, not verified here).
- I could not confirm the exact licence text for DeepFloorplan, R2V or the ArchCAD weights.
- I found no cross-style (train on A, test on B) wall-IoU study that I could verify beyond the 20.47% CVC snippet.

## 2. Papers on structural drawings (columns, beams, walls)

### Takeaway
Work that targets structural drawings directly is small-scale. The typical setup is about 500 images, 5 classes and YOLO, with "accuracy above 80%". None of the datasets or weights I found are clearly public. ArchCAD-400K is the only large benchmark that includes columns and beams.

### Cited Findings
- Zhao et al. (Applied Sciences, 2020), "A Deep Learning-Based Method to Detect Components from Scanned Structural Drawings for Reconstructing 3D Models":
  - Data: 500 images of 2D structural drawings with 5 classes (grid reference, column, horizontal beam, vertical beam, sloped beam).
  - Model: YOLO.
  - Results: average detection accuracy above 80%, at 0.71 s per image.
  - The MDPI page failed to load, so these details come from search snippets.
  — [MDPI](https://doi.org/10.3390/app10062066); [ResearchGate](https://www.researchgate.net/publication/340035388_A_Deep_Learning-Based_Method_to_Detect_Components_from_Scanned_Structural_Drawings_for_Reconstructing_3D_Models)
- A YOLO-based method to recognise structural components from 2D drawings (ASCE proceedings) reports average recognition accuracy above 80% for beams and columns, at 0.63 s per image (search snippet). — [ASCE](https://ascelibrary.org/doi/10.1061/9780784482865.080)
- ArchCAD-400K has column and beam classes on commercial-building CAD, with a panoptic symbol-spotting baseline (PQ 70.6 across all classes). Per-class column and beam PQ was not retrieved. — [arXiv 2503.22346](https://arxiv.org/html/2503.22346)
- Few-shot symbol detection on engineering drawings (Applied Artificial Intelligence, 2024) states that few-shot detection "has not yet been explored extensively" for engineering symbol digitisation. Rare symbols and annotation cost motivate it. — [Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/08839514.2024.2406712)
- There is a 2025 pipeline for parsing engineering drawings (mechanical-style): YOLOv11 with oriented bounding boxes, followed by transformer parsing. — [arXiv 2505.01530](https://arxiv.org/pdf/2505.01530)
- Roboflow Universe has small community datasets of structural drawings. One example is a 33-image set labelled "Structural" with the classes "ELM_Column" and "Drawing Title". — [Roboflow search class:structural](https://universe.roboflow.com/search?q=class%3Astructural); [class:column](https://universe.roboflow.com/search?q=class:column)
- A YOLOv8 repo detects columns, walls, doors and windows on architectural floor plans. — [sanatladkat/floor-plan-object-detection](https://github.com/sanatladkat/floor-plan-object-detection)

### Inferences
- The "above 80% accuracy" figures come from small, single-source datasets. They should not be read as a guide to performance on arbitrary drawing offices.
- Columns on structural plans are usually small, repeated filled or hatched squares or circles at grid intersections. That makes them the most tractable class for detection or template or exemplar matching.
- Beams, which show as dashed or double lines, and footings, which show as rectangles with hatching or tags, are harder for box detectors. They are closer to line or vector problems.

### Gaps
- I could not find a public structural-plan raster dataset with footing, raft or slab labels.
- I could not verify 2023–2026 Automation in Construction or AEI papers on column and beam detection with public code in this pass.

## 3. Zero-shot, open-vocabulary and visual-prompt models

### Takeaway
Text-prompted open-vocabulary detectors transfer very poorly to line drawings. Grounding DINO scores mAP@0.5 of about 0.0 on CubiCasa5K symbols. Visual-exemplar prompting ("find more like this") is the most promising zero-label route. SAM 3 supports it with open weights. T-Rex2 does too, but is API-only. Pure SAM gives masks without labels and needs guidance.

### Cited Findings
- Grounding DINO-tiny, zero-shot on CubiCasa5K (400 test plans, 10 classes, 1024 px input):
  - mAP@0.5 is about 0.0.
  - About 95% of door locations did get a prediction, but the boxes were 100–330 times too large, covering whole rooms.
  - Prompting "{class} symbol" lifted compact fixtures (toilet from 6% to 21%) but left doors and windows near 0%.
  - Fine-tuned Grounding DINO reached 39.3 mAP@0.5.
  - A small YOLOv8s trained in-domain reached 67.1, and the Apache-licensed YOLOX-s reached 65.7.
  — [heypaprika/gdino](https://github.com/heypaprika/gdino)
- Grounding DINO scores 52.5 AP zero-shot on COCO and 26.1 mean AP on ODinW. OWLv2 self-trains on pseudo-boxes from about 1 billion web images. — [Roboflow blog](https://blog.roboflow.com/grounding-dino-zero-shot-object-detection/); [Forasoft overview](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/open-vocabulary-detection-grounding-dino-florence-2-rtdetr-rfdetr)
- A 2026 comparison of YOLO-World, SAM3, Grounding DINO and OWLv2 (aerial imagery) found that "conditions that optimize one architecture can collapse another": prompt sensitivity is architecture-specific. — [arXiv 2601.22164](https://arxiv.org/pdf/2601.22164)
- **SAM 3** (Meta, Nov 2025, arXiv 2511.16719):
  - Promptable Concept Segmentation returns every instance of a concept. The concept can come from a text noun phrase, from image exemplars (boxes), or from both.
  - Negative exemplars can suppress false positives.
  - It was trained on 4M unique concept labels.
  - SAM 3.1 has since been released.
  — [arXiv 2511.16719](https://arxiv.org/abs/2511.16719); [Ultralytics docs](https://docs.ultralytics.com/models/sam-3); [Meta blog](https://ai.meta.com/blog/segment-anything-model-3/)
- **T-Rex2** (ECCV 2024): combines text and visual prompts. Visual prompts suit "novel objects" that are hard to describe in text. Weights are **API-only**: the repo is API client code under IDEA License 1.0 and needs a token from DeepDataSpace cloud. The same repo points to **Rex-Omni** as a "fully open-sourced" successor with detection, OCR and grounding. — [arXiv 2403.14610](https://arxiv.org/pdf/2403.14610); [IDEA-Research/T-Rex](https://github.com/IDEA-Research/T-Rex)
- SAM on floor plans: it cannot label its masks, and each prompt yields one object. The GMFS method (foundation models plus five reference samples plus post-processing) gave a 2.7-fold F1 gain over SAM's automatic mode for segmenting floor-plan doors. — [ScienceDirect S2772991524000562](https://www.sciencedirect.com/science/article/pii/S2772991524000562) (search snippet; the full text returned HTTP 403)
- VLMs on floor plans were studied at ACM DocEng 2025 ("Visual LLMs for graphics understanding: floorplan images"). Details were not retrieved. — [ACM DL](https://dl.acm.org/doi/10.1145/3704268.3748681)

### Inferences
- Do not expect usable zero-shot text prompts ("column", "wall") from Grounding DINO, OWLv2, YOLO-World or Florence-2 on plans. The only direct evidence is the CubiCasa result, and it is a near-total failure with grossly oversized boxes.
- The best no-label workflow is exemplar-based: a user or rule marks one or two columns per sheet, then SAM 3 exemplar prompts (open weights) or T-Rex2 (API) find the repeats. The next best is DINOv2 patch-feature similarity or classical template matching (normalised cross-correlation), since columns are near-identical glyphs.
- Exemplar results can then be pseudo-labels to train a small YOLOX or RT-DETR model.
- I found no published metric for SAM 3 or T-Rex2 exemplar mode on technical drawings. Validate on our own sheets before relying on either.

### Gaps
- I found no benchmark of OWLv2, Florence-2, YOLO-World, SAM 2/3 or DINOv2 matching specifically on construction or structural drawings.
- I did not verify Rex-Omni's licence or its performance on drawings.

## 4. Practical issues: huge sheets, thin lines, text clutter, colour independence, box versus vector precision

### Takeaway
There is little direct literature on this, but the evidence points one way. Sheets are 6,500–9,500 px or larger, so tiling is mandatory. Open-vocabulary detectors mislocalise thin-line symbols badly. Boxes and masks from raster models only approximate vector geometry, and need snapping or vectorisation afterwards.

### Cited Findings
- MLSTRUCT-FP images are 6,500–9,500 px. Wall-segmentation work on it extracted more than 1.3 million patches for training. — [MLStructFP](https://github.com/MLSTRUCT/MLStructFP); [PMC wall segmentation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10557507/)
- ArchCAD-400K splits whole drawings into 14 m × 14 m chunks because the sheets are too large. — [arXiv 2503.22346](https://arxiv.org/html/2503.22346)
- Grounding DINO at 1024 px on whole floor plans predicted boxes 100–330 times too large. Downscaling a whole plan clearly destroys small symbols. — [heypaprika/gdino](https://github.com/heypaprika/gdino)
- Rotation sensitivity: wall IoU reportedly falls from 95.99% to 70.39% on rotated CubiCasa (search snippet). — [arXiv 2504.03241](https://arxiv.org/pdf/2504.03241)
- Raster pipelines use segmentation followed by vectorisation and post-processing to recover wall geometry. — [arXiv 2408.01526](https://arxiv.org/html/2408.01526v1)
- On CVC-FP, detector mAP@0.5 is 0.82 but mAP@0.5:0.95 is only 0.48. Boxes are roughly right but not precise. — [FloorYOLO](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1911589/full)

### Inferences
- A0 at 300 dpi is about 9,900 × 14,000 px. Use overlapping tiles (for example 1024–1536 px with 15–25% overlap, SAHI-style) and merge with NMS or WBF. These tile values are a suggestion, not tested here.
- Converting to grayscale and binarising works with public datasets, which are mostly black and white, apart from MLSTRUCT-FP's colourised plans. It also satisfies the no-colour constraint.
- If the PDFs are vector, raster detections should act as seeds, and final geometry should be snapped to the vector lines. The low mAP@0.5:0.95 shows that box precision alone is weak for quantity takeoff.

### Gaps
- I found no published study quantifying the effect of dpi or tile size on structural drawings, or of text clutter on detection.

## 5. Fine-tuning YOLO or DETR: data needs and synthetic data

### Takeaway
Floor-plan symbols are very learnable with modest in-domain data. A few hundred to a few thousand plans give mAP@0.5 of about 0.65–0.99. Procedurally generated synthetic plans, with noise injected, are a proven way to bootstrap without manual labels.

### Cited Findings
- YOLOv8s trained on CubiCasa5K reached 67.1 mAP@0.5 on 10 classes. Doors and windows went from 0.0 AP zero-shot to 86.7 and 84.6 AP after in-domain training. YOLOX-s (Apache-2.0) matched it at 65.7, avoiding YOLOv8's AGPL-3.0 licence. — [heypaprika/gdino](https://github.com/heypaprika/gdino)
- FloorYOLO (YOLOv8n with ECA attention, Frontiers 2026):
  - SESYD (1,000 synthetic plans; 700 train): mAP@0.5 of 0.995.
  - CVC-FP (only 98 real training images): mAP@0.5 of 0.824 and mAP@0.5:0.95 of 0.48.
  — [Frontiers](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1911589/full)
- A procedural floor-plan generator with random perturbations gave YOLOv5 virtually unlimited training data. Adding real-world noise patches to synthetic images kept the drop small when moving to real plans (YOLO Nano). — [Object Detection in Floor Plans for VR (PDF)](https://pdfs.semanticscholar.org/558c/fec5a854a8c0e775c25bab2b8589097ce170.pdf)
- The structural-drawing YOLO work used 500 images for 5 classes and reached above 80% accuracy (search snippet). — [MDPI Applied Sciences 2020](https://doi.org/10.3390/app10062066)
- The Kaggle "Floor Plans 500" dataset is annotated in YOLO11 format. — [Kaggle](https://www.kaggle.com/datasets/umairinayat/floor-plans-500-annotated-object-detection)

### Inferences
- A realistic path with no labelled data:
  1. Generate synthetic structural plans: grids, columns (filled or hatched squares and circles), beam lines, footing rectangles, dimension and text clutter, rendered in black and white at the target dpi.
  2. Optionally rasterise the ArchCAD-400K column and beam labels, if the licence allows.
  3. Add exemplar-mined pseudo-labels from SAM 3 or template matching.
  4. Train a permissively licensed detector (YOLOX, RT-DETR or RF-DETR) on tiles.
- Expect a few hundred real, pseudo-labelled or verified sheets to be needed to close the synthetic-to-real gap. This is extrapolated from the CVC-FP result (98 images gave 0.82) and the structural paper (500 images).

### Gaps
- I found no study measuring how many labelled structural sheets reach a given column or beam accuracy.
- I found no public synthetic generator built specifically for structural (rather than architectural) plans.
