# SOTA benchmark research

Research date: 2026-06-15.

This note tracks face-vector, quality, and benchmark ideas that are useful for seju-face-lab.
It is implementation guidance for aggregate visual analysis, not biometric enrollment or identity verification.

## Primary Sources Checked

- NIST FRTE/FATE: official split between face recognition evaluation and face analysis evaluation.
  Source: https://www.nist.gov/programs-projects/face-technology-evaluations-frtefate
- NIST FRTE 1:1: standardized verification reports with FNMR/FMR, template generation, multi-face handling, and demographic/quality discussion.
  Source: https://pages.nist.gov/frvt/html/frvt11.html
- NIST FRTE 1:N: large-gallery identification benchmark track.
  Source: https://pages.nist.gov/frvt/html/frvt1N.html
- IJB-C: template-based unconstrained benchmark across still and video media.
  Source: https://biometrics.cse.msu.edu/Publications/Face/Mazeetal_IARPAJanusBenchmarkCFaceDatasetAndProtocol_ICB2018.pdf
- MegaFace: distractor-gallery stress benchmark for verification/identification.
  Source: https://arxiv.org/abs/1512.00596
- ArcFace: angular-margin face embedding family used by InsightFace/DeepFace ArcFace paths.
  Source: https://arxiv.org/abs/1801.07698
- AdaFace: quality-adaptive margin, using feature norms as an image-quality proxy.
  Source: https://arxiv.org/abs/2204.00964
- MagFace: embedding magnitude as recognition and quality signal.
  Source: https://arxiv.org/abs/2103.06627
- InsightFace: OSS face detection/alignment/recognition toolkit with ArcFace-family implementations.
  Source: https://github.com/deepinsight/insightface
- DeepFace: OSS wrapper for multiple detectors and recognition models.
  Source: https://github.com/serengil/deepface
- OpenCLIP and LAION CLIP benchmark: style/prompt/image-text evaluation path, separate from face geometry.
  Sources: https://github.com/mlfoundations/open_clip and https://github.com/LAION-AI/CLIP_benchmark
- Worldcoin Open IRIS: separate-modality reference for segmentation-template-match architecture.
  Source: https://github.com/worldcoin/open-iris

## Implementation Decisions

- Implemented now: `build --balance subject`.
  IJB-C-style template logic is now available for folder-based datasets. Images are averaged per subject folder first, then subjects are averaged. This reduces overweighting of subjects with many images.
- Implemented now: agency page evidence badges.
  The public page can show `real_centroid_baseline`, `real_and_generated`, `hypothesis_and_generated`, or `unverified` using the data-quality audit JSON.
- Kept: InsightFace/ArcFace and DeepFace/RetinaFace as neural cross-checks.
  They remain optional dependency backends because local deterministic vectors are useful for reproducible smoke tests.
- Kept separate: OpenCLIP style score.
  CLIP-style similarity is useful for prompt and visual style review, but it is not merged into face-vector claims.

## Next Algorithms To Add

- Quality-aware centroid weighting:
  Use detector acceptance, QA result, and neural embedding norm proxies as confidence bands. AdaFace/MagFace motivate this, but the score must stay visible rather than silently changing the centroid.
- Distractor-gallery review:
  Add MegaFace/FRTE-inspired subject review with distractor folders and CMC/FPIR-style rank curves.
- Local verification pairs:
  Add FRTE-style same/different pair exports with ROC-style CSV so backend changes can be evaluated beyond top-rank examples.
- Template diagnostics:
  For each subject/agency, export subject image count, accepted image count, per-subject vector norm, intra-subject spread, and final subject weight.
- Detector parity:
  Continue `compare-deepface-detectors` and promote only score changes that survive acceptance-count and rank-agreement checks.

## Boundary

The project should use these methods to improve aggregate-image research, prompt generation, and public-page evidence clarity.
It should not claim NIST-equivalent performance, infer identity, rank personal value, or mix iris templates with face vectors.
