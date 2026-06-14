# Research Tracking

Retrieval/design date: 2026-06-14.

## Current State

- Official seju source discovery is implemented through `sources discover`.
- Reviewed eligible image staging is implemented through `sources download`.
- Deterministic centroid build/evaluate is implemented and tested.
- OpenCV face-crop normalization is implemented as optional backend `opencv-face`.
- InsightFace adapter code is dependency-gated and reports ready only when `insightface` and `onnxruntime-gpu` are installed.
- DeepFace adapter code is dependency-gated and uses `DeepFace.represent` for neural cross-checking.
- Per-subject similarity review is implemented through `review-subjects`.
- SNS handle/engagement manifests and face-score correlation reports are implemented.
- OpenCLIP style-axis scoring is implemented through `style-evaluate`.
- `compare-runs` reports style and same-image combined scores when style outputs are present.
- Image-generation dry-run planning and local RTX 4090 Diffusers smoke runs are implemented.
- `generate --prompt-profile detector-friendly` records detector-oriented prompt settings for frontal, unobscured candidate batches.
- `qa-images` flags generated candidates that are collages, extreme crops, off-center faces, or missing a frontal OpenCV face.
- `compare-runs` now reads `quality/` outputs and reports QA-gated best centroid scores.

## GitHub Issue Plan

Create and track these issues:

- `P1 GPU face embeddings`: install and verify InsightFace/ONNXRuntime-GPU on RTX machines.
- `P1 Celebrity subject review workflow`: collect reviewed subject folders and run `review-subjects`.
- `P1 Generation loop`: connect `generation_manifest.json` to Diffusers or ComfyUI batches.
- `P1 SNS correlation workflow`: run handle extraction, engagement manifesting, and correlation reports.
- `P2 DeepFace adapter`: verify optional DeepFace install on local GPU/CPU and compare embeddings against InsightFace/deterministic scores.
- `P2 CLIP style axis`: verify optional OpenCLIP install and use `style-evaluate` alongside face geometry scores.
- `P2 Remote worker playbook`: document RTX 4090 and RTX 5060 Ti split-run commands.

## Local ToDo

- Keep `data/raw/`, `data/subjects/`, `data/processed/`, and `outputs/` out of Git.
- Record source URL, retrieval date, and permission notes for any comparison subject set.
- Prefer one subject per folder and multiple images per subject.
- Review `subject_reviews.md` before making any interpretation.
- Treat all similarity scores as local model measurements, not identity or objective labels.

## Next Experiment

1. Build the official seju centroid from reviewed local images.
2. Place comparison celebrity/public-figure image folders under `data/subjects/`.
3. Run `review-subjects` with the deterministic and `opencv-face` backends.
4. Run the same review with InsightFace or DeepFace once optional dependencies are installed.
5. Plan aggregate candidate faces with `generate --provider dry-run`.
6. For detector-visible scoring batches, use `generate --prompt-profile detector-friendly`.
7. Generate with Diffusers/ComfyUI on a GPU worker and score with `evaluate`.
8. Run `style-evaluate` so generated candidates have both face-geometry and style-axis scores.
9. Run `qa-images` before visual review so collages/extreme crops do not win on score alone.
10. Rank evaluated generated batches with `compare-runs`, including combined face/style scores when available.
11. Run SNS handle/engagement manifests and `analyze correlation` for reviewable metric joins.
12. Compare deterministic scores against InsightFace/DeepFace on the same ignored image sets.

## GPU Generation Notes

- RTX 4090 smoke generation succeeded with `.venv` Python 3.12.13, torch 2.12.0+cu126, and Diffusers 0.38.0.
- Small generated batches were evaluated locally; generated images and per-run scores remain ignored under `outputs/`.
- OpenCV face-crop build succeeded on the local official image set with 173 usable face crops from 259 source images.
- Detector-friendly RTX 4090 v2 produced one QA-passing candidate out of two; v3/v4 showed why QA is needed by producing extreme crops, off-center faces, and collages.
- The current committed route is detector-friendly generation, deterministic/OpenCV evaluation, `qa-images`, then `compare-runs` with QA-gated ranking before any visual interpretation.
- InsightFace sample build/evaluate succeeded on `data/raw/seju_official_sample` with 2 usable images and 512D embeddings.
- Current ONNXRuntime reports CUDA provider availability, but InsightFace execution fell back to CPU because `cublasLt64_12.dll` is missing.
- Full committed workflow notes are in `docs/gpu-generation-log.md`.
