# Research Tracking

Retrieval/design date: 2026-06-14.

## Current State

- Official seju source discovery is implemented through `sources discover`.
- Reviewed eligible image staging is implemented through `sources download`.
- Deterministic centroid build/evaluate is implemented and tested.
- Per-subject similarity review is implemented through `review-subjects`.
- Neural backends and DeepFace cross-checking are planned.
- Image-generation dry-run planning and local RTX 4090 Diffusers smoke runs are implemented.

## GitHub Issue Plan

Create and track these issues:

- `P1 GPU face embeddings`: implement InsightFace/ONNXRuntime-GPU backend for RTX machines.
- `P1 Celebrity subject review workflow`: collect reviewed subject folders and run `review-subjects`.
- `P1 Generation loop`: connect `generation_manifest.json` to Diffusers or ComfyUI batches.
- `P2 DeepFace adapter`: compare DeepFace-family embeddings against InsightFace/deterministic scores.
- `P2 CLIP style axis`: add OpenCLIP style scoring as a separate axis from face geometry.
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
3. Run `review-subjects` with the deterministic backend.
4. Run the same review with InsightFace or DeepFace once optional dependencies are installed.
5. Plan aggregate candidate faces with `generate --provider dry-run`.
6. Generate with Diffusers/ComfyUI on a GPU worker and score with `evaluate`.
7. Compare deterministic scores against InsightFace/DeepFace once neural backends are available.

## GPU Generation Notes

- RTX 4090 smoke generation succeeded with `.venv` Python 3.12.13, torch 2.12.0+cu126, and Diffusers 0.38.0.
- Small generated batches were evaluated locally; generated images and per-run scores remain ignored under `outputs/`.
- Full committed workflow notes are in `docs/gpu-generation-log.md`.
