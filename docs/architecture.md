# Architecture

Retrieval/design date: 2026-06-14.

## Pipeline

1. `sources discover`
   - official profile pages to JSONL URL manifest
   - robots checked
   - age unknown / under minimum age excluded by default
2. reviewed image staging
   - operator places consent-cleared crops in `data/raw/`
   - raw images stay Git-ignored
   - `sources download` can materialize reviewed manifest rows into `data/raw/seju_official/`
3. `build`
   - vector backend extracts per-image vectors
   - mean and median centroids are saved
   - mean/median appearance images are rendered
4. generation
   - `prompt.txt` and `generation_manifest.json` feed ComfyUI or another generator
5. `evaluate`
   - generated candidates are scored against mean and median vectors

## Backends

- `deterministic`: implemented. Uses local image statistics and needs only `numpy` + `Pillow`.
- `opencv-face`: planned. Face-box normalization and QA using `opencv-python`.
- `insightface`: planned. Face embedding adapter with `insightface` + `onnxruntime-gpu`.
- `clip-style`: planned. Secondary style similarity with `open-clip-torch`.

Keep geometry and style axes separate. A generated image can match the style prompt while missing face geometry, so evaluation should report both once neural backends are implemented.

## Folder Contract

- `configs/`: reproducible source and pipeline configuration.
- `data/raw/`: reviewed local reference images, ignored.
- `data/processed/`: manifests and intermediate derived data, ignored.
- `docs/`: design notes, retrieval evidence, and runbooks.
- `outputs/`: centroid models, prompts, generated images, evaluations, ignored.
- `src/seju_face_lab/`: package code.
- `tests/`: deterministic fixture tests.

## Commit Strategy

Use small logical commits:

1. scaffold package and deterministic centroid pipeline
2. add official source manifest discovery
3. add backend architecture and design docs

This keeps commit logs readable and lets later neural-backend work land as separate changes.
