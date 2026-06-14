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
6. `review-subjects`
   - per-person image folders are ranked against the local seju centroid
   - output is CSV, Markdown, and JSON for review and tracking

## Backends

- `deterministic`: implemented. Uses local image statistics and needs only `numpy` + `Pillow`.
- `opencv-face`: planned. Face-box normalization and QA using `opencv-python`.
- `insightface`: planned. Face embedding adapter with `insightface` + `onnxruntime-gpu`.
- `deepface`: planned. OSS face-model adapter for cross-checking and dataset QA.
- `clip-style`: planned. Secondary style similarity with `open-clip-torch`.
- `diffusion-generation`: planned. Diffusers/ComfyUI generation loop for prompt batches.

Keep geometry and style axes separate. A generated image can match the style prompt while missing face geometry, so evaluation should report both once neural backends are implemented.

## Subject Review Contract

Store reviewed comparison images under:

```text
data/subjects/<subject-name>/*.jpg
```

Then run:

```powershell
python -m seju_face_lab review-subjects --model outputs/seju_model --subjects data/subjects --out outputs/subject_reviews
```

The score is an approximate similarity to this local centroid only. It must not be treated as
identity recognition, attractiveness scoring, ethnicity classification, or an objective face-type label.

## GPU / Generation Plan

- RTX 4090 / RTX 5060 Ti nodes should run optional neural backends and generation batches only.
- Keep raw image sets and generated candidates Git-ignored.
- Use `insightface` or `deepface` for face-embedding cross-checks after deterministic results are stable.
- Use Diffusers or ComfyUI to generate candidates from `generation_manifest.json`, then score them with `evaluate`.
- Keep generated-image prompts aggregate-only; avoid copying a specific real person.

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
