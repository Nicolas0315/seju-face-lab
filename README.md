# seju-face-lab

`seju-face-lab` is a local pipeline for approximating a "seju face" profile from a curated image set.

It can:

- extract deterministic image vectors from input portraits
- compute mean and median centroid vectors
- render approximate mean/median face images
- discover official seju profile image candidates into a source manifest
- write generation prompts from centroid descriptors
- score generated images against the mean/median vectors
- review per-person image folders against the seju centroid

The current implementation is intentionally local and dependency-light: Python + `numpy` + `Pillow`. It does not identify people, infer identity, or claim that the result is a universal definition of "seju face"; it only summarizes the images you provide.

## Quick Start

From this repo root, expose the local `src` package:

```powershell
$env:PYTHONPATH=(Resolve-Path .\src)
```

Put consented reference images here:

```text
data/raw/
```

Build a centroid model:

```powershell
python -m seju_face_lab build --images data/raw --out outputs/seju_model
```

This writes:

```text
outputs/seju_model/centroids.npz
outputs/seju_model/profile.json
outputs/seju_model/mean_face.png
outputs/seju_model/median_face.png
outputs/seju_model/prompt.txt
outputs/seju_model/generation_manifest.json
outputs/seju_model/report.md
```

After generating candidate images with any image generator, place them in `outputs/generated/` and evaluate:

```powershell
python -m seju_face_lab evaluate --model outputs/seju_model --images outputs/generated --out outputs/evaluation
```

Plan a reproducible generation batch without downloading or running a model:

```powershell
python -m seju_face_lab generate --model outputs/seju_model --out outputs/generated --provider dry-run --count 8
```

Run a Diffusers batch after installing a CUDA-enabled PyTorch build and `.[generation]`.
`--variant auto` uses the `fp16` model variant when `--dtype float16` is selected:

```powershell
python -m seju_face_lab generate --model outputs/seju_model --out outputs/generated --provider diffusers --hf-model runwayml/stable-diffusion-v1-5 --count 8 --negative-prompt "copied identity"
```

Review other public-figure or celebrity image sets by folder:

```text
data/subjects/
  subject_a/
    image1.jpg
  subject_b/
    image1.jpg
```

```powershell
python -m seju_face_lab review-subjects --model outputs/seju_model --subjects data/subjects --out outputs/subject_reviews
```

This writes `subject_reviews.csv`, `subject_reviews.md`, and `subject_reviews.json`.

## Recommended Data

- Use images you have permission to analyze.
- Keep the dataset source-consistent: similar crop, lighting, camera distance, and expression.
- Start with 30+ images. The tool works with fewer, but the centroid is fragile.
- If full-body or crowded images are used, crop faces manually first. The default crop is a center crop, not a neural face detector.

## Web Source Discovery

The source discovery command creates a URL manifest first. It does not download images by default.

```powershell
python -m seju_face_lab sources discover --out data/processed/seju_sources.jsonl --as-of 2026-06-14
```

The manifest records:

- official profile page URL
- candidate image URL
- profile name and birth date when visible
- `eligible_for_analysis`, defaulting to `false` for under-18 or age-unknown profiles
- retrieval timestamp and source notes

Use this as a review queue. Download/analyze only images you have rights and consent to use.

See `docs/web-source-strategy.md` for the current site-structure analysis and extraction boundaries.

After reviewing the manifest, stage eligible images locally:

```powershell
python -m seju_face_lab sources download --manifest data/processed/seju_sources.jsonl --out data/raw/seju_official --max-count 50
```

Use `--dry-run` first to inspect planned local file names without downloading.

## Analysis Backends

```powershell
python -m seju_face_lab backends
```

Implemented now:

- `deterministic`: no neural dependency, good for local smoke tests and rough visual centroids

Designed next:

- `opencv-face`: face crop/QA normalization
- `insightface`: GPU face embeddings on RTX machines
- `deepface`: OSS face-model cross-checking and QA
- `clip-style`: secondary style similarity scoring
- `diffusion-generation`: Diffusers/ComfyUI candidate generation loop

See `docs/architecture.md` for the folder contract and backend plan.
See `docs/research-tracking.md` for the current GitHub Issue / ToDo breakdown.

## Output Meaning

- `mean_face.png`: pixel-wise mean of the normalized image crops.
- `median_face.png`: pixel-wise median, often more robust to outliers.
- `profile.json`: compact descriptor values and vector metadata.
- `prompt.txt`: a generation prompt based on observed centroid descriptors.
- generation `generation_run.json`: prompt, seed, provider, output paths, and evaluation command/argv.
- evaluation `scores.csv`: similarity of candidate generated images to the centroid vectors.
- evaluation `summary.json`: best/mean/median generated-image similarity for quick comparisons.
- subject review outputs: per-person approximate similarity rankings.

## Verification

```powershell
python -m unittest discover -s tests
```
