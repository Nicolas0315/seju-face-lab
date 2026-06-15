# seju-face-lab

`seju-face-lab` is a local pipeline for approximating a "seju face" profile from a curated image set.

It can:

- extract deterministic image vectors from input portraits
- compute mean and median centroid vectors
- render approximate mean/median face images
- discover official seju profile image candidates into a source manifest
- write generation prompts from centroid descriptors
- score generated images against the mean/median vectors
- score generated images on a separate OpenCLIP style axis
- review per-person image folders against the seju centroid
- extract SNS handles/engagement manifests and correlate them with face-score outputs

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
outputs/seju_model/vectors/image_vector_failures.json  # only when some images fail
```

Audit mean/median centroid vectors and descriptor deltas:

```powershell
python -m seju_face_lab audit-model --model outputs/seju_model --out outputs/model_audit
```

Export the full mean/median embedding vectors for external analysis or generation tooling:

```powershell
python -m seju_face_lab export-vectors --model outputs/seju_model --out outputs/seju_vectors.json
python -m seju_face_lab export-vectors --model outputs/seju_model --out outputs/seju_vectors.csv --format csv
```

After generating candidate images with any image generator, place them in `outputs/generated/` and evaluate:

```powershell
python -m seju_face_lab evaluate --model outputs/seju_model --images outputs/generated --out outputs/evaluation
```

Evaluate the same generated images on a separate OpenCLIP image-style axis after installing `.[clip]`:

```powershell
python -m pip install -e ".[clip]"
python -m seju_face_lab style-evaluate --model outputs/seju_model --images outputs/generated --out outputs/style_evaluation
```

Compare evaluated generation runs:

```powershell
python -m seju_face_lab evaluate --model outputs/seju_model --images outputs/generated_a --out outputs/generated_a/evaluation
python -m seju_face_lab style-evaluate --model outputs/seju_model --images outputs/generated_a --out outputs/generated_a/style_evaluation
python -m seju_face_lab evaluate --model outputs/seju_model --images outputs/generated_b --out outputs/generated_b/evaluation
python -m seju_face_lab style-evaluate --model outputs/seju_model --images outputs/generated_b --out outputs/generated_b/style_evaluation
python -m seju_face_lab compare-runs --runs outputs/generated_a outputs/generated_b --out outputs/run_reviews
```

`compare-runs` also accepts evaluation output directories that contain `summary.json`. When a run has
`style_evaluation/style_summary.json` and `style_evaluation/style_scores.csv`, the report includes
face, style, and per-image combined scores with the matched image path.
When a run has `quality/image_quality.json` and `quality/image_quality.csv`, the report also includes
QA pass counts and the best centroid score among QA-passing generated images.

Plan a reproducible generation batch without downloading or running a model:

```powershell
python -m seju_face_lab generate --model outputs/seju_model --out outputs/generated --provider dry-run --count 8
```

Run a Diffusers batch after installing a CUDA-enabled PyTorch build and `.[generation]`.
`--variant auto` uses the `fp16` model variant when `--dtype float16` is selected:

```powershell
python -m seju_face_lab generate --model outputs/seju_model --out outputs/generated --provider diffusers --hf-model runwayml/stable-diffusion-v1-5 --count 8 --negative-prompt "copied identity"
```

For batches meant to pass face detectors before scoring, use the detector-friendly prompt profile:

```powershell
python -m seju_face_lab generate --model outputs/seju_model --out outputs/generated_detector --provider diffusers --prompt-profile detector-friendly --count 8
```

To generate and immediately run the standard review after a real Diffusers batch, add `--review`.
Dry-run plans still only write the plan:

```powershell
python -m seju_face_lab generate --model outputs/seju_model --out outputs/generated_detector --provider diffusers --prompt-profile detector-friendly --count 8 --review --review-out outputs/generated_detector/review
```

Run the OpenCV generated-image QA gate before trusting a high score from a generated batch:

```powershell
python -m seju_face_lab qa-images --images outputs/generated_detector --out outputs/generated_detector/quality
```

This writes `image_quality.csv`, `image_quality.md`, and `image_quality.json`. A candidate passes only
when OpenCV sees one centered frontal face with a usable crop size.

Or run the standard generated-image review in one command:

```powershell
python -m seju_face_lab review-generated --model outputs/seju_model --images outputs/generated_detector --out outputs/generated_detector/review
```

This writes `evaluation/`, `quality/`, and a one-run `generation_run_reviews.*` report.

Summarize the centroid model, generated-image precision review, QA gate, optional
subject-review evidence, and backend agreement into one reviewable bundle:

```powershell
python -m seju_face_lab precision-report --model outputs/seju_model --model-audit outputs/model_audit --vector-export outputs/seju_vectors.json --generation-review outputs/generated_detector/review --subject-review outputs/subject_reviews --backend-comparison outputs/backend_compare --subject-backend-comparison outputs/subject_backend_compare --out outputs/precision_report
```

This writes `precision_report.json` and `precision_report.md`, including the optional
mean/median vector audit and vector-export evidence when `--model-audit` and
`--vector-export` point at files or directories.

Run a reproducible local pipeline from a JSON config:

```powershell
python -m seju_face_lab run-pipeline --config configs/pipelines/full-local-review.example.json --out outputs/local_pipeline_run
```

The runner executes the configured build/audit-model/evaluate/review/backend-comparison/
subject-backend-comparison/precision steps and writes `pipeline_run.json` plus `pipeline_run.md`.
Add a `style_evaluation` config block after installing `.[clip]` to include the optional
`style-evaluate` OpenCLIP scoring step in the same run.
Use `configs/pipelines/full-retinaface-review.example.json` when the same run should include
the audited `deepface-retinaface` rank-agreement backend in the final precision bundle.

To track multiple generation settings as one experiment, use a generation sweep:

```powershell
python -m seju_face_lab run-pipeline --config configs/pipelines/generation-sweep.example.json --out outputs/generation_sweep_pipeline
```

Each sweep run writes `outputs/generation_sweep/<run-name>/generation_run.json`.
When `review=true` and `compare_runs=true`, the pipeline reviews each generated batch and writes
a shared `generation_run_reviews.*` bundle for the precision report.

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

This writes `subject_reviews.csv`, `subject_reviews.md`, `subject_reviews.json`, and
`subject_reviews.html` with local thumbnail cards for reviewing each subject's nearest image.

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

To reorganize downloaded official images into per-talent folders for subject review:

```powershell
python scripts/organize_by_talent.py --src data/raw/seju_official --dst data/raw/seju_by_talent --dry-run
```

SNS handle and public engagement collection are separate, reviewable manifests:

```powershell
python -m seju_face_lab sources scrape-handles --manifest data/processed/seju_sources.jsonl --out data/processed/sns_handles.jsonl
python -m seju_face_lab sources fetch-engagement --handles data/processed/sns_handles.jsonl --out data/processed/sns_engagement.jsonl --platforms instagram tiktok
```

The SNS fetchers are best-effort public-page readers. Treat blocked/partial rows as expected data quality signals.

When public pages are blocked, create a manual CSV template and import reviewed values:

```powershell
python scripts/generate_engagement_csv_template.py --handles data/processed/sns_handles.jsonl --out data/processed/sns_engagement_manual.csv
python -m seju_face_lab sources import-engagement --csv data/processed/sns_engagement_manual.csv --out data/processed/sns_engagement.jsonl
```

## Correlation Analysis

After running `review-subjects`, join those face scores to SNS engagement:

```powershell
python -m seju_face_lab analyze correlation --face-scores outputs/subject_reviews/subject_reviews.json --engagement data/processed/sns_engagement.jsonl --out outputs/correlation
```

This writes `correlation_dataset.csv`, `correlations.csv`, `correlation_report.md`, and `correlation_summary.json`.

## Analysis Backends

```powershell
python -m seju_face_lab backends
python -m seju_face_lab backend-diagnostics --out outputs/backend_diagnostics
```

Implemented now:

- `deterministic`: no neural dependency, good for local smoke tests and rough visual centroids
- `opencv-face`: optional `.[vision]` backend for OpenCV face crop normalization
- `insightface`: optional `.[face]` backend; shown as ready only when `insightface` and `onnxruntime-gpu` are installed
- `deepface`: optional `.[deepface]` backend using `DeepFace.represent`, defaulting to ArcFace
- `deepface-retinaface`: optional `.[deepface]` backend using ArcFace embeddings with the RetinaFace detector; current DeepFace face-validated cross-check candidate after the v7 detector audit
- `clip-style`: optional `.[clip]` image-style scoring via `style-evaluate`; kept separate from face geometry

On Windows, the InsightFace backend automatically adds torch's bundled CUDA DLL directory
to the process search path when it exists, so ONNXRuntime-GPU can use `CUDAExecutionProvider`
without hand-editing `PATH`.
The DeepFace extra includes `tf-keras` for TensorFlow/Keras 3 compatibility and the
backend switches Windows console streams to UTF-8 before importing DeepFace, avoiding
Unicode logging failures during first-time weight downloads.

Use the OpenCV face-crop backend after installing the optional vision dependencies:

```powershell
python -m pip install -e ".[vision]"
python -m seju_face_lab build --images data/raw --out outputs/seju_model_facecrop --backend opencv-face
python -m seju_face_lab evaluate --model outputs/seju_model_facecrop --images outputs/generated --out outputs/evaluation_facecrop --backend opencv-face
```

Use the DeepFace backend as a neural cross-check after installing its optional dependencies:

```powershell
python -m pip install -e ".[deepface]"
python -m seju_face_lab build --images data/raw --out outputs/seju_model_deepface --backend deepface
python -m seju_face_lab evaluate --model outputs/seju_model_deepface --images outputs/generated --out outputs/evaluation_deepface --backend deepface
python -m seju_face_lab build --images data/raw --out outputs/seju_model_deepface_retinaface --backend deepface-retinaface
python -m seju_face_lab evaluate --model outputs/seju_model_deepface_retinaface --images outputs/generated --out outputs/evaluation_deepface_retinaface --backend deepface-retinaface
```

Compare multiple backend rankings on the same local reference and target image sets:

```powershell
python -m seju_face_lab compare-backends --reference-images data/raw/seju_official --images outputs/generated_detector --out outputs/backend_compare --backends deterministic opencv-face insightface deepface deepface-retinaface
python -m seju_face_lab compare-subject-backends --reference-images data/raw/seju_official --subjects data/subjects --out outputs/subject_backend_compare --backends deterministic opencv-face insightface deepface-retinaface
```

This writes one model/evaluation folder per backend plus `backend_comparison.json` and
`backend_comparison.md`. Score scales are backend-specific; use the rank-agreement section to
review whether deterministic, face-crop, InsightFace, and DeepFace detector variants choose the same candidates.
For celebrity/public-figure folders under `data/subjects`, `compare-subject-backends` writes one
subject review per backend plus `subject_backend_comparison.json` and `.md` so per-subject
near-face rankings can be reviewed across deterministic and neural embeddings.

When DeepFace rejects many reference images, sweep its detector choices directly:

```powershell
python -m seju_face_lab compare-deepface-detectors --reference-images data/raw/seju_official --images outputs/generated_detector --out outputs/deepface_detector_compare --detectors opencv mtcnn retinaface skip
```

This writes `deepface_detector_comparison.json` and `.md`, plus per-detector model/evaluation
folders. Use it to compare detector acceptance counts before interpreting DeepFace rank divergence.
For long sweeps, rerun with `--reuse-existing` so completed detector folders are kept and only
missing detector outputs are computed.
For slow detectors, start with `--max-reference-images 50 --max-images 6`, then rerun without
limits once the detector looks useful.

Record local RTX 4090 and optional SSH remote-GPU readiness before split-run planning:

```powershell
python -m seju_face_lab worker-diagnostics --out outputs/worker_diagnostics
python -m seju_face_lab worker-diagnostics --out outputs/worker_diagnostics_fleet --include-remote
```

Run local explicit-worker evaluation when you want an auditable chunk assignment before
trying remote split runs:

```powershell
python -m seju_face_lab distributed-evaluate --model outputs/seju_model --images outputs/generated --out outputs/distributed_evaluation --backend deterministic
```

This writes merged `scores.csv`, `summary.json`, `distributed_scores.json`, and per-worker
assignment/output files under `.worker_tmp/`. `--include-remote` remains a readiness guard
until a reviewed shared-path or sync manifest is configured.

Implemented generation providers:

- `dry-run`: writes prompt, seed, and evaluation plan without running an image model.
- `diffusers`: runs local image generation through `generate --provider diffusers` with `.[generation]`.

See `docs/architecture.md` for the folder contract and backend plan.
See `docs/research-tracking.md` for the current GitHub Issue / ToDo breakdown.
See `docs/gpu-generation-log.md` for RTX 4090 generation smoke results.

## Output Meaning

- `mean_face.png`: pixel-wise mean of the normalized image crops.
- `median_face.png`: pixel-wise median, often more robust to outliers.
- `profile.json`: compact descriptor values and vector metadata.
- model audit `model_audit.json`: mean/median vector hashes, norms, cosine/euclidean distance, and descriptor deltas.
- vector export `export-vectors`: full mean/median embedding values as JSON or UTF-8 BOM CSV for external generation/scoring workflows.
- `prompt.txt`: a generation prompt based on observed centroid descriptors.
- generation `generation_run.json`: prompt, seed, provider, output paths, and evaluation command/argv.
- generation `prompt_profile`: `balanced` by default, or `detector-friendly` for frontal, unobscured candidate batches.
- evaluation `scores.csv`: similarity of candidate generated images to the centroid vectors.
- evaluation `summary.json`: best/mean/median generated-image similarity for quick comparisons.
- generated review `generation_run_reviews.csv`: one-command generated-image evaluation + QA + run review via `review-generated`, or directly after Diffusers generation with `generate --review`; includes provider, model, prompt profile, seed, count, steps, size, device, and dtype when `generation_run.json` is present.
- precision report `precision_report.json`: model centroid, optional `model_audit.json` mean/median vector distance summary, generation settings, generated-image mean/median score components, QA, subject-review, backend-comparison, and subject-backend-comparison summary via `precision-report`.
- pipeline run `pipeline_run.json`: configured build/audit-model/export-vectors/evaluate/style-evaluate/review/backend-comparison/subject-backend-comparison/precision orchestration via `run-pipeline`.
- generation sweep `configs/pipelines/generation-sweep.example.json`: repeatable seed/profile generation experiments with per-run manifests and optional shared run comparison.
- pipeline config `configs/pipelines/full-retinaface-review.example.json`: deterministic continuity plus `deepface-retinaface` neural rank agreement for the precision bundle.
- backend diagnostics `backend_diagnostics.json`: optional dependency, CUDA, vector backend, and generation-provider visibility.
- worker diagnostics `worker_diagnostics.json`: local/SSH Python, CUDA, torch, and package readiness for GPU split-run planning.
- backend comparison `backend_comparison.json`: per-backend model/evaluation outputs and same-image rank agreement.
- subject backend comparison `subject_backend_comparison.json`: per-backend celebrity/public-figure subject rankings and rank agreement.
- subject review `subject_reviews.html`: local thumbnail cards for per-person approximate similarity review.
- style evaluation `style_scores.csv`: OpenCLIP image-style similarity to mean/median renderings.
- style evaluation `style_summary.json`: best/mean/median style-axis scores.
- image quality `image_quality.csv`: OpenCV single-face QA for generated candidates.
- generation run reviews: rank evaluated candidate batches by local centroid scores.
- generation run review `generation_run_reviews.html`: local thumbnail review cards with face/style/combined/QA evidence.
- generation run reviews include QA-gated face scores, style scores, and per-image combined scores when those outputs exist.
- subject review outputs: per-person approximate similarity rankings.

## Verification

```powershell
python -m unittest discover -s tests
```
