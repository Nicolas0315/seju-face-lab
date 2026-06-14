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
   - `prompt.txt` and `generation_manifest.json` feed `generate`
   - `generate --provider dry-run` writes a reproducible run plan
   - `generate --provider diffusers` runs an optional local Diffusers batch
   - `--dry-run` always records `provider: dry-run`, even if the requested provider was diffusers
   - `--variant auto` maps `--dtype float16` to the Diffusers `fp16` variant
   - `--prompt-profile detector-friendly` steers aggregate prompts toward frontal, unobscured faces for detector/evaluation passes
   - `--review` runs the standard generated-image review after a real generation run produces files
5. `evaluate`
   - generated candidates are scored against mean and median vectors
6. `style-evaluate`
   - generated candidates are scored against mean/median rendered appearances with OpenCLIP image embeddings
   - this is a style/photographic axis, not a face-geometry score
7. `qa-images`
   - generated candidates are checked for exactly one centered frontal OpenCV face
   - this catches collages, extreme crops, off-center faces, and no-face detector failures before review
8. `compare-runs`
   - generation batches are ranked by QA-gated face score when quality outputs are present
   - otherwise batches are ranked by face score, or by best per-image combined face/style score when style outputs are present
9. `review-generated`
   - convenience command that runs `evaluate`, `qa-images`, and one-run `compare-runs` for a generated directory
   - also used by `generate --review` so generated batches can be scored immediately
10. `precision-report`
   - consolidates centroid metadata, generated-image review, QA, subject-review, and backend-comparison summaries for tracking
11. `run-pipeline`
   - executes configured build, generation, evaluation, review, backend comparison, and precision-report steps from JSON
   - writes `pipeline_run.json` and `pipeline_run.md` as the orchestration trace
12. `review-subjects`
   - per-person image folders are ranked against the local seju centroid
   - output is CSV, Markdown, and JSON for review and tracking
13. SNS and correlation analysis
   - `sources scrape-handles` writes reviewed SNS handle manifests
   - `sources fetch-engagement` writes best-effort public engagement manifests
   - `analyze correlation` joins `subject_reviews.json` to SNS metrics

## Backends

- `deterministic`: implemented. Uses local image statistics and needs only `numpy` + `Pillow`.
- `opencv-face`: implemented optional `vision` extra. Uses OpenCV Haar face boxes before deterministic vectors.
- `insightface`: implemented dependency-gated adapter with `insightface` + `onnxruntime-gpu`; listed as planned when optional dependencies are absent.
- `deepface`: implemented dependency-gated adapter with `DeepFace.represent`; defaults to ArcFace and reports no-face images as vectorization failures.
- `clip-style`: implemented as `style-evaluate`. Uses optional `open-clip-torch` image embeddings as a secondary style axis.
- `diffusion-generation`: planned. Diffusers/ComfyUI generation loop for prompt batches.

Keep geometry and style axes separate. A generated image can match the style prompt while missing face geometry, so evaluation should report neural face-embedding scores and style scores separately.

Backend diagnostics and comparison:

```powershell
python -m seju_face_lab backend-diagnostics --out outputs/backend_diagnostics
python -m seju_face_lab compare-backends --reference-images data/raw/seju_official --images outputs/generated_detector --out outputs/backend_compare --backends deterministic opencv-face insightface deepface
python -m seju_face_lab compare-deepface-detectors --reference-images data/raw/seju_official --images outputs/generated_detector --out outputs/deepface_detector_compare --detectors opencv mtcnn retinaface skip
```

`backend-diagnostics` records dependency/provider visibility. `compare-backends` builds a separate
centroid per backend and compares same-image rankings, avoiding cross-backend embedding averaging.
`compare-deepface-detectors` keeps the DeepFace model fixed and sweeps detector backends so
reference acceptance counts and generated-image ranks can be audited before trusting a DeepFace run.
Use `--reuse-existing` when a long detector sweep has completed some per-detector folders already.
Use `--max-reference-images` and `--max-images` for slow detector smoke audits before committing
to a full reference-set run.
Pass the comparison output into `precision-report --backend-comparison` or set
`backend_comparison.out` in a pipeline config so final precision bundles include backend agreement.

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
- Use Diffusers or ComfyUI to generate candidates from `generation_manifest.json`, then score them with `evaluate` and `style-evaluate`.
- Run `qa-images` before trusting generated-image scores; a collage can score well if one crop matches the centroid.
- Keep generated-image prompts aggregate-only; avoid copying a specific real person.
- `seju_face_lab.workers` contains local/SSH worker helpers; treat remote writes as explicit ops steps, not default CLI behavior.

Dry-run planning:

```powershell
python -m seju_face_lab generate --model outputs/seju_model --out outputs/generated --provider dry-run --count 8
```

Diffusers execution:

```powershell
python -m seju_face_lab generate --model outputs/seju_model --out outputs/generated --provider diffusers --hf-model runwayml/stable-diffusion-v1-5 --count 8 --device cuda --negative-prompt "copied identity"
```

Detector-friendly generation pass:

```powershell
python -m seju_face_lab generate --model outputs/seju_model --out outputs/generated_detector --provider diffusers --prompt-profile detector-friendly --count 8 --device cuda
```

Generated-image QA:

```powershell
python -m seju_face_lab qa-images --images outputs/generated_detector --out outputs/generated_detector/quality
```

One-command generated-image review:

```powershell
python -m seju_face_lab review-generated --model outputs/seju_model --images outputs/generated_detector --out outputs/generated_detector/review
```

Style-axis evaluation:

```powershell
python -m seju_face_lab style-evaluate --model outputs/seju_model --images outputs/generated --out outputs/style_evaluation
```

Combined run review:

```powershell
python -m seju_face_lab compare-runs --runs outputs/generated_a outputs/generated_b --out outputs/run_reviews
```

If `outputs/generated_a/quality/image_quality.csv` exists, the run review includes
`qa_pass_count`, `qa_pass_rate`, and `best_qa_centroid_score`.

## Folder Contract

- `configs/`: reproducible source and pipeline configuration.
- `data/raw/`: reviewed local reference images, ignored.
- `data/processed/`: manifests and intermediate derived data, ignored.
- `data/processed/sns_*.jsonl`: SNS handles/engagement manifests, ignored.
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
