# Research Tracking

Retrieval/design date: 2026-06-14.

## Current State

- Official seju source discovery is implemented through `sources discover`.
- Reviewed eligible image staging is implemented through `sources download`.
- Deterministic centroid build/evaluate is implemented and tested.
- OpenCV face-crop normalization is implemented as optional backend `opencv-face`.
- InsightFace adapter code is dependency-gated and has a sample-verified 512D comparison path when `insightface` and ONNXRuntime providers are available.
- DeepFace adapter code is dependency-gated and uses `DeepFace.represent` for neural cross-checking.
- `backend-diagnostics` records optional dependency, CUDA, and ONNXRuntime provider visibility.
- `benchmark-research` records benchmark/OSS adoption notes for NIST FRTE/FATE, NEC context, InsightFace, DeepFace, OpenCLIP, and Worldcoin Open IRIS; Open IRIS remains separate-modality architecture guidance, not a face-vector backend.
- `compare-backends` builds/evaluates multiple vector backends on the same local image sets and reports rank agreement.
- `compare-deepface-detectors` sweeps DeepFace detector choices on the same local image sets and reports detector acceptance/rank agreement.
- Per-subject similarity review is implemented through `review-subjects`.
- `review-subjects` writes a local HTML thumbnail review so celebrity/public-figure nearest-image evidence can be checked per subject.
- `review-subjects` now summarizes stable mean-score leaders, peak best-image leaders, single-image lift, and mean/median centroid affinity for member-level vector analysis.
- Per-subject backend rank agreement is implemented through `compare-subject-backends`.
- SNS handle/engagement manifests, cache-backed SNS exploration, and face-score correlation reports are implemented.
- OpenCLIP style-axis scoring is implemented through `style-evaluate`.
- `compare-runs` reports style and same-image combined scores when style outputs are present.
- `compare-runs` writes a local HTML thumbnail review so generated candidates can be visually checked beside face/style/QA evidence.
- Image-generation dry-run planning and local RTX 4090 Diffusers smoke runs are implemented.
- OpenAI GPT Image generation is implemented as `generate --provider openai-image` with optional `.[openai]`.
- `generate --prompt-profile detector-friendly` records detector-oriented prompt settings for frontal, unobscured candidate batches.
- `generate --centroid-kind mean|median` records whether the prompt was derived from mean or median centroid descriptors.
- `compare-runs` and `precision-report` now summarize generated candidates by centroid kind so mean-vs-median generation experiments can be reviewed without mixing their winners.
- `qa-images` flags generated candidates that are collages, extreme crops, off-center faces, or missing a frontal OpenCV face.
- `compare-runs` now reads `quality/` outputs and reports QA-gated best centroid scores.
- `review-generated` runs generated-image evaluation, QA, and one-run comparison as the standard precision-review shortcut.
- `generate --review` now chains real Diffusers generation into the same generated-image review shortcut.
- `generate --review` also chains OpenAI Image API generation into the same generated-image review shortcut.
- `precision-report` consolidates a workflow readiness checklist, centroid, optional `model_audit.json` mean/median vector distance metadata, vector export, face-ingredient decomposition, benchmark/OSS research, generated-image mean/median score components, QA, subject-review, backend-comparison, subject-backend-comparison, and optional correlation evidence into a review bundle.
- `audit-model` writes standalone mean/median vector hashes, norms, distance metrics, and descriptor deltas for centroid-model review.
- `ingredients-report` decomposes mean/median descriptors into face-part, color-tone, makeup-texture, hair-signal, and generation-guidance notes.
- `review-agencies` creates official-source agency average face parameters, 8-axis vectors, Image Gen prompts, and seju descriptor-similarity rankings from `configs/agencies/seju_like_agencies.json`.
- `face-axes` maps local or generated images to 4+4 axes, quadrant/corner/cross labels, outlier scores, and presentation-state flags without applying insulting person labels.
- `enhance-agencies` fuses agency descriptor hypotheses, generated-image centroid scores, and observed 8-axis vectors into an enhancement score, confidence label, and prompt-improvement actions.
- `calibrate-agency-generation` converts measured enhancement gaps into per-agency calibrated prompts, negative prompts, seed plans, and priority labels for the next generation round.
- `export-vectors` writes full mean/median centroid embedding values as JSON or CSV for external analysis and generation tooling.
- `run-pipeline` executes JSON-configured build/audit-model/export-vectors/ingredients-report/benchmark-research/evaluate/style-evaluate/review/backend-comparison/subject-backend-comparison/SNS-engagement/correlation/precision steps and records a pipeline run manifest.
- `generation_sweep` pipeline configs execute repeatable multi-seed/multi-profile generation experiments with per-run manifests and optional shared generated-run comparison.
- `configs/pipelines/generation-openai-image.example.json` captures the GPT Image API generation + generated-review + precision-report workflow.
- `configs/pipelines/sns-correlation.example.json` captures the repeatable subject-review + cached SNS exploration + face-score/SNS correlation + precision-report workflow.
- `configs/pipelines/full-retinaface-review.example.json` captures the current full review path with deterministic continuity plus `deepface-retinaface` backend rank agreement.
- `worker-diagnostics` records local RTX 4090 and optional SSH remote-GPU Python/CUDA/torch/package readiness without remote writes.
- `distributed-evaluate` runs explicit local worker-chunk scoring and records merged score outputs plus worker assignment artifacts before any remote split-run.
- `audit_research_data_quality.py` checks real-vs-hypothesis evidence type, subject/image counts, image imbalance, duplicates, small images, aspect outliers, agency generated-score coverage, and quadrant separability before strong claims.
- `docs/strengthening-next-plan.md` records the next strengthening engine execution order, output paths, promotion rules, and verification commands.

## GitHub Issue Plan

Create and track these issues:

- `P1 GPU face embeddings`: expand multi-candidate InsightFace/ONNXRuntime-GPU comparisons beyond the current 6-image batch and investigate DeepFace ranking divergence.
- `P1 Benchmark/OSS vectorization review`: keep `benchmark-research` current and map OSS benchmark lessons into local backend comparisons.
- `P1 Celebrity subject review workflow`: collect reviewed subject folders and run `review-subjects` plus `compare-subject-backends`.
- `P1 Generation loop`: expand reviewed Diffusers batches and compare generated candidates across face, style, QA, and backend axes.
- `P1 SNS correlation workflow`: run handle extraction, engagement manifesting, and correlation reports.
- `P2 DeepFace adapter`: run the committed detector sweep after the default DeepFace/OpenCV detector accepted only `139/259` official references.
- `P2 CLIP style axis`: verify optional OpenCLIP install and use `style-evaluate` alongside face geometry scores.
- `P2 Remote worker playbook`: document RTX 4090 local worker chunking and SSH remote-GPU split-run commands.
- `P0 Data quality gates`: wire `scripts/audit_research_data_quality.py` into repeatable pipelines and public page badges.
- `P0 Real agency data collection`: collect per-talent non-seju agency image sets with source manifests before calling them real agency averages.
- `P0 Subject-balanced centroid`: build a centroid mode that averages per subject before computing the global centroid.
- `P0 Strengthening next plan`: execute `docs/strengthening-next-plan.md` from P0 evidence gates through P2 visualization.

## Local ToDo

- Keep `data/raw/`, `data/subjects/`, `data/processed/`, and `outputs/` out of Git.
- Record source URL, retrieval date, and permission notes for any comparison subject set.
- Prefer one subject per folder and multiple images per subject.
- Review `subject_reviews.md` before making any interpretation.
- Treat all similarity scores as local model measurements, not identity or objective labels.

## Next Experiment

1. Build the official seju centroid from reviewed local images.
2. Place comparison celebrity/public-figure image folders under `data/subjects/`.
3. Run `review-subjects` and `compare-subject-backends` with deterministic, `opencv-face`, and neural backends.
4. Run the same review with InsightFace or DeepFace once optional dependencies are installed.
5. Plan aggregate candidate faces with `generate --provider dry-run`.
6. For detector-visible scoring batches, use `generate --prompt-profile detector-friendly`.
7. Generate with Diffusers/ComfyUI on a GPU worker, or `openai-image` through the OpenAI Images API, and score with `generate --review` or `review-generated`.
8. For seed/profile/mean-vs-median iteration, run `configs/pipelines/generation-sweep.example.json` so all candidate settings have per-run manifests and a shared run comparison.
9. Run `style-evaluate` so generated candidates have both face-geometry and style-axis scores.
10. Run `qa-images` or `review-generated` before visual review so collages/extreme crops do not win on score alone.
11. Rank evaluated generated batches with `compare-runs`, including QA-gated and combined face/style scores when available.
12. Write an `ingredients-report` and `precision-report` for the model, model audit, vector export, best generation review, QA, subject-review, backend-comparison, and subject-backend-comparison outputs.
13. Use `run-pipeline` for repeatable local build/audit-model/export-vectors/evaluate/style-evaluate/review/backend-agreement/precision runs from config.
14. Run SNS handle/engagement manifests and `analyze correlation`, or `configs/pipelines/sns-correlation.example.json`, for reviewable metric joins.
15. Run `backend-diagnostics` and `worker-diagnostics --include-remote` on RTX nodes and archive the ignored output paths in the Issue comment.
16. Run `benchmark-research` to refresh benchmark/OSS adoption notes before adding or changing vector backends.
17. Run `distributed-evaluate` locally before remote split-runs so chunk assignment and merged score outputs are reviewable.
18. Compare deterministic scores against InsightFace/DeepFace on the same ignored image sets with `compare-backends`.
19. When DeepFace diverges, run `compare-deepface-detectors` with `opencv mtcnn retinaface skip` and compare acceptance counts before reviewing ranks.

## GPU Generation Notes

- RTX 4090 smoke generation succeeded with `.venv` Python 3.12.13, torch 2.12.0+cu126, and Diffusers 0.38.0.
- Small generated batches were evaluated locally; generated images and per-run scores remain ignored under `outputs/`.
- OpenCV face-crop build succeeded on the local official image set with 173 usable face crops from 259 source images.
- Detector-friendly RTX 4090 v2 produced one QA-passing candidate out of two; v3/v4 showed why QA is needed by producing extreme crops, off-center faces, and collages.
- The current committed route is detector-friendly generation, then `review-generated` or deterministic/OpenCV evaluation + `qa-images` + `compare-runs` with QA-gated ranking before any visual interpretation.
- InsightFace sample `compare-backends` succeeded on `data/raw/seju_official_sample` against `outputs/generated_detector_v5`: 2 usable reference images, 1 generated image, 512D embeddings, best generated image `candidate_0001_seed_260623`, best InsightFace centroid score `0.052322`.
- Full-set InsightFace `compare-backends` succeeded against `outputs/generated_detector_v5` after Windows torch CUDA DLL path preparation: deterministic refs `259/259`, InsightFace refs `224/259`, generated images `1/1`, InsightFace embedding dimension `512`, best generated image `candidate_0001_seed_260623`, best InsightFace centroid score `0.038486`.
- Current ONNXRuntime reports `TensorrtExecutionProvider`, `CUDAExecutionProvider`, and `CPUExecutionProvider`; the full-set comparison log confirms applied providers `CUDAExecutionProvider, CPUExecutionProvider`.
- v7 RTX 4090 detector-friendly batch generated 6 candidates and reviewed them in one pass: deterministic evaluated `6/6`, QA pass `3/6`, best QA image `candidate_0003_seed_260702`, best deterministic score `0.408002`.
- v7 full-set InsightFace comparison used ONNXRuntime `CUDAExecutionProvider`: deterministic refs `259/259`, InsightFace refs `224/259`, InsightFace generated images `5/6`, failed generated image `candidate_0001_seed_260700`, best image matched deterministic at `candidate_0003_seed_260702`, best InsightFace score `0.105692`, deterministic-vs-InsightFace Spearman rank `0.300000` over 5 common images.
- DeepFace runtime is verified locally after adding `tf-keras` to the optional extra and preparing UTF-8 console output on Windows; sample comparison produced 512D embeddings for 2 reference images and 1 generated image.
- v7 all-backend comparison completed: deterministic refs `259/259`, OpenCV refs `173/259`, InsightFace refs `224/259`, DeepFace refs `139/259`; deterministic/OpenCV/InsightFace all picked `candidate_0003_seed_260702`, while DeepFace picked `candidate_0005_seed_260704`.
- v7 pairwise rank agreement over 5 common generated images: deterministic-vs-OpenCV `0.900000`, deterministic-vs-InsightFace `0.300000`, InsightFace-vs-OpenCV `0.400000`, DeepFace-vs-deterministic `-0.700000`, DeepFace-vs-InsightFace `-0.200000`, DeepFace-vs-OpenCV `-0.400000`.
- `compare-backends` is now the committed review path for checking whether deterministic/OpenCV and neural embeddings rank the same generated or subject images.
- `compare-deepface-detectors` is now the committed audit path for checking whether DeepFace's detector choice, rather than ArcFace scoring alone, explains low reference acceptance or rank divergence.
- v7 `compare-deepface-detectors` confirmed `deepface-opencv` acceptance/ranking in a reusable detector report: refs `139/259`, generated images `5/6`, best image `candidate_0005_seed_260704`, best score `0.567757`. The four-detector local sweep hit a 20-minute turn timeout after OpenCV, so resume the remaining detectors with `--reuse-existing`.
- v7 `deepface-skip` accepted refs `259/259` and generated images `6/6`, best image `candidate_0006_seed_260705`, best score `0.986008`, and Spearman rank vs `deepface-opencv` was `0.600000` over 5 common images. Treat `skip` as a detector-rejection pressure check only, not as a replacement for face-validated scoring.
- A 25-reference/6-generated `mtcnn` smoke audit is now possible through `--max-reference-images` / `--max-images`: same first-25 refs gave OpenCV `18/25`, MTCNN `22/25`, both generated `5/6`, bests diverged, and Spearman was `0.100000`.
- The same first-25 smoke audit with `retinaface` matched MTCNN on acceptance (`22/25`) and generated best image (`candidate_0006_seed_260705`); MTCNN-vs-RetinaFace Spearman was `1.000000`, while OpenCV-vs-neural detector Spearman stayed `0.100000`.
- Full v7 `retinaface` audit accepted refs `221/259` and generated images `5/6`, best image `candidate_0005_seed_260704`, best score `0.574721`; it improved on DeepFace/OpenCV's `139/259` refs while keeping Spearman `0.800000` vs OpenCV over 5 common generated images.
- `deepface-retinaface` is now registered as a normal backend, so backend comparison and model/evaluation runs can use the audited DeepFace RetinaFace path directly.
- Registered-backend smoke check completed on `data/raw/seju_official_sample` vs `outputs/generated_detector_v5`: `deepface-retinaface` refs `2/2`, generated images `1/1`, best image `candidate_0001_seed_260623`, best score `0.293422`.
- `precision-report` vector audit smoke completed on `outputs/seju_model_official`: model images `259`, mean/median embedding shapes `[1073]`, both L2 norms `1.0`, mean SHA-256 prefix `bc562e5703aa`, median SHA-256 prefix `0b9b938d2460`.
- `precision-report` generated-score component smoke completed after regenerating `outputs/evaluation_v5_score_components_smoke`: best image `candidate_0001_seed_260623`, centroid score `0.168881`, cosine-to-mean `0.195562`, cosine-to-median `0.1422`, euclidean-to-mean `1.268415`, euclidean-to-median `1.309809`.
- Complete official precision bundle smoke completed at `outputs/precision_report_official_complete`: readiness `8/8`, optional readiness `5/5`, model images `259`, embedding dim `1073`, mean/median embedding cosine `0.733644`, generated best score `0.408002`, top subject `kasumi-mori` with mean score `0.464969` and best score `0.594015`, subject backends `deterministic, opencv-face`, generation backends `deterministic, opencv-face, insightface, deepface`; World/Open IRIS remains logged as out-of-scope for face-vector scoring.
- Agency Image Gen smoke completed with five fictional aggregate samples in `outputs/agency_imagegen_samples`: seju centroid scores were `platinum 0.346811`, `lespros 0.214772`, `trustar 0.152418`, `seju 0.073247`, `asia-promotion -0.270030`; 8-axis summary quadrant was `defined_bright`, and low-presentation cases are recorded as image-state flags such as `dark_or_underlit_image`, not as person labels.
- Agency enhancement smoke completed at `outputs/agency_enhancement`: fused ranking was `platinum 0.754879`, `lespros 0.717206`, `trustar 0.694741`, `seju 0.681432`, `asia-promotion 0.581166`; top improvement actions were prompt-axis adjustment, second-seed/regeneration, even front lighting, centered frontal crop, and reduced hair shadow/edge noise.
- Agency generation calibration completed at `outputs/agency_generation_calibration`: target image score `0.35`, target axis alignment `0.62`, target enhancement score `0.76`; priority counts were `baseline_control 3`, `regenerate 2`, with `seju` and `asia-promotion` selected for first regeneration because their image-score gaps were `0.276753` and `0.620030`.
- Data quality audit v1 completed at `outputs/data_quality_audit_v1`: risk level `needs_real_data_before_strong_claims`; issues were low seju subject count for robust generalization, subject image imbalance ratio `3.6`, 34 small images, 34 aspect outliers, 7 non-seju agencies still hypothesis/generated-only, and low quadrant separability.
- Full committed workflow notes are in `docs/gpu-generation-log.md`.
