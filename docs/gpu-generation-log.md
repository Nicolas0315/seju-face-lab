# GPU Generation Log

Run date: 2026-06-15.

## Verified Environment

- host: local RTX 4090 workstation
- Python env: ignored `.venv`, CPython 3.12.13
- torch: 2.12.0+cu126
- diffusers: 0.38.0
- generator family: Diffusers text-to-image pipeline
- source model: ignored `outputs/seju_model_official`

## Verified Workflow

- Built the official seju centroid model from the ignored local source image set.
- Generated small GPU batches from `generation_manifest.json`.
- Evaluated generated candidates with `seju_face_lab evaluate`.
- Compared evaluated local batches with `seju_face_lab compare-runs`.
- Built and evaluated an optional OpenCV face-crop centroid model as a detector-visible face QA gate.
- Added and verified `qa-images` to reject generated collages, extreme crops, and off-center faces before visual review.
- Added and sample-verified the dependency-gated InsightFace backend path for 512D neural embedding comparison.
- Confirmed generated images, manifests, and scores stay ignored under `outputs/`.
- Confirmed the shortened default prompt avoids CLIP prompt truncation in the smoke run.
- The detector-friendly profile initially emitted a CLIP token-length warning in Diffusers v1.5 runs; v5 compacted the detector-friendly negative prompt and removed that warning in a one-image RTX 4090 run.

## 2026-06-15 Detector-Friendly RTX 4090 Pass

Commands used:

```powershell
python -m seju_face_lab generate --model outputs/seju_model_official --out outputs/generated_detector_v2 --provider diffusers --count 2 --seed 260617 --steps 20 --prompt-profile detector-friendly --device cuda --dtype float16
python -m seju_face_lab evaluate --model outputs/seju_model_official --images outputs/generated_detector_v2 --out outputs/generated_detector_v2/evaluation
python -m seju_face_lab evaluate --model outputs/seju_model_opencv_official --backend opencv-face --images outputs/generated_detector_v2 --out outputs/generated_detector_v2/evaluation_opencv
python -m seju_face_lab qa-images --images outputs/generated_detector_v2 --out outputs/generated_detector_v2/quality
```

Observed local results:

- v1 detector-friendly prompt: deterministic `image_count=2`, OpenCV `image_count=0`, `failed_count=2`; images were too cropped for OpenCV.
- v2 detector-friendly prompt: deterministic best `candidate_0002_seed_260618`, best score `0.363268`; OpenCV best `candidate_0002_seed_260618`, best score `0.641401`; QA pass `1/2`.
- v2 QA rejected `candidate_0001_seed_260617` because OpenCV detected 4 faces, matching the visible grid/collage failure.
- v3 shorter ID-headshot prompt: deterministic `image_count=2`, OpenCV `image_count=2`, QA pass `0/2`; failures were extreme crop and off-center/small face.
- v4 v2-style prompt plus stronger collage negatives: deterministic `image_count=2`, OpenCV `image_count=1`, QA pass `0/2`; failures were 4-face collage and no detected frontal face.
- v5 compact detector-friendly negative prompt: no CLIP token-length warning, deterministic best `0.168881`, OpenCV best `0.487359`, QA pass `1/1`.
- v6 `generate --review` smoke run completed generation, evaluation, QA, and one-run review in one command; deterministic best `0.044308`, QA pass `0/1` because OpenCV detected no face.
- QA-gated `compare-runs` over v2-v5 still ranks v2 first: deterministic QA best `0.363268`; OpenCV QA best `0.641401`.
- `review-generated` on v5 reproduced the standard generated-image review in one command: deterministic `image_count=1`, QA pass `1/1`, one-run review best QA score `0.168881`.
- `compare-backends` on the v5 sample completed both `deterministic` and `insightface`: 2 reference images, 1 generated image, InsightFace embedding dimension `512`, best generated image `candidate_0001_seed_260623`, best InsightFace centroid score `0.052322`.
- After adding automatic torch CUDA DLL path preparation on Windows, full-set `compare-backends` completed with ONNXRuntime `CUDAExecutionProvider`: deterministic refs `259/259`, InsightFace refs `224/259`, generated images `1/1`, InsightFace embedding dimension `512`, best generated image `candidate_0001_seed_260623`, best InsightFace centroid score `0.038486`.
- `precision-report` bundled the full-set CUDA backend comparison with the v5 generated review: model images `259`, best generated deterministic score `0.168881`, completed backends `deterministic, insightface`.
- v7 rank batch generated 6 detector-friendly candidates on RTX 4090; deterministic review evaluated `6/6`, QA pass `3/6`, best QA image `candidate_0003_seed_260702`, best deterministic score `0.408002`.
- v7 full-set backend comparison completed with ONNXRuntime `CUDAExecutionProvider`: deterministic refs `259/259`, InsightFace refs `224/259`, InsightFace generated images `5/6`, failed image `candidate_0001_seed_260700`, best image matched deterministic at `candidate_0003_seed_260702`, best InsightFace score `0.105692`, deterministic-vs-InsightFace Spearman rank `0.300000` over 5 common images.
- `precision-report` bundled the v7 generated review and backend comparison: model images `259`, best generated score `0.408002`, QA pass `3/6`, completed backends `deterministic, insightface`, rank agreement common images `5`.
- DeepFace runtime was verified after adding `tf-keras` to the optional extra and preparing UTF-8 console output on Windows. Sample DeepFace comparison completed with 2 reference images, 1 generated image, 512D embeddings, best score `0.017218`.
- v7 all-backend comparison completed for `deterministic`, `opencv-face`, `insightface`, and `deepface`: deterministic/opencv/InsightFace all chose `candidate_0003_seed_260702`; DeepFace chose `candidate_0005_seed_260704`; pairwise Spearman over 5 common images was deterministic-vs-OpenCV `0.900000`, deterministic-vs-InsightFace `0.300000`, InsightFace-vs-OpenCV `0.400000`, DeepFace-vs-deterministic `-0.700000`, DeepFace-vs-InsightFace `-0.200000`, and DeepFace-vs-OpenCV `-0.400000`.
- Added `compare-deepface-detectors` so the next DeepFace audit can hold ArcFace constant and sweep `opencv`, `mtcnn`, `retinaface`, and `skip` detector backends on the same reference/generated image sets.
- First v7 detector audit confirmed `deepface-opencv` exactly reproduces the prior DeepFace acceptance/ranking: official refs `139/259`, generated images `5/6`, failed generated image `candidate_0001_seed_260700`, best image `candidate_0005_seed_260704`, best score `0.567757`. The full four-detector sweep exceeded a 20-minute local turn timeout after finishing OpenCV, so the committed `--reuse-existing` flag should be used to resume the remaining detector runs without recomputing OpenCV.
- Resumed the v7 detector audit with `skip`: `deepface-skip` accepted official refs `259/259` and generated images `6/6`, best image `candidate_0006_seed_260705`, best score `0.986008`, and Spearman rank vs `deepface-opencv` was `0.600000` over 5 common generated images. Boundary: `skip` bypasses face detection, so it is useful for testing detector rejection pressure but should not replace detector-validated face scoring.

Current best local generated candidate by deterministic QA-gated score is:

```text
outputs/generated_detector_v7_rank/candidate_0003_seed_260702.png
```

Boundary: these are approximate local scores against ignored local centroid models, not identity, attractiveness, ethnicity, or objective face-type labels.

## Prompt Policy

- Keep the default prompt descriptor-derived and dataset-neutral.
- Keep the prompt short enough for Stable Diffusion v1 CLIP limits.
- State aggregate/new-fictional-person/no-celebrity-likeness early.
- Put quality failures such as hair-obscured face, illustration, doll/mannequin, and copied identity in the negative prompt.
- Use `generate --prompt` for study-specific target styling instead of hard-coding traits in `prompt_from_descriptors`.
- Use `generate --prompt-profile detector-friendly` for the next face-detector-visible scoring batch.

```powershell
python -m seju_face_lab generate --model outputs/seju_model_official --out outputs/generated_detector --provider diffusers --prompt-profile detector-friendly --count 8 --device cuda --review
```

## Next Steps

- Run larger ignored GPU batches and keep only summarized findings in committed docs.
- Run larger detector-friendly batches beyond 6 images to reduce the noise in backend rank agreement.
- Resume `compare-deepface-detectors --reuse-existing` for `mtcnn retinaface` to investigate whether alternative face detectors preserve face-validated scoring while improving on DeepFace/OpenCV `139/259` acceptance.
- Treat ONNXRuntime CUDA provider visibility as environment evidence only; record backend vectorization results separately for each image set.
- Compare deterministic, neural face-embedding, and visual-review rankings before closing the generation-loop issue.
