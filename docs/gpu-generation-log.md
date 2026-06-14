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

Current best local candidate for review remains:

```text
outputs/generated_detector_v2/candidate_0002_seed_260618.png
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
- Run larger detector-friendly batches, then score with `review-generated` plus OpenCV/InsightFace or DeepFace where available.
- Run larger generated batches so backend rank agreement has more than one common generated image.
- Verify DeepFace installation/runtime and compare its rankings with deterministic/OpenCV/InsightFace.
- Treat ONNXRuntime CUDA provider visibility as environment evidence only; record backend vectorization results separately for each image set.
- Compare deterministic, neural face-embedding, and visual-review rankings before closing the generation-loop issue.
