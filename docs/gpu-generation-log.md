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
- Added dependency-gated InsightFace backend code for later RTX neural embedding verification.
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
- QA-gated `compare-runs` over v2-v5 still ranks v2 first: deterministic QA best `0.363268`; OpenCV QA best `0.641401`.

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
python -m seju_face_lab generate --model outputs/seju_model_official --out outputs/generated_detector --provider diffusers --prompt-profile detector-friendly --count 8 --device cuda
```

## Next Steps

- Run larger ignored GPU batches and keep only summarized findings in committed docs.
- Run larger detector-friendly batches, then score with deterministic/OpenCV/InsightFace or DeepFace where available and filter with `qa-images`.
- Fix ONNXRuntime CUDA DLL availability for InsightFace; current sample build/evaluate succeeds through CPU fallback.
- Add full-set InsightFace or DeepFace scoring before treating deterministic scores as face-geometry quality.
- Compare deterministic, neural face-embedding, and visual-review rankings before closing the generation-loop issue.
