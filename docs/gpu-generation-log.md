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
- Confirmed generated images, manifests, and scores stay ignored under `outputs/`.
- Confirmed the shortened default prompt avoids CLIP prompt truncation in the smoke run.

## Prompt Policy

- Keep the default prompt descriptor-derived and dataset-neutral.
- Keep the prompt short enough for Stable Diffusion v1 CLIP limits.
- State aggregate/new-fictional-person/no-celebrity-likeness early.
- Put quality failures such as hair-obscured face, illustration, doll/mannequin, and copied identity in the negative prompt.
- Use `generate --prompt` for study-specific target styling instead of hard-coding traits in `prompt_from_descriptors`.

## Next Steps

- Run larger ignored GPU batches and keep only summarized findings in committed docs.
- Add InsightFace or DeepFace scoring before treating deterministic scores as face-geometry quality.
- Compare deterministic, neural face-embedding, and visual-review rankings before closing the generation-loop issue.
