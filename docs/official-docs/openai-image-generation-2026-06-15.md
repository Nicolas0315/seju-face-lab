# OpenAI Image Generation API Evidence

- retrieval_date: 2026-06-15
- source: https://developers.openai.com/api/docs/guides/image-generation
- source: https://developers.openai.com/api/reference/resources/images/methods/generate/
- local_version: optional `openai` SDK is dependency-gated as `seju-face-lab[openai]`
- decision: add `generate --provider openai-image` as an optional image-generation provider.
- API surface used: `OpenAI().images.generate(model=..., prompt=..., n=..., size=..., quality=..., output_format=...)`
- output handling: write `b64_json` image data when present; fall back to returned URL only if a model/provider supplies one.
- verification: mocked SDK response writes a local generated image and `generation_run.json`.
- risk: OpenAI model access may require organization verification and `OPENAI_API_KEY`.
- rollback: remove provider dispatch in `src/seju_face_lab/cli.py`, `run_openai_image_generation` in `src/seju_face_lab/generation.py`, and the `openai` extra in `pyproject.toml`.
- next_refresh_date: 2026-09-15
