# Loop State — Seju Face Lab

Last run: 2026-08-25 (Asia/Tokyo)

## Current Decision

- Vector Model v1: `promoted`
- Selected scoring algorithm: `B1` (robust global center)
- `C1` tangent component model: evaluated but not selected because its improvement confidence interval crossed zero and bootstrap stability was below the predefined gate.
- Fictional-candidate generation: not started. The planner and blinded-review interfaces exist, but generated faces must not be produced or published until a separate candidate-generation run satisfies QA, support, diversity, and human-review gates.

## Verified

- Clean source audit: 259 eligible downloads, 34 signature assets rejected, 225 face candidates.
- CUDA InsightFace observations: 224 accepted, 1 no-face rejection, 34 subjects; detector coverage 0.9956.
- Subject-disjoint LOSO: 34 folds; score median 49.95; worst decile 9.77.
- Six perturbation families: all gates passed; 224 clean observations; runtime 283.36 seconds.
- Repository suite: 157 tests, exit 0 in the repository virtual environment.
- Japanese specification: 8 Mermaid diagrams extracted and rendered.

## Boundaries / Next Gate

- The 96.97 score smoke is a reference-image execution check, not an independent accuracy result.
- No raw images, embeddings, generated portraits, or individual ranking artifacts are tracked by Git.
- Candidate generation and browser/public deployment remain separate, operator-gated delivery work.

## Fictional Candidate Run (2026-08-25)

- Generated five text-only, adult fictional candidates; no real-person image or name was used as input.
- OpenCV QA: 5/5 single centered faces passed.
- CUDA B1 scoring: one candidate entered the predeclared 70–90 band (84.89); four were rejected as outside the band.
- The surviving candidate is inside component support. A one-candidate frontier is a technical preview only; it cannot support a preference or popularity conclusion.
- Public-safe preview: `https://seju-face-lab-fictional-candidates.pages.dev/`; the first image was withdrawn for presentation-quality revalidation. The page now shows only the synthetic/revalidation boundary and exposes no coordinates, raw score, prompts, seeds, or source-person data.
- Strengthening-loop design: `docs/seju-vector-strengthening-loop-ja.md`.

---
Run log: `docs/validation/seju-vector-model-v1-2026-08-25.md`
