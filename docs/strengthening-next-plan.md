# strengthening next plan

Plan date: 2026-06-15.

This plan turns the current data-quality review into the next strengthening engine workstream.
The rule is simple: evidence quality gates run before public scores or generated-image conclusions.

## Current Gate Result

- latest audit command: `python scripts/audit_research_data_quality.py --out outputs/data_quality_audit_v1`
- latest audit report: `outputs/data_quality_audit_v1/data_quality_audit.md`
- current risk level: `needs_real_data_before_strong_claims`
- current blocker: 7 non-seju agencies are still hypothesis/generated-only.
- safe current claim: seju has a local real-image centroid; cross-agency views are experimental maps.
- unsafe current claim: non-seju office rows are real agency average faces.

## Execution Order

### P0. Evidence Gates

Goal: make non-essential or weak data visible before any score is interpreted.

Items:

1. `data-quality-audit` pipeline gate
   - source: `scripts/audit_research_data_quality.py`
   - output: `outputs/data_quality_audit_v*/`
   - page badge: show `real_centroid_baseline`, `hypothesis_and_generated`, or `real_and_generated`
   - done when: public page and `data.json` expose evidence type per agency.
   - status 2026-06-15: implemented in `scripts/build_agency_site.py` via optional `--data-quality`.

2. Source manifest
   - target file shape: `source_manifest.jsonl`
   - required fields: `image_path`, `sha256`, `source_url`, `retrieved_at`, `source_type`, `permission_note`, `width`, `height`, `subject_slug`, `agency_slug`
   - done when: every new real image batch can be audited without guessing where images came from.

3. Real agency data collection
   - target root: `data/raw/agencies/<agency>/<subject>/`
   - first agencies: `platinum`, `trustar`, `asia-promotion`, `lespros`, `lvs`, `sgmedia`, `twin-planet`
   - done when: each agency has a manifest and enough per-subject images to build a real centroid.

4. Subject-balanced centroid
   - target behavior: average per subject first, then average subjects.
   - reason: current seju subject image imbalance is 3.6, so image-heavy subjects can overweight the centroid.
   - done when: model build can emit `centroid_mode=image_weighted|subject_balanced`.
   - status 2026-06-15: implemented as `python -m seju_face_lab build --balance subject`.

Verification:

```powershell
python scripts/audit_research_data_quality.py --out outputs/data_quality_audit_v_next
python -m seju_face_lab review-subjects --model outputs/seju_model_official --subjects data/raw/seju_by_talent --out outputs/seju_subject_reviews_next
ruff check .
python -m unittest discover -s tests
python -m compileall -q src tests scripts
git diff --check
```

### P1. Scoring Robustness

Goal: stop single-backend or single-seed artifacts from looking like stable signal.

Items:

1. Multi-backend promotion gate
   - required backends: `deterministic`, `opencv-face`
   - preferred neural checks: `insightface`, `deepface-retinaface`
   - done when: a score promotion includes backend acceptance counts and rank agreement.

2. Detector acceptance report
   - fields: backend, accepted count, failed count, face box where available, failure reason.
   - done when: `review-subjects`, `evaluate`, and agency reviews can show detector coverage.

3. Agency generation sweep
   - target: N seeds per agency, QA-gated, with selected-candidate manifest.
   - output: `outputs/agency_generation_sweeps/<version>/`
   - done when: agency page uses the best QA-passing candidate per agency, not a one-off image.

4. Style axis separation
   - backend: OpenCLIP via `style-evaluate`
   - rule: style score stays separate from face-vector score.
   - done when: page and reports show face, style, QA, and combined candidate evidence separately.

5. SOTA benchmark adoption map
   - source: `docs/sota-benchmark-research.md`
   - implemented now: IJB-C-style subject-balanced templates and page evidence badges.
   - next: AdaFace/MagFace-inspired quality confidence, MegaFace/FRTE-inspired distractor-gallery reports.

Verification:

```powershell
python -m seju_face_lab compare-backends --reference-images data/raw/seju_official --images outputs/agency_generation_refined/v2_contrast/images --out outputs/backend_compare_agency_v_next --backends deterministic opencv-face
python -m seju_face_lab qa-images --images outputs/agency_generation_refined/v2_contrast/images --out outputs/agency_generation_refined/v2_contrast/quality
python -m seju_face_lab enhance-agencies --model outputs/seju_model_official --agencies configs/agencies/seju_like_agencies.json --images outputs/agency_generation_refined/v2_contrast/images --out outputs/agency_generation_refined/v2_contrast/enhancement_next
```

### P2. Interpretation and Visualization

Goal: make the visual map show real separability instead of compressing everything into one quadrant.

Items:

1. Adaptive 8-axis projection
   - method: PCA first; UMAP optional if dependency is available.
   - keep fixed quadrant map as reference.
   - done when: page shows fixed quadrant plus adaptive projection.

2. Prompt attribution
   - table: descriptor delta -> 8-axis delta -> prompt clause -> measured effect.
   - done when: a generated image can be reviewed by which prompt clauses moved which axes.

3. Condition-stratified robustness
   - strata: source type, resolution, crop quality, aspect ratio, detector acceptance, lighting/presentation flags.
   - done when: reports show whether scores are stable across quality strata.

4. Drift monitor
   - inputs: official agency roster URLs and retrieval dates.
   - done when: changed official pages create a refresh task before old names/source examples are reused.

Verification:

```powershell
python -m seju_face_lab face-axes --images outputs/agency_generation_refined/v2_contrast/images --out outputs/agency_generation_refined/v2_contrast/face_axes_next
python scripts/build_agency_site.py --average-params outputs/agency_reviews/seju_like_v2/agency_average_params.json --enhancement outputs/agency_generation_refined/v2_contrast/enhancement/agency_enhancement_report.json --images outputs/agency_generation_refined/v2_contrast/images
```

## Promotion Rules

- A page update may show generated agency contrast samples only with `hypothesis_and_generated` evidence labels.
- A non-seju agency may be called a real agency average only after real per-talent images and source manifests exist.
- A score may be promoted only after data-quality audit, QA gate, and at least deterministic + OpenCV agreement are recorded.
- A neural face-embedding result may be used only with detector acceptance counts.
- Style and face-geometry scores stay separate unless a combined score explicitly lists its weights.

## Versioned Outputs

- Audit: `outputs/data_quality_audit_v*/`
- Real agency subject reviews: `outputs/agency_real_subject_reviews/<agency>/`
- Agency centroids: `outputs/agency_real_centroids/<agency>/`
- Generation sweeps: `outputs/agency_generation_sweeps/<version>/`
- Site: `outputs/agency_site/`
- Public docs and configs: committed under `docs/`, `configs/`, `scripts/`, and `tests/`

## Done Criteria For The Next Engine

- `scripts/audit_research_data_quality.py` is wired into the normal runbook.
- Public page includes evidence-type badges.
- At least one non-seju agency has a real per-talent image set and source manifest.
- Subject-balanced centroid mode exists and is compared against image-weighted centroid mode.
- Backend agreement is reported before ranking claims.
- Multi-seed agency generation sweep replaces one-off generated images.
- All new behavior has tests and passes:
  - `ruff check .`
  - `python -m unittest discover -s tests`
  - `python -m compileall -q src tests scripts`
  - `git diff --check`

## Safety Boundary

All outputs are local research evidence only. Do not use them for identity, attractiveness,
popularity, ethnicity, cleanliness, personality, or personal-value labels.
