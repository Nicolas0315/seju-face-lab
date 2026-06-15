# strengthening engine review

Review date: 2026-06-15.

This review separates what is backed by local real-image vectors from what is still
official-source hypothesis or generated-image evidence.

## Current Evidence Snapshot

- seju real-image set: `data/raw/seju_by_talent/`
- seju subjects: 34
- seju reviewed images: 259
- seju failed images in subject review: 0
- seju centroid model: `outputs/seju_model_official/centroids.npz`
- agency contrast set: 8 agencies in `configs/agencies/seju_like_agencies.json`
- agency contrast evidence: official-source descriptor hypotheses plus generated aggregate image scoring
- non-seju real per-talent agency image averages: not yet collected

## Data Quality Audit v1

- command: `python scripts/audit_research_data_quality.py --out outputs/data_quality_audit_v1`
- report: `outputs/data_quality_audit_v1/data_quality_audit.md`
- risk_level: `needs_real_data_before_strong_claims`
- issue_count: 6
- engine_items: 6
- seju subjects/images: 34 subjects / 259 images
- seju subject image imbalance: min 5, max 18, ratio 3.6
- exact duplicate groups: 0
- small image count: 34 with min side below 384 px
- aspect outlier count: 34
- agency evidence gap: 7 agencies are still `hypothesis_and_generated`

Audit conclusion:

- The seju centroid is a real local-image average, but it is not yet a fully robust population-scale
  model because subject count, image balance, and crop/size quality still need gates.
- The cross-agency page is useful as an experimental map, not as real agency average-face evidence,
  until non-seju agencies have per-talent image sets and source manifests.
- The next strengthening engine must promote evidence type and data-quality gates ahead of score display.

## 10 Weaknesses To Turn Into Engine Items

1. Real data coverage is narrow.
   - Evidence: only seju currently has a real per-talent image set with 34 subjects and 259 reviewed images.
   - Risk: agency comparisons can look precise while actually comparing seju real data against other-agency hypotheses.
   - Engine item: add `agency-data-coverage-audit` that reports real subjects/images per agency before any ranking.

2. Non-seju agencies are hypothesis-backed, not real-image averaged.
   - Evidence: `configs/agencies/seju_like_agencies.json` uses descriptor offsets; v2 generated images are scored, but not built from real per-talent image folders.
   - Risk: the map shows prompt/generation behavior, not actual agency visual distributions.
   - Engine item: add `build-agency-centroids --subjects-root data/raw/agencies/<agency>/`.

3. Image source quality and permission metadata are under-modeled.
   - Evidence: local paths exist, but source URL/license/permission quality is not attached per image in the scoring output.
   - Risk: hard to audit which images are official, duplicated, cropped, stale, low-res, or unsuitable.
   - Engine item: add per-image `source_manifest.jsonl` with URL, retrieval date, source type, permission note, hash, dimensions, and duplicate group.

4. Subject balance is not enforced.
   - Evidence: the seju set has 34 folders/259 images, but current centroid build treats all images together.
   - Risk: subjects with many photos can dominate the mean/median centroid.
   - Engine item: add subject-balanced centroids: average per subject first, then average subjects.

5. Backend disagreement is known but not yet policy-gated.
   - Evidence: research tracking records deterministic/OpenCV/InsightFace/DeepFace rank divergence, especially DeepFace detector effects.
   - Risk: one backend can overstate a result.
   - Engine item: require backend agreement bands before promoting a score: deterministic + OpenCV + InsightFace/DeepFace-retinaface where available.

6. Detector acceptance bias is still a major confounder.
   - Evidence: prior logs show OpenCV and DeepFace detector acceptance vary substantially across the same reference set.
   - Risk: rankings reflect detector acceptance or crop behavior more than face similarity.
   - Engine item: add detector acceptance, crop box, face-size, pose/occlusion, and rejection reason columns to every review.

7. 8-axis map compresses too much into one quadrant.
   - Evidence: v2 contrast agencies still share `defined_bright`; the useful spread appears in secondary axes such as `muted_vivid`, `natural_styled`, and outlier score.
   - Risk: quadrant labels hide meaningful differences.
   - Engine item: add adaptive PCA/UMAP projection over 8-axis vectors plus small-multiple axis bars; keep fixed quadrants as a secondary view.

8. Prompt generation is too coarse for contrast agencies.
   - Evidence: auto prompts from descriptor offsets can still collapse into the same detector-friendly seju-like language.
   - Risk: generated contrast samples converge toward the same look and reduce separability.
   - Engine item: add prompt-token attribution from descriptor deltas, with explicit axis-to-prompt clauses and negative prompts per target axis.

9. Generated-image evaluation is not yet a closed-loop optimizer.
   - Evidence: v1/v2 images were generated and scored, but there is no automated multi-seed selection policy in the agency page flow.
   - Risk: one lucky or unlucky image can move the agency score.
   - Engine item: add `agency-generation-sweep` with N seeds per agency, QA gate, backend agreement, style axis, and selected-candidate manifest.

10. Bias/fairness and applicability are documented but not measured locally.
    - Evidence: boundaries are written, but local reports do not yet quantify performance by source type, lighting, crop, age bracket, or presentation condition.
    - Risk: the engine may be robust only for clean, frontal, similar-looking images.
    - Engine item: add condition-stratified score reports: source type, lighting, crop quality, face size, hair occlusion, expression, and image age.

## Next Strengthening Engine Backlog

P0:

- `agency-data-coverage-audit`: fail or warn when an agency has no real per-talent image set.
- `source_manifest.jsonl`: hash, URL, retrieved_at, source type, dimensions, duplicate group.
- Subject-balanced centroid mode: `--balance subject`.
- Detector acceptance report in every review bundle.
- `data-quality-audit`: implemented as `scripts/audit_research_data_quality.py`; next step is wiring it into pipelines and page badges.

P1:

- Multi-backend promotion gate: deterministic + opencv-face + insightface/deepface-retinaface agreement.
- `agency-generation-sweep`: multi-seed generated candidate scoring per agency.
- Adaptive 8-axis projection view: PCA/UMAP + fixed 8-axis bars.
- Prompt attribution table: descriptor delta -> axis delta -> prompt clause -> measured effect.

P2:

- Condition-stratified robustness report.
- Duplicate/near-duplicate detection.
- Real non-seju agency collection workflow with official-source manifests.
- Drift monitor for agency rosters and source pages.

## Research Notes

- NIST FRTE/FATE split identity verification and face analysis tracks. For this project, identity-style
  verification should stay separate from presentation/analysis axes.
- NIST demographic-effect reports are a reminder that face systems need subgroup and condition
  reporting; this project should avoid demographic labels unless there is a legitimate, consented,
  and carefully scoped evaluation need.
- ArcFace/InsightFace remains a strong reference family for face embeddings, but detector choice and
  data quality can change acceptance and rankings; local backend agreement is mandatory before claims.
- CLIP/OpenCLIP is useful for style/photographic axis scoring, but it should remain separate from
  face-geometry scores because text-image alignment can encode broad style rather than facial structure.

## Sources

- NIST FRTE 1:1 Verification: https://pages.nist.gov/frvt/html/frvt11.html
- NIST Face Projects and demographic effects overview: https://www.nist.gov/programs-projects/face-projects
- NIST FRTE demographic effects page: https://pages.nist.gov/frvt/html/frvt_demographics.html
- NIST FRTE/FATE split: https://www.nist.gov/programs-projects/face-technology-evaluations-frtefate
- ArcFace paper: https://arxiv.org/abs/1801.07698
- InsightFace project: https://github.com/deepinsight/insightface
- CLIP introduction: https://openai.com/index/clip/
- LAION CLIP benchmark: https://github.com/LAION-AI/CLIP_benchmark
