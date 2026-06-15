# Agency Image Versioning

This document tracks public-page image update versions for the agency average-face research page.

## v0 baseline

- images: `outputs/agency_imagegen_samples/`
- enhancement: `outputs/agency_enhancement/`
- page assets: `outputs/agency_site/assets/`
- deployed page: `https://seju-face-lab-agency-research.pages.dev/`

## agency-refinement-v1

- config: `configs/generation_updates/agency_refinement_v1.json`
- source calibration: `outputs/agency_generation_calibration/generation_calibration.json`
- selected updates: `seju`, `asia-promotion`
- updated images: `outputs/agency_generation_refined/v1/images/`
- evaluation output: `outputs/agency_generation_refined/v1/evaluation/`
- axis output: `outputs/agency_generation_refined/v1/face_axes/`
- enhancement output: `outputs/agency_generation_refined/v1/enhancement/`
- page assets after promotion: `outputs/agency_site/assets/`
- generated image source cache id: `019ec65e-dcff-7ef2-a906-a2501156d96e` (local Codex cache, not portable)
- production URL: `https://seju-face-lab-agency-research.pages.dev/`
- deployment URL: `https://302d751d.seju-face-lab-agency-research.pages.dev/`

Measured results:

- `seju`: enhancement `0.681432` -> `0.783322` (`+0.101890`), target met.
- `asia-promotion`: enhancement `0.581166` -> `0.707412` (`+0.126246`), improved but still below the `0.76` target.
- `platinum`: carried forward at `0.754879`.
- `lespros`: carried forward at `0.716674`.
- `trustar`: carried forward at `0.689821`.

Current v1 rank:

1. `seju`: `0.783322`
2. `platinum`: `0.754879`
3. `lespros`: `0.716674`
4. `asia-promotion`: `0.707412`
5. `trustar`: `0.689821`

Verification:

- Site build: `python scripts/build_agency_site.py --images outputs/agency_generation_refined/v1/images`
- Deployed with Cloudflare Pages project `seju-face-lab-agency-research`.
- Remote image hash check: `seju.png` and `asia-promotion.png` matched v1 local images.
- Browser check: 5 cards, 5 images, all images loaded after lazy-load scroll.

## agency-direction-map-v1

- page section: `8軸方向性マップ`
- implementation: `scripts/build_agency_site.py`
- source vectors: `outputs/agency_generation_refined/v1/enhancement/agency_enhancement_report.json`
- rendered page: `outputs/agency_site/index.html`
- data export: `outputs/agency_site/data.json`
- deployment URL: `https://02230471.seju-face-lab-agency-research.pages.dev/`
- encoding:
  - main quadrant: average of `soft_defined` + `natural_styled` by average of `deep_bright` + `cool_warm`
  - cross-axis notes: `dynamic_symmetric` and `light_dark_hair`
  - detailed rows: all 8 observed axes per agency

## agency-subject-review-v1

- output: `outputs/agency_subject_reviews/subject_reviews.html`
- csv: `outputs/agency_subject_reviews/subject_reviews.csv`
- json: `outputs/agency_subject_reviews/subject_reviews.json`
- input grouping: `outputs/agency_subject_reviews/input_by_agency/`
- source images: `outputs/agency_generation_refined/v1/images/`
- note: this is a subject-review-format comparison of generated agency aggregate images, not a per-talent review for non-seju agencies.
- current ranking by generated aggregate image score: `seju`, `platinum`, `lespros`, `trustar`, `asia-promotion`

## agency-contrast-v2

- purpose: add higher-styling contrast agencies so the 8-axis map shows more spread than the close seju-like set.
- added agencies: `lvs`, `sgmedia`, `twin-planet`
- config: `configs/agencies/seju_like_agencies.json`
- average params: `outputs/agency_reviews/seju_like_v2/agency_average_params.json`
- generated images: `outputs/agency_generation_refined/v2_contrast/images/`
- evaluation: `outputs/agency_generation_refined/v2_contrast/evaluation/`
- face axes: `outputs/agency_generation_refined/v2_contrast/face_axes/`
- enhancement: `outputs/agency_generation_refined/v2_contrast/enhancement/`
- subject-review-format comparison: `outputs/agency_subject_reviews_v2_contrast/subject_reviews.html`
- page build: `outputs/agency_site/index.html`
- deployment URL: `https://19b8ac6a.seju-face-lab-agency-research.pages.dev/`
- evidence-badge deployment URL: `https://bb8af4e9.seju-face-lab-agency-research.pages.dev/`
- verification screenshots: `outputs/agency_site/browser_verify.png`, `outputs/agency_site/deploy_verify.png`
- current top enhancement ranking: `sgmedia`, `seju`, `platinum`, `lvs`, `lespros`, `twin-planet`, `asia-promotion`, `trustar`
- largest visible axis spread: `lvs` has stronger vivid/styled/outlier signals than the close seju-like set.
- evidence boundary: v2 contrast agencies are official-source descriptor hypotheses plus generated aggregate image scoring; they are not real per-talent image averages yet.

Version rules:

- Generated images stay under `outputs/` and are not committed.
- Configs, docs, scripts, score summaries, and path manifests are committed.
- Every image update version records source prompts, source scores, output image paths, evaluation paths, and page promotion paths.
- A version is promoted to the public page only after `evaluate`, `face-axes`, `enhance-agencies`, local browser verification, and Cloudflare verification run successfully.

Safety boundary:

- Images are fictional aggregate samples, not real-person likenesses.
- Scores are local centroid similarity and axis-alignment evidence only.
- Do not use these outputs as identity, attractiveness, popularity, ethnicity, or personal-value labels.
