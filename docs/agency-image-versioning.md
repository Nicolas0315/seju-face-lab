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

Version rules:

- Generated images stay under `outputs/` and are not committed.
- Configs, docs, scripts, score summaries, and path manifests are committed.
- Every image update version records source prompts, source scores, output image paths, evaluation paths, and page promotion paths.
- A version is promoted to the public page only after `evaluate`, `face-axes`, `enhance-agencies`, local browser verification, and Cloudflare verification run successfully.

Safety boundary:

- Images are fictional aggregate samples, not real-person likenesses.
- Scores are local centroid similarity and axis-alignment evidence only.
- Do not use these outputs as identity, attractiveness, popularity, ethnicity, or personal-value labels.
