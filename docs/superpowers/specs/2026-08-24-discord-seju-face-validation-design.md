# Discord Evidence to Seju Face Artifact Validation

Status: design for review

Date: 2026-08-24

Target repository: `seju-face-lab`

Target Discord guild: local private configuration only

## 1. Outcome

Build a reproducible local pipeline that turns authorized Discord conversation evidence into:

1. bounded research hypotheses;
2. an explicit validation protocol;
3. a quality-aware aggregate face template;
4. clearly fictional generated candidates;
5. technical, statistical, and publication-safety evidence for each candidate.

The pipeline must not identify a person, match a person, infer demographics, score attractiveness, or claim that a local aggregate represents a population. It may describe only the consent-cleared local image set and the generated artifacts derived from that set.

## 2. Current Evidence

### 2.1 Fleet and repository

- The complete runnable environment is currently present only on `rtx4090`.
- The local reference set has 34 subjects and 259 images, with 5 to 18 images per subject.
- The landmark-aligned build accepts 224 of 259 images, so detector dropout is material.
- Existing deterministic tests pass, but the current profile contract does not identify every backend, model, detector, alignment, crop, version, and model hash.
- The current random-unit-vector null is not a face-manifold control and cannot support a strong superiority claim.

### 2.2 Discord source audit

Read-only checks on 2026-08-24 established:

- the existing local Discord SQLite archive is healthy but stale by about 58 hours;
- it contains zero messages whose `guild_id` equals the target guild;
- an authenticated browser can access the target guild;
- the guild exposes two text channels to the current account;
- a guild search for `seju` returns 23 hits spanning 2026-04-21 through 2026-08-24.

The relevant conversation themes are:

- the working definition of a “Seju face” is unclear;
- participants want an aggregate or representative fictional face rather than a real-person identification result;
- vector proximity and generated-image validation are desired;
- a fictional face intended to sit alongside well-known creators is discussed;
- visual proportion language such as “golden ratio” appears;
- generated images and a parody-like site are discussed, including a suggestion to test whether people could be deceived.

The last two themes are constraints, not acceptance criteria. “Golden ratio” must not become an attractiveness score, and deception must not become a product goal. Any public artifact must be conspicuously labeled as synthetic and fictional.

## 3. Non-goals and Hard Boundaries

The implementation must reject or omit:

- identity recognition or identity verification;
- “who is this?” or nearest-person results;
- membership prediction for a real person;
- demographic, gender, age, ethnicity, personality, or attractiveness labels;
- face swap or copying a named real person;
- an unlabeled synthetic image or page that could reasonably be mistaken for an official Seju asset;
- public Discord message text, usernames, message IDs, attachments, raw face images, embeddings, or generated model outputs;
- a claim that the method is globally better than InsightFace.
- commercial deployment using InsightFace's restricted pretrained models without a separately verified license.

The safe research claim is narrower: a quality-aware robust aggregation and validation protocol may outperform the repository's current simple aggregation baseline for this local, consent-cleared dataset under predefined holdout and perturbation metrics.

## 4. Design Principles

### 4.1 Conversation is evidence, not truth

Discord messages create candidate hypotheses. They do not create ground-truth labels. A hypothesis must be converted into a machine-readable statement with:

- source scope and timestamp range;
- supporting message count;
- contradictory or uncertainty evidence;
- permitted measurable interpretation;
- rejected interpretations;
- operator disposition: `accepted`, `rejected`, or `needs_review`.

### 4.2 Geometry, presentation, and publication risk stay separate

The pipeline reports independent axes:

- aggregate face-embedding agreement;
- detector and alignment quality;
- image presentation/style quality;
- duplicate/copy risk;
- publication-safety status.

No weighted “overall beauty” or “Seju membership” score is produced.

### 4.3 Subject-disjoint evidence is mandatory

An image or subject used to build a fold template must not validate that same fold. All comparisons use identical subject-disjoint splits across the baseline and candidate algorithm.

### 4.4 Fail closed

A model-contract mismatch, multi-face image, missing provenance, insufficient holdout units, missing confidence interval, or unsafe publication label blocks promotion.

## 5. Architecture

```text
authorized Discord source
        |
        v
local immutable evidence snapshot
        |
        v
hypothesis extraction + policy classification
        |
        +--------------------+
        |                    |
        v                    v
accepted measurable     rejected/needs-review
hypotheses              evidence
        |
        v
consent-cleared local image manifest
        |
        v
quality + detector evidence -> robust subject templates
        |                              |
        +------------------------------+
                       |
                       v
          LOSO + perturbation evaluation
                       |
                       v
             fictional generation sweep
                       |
                       v
       artifact QA + copy-risk + safety gate
                       |
                       v
       local report -> aggregate-only public bundle
```

## 6. Discord Evidence Layer

### 6.1 Acquisition adapters

Implement an adapter interface, not a browser-specific core:

```python
class DiscordEvidenceSource(Protocol):
    def iter_messages(self, guild_id: str) -> Iterable[DiscordMessage]: ...
```

Adapters:

1. `sqlite-readonly`: preferred for repeatable offline runs; opens an operator-provided snapshot using SQLite immutable/read-only mode and filters exact `guild_id`.
2. `jsonl-import`: immediate fallback for an authorized export; validates every row before local storage.
3. `discord-api`: optional later adapter using an operator-managed, least-privilege official Discord integration. Authentication and persistent permissions remain an operator gate.

Browser automation is an audit and bootstrap surface only. It is too stateful to be the canonical training or evaluation input. A one-time authorized browser audit may be normalized into a private JSONL snapshot, after which every research run consumes the immutable local snapshot rather than live page state.

### 6.2 Snapshot contract

Canonical private JSONL fields:

```text
schema_version
guild_id
channel_id
thread_id
message_id
timestamp
edited_timestamp
author_pseudonym
author_bot
content
attachment_count
attachment_media_types
referenced_message_id
source_adapter
snapshot_id
```

Rules:

- raw usernames are replaced with run-local pseudonyms;
- raw text is stored only under an ignored local output directory;
- attachments are not downloaded by default;
- message and channel IDs never enter a public report;
- each snapshot records row count, min/max timestamp, channel count, and a deterministic content digest;
- duplicate message IDs collapse to the latest edited version while retaining edit provenance;
- import refuses a row from a different guild.

### 6.3 Hypothesis contract

Private `hypotheses.json` fields:

```text
hypothesis_id
summary
support_count
uncertainty_count
source_time_range
permitted_measurement
rejected_measurements
policy_status
operator_status
```

Initial hypotheses derived from the audited conversation:

- `H1`: the project needs an explicit operational definition instead of an intuitive label;
- `H2`: generated artifacts should be fictional aggregate samples, not real-person reproductions;
- `H3`: vector agreement must be validated using subject-disjoint evidence;
- `H4`: generated images require technical QA and visible synthetic provenance;
- `H5`: social-media or branding ideas are downstream presentation research and must not alter face-geometry ground truth.

Rejected hypotheses:

- real-person membership classification;
- attractiveness or “golden ratio” optimization;
- deception success as an experiment metric.

## 7. Model Contract

Every built model must store and validate:

```text
schema_version
backend_name
recognition_model_name
recognition_model_sha256
detector_name
detector_model_sha256
alignment_method
landmark_schema
crop_size
color_space
preprocessing_version
embedding_dimension
embedding_normalization
reference_manifest_sha256
subject_balance_mode
quality_model_version
aggregation_method
code_commit
```

Evaluation refuses any mismatch other than run-specific output paths and timestamps. Equal embedding dimensions are never treated as proof of compatibility.

## 8. Candidate Algorithm: Quality-aware Robust Spherical Template

InsightFace remains the embedding extractor baseline. The new contribution is the local aggregation and validation layer, not an unsubstantiated replacement recognition network.

### 8.1 Per-image processing

For each consent-cleared reference image:

1. detect all faces;
2. reject zero-face and multi-face images from automatic training;
3. retain detector confidence, bounding box, relative face area, five-point landmarks, yaw/pitch/roll estimate, and rejection reason;
4. align to the model's canonical crop;
5. produce a unit-normalized neural embedding;
6. calculate quality evidence: blur/focus, exposure clipping, pose magnitude, landmark reprojection error, crop margin, and resolution;
7. detect exact and perceptual duplicates before aggregation.

Quality evidence is reported as a vector. It is not a personal attribute.

### 8.2 Fold-fitted quality weight

For image `i`, define a bounded weight:

```text
q_i = clip(exp(-beta dot z_i), q_min, 1)
```

where `z_i` is the non-negative defect vector normalized using training-fold statistics only. `beta` is selected using nested folds or a predeclared small grid on training subjects. Held-out subjects never tune the weight.

### 8.3 Robust subject template

For each subject, calculate the weighted spherical geometric median of the accepted unit embeddings. Apply one deterministic angular-outlier pass using a threshold learned only from the training fold. If fewer than the configured minimum independent images remain, mark the subject insufficient instead of silently falling back.

This prevents image-heavy subjects and burst duplicates from dominating the result.

### 8.4 Equal-subject global template

Aggregate the valid subject templates with an equal-subject spherical geometric median. Record:

- subject count and accepted-image count;
- per-subject influence under delete-one analysis;
- bootstrap 95% confidence interval over subjects;
- detector dropout and quality-weight distribution.

No individual subject name or influence value is included in a public artifact.

### 8.5 Baselines

Compare on the same folds:

- B0: current image-weighted mean;
- B1: subject-balanced arithmetic mean;
- B2: subject-balanced spherical mean;
- C1: proposed quality-aware robust spherical template.

The same InsightFace pack, detector, alignment, and reference manifest must be used for all four.

## 9. Evaluation Protocol

### 9.1 Leave-one-subject-out

For each subject:

- build a template from the other subjects;
- evaluate only the held-out subject's accepted images and subject template;
- retain fold-level coverage and quality evidence;
- never evaluate a fold on images used to build it.

### 9.2 Perturbation matrix

Run deterministic perturbations that reflect observed failure modes:

- mild and severe downscale;
- blur;
- exposure shift;
- JPEG compression;
- partial crop;
- detector change with a fixed recognition model, plus separately reported backend changes where compatible.

Report clean-to-perturbed degradation, acceptance dropout, and rank stability. Never average or directly compare coordinates from different embedding spaces. Perturbed images are evaluation artifacts, not new training samples.

### 9.3 Metrics

Primary metrics:

- LOSO held-out template agreement, median and bootstrap 95% CI;
- worst-decile held-out agreement;
- clean-to-perturbed agreement drop;
- detector acceptance rate and dropout delta;
- fold-to-fold rank correlation;
- delete-one subject influence maximum;
- runtime and peak GPU memory.

Secondary metrics:

- centroid bootstrap stability;
- quality-weight sensitivity;
- exact/perceptual duplicate prevalence;
- agreement across compatible neural backends.

If a consent-cleared, subject-disjoint external control set is later available, add calibration metrics in a separately labeled experiment. Until then, do not report a “membership probability,” FMR, or superiority over a general recognition benchmark.

### 9.4 Promotion rule

C1 is promoted only when all conditions hold:

- the lower confidence bound of LOSO improvement over B1 is positive for the predefined primary agreement metric;
- worst-decile performance does not regress beyond a predefined tolerance;
- perturbation degradation improves or remains within tolerance;
- detector coverage does not regress materially;
- no fold violates the model contract;
- all results are reproducible from a manifest and commit hash.

Otherwise retain the simpler subject-balanced baseline.

## 10. Generated Artifact Protocol

### 10.1 Generation

- use aggregate-only prompt descriptors;
- prohibit named-person likeness prompts;
- generate multiple seeds and keep every run manifest;
- include `synthetic`, `fictional`, model/provider, seed, prompt hash, source model hash, and generation timestamp in the private manifest;
- never use Discord attachments directly as generation input without a separate consent-cleared intake step.

### 10.2 Technical QA

Each candidate must pass:

- exactly one detected face;
- minimum detector confidence and relative face area;
- bounded yaw/pitch/roll;
- landmark reprojection threshold;
- no severe blur, clipping, crop, or malformed facial structure signal;
- no exact or perceptual duplicate of any source image;
- successful evaluation under at least two compatible detector paths when available.

### 10.3 Aggregate agreement

Report agreement to the final promoted aggregate template and its subject-bootstrap uncertainty. Do not list nearest people or reference images. Geometry and style remain separate axes.

### 10.4 Publication-safety gate

Every published image and page must include:

- a visible “AI生成・架空” label adjacent to the image;
- metadata that identifies the artifact as synthetic;
- a statement that it is not an official Seju asset and depicts no identified person;
- no claim of affiliation, endorsement, or real-person membership;
- no copied official visual identity that would create a deceptive official-looking page;
- aggregate-only metrics and dataset-scope caveats.

Deployment is a non-commercial research preview unless a separate model-license review permits more. The generated-image bundle is staged in an ignored local directory and uploaded without committing the portraits to Git.

The public bundle contains generated images only after operator review. It contains no raw Discord content, usernames, source photos, embeddings, per-subject scores, or private output paths.

## 11. Proposed Modules and CLI

New modules:

```text
src/seju_face_lab/discord_evidence.py
src/seju_face_lab/hypotheses.py
src/seju_face_lab/model_contract.py
src/seju_face_lab/detector_evidence.py
src/seju_face_lab/robust_templates.py
src/seju_face_lab/holdout_evaluation.py
src/seju_face_lab/artifact_policy.py
```

CLI additions:

```text
discord-import
discord-summarize
build-robust
evaluate-robust
review-generated-v2
build-public-evidence
```

Existing `build`, `evaluate`, `qa-images`, `review-generated`, `precision-report`, and pipeline orchestration should call the new primitives rather than duplicate logic.

## 12. Storage and Privacy

Private ignored paths:

```text
outputs/discord_evidence/<snapshot_id>/
outputs/robust_models/<run_id>/
outputs/robust_evaluation/<run_id>/
outputs/generated/<run_id>/
outputs/artifact_reviews/<run_id>/
```

Versioned repository content may include schemas, example fixtures using invented messages, algorithms, tests, and aggregate report templates. It must not include the real guild ID, real messages, real usernames, source images, embeddings, or generated portraits.

## 13. Test Strategy

### 13.1 Deterministic unit tests

- exact guild filter and cross-guild refusal;
- SQLite immutable/read-only opening;
- edit deduplication and snapshot digest;
- pseudonymization and public-redaction guarantees;
- hypothesis policy classification, including rejection of identity, attractiveness, and deception goals;
- model-contract mismatch rejection for same-dimension incompatible backends;
- quality-vector bounds and fold-only fitting;
- spherical median behavior and equal-subject weighting;
- duplicate/burst suppression;
- LOSO split leakage prevention;
- bootstrap determinism under a seed;
- perturbation manifest reproducibility;
- public bundle exclusion of private fields and unlabeled images.

### 13.2 Integration tests

- invented Discord JSONL -> hypotheses -> accepted measurement plan;
- fixture images -> robust model -> LOSO report;
- generated fixture -> QA -> publication gate;
- baseline and candidate use identical fold manifests;
- a deliberately mismatched detector/model profile fails before scoring.

### 13.3 Real-path validation

On `rtx4090`:

1. import the authorized target-guild snapshot locally;
2. confirm row/channel/time-range counts against the Discord readback;
3. build B0, B1, B2, and C1 from the same 34-subject manifest;
4. run all 34 LOSO folds and the perturbation matrix;
5. generate a multi-seed fictional batch;
6. run technical QA, aggregate agreement, duplicate risk, and publication policy;
7. visually inspect the selected images;
8. build an aggregate-only public preview;
9. run a browser smoke test against the deployed preview after explicit deployment authorization.

## 14. Acceptance Criteria

The implementation is complete only when:

- Discord evidence can be reproduced from an authorized local snapshot without browser state;
- real Discord content remains local and public-output tests prove redaction;
- every model/evaluation enforces the full model contract;
- 34-subject LOSO completes without train/test subject leakage;
- candidate-vs-baseline comparison includes bootstrap confidence intervals and perturbation evidence;
- no global “better than InsightFace” claim is made;
- selected generated images pass technical and publication-safety gates;
- the public preview visibly labels every image as fictional and synthetic;
- unit, integration, full deterministic, GPU smoke, and deployed browser smoke evidence are recorded separately.

## 15. Delivery Sequence

1. evidence schemas, local adapters, and redaction tests;
2. model contract and fail-closed compatibility checks;
3. detector/quality evidence and robust subject aggregation;
4. LOSO and perturbation evaluator;
5. generation review v2 and publication-safety gate;
6. real 34-subject comparison on `rtx4090`;
7. fictional generation sweep and operator visual review;
8. aggregate-only public preview and deployment smoke test.

Implementation starts only after this design is reviewed. Each delivery slice uses tests first and preserves the existing ignored-data boundaries.

## 16. Research Basis

- [InsightFace Model Zoo](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md) defines the available recognition/detection packs, benchmark figures, and non-commercial-research model restriction.
- [NIST FATE Quality](https://pages.nist.gov/frvt/html/frvt_quality.html) evaluates explicit image defects and unified quality behavior; quality must be treated as measured evidence rather than a subjective label.
- [MagFace (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Meng_MagFace_A_Universal_Representation_for_Face_Recognition_and_Quality_Assessment_CVPR_2021_paper.html) and [AdaFace (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_AdaFace_Quality_Adaptive_Margin_for_Face_Recognition_CVPR_2022_paper.html) motivate quality-aware representation or margin handling, but this project applies the idea conservatively at the aggregation layer unless a separately licensed training effort is approved.
- The repository's existing architecture already separates face geometry, style, generation QA, and precision reporting; the design strengthens those boundaries rather than merging them into a single score.
