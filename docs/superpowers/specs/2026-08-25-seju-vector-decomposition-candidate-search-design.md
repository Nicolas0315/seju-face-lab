# Seju Face Vector Decomposition and Fictional Candidate Search

Status: design for review

Date: 2026-08-25

Supersedes: `2026-08-24-discord-seju-face-validation-design.md`

## 1. Objective

Refocus `seju-face-lab` on one research problem:

1. build a provenance-verified, face-only dataset from the local Seju official-site snapshot;
2. represent each accepted face with compatible geometry and neural vectors;
3. separate within-person variation from stable between-person common structure;
4. retain an interpretable `Seju approximation score`;
5. explore diverse fictional third-person candidates inside the stable Seju vector region;
6. evaluate candidate preference independently through blinded human review.

Discord ingestion, public-site deployment, cross-agency comparison, social engagement collection, and automatic popularity prediction are outside this implementation slice.

## 2. Verified Starting Point

The 2026-06-14 local snapshot contains:

- 265 discovered rows from 35 profiles;
- 259 eligible rows from 34 profiles;
- 6 rows from one age-unknown profile excluded before download;
- 259 downloaded files, with zero missing files and zero SHA-256 mismatches;
- 259 of 259 profile and image URLs hosted by `seju.tokyo`;
- 34 signature images incorrectly admitted as analyzable images;
- 34 `og:image` rows, including 33 thumbnail URLs that may duplicate or resize another portrait;
- 192 other eligible image URLs;
- 5 to 18 files per subject.

The source is Seju-only by domain and profile provenance, but it is not yet a clean face-only dataset.

Current model behavior confirms the contamination:

- the deterministic subject-balanced model used all 259 images, including signature images;
- InsightFace rejected 35 images: all 34 signatures plus one ordinary image with no detectable face;
- the InsightFace aggregate retained 224 images;
- exact file hashes are unique, but perceptual and resized duplicates have not been eliminated;
- current model profiles do not fully bind backend, detector, alignment, model pack, hashes, and preprocessing.

The previous generated portrait is rejected. It was produced from generic textual appearance descriptors, not by decoding or conditioning on the measured face-vector space, and therefore does not demonstrate Seju-specific reconstruction.

## 3. Boundaries

### 3.1 Permitted outputs

- per-image detector, alignment, quality, and provenance evidence;
- local unit-normalized face embeddings;
- local landmark-derived geometry vectors;
- subject-level robust templates;
- aggregate components and uncertainty;
- Seju approximation score with explicit local-percentile semantics;
- fictional candidate vectors, generated images, and blinded preference results;
- aggregate research reports without raw embeddings or personal rankings.

### 3.2 Prohibited outputs

- identity recognition or nearest-person results;
- a prediction that a real person belongs to Seju;
- attractiveness, beauty, worth, personality, demographic, or protected-attribute labels;
- a claim that face geometry alone predicts popularity;
- per-person public scores or component coordinates;
- named-person likeness generation, face swap, or official-looking deceptive assets;
- a claim that this local algorithm globally outperforms InsightFace;
- commercial use of restricted pretrained models without a separate license review.

`Seju approximation score` means similarity to the cleaned local Seju snapshot. It is not a probability, identity decision, population statistic, or measure of personal value.

## 4. System Decomposition

```text
official-source manifest
        |
        v
provenance + content gate
        |
        v
single-face detector + alignment + quality evidence
        |
        +-------------------------+
        |                         |
        v                         v
512D neural embedding     landmark geometry vector
        |                         |
        +------------+------------+
                     |
                     v
       robust per-subject templates
                     |
                     v
      spherical mean + tangent-space components
                     |
          +----------+----------+
          |                     |
          v                     v
  approximation score    component decomposition
          |                     |
          +----------+----------+
                     |
                     v
       fictional candidate search loop
                     |
                     v
       QA + score + diversity frontier
                     |
                     v
          blinded human preference
```

Each stage writes an immutable local run manifest. A downstream stage refuses incompatible or incomplete upstream evidence.

## 5. Clean Dataset Contract

### 5.1 Source allowlist

An accepted row must have:

- exact allowed profile host `seju.tokyo`;
- exact allowed image host `seju.tokyo`;
- source profile URL, image URL, retrieval timestamp, subject slug, and SHA-256;
- an eligible adult-status decision from the reviewed source snapshot;
- a local file whose SHA-256 matches the manifest.

The domain allowlist proves source scope, not face validity.

### 5.2 Pre-detector exclusions

Reject before vectorization:

- URL or filename classified as a signature asset;
- known navigation, logo, icon, or non-portrait asset;
- corrupt or unsupported image;
- missing or mismatched manifest evidence.

The initial expected exclusion count is at least 34 signature images. Tests must derive the count from manifest classification rather than hard-code it.

### 5.3 Face gate

For every remaining image, record:

```text
image_id
subject_id
face_count
detector_confidence
bbox
relative_face_area
landmarks_5
yaw_pitch_roll
landmark_reprojection_error
blur
exposure_clipping
crop_margin
resolution
accepted
rejection_reason
```

Automatic training accepts exactly one face. Zero-face and multi-face images fail closed. The system never silently selects the largest face for training.

### 5.4 Near-duplicate gate

Within each declared subject only:

1. cluster exact duplicates by SHA-256;
2. cluster resized or recompressed duplicates using perceptual hash and structural similarity on the aligned crop;
3. retain the highest-quality representative;
4. record every suppressed image and reason.

The 33 thumbnail URLs are treated as suspected duplicates until this gate proves otherwise. No cross-person identity search is performed.

### 5.5 Minimum evidence

A subject needs at least three independent accepted portrait clusters. A run with fewer than 20 valid subjects is diagnostic only and cannot produce a promoted approximation model.

## 6. Model Contract

Every model and evaluation stores:

```text
schema_version
source_manifest_sha256
clean_manifest_sha256
code_commit
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
quality_definition_version
duplicate_definition_version
subject_aggregation
global_aggregation
component_method
component_count
score_definition_version
```

Same-sized vectors are not assumed compatible. Any contract mismatch blocks scoring.

## 7. Dual Face Representation

### 7.1 Neural face representation

Use InsightFace `buffalo_l` as the initial 512-dimensional baseline extractor:

- five-point aligned crop;
- unit-normalized ArcFace-family embedding;
- detector and recognition models explicitly hashed;
- no gender or age attribute module;
- no per-person matching output.

The embedding was trained for face recognition and is identity-sensitive. This project uses it only to build local aggregate templates and component evidence; it does not expose recognition or nearest-person operations.

The extractor is replaceable. The evaluation protocol compares aggregation methods under an identical extractor before comparing extractors.

### 7.2 Interpretable geometry representation

Use a separately versioned 106-point landmark module for geometry decomposition. The five detection keypoints remain alignment inputs only and are insufficient for jaw, cheek, or detailed feature measurements. Derive a dimensionless geometry vector from the 106-point observations, such as:

- face width-to-height ratio;
- normalized inter-eye distance;
- eye-line to brow, nose, and mouth positions;
- nose and mouth width relative to face width;
- left-right landmark symmetry residual;
- jaw and cheek contour coefficients;
- crop and pose residuals used as quality evidence, not personal traits.

Geometry fields describe image measurements. They are not beauty ratios or demographic attributes.

The neural and geometry vectors remain separate in storage and reporting. Their evidence may be viewed together, but they are never concatenated without a versioned experiment.

## 8. Robust Subject Templates

### 8.1 Image weights

Convert non-negative defect measurements into a bounded training weight:

```text
w_i = clip(exp(-beta dot defect_i), w_min, 1)
```

Normalization statistics and `beta` are fitted only on training subjects inside each evaluation fold. Quality never increases a score; it controls evidence reliability.

### 8.2 Subject aggregation

For each subject:

1. group near-duplicates and keep one representative;
2. calculate a quality-weighted spherical geometric median of unit embeddings;
3. apply one deterministic angular-outlier pass using a training-fold threshold;
4. calculate a robust median geometry vector;
5. retain uncertainty and accepted-image count.

Each subject contributes exactly one template to the global model.

### 8.3 Global spherical center

Calculate the equal-subject spherical geometric median `mu`. Record delete-one-subject influence and subject-bootstrap confidence intervals. No subject may dominate because it has more profile images.

## 9. Stable Component Decomposition

Map each subject template into the tangent space at `mu`, then fit robust principal components over the subject templates.

Component selection:

- maximum eight components for the first version;
- choose the smallest count that reaches the predefined explained-variance threshold and remains stable under subject bootstrap;
- align component sign and order across bootstrap samples before calculating stability;
- reject an unstable component instead of assigning it a narrative label.

Outputs:

- anonymous subject-template scatter in component space;
- component eigenvalue and bootstrap stability;
- geometry-vector correlations for interpretation;
- local mean and 5th/95th-percentile intervals;
- held-out reconstruction residual.

Components are named `C1`, `C2`, and so on. Human-readable descriptions may state measured geometric correlations, not attractiveness or personality interpretations.

## 10. Seju Approximation Score

### 10.1 Definition

For a quality-passing face vector `x`:

1. calculate cosine agreement with the global spherical center;
2. map `x` into the stable component space;
3. calculate robust subspace reconstruction residual;
4. calculate a component-support flag from training-fold coordinate intervals;
5. calibrate center agreement and reconstruction residual against leave-one-subject-out Seju subject templates;
6. combine the two non-compensating percentile scores with a harmonic mean.

```text
center_percentile = ECDF_LOSO(cosine(x, mu)) * 100
manifold_percentile = (1 - ECDF_LOSO(residual(x))) * 100
seju_approximation = harmonic_mean(center_percentile, manifold_percentile)
```

Quality is a prerequisite and confidence indicator, not a term in the approximation score.
The component-support flag is also a gate rather than a compensating score term. A candidate outside the training-fold component envelope is reported as out of support even if its scalar score is high.

### 10.2 Meaning

A score of 80 means the candidate is more central and better contained by the cleaned local Seju component model than roughly 80 percent of held-out Seju subject templates under this score definition. It does not mean “80% likely to be Seju.”

Because there are only 34 subjects, publish the score with:

- bootstrap 95% interval;
- score-definition version;
- accepted subject count;
- center and manifold component percentiles;
- out-of-support flag;
- explicit local-snapshot boundary.

### 10.3 No world calibration in version one

Version one has no consent-cleared general-population control set. It therefore cannot estimate false-match rates, group membership probabilities, or population percentiles. A separate control-set experiment requires a new design review.

## 11. Evaluation Protocol

### 11.1 Baselines

Use the same clean manifest and folds:

- B0: image-weighted arithmetic mean;
- B1: equal-subject arithmetic mean;
- B2: equal-subject spherical mean;
- C1: quality-aware spherical geometric median plus robust components.

### 11.2 Subject-disjoint evaluation

Run leave-one-subject-out across every valid subject. For each fold, the held-out subject is excluded from:

- quality normalization;
- duplicate and outlier thresholds;
- center calculation;
- component fitting;
- score calibration.

Tests assert no subject ID appears in both fit and evaluation manifests.

### 11.3 Perturbations

Evaluate deterministic image degradation:

- downscale;
- defocus and motion blur;
- exposure shift and clipping;
- JPEG recompression;
- partial crop;
- detector change with a fixed recognition model.

Perturbations are never included in training.

### 11.4 Primary metrics

- LOSO score median and bootstrap 95% interval;
- worst-decile LOSO score;
- clean-to-perturbed score degradation;
- face-gate acceptance and dropout;
- component stability under subject bootstrap;
- delete-one-subject center influence;
- score rank stability;
- runtime and peak GPU memory.

### 11.5 Promotion gate

C1 replaces B1 only if:

- its predefined LOSO stability metric improves with a positive bootstrap lower bound;
- worst-decile performance stays within tolerance;
- perturbation degradation does not regress materially;
- detector coverage does not regress materially;
- all component and model contracts validate;
- results reproduce from the same clean manifest and commit.

If these conditions fail, retain the simpler equal-subject baseline.

## 12. Fictional Third-Candidate Search

Face embeddings are not directly invertible. Candidate discovery therefore uses a generate-measure-select loop rather than pretending to decode the 512-dimensional center.

### 12.1 Search target

Do not maximize the approximation score to 100. That would favor an overly average, potentially bland center. Search for candidates that:

- fall initially in the 70 to 90 Seju approximation band;
- remain inside the stable component support;
- occupy different component-space regions from other fictional candidates;
- pass face, image, and provenance QA;
- contain no named-person or official-brand prompt reference.

The band is configuration, not a scientific constant, and may be revised after blinded review.

### 12.2 Generation controls

Use the interpretable geometry intervals and aggregate presentation constraints to create prompt or local-generator variations. Never translate a component into a beauty, ethnicity, personality, or named-person instruction.

Every generated candidate records:

```text
candidate_id
generator
generator_model
seed
prompt_hash
model_contract_hash
source_component_target
synthetic_label
generated_at
```

Generated images remain local and visibly marked as synthetic during review.

### 12.3 Candidate frontier

Keep a Pareto frontier rather than one opaque combined score:

- Seju approximation;
- component-support confidence;
- image and detector QA;
- diversity from other fictional candidates;
- blinded preference score.

No candidate is selected solely because its Seju approximation is highest.

## 13. Blinded Preference Evaluation

The project cannot infer popularity from a face vector. It can measure how a defined review panel responds to fictional candidates.

Review presentation:

- randomize candidate order;
- hide generator, seed, approximation score, and component target;
- show the same crop, resolution, background, and synthetic label;
- ask separate questions: `印象に残る`, `応援したい`, `SNSで見たい`;
- allow `同程度` and abstention;
- never show real-person comparison images in the voting screen.

Aggregate pairwise responses with a versioned Bradley-Terry-style model or win-rate baseline. Report reviewer count, comparison count, uncertainty, and panel scope. A single-operator review is a personal preference result, not a public popularity claim.

Preference evidence and Seju approximation remain separate columns and separate conclusions.

## 14. Proposed Code Boundaries

New focused modules:

```text
src/seju_face_lab/data_gate.py
src/seju_face_lab/model_contract.py
src/seju_face_lab/face_observations.py
src/seju_face_lab/geometry_vectors.py
src/seju_face_lab/robust_templates.py
src/seju_face_lab/component_model.py
src/seju_face_lab/approximation_score.py
src/seju_face_lab/candidate_search.py
src/seju_face_lab/preference_review.py
```

Existing generation, SNS, agency, Discord, and site code is not removed in this slice. It is disconnected from the new promoted research pipeline until separately reviewed.

Proposed CLI surface:

```text
audit-face-dataset
build-face-observations
build-vector-model
decompose-vectors
evaluate-vector-model
score-face
plan-fictional-candidates
review-fictional-candidates
```

The first delivery ends at `evaluate-vector-model`. Candidate generation starts only after the score model passes its promotion gate.

## 15. Test Strategy

### 15.1 Unit tests

- reject signature URLs before vectorization;
- reject non-allowlisted hosts and manifest hash mismatches;
- reject zero-face and multi-face images;
- record detector evidence and rejection reason;
- deterministically cluster resized duplicates within a subject;
- keep the highest-quality duplicate representative;
- reject a subject with fewer than three independent portraits;
- refuse same-dimension vectors with different model contracts;
- enforce unit normalization;
- verify equal-subject weighting;
- verify spherical median robustness to an outlier;
- prevent LOSO subject leakage;
- align bootstrap component signs and ordering;
- reproduce score calibration under a fixed seed;
- verify score semantics and out-of-support flags;
- ensure quality changes confidence but not the score formula;
- keep preference fields out of approximation scoring;
- keep approximation fields hidden during blinded review.

### 15.2 Integration tests

- invented source manifest -> clean manifest -> observations;
- fixture faces -> subject templates -> components -> score;
- baseline and candidate algorithms consume identical fold manifests;
- contract mismatch stops before scoring;
- fictional candidate manifest -> score/QA -> blinded review bundle;
- public-safe aggregate report contains no raw vectors, names, or per-person rankings.

### 15.3 Real-path verification on `rtx4090`

1. rerun the 259-row source and SHA-256 audit;
2. confirm 34 signature exclusions and classify every other failure;
3. calculate perceptual duplicate clusters, especially the 33 thumbnails;
4. confirm final independent-image and subject counts;
5. build B0, B1, B2, and C1 with InsightFace CUDA;
6. execute all valid LOSO folds and perturbations;
7. save confidence intervals, component stability, runtime, and GPU memory;
8. select or reject C1 using the predefined gate;
9. only after promotion, generate a small fictional candidate batch;
10. run blinded operator review without exposing scores.

## 16. Acceptance Criteria

The vector-algorithm slice is complete only when:

- every included file has Seju-source provenance and matching SHA-256;
- signatures, non-faces, multi-faces, and near-duplicate extras are excluded with recorded reasons;
- the final dataset has at least 20 subjects and three independent portraits per subject;
- every embedding has a complete compatible model contract;
- neural and geometry representations are separately versioned;
- LOSO manifests prove subject disjointness;
- component stability and score uncertainty are reported;
- approximation-score semantics are tested and shown to users;
- no identity, attractiveness, popularity, demographic, or population claim is emitted;
- the promoted method beats or cleanly defers to the equal-subject baseline under the predefined gate;
- deterministic tests, full unit tests, lint, compilation, CUDA smoke, and real LOSO evidence are recorded separately.

The fictional-candidate slice is complete only when:

- the vector model has already passed promotion;
- candidates are synthetic, locally stored, and provenance-tracked;
- candidates occupy the configured score band and pass component-support and image QA;
- candidate diversity is measured only among fictional candidates;
- preference review is blinded and separate from approximation scoring;
- results are reported as panel preference, never inherent popularity.

## 17. Delivery Order

1. source/content gate and clean manifest;
2. model contract and detector observations;
3. geometry vectors and robust subject templates;
4. tangent-space component model;
5. LOSO-calibrated approximation score;
6. baseline, perturbation, and CUDA evaluation;
7. promotion decision;
8. fictional candidate planner and small generation loop;
9. blinded preference review.

Implementation starts only after this written specification is reviewed and approved.
