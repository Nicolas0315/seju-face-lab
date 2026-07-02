# Consented Subject Intake

Purpose: prepare `data/subjects/` for real, consent-cleared local review without committing private
images, vectors, generated portraits, or identity claims.

## Boundary

- Use only images where the operator has rights and explicit consent for local analysis.
- Keep image binaries under `data/subjects/`; this directory is Git-ignored.
- Do not store personal names, contact details, IDs, consent forms, or image URLs in this repo.
- Use pseudonymous subject slugs such as `subject_001`.
- Do not use the output for identity recognition, attractiveness scoring, demographic labeling,
  person matching, or claims about a real population.
- Treat every vector/report as an approximation of the provided local image set only.

## Folder Shape

```text
data/subjects/
  subject_001/
    img_001.jpg
    img_002.jpg
  subject_002/
    img_001.jpg
```

Recommended minimum per subject:

- 3+ images when available.
- Similar crop/framing across subjects.
- Exclude screenshots, watermarked images, duplicate near-identical frames, and images with multiple
  visible faces unless the analysis goal explicitly allows manual crop review.

## Intake Manifest

Copy `configs/subject_intake.example.json` to a local ignored path before filling it, for example:

```powershell
Copy-Item configs\subject_intake.example.json data\processed\subject_intake.local.json
```

Keep the filled manifest local. The manifest should record only safe metadata:

- pseudonymous subject slug
- consent record reference stored outside this repo
- local folder path
- allowed local uses
- image count
- review date
- retention review date

Do not add:

- real names
- email, phone, address, account IDs, or government IDs
- raw consent text or signatures
- public image URLs
- image hashes if those hashes are treated as personal data in the operating context

## Verification Commands

After placing images locally:

```powershell
python -m seju_face_lab vectorize-subjects --subjects data/subjects --out outputs/subject_vectors
python -m seju_face_lab review-subjects --model outputs/seju_model --subjects data/subjects --out outputs/subject_reviews
python -m seju_face_lab compare-subject-backends --reference-images data/raw/seju_official --subjects data/subjects --out outputs/subject_backend_compare --backends deterministic opencv-face
```

For repo-safe verification without private images:

```powershell
ruff check .
python -m unittest discover -s tests
```

## Operator Gate

The task is complete only after the operator places consent-cleared images under `data/subjects/`,
keeps the completed intake manifest local, runs the verification commands, and stores only summary
evidence that excludes images, vectors, personal data, local URLs, and raw consent records.
