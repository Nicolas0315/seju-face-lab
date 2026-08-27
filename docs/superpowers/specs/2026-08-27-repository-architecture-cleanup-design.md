# seju-face-lab repository architecture cleanup design

Date: 2026-08-27

## Goal

Bring documentation and verification in line with the current implementation,
then reduce the highest-risk coupling around the command-line entry point without
changing scoring behavior or touching local research artifacts.

## Scope

- Update the Japanese system overview so implemented data gates, subject-disjoint
  LOSO evaluation, promotion decisions, and fictional-candidate gates are not
  described as unimplemented.
- Clearly separate implemented-and-unit-tested behavior from real-data,
  multi-backend, GPU, and promotion evidence that remains unverified.
- Add a reproducible development verification entry point covering Ruff,
  unittest, compileall, architecture checks, and diff checks.
- Begin CLI decomposition with one low-risk domain boundary: move SNS command
  parser registration and dispatch wiring behind an internal module while
  preserving public command names and arguments.
- Document SNS access as optional external I/O that is not required by the local
  face-vector model or promotion path.

## Non-goals

- Do not change vector math, thresholds, model contracts, promotion criteria, or
  generated-candidate ranking.
- Do not download images, query SNS services, invoke remote workers, use API keys,
  or run paid generation.
- Do not move or commit files under `data/`, `outputs/`, `.tmp/`, or model caches.
- Do not combine PCC-NH prompt evidence with face embeddings or SNS popularity
  into a single score.
- Do not complete the entire 2,251-line CLI decomposition in one change.

## Architecture boundary

The core remains a local pipeline:

`consented manifest -> data gate -> observation/vector contract -> subject model
-> LOSO/promotion -> synthetic candidate gate -> separate human review`.

Generation providers, remote workers, public-source discovery, and SNS collection
are adapters around that core. SNS metrics remain an optional reporting input and
must not promote a face model or alter the approximation score.

PCC-NH may later supply a versioned prompt/evaluation JSON contract. The two
repositories remain independently testable and share no raw images, vectors,
person identifiers, or implicit Python imports.

## Failure behavior

- Core deterministic verification runs without network, credentials, or GPU.
- Missing Ruff is reported as an environment failure, not silently omitted.
- Optional adapters remain fail-closed and report unavailable/blocked status.
- Architecture documentation labels historical measurements by date and never
  treats unit-test coverage as real-data promotion evidence.

## TDD and verification

1. Add characterization tests for the existing SNS parser surface and dispatch.
2. Observe a focused test fail when it expects the new module boundary.
3. Extract only enough registration/dispatch code to pass while keeping CLI output
   and argument behavior unchanged.
4. Run all current unittests (baseline: 158 passing on 2026-08-27).
5. Run Ruff, compileall, architecture structural review, Mermaid block checks,
   npm audit, and `git diff --check`.
6. Confirm no data, output, cache, or worktree artifact entered the diff.

## Staged follow-up

Later changes may separate generation, worker, and pipeline commands using the
same characterization-test pattern. They are intentionally excluded from this
first cleanup to keep review and rollback bounded.
