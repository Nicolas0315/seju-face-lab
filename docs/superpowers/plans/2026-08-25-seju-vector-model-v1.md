# Seju Vector Model v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Seju公式画像の出自・顔・近重複を検査し、人物分離LOSOで校正したSeju近似スコアと、promotion後だけ利用できる架空候補レビュー基盤を構築する。

**Architecture:** 既存の収集・InsightFace backendを再利用し、新しい研究パイプラインを独立モジュールとして追加する。各段階はJSON/JSONLの不変証跡とモデル契約hashを受け渡し、契約不一致、人物漏洩、証拠不足をfail closedにする。raw画像、ベクトル、生成画像はローカル出力のままGitへ入れない。

**Tech Stack:** Python 3.12、NumPy、Pillow、標準`unittest`、任意のInsightFace/ONNX Runtime CUDA。

**Spec:** `docs/superpowers/specs/2026-08-25-seju-vector-decomposition-candidate-search-design.md`（日本語レビュー: `docs/seju-vector-system-overview-ja.md`）

## Global Constraints

- 本人識別、nearest-person、魅力度、人格、人口統計、顔からの人気予測を実装しない。
- `Seju近似スコア`はcleanなローカルSeju snapshot内のLOSO百分位であり、所属確率ではない。
- 学習可能条件は20人物以上、各人物3独立portrait cluster以上。
- neural vectorとgeometry vectorを連結せず、別schema・別保存にする。
- 同じ次元でもmodel contract hashが異なるvectorを混合しない。
- すべてのfoldでheld-out subjectを品質正規化、外れ値閾値、中心、成分、校正から除外する。
- 候補探索はpromoted modelだけを受け付け、近似・support・QA・多様性・選好を単一スコアへ潰さない。

---

### Task 1: 日本語仕様と図の検証可能化

**Files:**
- Modify: `docs/seju-vector-system-overview-ja.md`
- Modify: `docs/superpowers/specs/2026-08-25-seju-vector-decomposition-candidate-search-design.md`
- Create: `scripts/verify_mermaid_blocks.py`
- Test: `tests/test_vector_model_v1.py`

**Interfaces:**
- Consumes: Markdown内の fenced `mermaid` block。
- Produces: `extract_mermaid_blocks(path: Path, out_dir: Path) -> list[Path]`。

- [ ] **Step 1: 失敗テストを書く**

```python
def test_extract_mermaid_blocks_rejects_unclosed_block(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "bad.md"
        source.write_text("```mermaid\nflowchart TD\nA-->B\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unclosed mermaid block"):
            extract_mermaid_blocks(source, root / "out")
```

- [ ] **Step 2: REDを確認する**

Run: `python -m unittest tests.test_vector_model_v1.VectorModelV1Tests.test_extract_mermaid_blocks_rejects_unclosed_block -v`  
Expected: import failure because `scripts.verify_mermaid_blocks` does not exist.

- [ ] **Step 3: 最小実装を書く**

`scripts/verify_mermaid_blocks.py`にMarkdown fenceを順序どおり抽出し、未閉鎖、空block、0 blockを`ValueError`にする実装とCLIを追加する。出力名は`diagram-01.mmd`からの連番とする。

- [ ] **Step 4: GREENと実資料を確認する**

Run: `python -m unittest tests.test_vector_model_v1.VectorModelV1Tests.test_extract_mermaid_blocks_rejects_unclosed_block -v`  
Run: `python scripts/verify_mermaid_blocks.py docs/seju-vector-system-overview-ja.md --out .tmp/mermaid`  
Expected: test PASS、7個以上の`.mmd`を抽出、exit 0。

- [ ] **Step 5: コミットする**

```bash
git add docs scripts/verify_mermaid_blocks.py tests/test_vector_model_v1.py
git commit -m "docs: add Japanese vector system diagrams"
```

### Task 2: モデル契約と互換性ゲート

**Files:**
- Create: `src/seju_face_lab/model_contract.py`
- Modify: `tests/test_vector_model_v1.py`

**Interfaces:**
- Produces: `ModelContract.from_mapping(value)`, `ModelContract.sha256()`, `assert_compatible(left, right)`。
- Contract fields are the exact fields listed in specification section 6.

- [ ] **Step 1: 失敗テストを書く**

```python
def test_same_dimension_different_model_hash_is_incompatible(self) -> None:
    left = ModelContract.from_mapping(contract_fixture(recognition_model_sha256="a" * 64))
    right = ModelContract.from_mapping(contract_fixture(recognition_model_sha256="b" * 64))
    with self.assertRaisesRegex(ValueError, "model contract mismatch"):
        assert_compatible(left, right)
```

- [ ] **Step 2: REDを確認する**

Run: `python -m unittest tests.test_vector_model_v1.VectorModelV1Tests.test_same_dimension_different_model_hash_is_incompatible -v`  
Expected: import failure because `model_contract` is absent.

- [ ] **Step 3: 最小実装を書く**

Frozen dataclassで全必須fieldを検証し、canonical JSON（UTF-8、key sort、compact separator）のSHA-256を返す。未知fieldと欠損fieldを拒否し、hash一致以外は互換とみなさない。

- [ ] **Step 4: GREENを確認する**

Run: `python -m unittest tests.test_vector_model_v1 -v`  
Expected: all current Task 1-2 tests PASS.

- [ ] **Step 5: コミットする**

```bash
git add src/seju_face_lab/model_contract.py tests/test_vector_model_v1.py
git commit -m "feat: enforce vector model contracts"
```

### Task 3: 出自・asset・近重複データゲート

**Files:**
- Create: `src/seju_face_lab/data_gate.py`
- Modify: `src/seju_face_lab/cli.py`
- Modify: `tests/test_vector_model_v1.py`

**Interfaces:**
- Produces: `audit_face_dataset(source_manifest: Path, download_manifest: Path, image_root: Path, out_dir: Path) -> DatasetAudit`。
- Produces CLI: `audit-face-dataset --source-manifest --download-manifest --images --out`。
- Output: `clean_manifest.jsonl`, `rejected_manifest.jsonl`, `dataset_audit.json`。

- [ ] **Step 1: signatureとhost/hash mismatchの失敗テストを書く**

```python
def test_dataset_gate_rejects_signature_before_face_detection(self) -> None:
    audit = audit_face_dataset_from_rows(
        [download_row(url="https://seju.tokyo/x/signature.png", sha256=fixture_hash())],
        allowed_hosts={"seju.tokyo"},
    )
    self.assertEqual(audit.rejected[0].reason, "signature_asset")
    self.assertEqual(audit.clean, [])
```

- [ ] **Step 2: REDを確認する**

Run: `python -m unittest tests.test_vector_model_v1.VectorModelV1Tests.test_dataset_gate_rejects_signature_before_face_detection -v`  
Expected: import failure because `data_gate` is absent.

- [ ] **Step 3: 最小実装を書く**

URL host完全一致、eligible、local path、SHA-256、signature/navigation/logo/icon分類を順に検査する。人物内だけで64-bit perceptual hashと整列cropのSSIMを用いた近重複clusterを作り、品質tupleが最大の代表を残す純粋関数を追加する。cross-subject比較は実装しない。

- [ ] **Step 4: GREENとCLI統合を確認する**

Run: `python -m unittest tests.test_vector_model_v1 -v`  
Expected: signature、host、hash、近重複代表選択の各test PASS。

- [ ] **Step 5: コミットする**

```bash
git add src/seju_face_lab/data_gate.py src/seju_face_lab/cli.py tests/test_vector_model_v1.py
git commit -m "feat: add fail-closed face dataset gate"
```

### Task 4: 顔観測と106点geometry vector

**Files:**
- Create: `src/seju_face_lab/face_observations.py`
- Create: `src/seju_face_lab/geometry_vectors.py`
- Modify: `src/seju_face_lab/backends.py`
- Modify: `src/seju_face_lab/cli.py`
- Modify: `tests/test_vector_model_v1.py`

**Interfaces:**
- Produces: `FaceObservation`, `extract_face_observation(path, extractor, contract)`。
- Produces: `normalize_landmarks_106(points: np.ndarray) -> np.ndarray` and `geometry_vector(points) -> GeometryVector`。
- Produces CLI: `build-face-observations --manifest --images --contract --out`。

- [ ] **Step 1: zero/multi-face fail-closedと幾何不変性の失敗テストを書く**

```python
def test_face_observation_requires_exactly_one_face(self) -> None:
    with self.assertRaisesRegex(ValueError, "expected exactly one face"):
        extract_face_observation(Path("fixture.png"), FakeExtractor(faces=[]), contract_fixture())

def test_geometry_vector_is_translation_and_scale_invariant(self) -> None:
    points = landmark_fixture_106()
    actual = geometry_vector(points * 3.0 + np.array([90.0, -40.0])).values
    np.testing.assert_allclose(actual, geometry_vector(points).values, atol=1e-8)
```

- [ ] **Step 2: REDを確認する**

Run: `python -m unittest tests.test_vector_model_v1.VectorModelV1Tests.test_face_observation_requires_exactly_one_face tests.test_vector_model_v1.VectorModelV1Tests.test_geometry_vector_is_translation_and_scale_invariant -v`  
Expected: missing module imports.

- [ ] **Step 3: 最小実装を書く**

InsightFace adapterは`face.embedding`、`face.kps`、`face.landmark_2d_106`を読み、1顔以外を拒否する。geometryは106点を重心移動、inter-ocular相当のrobust scaleで正規化し、正規化shape 212値と左右反転Procrustes residualをversion付きで保存する。5点は整列専用に保持する。

- [ ] **Step 4: GREENを確認する**

Run: `python -m unittest tests.test_vector_model_v1 -v`  
Expected: face count、unit norm、106点必須、平行移動・scale不変test PASS。

- [ ] **Step 5: コミットする**

```bash
git add src/seju_face_lab/face_observations.py src/seju_face_lab/geometry_vectors.py src/seju_face_lab/backends.py src/seju_face_lab/cli.py tests/test_vector_model_v1.py
git commit -m "feat: record dual face observations"
```

### Task 5: robust人物テンプレートとtangent component model

**Files:**
- Create: `src/seju_face_lab/robust_templates.py`
- Create: `src/seju_face_lab/component_model.py`
- Modify: `src/seju_face_lab/cli.py`
- Modify: `tests/test_vector_model_v1.py`

**Interfaces:**
- Produces: `spherical_geometric_median(vectors, weights)`, `build_subject_template(...)`。
- Produces: `fit_component_model(subject_templates, max_components=8, variance_threshold=0.9)` and `project_component(model, vector)`。
- Produces CLI: `build-vector-model`, `decompose-vectors`。

- [ ] **Step 1: 人物均等と外れ値耐性の失敗テストを書く**

```python
def test_global_center_weights_subjects_equally(self) -> None:
    many_images = {"a": repeated_vectors([1.0, 0.0], 20), "b": repeated_vectors([0.0, 1.0], 3)}
    center = build_global_center(build_templates(many_images))
    np.testing.assert_allclose(center, unit([1.0, 1.0]), atol=1e-6)
```

- [ ] **Step 2: REDを確認する**

Run: `python -m unittest tests.test_vector_model_v1.VectorModelV1Tests.test_global_center_weights_subjects_equally -v`  
Expected: missing robust template implementation.

- [ ] **Step 3: 最小実装を書く**

Weiszfeld型の正規化反復で球面幾何中央値を求め、決定論的角度外れ値passを一度だけ適用する。中心`mu`へのlog map、NumPy SVD、variance thresholdと上限8による成分選択、bootstrap sign alignmentとstabilityを実装する。

- [ ] **Step 4: GREENを確認する**

Run: `python -m unittest tests.test_vector_model_v1 -v`  
Expected: equal-subject、outlier、component count、sign alignment test PASS。

- [ ] **Step 5: コミットする**

```bash
git add src/seju_face_lab/robust_templates.py src/seju_face_lab/component_model.py src/seju_face_lab/cli.py tests/test_vector_model_v1.py
git commit -m "feat: build robust subject component model"
```

### Task 6: LOSO校正Seju近似スコアとpromotion

**Files:**
- Create: `src/seju_face_lab/approximation_score.py`
- Create: `src/seju_face_lab/vector_evaluation.py`
- Modify: `src/seju_face_lab/cli.py`
- Modify: `tests/test_vector_model_v1.py`

**Interfaces:**
- Produces: `score_vector(vector, model, calibration) -> ApproximationScore`。
- Produces: `build_loso_folds(subject_ids)`, `evaluate_vector_model(dataset, seed) -> EvaluationReport`。
- Produces CLI: `evaluate-vector-model`, `score-face`。

- [ ] **Step 1: score意味とLOSO漏洩防止の失敗テストを書く**

```python
def test_loso_fold_never_contains_held_out_subject_in_fit(self) -> None:
    for fold in build_loso_folds(["a", "b", "c"]):
        self.assertNotIn(fold.held_out_subject_id, fold.fit_subject_ids)

def test_approximation_is_harmonic_mean_of_two_percentiles(self) -> None:
    result = combine_percentiles(center_percentile=80.0, manifold_percentile=20.0)
    self.assertEqual(result, 32.0)
```

- [ ] **Step 2: REDを確認する**

Run: `python -m unittest tests.test_vector_model_v1.VectorModelV1Tests.test_loso_fold_never_contains_held_out_subject_in_fit tests.test_vector_model_v1.VectorModelV1Tests.test_approximation_is_harmonic_mean_of_two_percentiles -v`  
Expected: missing evaluation/scoring implementation.

- [ ] **Step 3: 最小実装を書く**

foldごとに品質統計、中心、成分をfit subjectだけから再計算し、held-out scoreを保存する。ECDFは`<=`の経験順位、manifoldは残差の反転順位、両者を調和平均にする。support interval、bootstrap CI、B0/B1/B2/C1、worst-decile、摂動差、promotion理由を別fieldで保存する。

- [ ] **Step 4: GREENを確認する**

Run: `python -m unittest tests.test_vector_model_v1 -v`  
Expected: leakage、ECDF、harmonic mean、support、deterministic seed、promotion/defer test PASS。

- [ ] **Step 5: コミットする**

```bash
git add src/seju_face_lab/approximation_score.py src/seju_face_lab/vector_evaluation.py src/seju_face_lab/cli.py tests/test_vector_model_v1.py
git commit -m "feat: calibrate Seju approximation with LOSO"
```

### Task 7: promotion限定の架空候補とブラインド選好

**Files:**
- Create: `src/seju_face_lab/candidate_search.py`
- Create: `src/seju_face_lab/preference_review.py`
- Modify: `src/seju_face_lab/cli.py`
- Modify: `tests/test_vector_model_v1.py`

**Interfaces:**
- Produces: `plan_candidates(promoted_model, measured_candidates, score_band=(70, 90)) -> CandidateFrontier`。
- Produces: `make_blind_bundle(candidates, seed)`, `aggregate_pairwise(responses)`。
- Produces CLI: `plan-fictional-candidates`, `review-fictional-candidates`。

- [ ] **Step 1: promotionとblind leakageの失敗テストを書く**

```python
def test_candidate_planner_rejects_unpromoted_model(self) -> None:
    with self.assertRaisesRegex(ValueError, "promoted model required"):
        plan_candidates({"promotion": "diagnostic"}, [])

def test_blind_bundle_hides_score_seed_and_generator(self) -> None:
    bundle = make_blind_bundle([candidate_fixture()], seed=7)
    serialized = json.dumps(bundle)
    for forbidden in ("approximation_score", "generator", "seed", "component_target"):
        self.assertNotIn(forbidden, serialized)
```

- [ ] **Step 2: REDを確認する**

Run: `python -m unittest tests.test_vector_model_v1.VectorModelV1Tests.test_candidate_planner_rejects_unpromoted_model tests.test_vector_model_v1.VectorModelV1Tests.test_blind_bundle_hides_score_seed_and_generator -v`  
Expected: missing candidate/review implementation.

- [ ] **Step 3: 最小実装を書く**

score帯、support、QAをhard gateにし、候補間component距離で非支配frontierを作る。blind bundleはopaque review IDと同一表示条件だけを含め、response集計は質問別win/tie/abstainとWilson intervalを返す。近似値を選好集計へ入力しない。

- [ ] **Step 4: GREENを確認する**

Run: `python -m unittest tests.test_vector_model_v1 -v`  
Expected: promotion、frontier、blind fields、質問分離、panel scope test PASS。

- [ ] **Step 5: コミットする**

```bash
git add src/seju_face_lab/candidate_search.py src/seju_face_lab/preference_review.py src/seju_face_lab/cli.py tests/test_vector_model_v1.py
git commit -m "feat: gate fictional candidate review on promotion"
```

### Task 8: 全体検証・RTX4090実データ・安全監査

**Files:**
- Modify: `README.md`
- Modify: `STATE.md`
- Create local only: `outputs/vector_model_v1/**`

**Interfaces:**
- Consumes all previous CLI commands.
- Produces deterministic reports and a documented promoted/deferred/blocked decision.

- [ ] **Step 1: 決定論的全検証を実行する**

Run: `python -m compileall -q src tests scripts`  
Run: `ruff check .`  
Run: `python -m unittest discover -s tests`  
Expected: exit 0 and zero failed tests.

- [ ] **Step 2: 日本語図を検証する**

Run: `python scripts/verify_mermaid_blocks.py docs/seju-vector-system-overview-ja.md --out .tmp/mermaid`  
Run: `Get-ChildItem .tmp/mermaid/*.mmd | ForEach-Object { npx --yes @mermaid-js/mermaid-cli -i $_.FullName -o (Join-Path $_.DirectoryName ($_.BaseName + '.svg')) -b transparent }`  
Expected: all blocks parse; no unclosed or empty diagram.

- [ ] **Step 3: RTX4090で実データgateとCUDA実行を行う**

Run: `python -m seju_face_lab audit-face-dataset --source-manifest C:/Users/ogosh/work/seju-face-lab/data/processed/seju_sources_official_2026-06-14.jsonl --download-manifest C:/Users/ogosh/work/seju-face-lab/data/raw/seju_official/download_manifest.jsonl --images C:/Users/ogosh/work/seju-face-lab/data/raw/seju_official --out C:/Users/ogosh/work/seju-face-lab/outputs/vector_model_v1/dataset_audit`。  
Run: `python -m seju_face_lab build-face-observations --manifest C:/Users/ogosh/work/seju-face-lab/outputs/vector_model_v1/dataset_audit/clean_manifest.jsonl --images C:/Users/ogosh/work/seju-face-lab/data/raw/seju_official --backend insightface --out C:/Users/ogosh/work/seju-face-lab/outputs/vector_model_v1/observations`。  
Run: `python -m seju_face_lab build-vector-model --observations C:/Users/ogosh/work/seju-face-lab/outputs/vector_model_v1/observations --out C:/Users/ogosh/work/seju-face-lab/outputs/vector_model_v1/model`。  
Run: `python -m seju_face_lab evaluate-vector-model --model C:/Users/ogosh/work/seju-face-lab/outputs/vector_model_v1/model --out C:/Users/ogosh/work/seju-face-lab/outputs/vector_model_v1/evaluation --seed 20260825`。  
Record source rows、signature/non-face/multi-face/near-duplicate reasons、valid subjects、independent clusters、CUDA provider、runtime、peak GPU memory、all LOSO folds。

- [ ] **Step 4: promotion結果に従う**

If all minimum-evidence and statistical gates pass, mark the selected algorithm `promoted`. Otherwise mark it `deferred` or `blocked` with exact failed gates; do not generate candidate images when promotion is absent.

- [ ] **Step 5: Git安全監査と最終コミットを行う**

Run: `git status --short`、`git diff --check`、`git ls-files data outputs '*.npy' '*.npz'`。  
Expected: raw画像、vectors、generated portraits、model outputs are untracked and absent from commit.

```bash
git add README.md STATE.md docs src scripts tests
git commit -m "feat: complete Seju vector model v1 validation"
```
