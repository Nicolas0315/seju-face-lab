# Development Runbook / 開発・研究の進め方

**EN**: Use this runbook to set up a clean environment, run the pipeline, and keep local research
artifacts out of Git.

**JA**: このメモは、クリーンな環境構築、研究パイプライン実行、ローカル成果物のGit除外を
同じ手順で進めるためのものです。

## 1. Workspace / 作業ディレクトリ

```powershell
cd seju-face-lab
git status --short --branch
```

Local images and generated outputs are not included in the repository. Prepare your own consent-cleared data.

ローカル画像と生成物はリポジトリに含まれません。許諾済みデータを各自で準備してください。

## 2. Environment / 環境構築

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e . ruff
```

Optional GPU, face-recognition, style, or generation dependencies:

GPU、顔認識、スタイル、生成系の任意依存:

```powershell
python -m pip install -e ".[vision]"
python -m pip install -e ".[deepface]"
python -m pip install -e ".[face]"
python -m pip install -e ".[clip]"
python -m pip install -e ".[generation]"
```

For OpenAI Images API, install `.[openai]` and provide `OPENAI_API_KEY` through the environment or
1Password. Do not write keys into files, docs, prompts, or logs.

OpenAI Images API を使う場合は `.[openai]` を入れ、APIキーは環境変数か1Password参照で扱います。
キーをファイル、ドキュメント、プロンプト、ログに書かないでください。

## 3. Smoke Checks / 最小スモーク

These checks work without private images.

画像なしでも確認できます。

```powershell
python -m seju_face_lab backends
ruff check .
python -m unittest discover -s tests
```

With consented reference images:

許諾済み画像がある場合:

```powershell
python -m seju_face_lab build --images data/raw --out outputs/seju_model
python -m seju_face_lab audit-model --model outputs/seju_model --out outputs/model_audit
python -m seju_face_lab ingredients-report --model outputs/seju_model --out outputs/face_ingredients
```

## 4. Research Pipeline / 研究パイプライン

Agency-level average-face parameters:

事務所別平均顔パラメータ:

```powershell
python -m seju_face_lab review-agencies --model outputs/seju_model --agencies configs/agencies/seju_like_agencies.json --out outputs/agency_reviews/seju_like
```

8-axis evaluation for generated or candidate images:

生成サンプルや候補画像の8軸評価:

```powershell
python -m seju_face_lab face-axes --images outputs/generated --out outputs/face_axes
python -m seju_face_lab evaluate --model outputs/seju_model --images outputs/generated --out outputs/evaluation
```

Configured pipeline:

一括実行:

```powershell
python -m seju_face_lab run-pipeline --config configs/pipelines/full-local-review.example.json --out outputs/local_pipeline_run
```

## 5. Labeling Policy / ラベル設計

Describe image state, not personal value.

人の価値ではなく、画像状態を記録します。

- OK: `underlit`, `high_contrast`, `occluded`, `off_center`, `hair_shadow`, `strong_styling`
- NG: `ugly`, `dirty`, `unclean`, `low value`, attractiveness rank

This policy is implemented in `face-axes` presentation flags and documented in
`docs/agency-research-flow.md`.

この方針は `face-axes` の presentation flags と `docs/agency-research-flow.md` に反映されています。

## 6. Keeping the Repo Clean / リポジトリをクリーンに保つ

Before and after work:

作業前後:

```powershell
git status --short --branch
git diff --check
```

Check ignored local artifacts without deleting them:

削除せずにローカル成果物を確認:

```powershell
git clean -ndX
```

`data/` and `outputs/` can contain non-recoverable local artifacts. Confirm before deleting them.

`data/` と `outputs/` には再取得できないローカル成果物が入ることがあります。削除前に確認してください。
