# seju-face-lab

**EN**: Local research pipeline for building an approximate aggregate "seju-face" profile from
consented image sets, then evaluating generated or comparison images against that local centroid.

**JA**: 許諾済みの顔画像セットから「seju顔」風のローカルな平均特徴を作り、生成画像や比較対象画像との
近似度を検証する研究用パイプラインです。

This project does **not** identify people, score attractiveness, or judge personal value.
Scores only mean similarity to a centroid built from the images you provide.

このプロジェクトは、本人識別・魅力度判定・人格評価を目的にしていません。出力スコアは
「投入した画像セットから作った重心にどれくらい近いか」を示す実験値です。

## Contents / 目次

- [What It Does / できること](#what-it-does--できること)
- [Safety Boundary / 重要な境界](#safety-boundary--重要な境界)
- [Setup / セットアップ](#setup--セットアップ)
- [Quick Start / 最短実行](#quick-start--最短実行)
- [Common Commands / よく使うコマンド](#common-commands--よく使うコマンド)
- [Repository Layout / ディレクトリ構成](#repository-layout--ディレクトリ構成)
- [Development / 開発](#development--開発)
- [Docs / 関連ドキュメント](#docs--関連ドキュメント)

## What It Does / できること

**EN**

- Build mean and median face-vector centroids from a local image set.
- Render approximate mean/median face images and generation prompts.
- Decompose aggregate features into face parts, color tone, makeup texture, and hair signals.
- Score generated or candidate images against the local centroid.
- Map images onto 4+4 visual axes: quadrant, corner, cross-axis, outlier score, and presentation flags.
- Build agency-level average-face parameter hypotheses and Image Gen prompts.
- Extend validation with optional DeepFace, InsightFace, OpenCV, OpenCLIP, Diffusers, and OpenAI Images backends.

**JA**

- ローカル画像セットから平均・中央値の顔ベクトル重心を作る
- 平均顔・中央値顔の近似画像と生成プロンプトを出力する
- 顔パーツ、色味、メイク質感、髪まわりの集約特徴を分解する
- 生成画像や候補画像をローカル重心に対してスコアリングする
- 4+4の8軸で、象限・四隅・十字軸・外れ値・画像状態フラグを出す
- 事務所別の平均顔パラメータ仮説と Image Gen 用プロンプトを作る
- DeepFace、InsightFace、OpenCV、OpenCLIP、Diffusers、OpenAI Images で検証を拡張する

## Safety Boundary / 重要な境界

**EN**

- Use only images you have rights and consent to analyze.
- Do not commit private images, generated portraits, embeddings, SNS data, or model outputs.
- Do not generate or request a specific real person's likeness.
- Do not use insulting labels such as ugly, dirty, or unclean. Record neutral image-state flags instead:
  underlit, occluded, high contrast, strong hair shadow, off-center, high texture, strong styling.
- Do not merge face vectors, style vectors, SNS metrics, and iris templates into one identity-like score.

**JA**

- 解析する画像は、権利・利用許諾・同意を確認したものだけを使ってください。
- 個人画像、生成画像、埋め込み、SNSデータ、モデル出力をコミットしないでください。
- 実在人物の似顔コピーや、特定個人の再現を目的にした生成は扱いません。
- 「ブス」「不潔」など人を傷つけるラベルは使いません。暗い、顔が隠れている、コントラストが強い、
  髪影が強い、中心からずれている、質感が強い、スタイリングが強い、といった中立的な画像状態として扱います。
- 顔ベクトル、スタイルベクトル、SNS指標、虹彩テンプレートを、単一の人物評価スコアに混ぜません。

## Setup / セットアップ

PowerShell:

```powershell
git clone https://github.com/Nicolas0315/seju-face-lab.git
cd seju-face-lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

Development tools / 開発ツール:

```powershell
python -m pip install -e . ruff
```

Optional extras / 任意バックエンド:

```powershell
python -m pip install -e ".[vision]"      # OpenCV face crop / OpenCV 顔クロップ
python -m pip install -e ".[deepface]"    # DeepFace / ArcFace
python -m pip install -e ".[face]"        # InsightFace + ONNXRuntime GPU
python -m pip install -e ".[clip]"        # OpenCLIP style axis / スタイル軸
python -m pip install -e ".[generation]"  # Diffusers local generation / ローカル生成
python -m pip install -e ".[openai]"      # OpenAI Images API
```

## Quick Start / 最短実行

Place consented reference images under `data/raw/`.

許諾済みの参照画像を `data/raw/` に置きます。

```text
data/raw/
  sample_001.jpg
  sample_002.jpg
```

Build a centroid model / 平均・中央値ベクトルを作成:

```powershell
python -m seju_face_lab build --images data/raw --out outputs/seju_model
```

Main outputs / 主な出力:

```text
outputs/seju_model/centroids.npz
outputs/seju_model/profile.json
outputs/seju_model/mean_face.png
outputs/seju_model/median_face.png
outputs/seju_model/prompt.txt
outputs/seju_model/report.md
```

Audit and decompose / 監査と要素分解:

```powershell
python -m seju_face_lab audit-model --model outputs/seju_model --out outputs/model_audit
python -m seju_face_lab ingredients-report --model outputs/seju_model --out outputs/face_ingredients
```

Evaluate generated or candidate images / 生成画像・候補画像を評価:

```powershell
python -m seju_face_lab evaluate --model outputs/seju_model --images outputs/generated --out outputs/evaluation
```

Map images onto 8 visual axes / 8軸の象限・外れ値・画像状態フラグ:

```powershell
python -m seju_face_lab face-axes --images outputs/generated --out outputs/face_axes
```

Build agency-level average-face parameters / 事務所別平均顔パラメータ仮説:

```powershell
python -m seju_face_lab review-agencies --model outputs/seju_model --agencies configs/agencies/seju_like_agencies.json --out outputs/agency_reviews/seju_like
```

## Common Commands / よく使うコマンド

Backend visibility / バックエンド確認:

```powershell
python -m seju_face_lab backends
python -m seju_face_lab backend-diagnostics --out outputs/backend_diagnostics
```

Generation dry-run / 生成計画だけを作る:

```powershell
python -m seju_face_lab generate --model outputs/seju_model --out outputs/generated --provider dry-run --count 8
```

Diffusers generation / Diffusers 生成:

```powershell
python -m seju_face_lab generate --model outputs/seju_model --out outputs/generated --provider diffusers --hf-model runwayml/stable-diffusion-v1-5 --count 8 --negative-prompt "copied identity"
```

OpenAI Images API generation / OpenAI Images API 生成:

```powershell
$env:OPENAI_API_KEY="op://... or an exported environment variable"
python -m seju_face_lab generate --model outputs/seju_model --out outputs/generated_openai --provider openai-image --image-model gpt-image-2 --width 1024 --height 1024 --quality medium --count 4 --review
```

Generated-image QA and review / 生成画像QAとレビュー:

```powershell
python -m seju_face_lab qa-images --images outputs/generated --out outputs/generated/quality
python -m seju_face_lab review-generated --model outputs/seju_model --images outputs/generated --out outputs/generated/review
```

Subject-folder review / 人物フォルダ別レビュー:

```text
data/subjects/
  subject_a/
    image1.jpg
  subject_b/
    image1.jpg
```

```powershell
python -m seju_face_lab review-subjects --model outputs/seju_model --subjects data/subjects --out outputs/subject_reviews
```

Run a JSON pipeline / JSON設定で一括実行:

```powershell
python -m seju_face_lab run-pipeline --config configs/pipelines/full-local-review.example.json --out outputs/local_pipeline_run
```

## Repository Layout / ディレクトリ構成

- `configs/`: reproducible source, agency, and pipeline configs / ソース・事務所・パイプライン設定
- `data/`: local data only; private and ignored / ローカルデータ置き場、Git管理外
- `docs/`: design notes, research logs, runbooks / 設計、研究ログ、運用メモ
- `outputs/`: models, generated images, evaluations; ignored / モデル、生成画像、評価結果、Git管理外
- `scripts/`: helper scripts / 補助スクリプト
- `src/seju_face_lab/`: package code / パッケージ本体
- `tests/`: deterministic unit tests / 決定論的ユニットテスト

## Development / 開発

Minimum checks / 最小検証:

```powershell
ruff check .
python -m compileall -q src tests scripts
python -m unittest discover -s tests
git diff --check
```

GitHub Actions runs the same class of checks. See [`CONTRIBUTING.md`](CONTRIBUTING.md)
and [`docs/development.md`](docs/development.md) for the contributor workflow.

GitHub Actions でも同系統の検証を走らせます。開発手順は
[`CONTRIBUTING.md`](CONTRIBUTING.md) と [`docs/development.md`](docs/development.md) を見てください。

## Docs / 関連ドキュメント

- [`docs/architecture.md`](docs/architecture.md): pipeline and backend design / パイプラインとバックエンド設計
- [`docs/agency-research-flow.md`](docs/agency-research-flow.md): agency average-face research flow / 事務所別平均顔研究フロー
- [`docs/research-tracking.md`](docs/research-tracking.md): issues, TODO, evidence / 研究Issue、ToDo、検証ログ
- [`docs/web-source-strategy.md`](docs/web-source-strategy.md): web source boundaries / Webソース収集境界
- [`docs/gpu-generation-log.md`](docs/gpu-generation-log.md): GPU generation logs / GPU生成・評価ログ

## License and Data / ライセンスとデータ

This repository is for code, configs, and documentation. Keep images, generated portraits,
vectors, SNS artifacts, credentials, and private notes local.

このリポジトリではコード、設定、ドキュメントだけを共有します。画像、生成物、ベクトル、
SNS収集結果、認証情報、私的メモはローカルに保持してください。
