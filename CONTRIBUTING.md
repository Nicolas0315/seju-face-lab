# Contributing / 開発ガイド

**EN**: This repository is a research workspace for reproducible face-vector and generated-image
experiments. Keep changes small, tested, and respectful of the data boundary.

**JA**: このリポジトリは、顔ベクトルと生成画像評価を再現可能に進めるための研究作業場です。
変更は小さく、テスト付きで、データ境界を守って進めてください。

## Setup / 初回セットアップ

```powershell
git clone https://github.com/Nicolas0315/seju-face-lab.git
cd seju-face-lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e . ruff
```

Install optional extras only when needed.

任意バックエンドは必要な時だけ入れます。

```powershell
python -m pip install -e ".[vision]"
python -m pip install -e ".[deepface]"
python -m pip install -e ".[face]"
python -m pip install -e ".[clip]"
python -m pip install -e ".[generation]"
python -m pip install -e ".[openai]"
```

## Before You Work / 作業前チェック

```powershell
git status --short --branch
ruff check .
python -m unittest discover -s tests
```

If the worktree has changes, confirm they are yours before editing.

未コミット差分がある場合は、自分の変更かどうかを確認してから編集してください。

## Coding Rules / 実装方針

- Follow the existing CLI, JSON config, and output-file patterns.
- Add or update tests in the same change when behavior changes.
- Do not commit images, generated portraits, embeddings, SNS artifacts, or model outputs.
- Do not add labels that imply identity, attractiveness, personality, ethnicity, or personal value.
- Explain scores as similarity to a local centroid, not as an objective ranking.

- 既存のCLI、JSON設定、出力ファイル形式に合わせる。
- 挙動を変える場合は同じ変更内でテストを追加・更新する。
- 画像、生成画像、埋め込み、SNS成果物、モデル出力をコミットしない。
- 本人識別、魅力度、人格、民族性、個人価値を示すラベルを追加しない。
- スコアは客観ランキングではなく、ローカル重心への近似として説明する。

## Verification / 検証

```powershell
ruff check .
python -m compileall -q src tests scripts
python -m unittest discover -s tests
git diff --check
```

When touching optional backends:

任意バックエンドを触った場合:

```powershell
python -m seju_face_lab backends
python -m seju_face_lab backend-diagnostics --out outputs/backend_diagnostics
```

## Data Handling / データの扱い

- `data/raw/`: consent-cleared reference images / 許諾済み参照画像
- `data/subjects/`: subject folders for local comparison / 比較対象の人物別フォルダ
- `data/processed/`: manifests and intermediate files / マニフェスト・中間データ
- `outputs/`: models, generated images, evaluations / モデル・生成画像・評価結果

These paths are ignored by Git. Share only summaries that do not contain private images or personal data.

これらは Git 管理外です。共有する場合も、個人画像や個人データを含まない要約にしてください。

## Pull Requests / PRとコミット

- Keep one commit focused on one purpose.
- Record research flow or evidence updates under `docs/`.
- Summarize important command results instead of pasting long logs.
- If CI fails, state the failing command and the next check.

- 1コミット1目的を基本にする。
- 研究フローや検証ログは `docs/` に残す。
- 長いログは貼らず、重要行だけ要約する。
- CIが落ちた場合は、失敗コマンドと次の確認点を明記する。
