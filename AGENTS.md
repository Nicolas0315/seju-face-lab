# seju-face-lab

Global baseline: `C:\Users\ogosh\AGENTS.md`.

Repo-specific rules:

- Keep face vectors, generated portraits, personal image datasets, embeddings, and model outputs local.
- 顔ベクトル、生成画像、個人画像セット、埋め込み、モデル出力はローカルに保持し、コミットしない。
- Use deterministic verification first: `ruff check .` and `python -m unittest discover -s tests`.
- 検証は決定論的コマンドを優先する。
- Do not claim vectors represent a real population. They approximate only the provided image set.
- ベクトルは実人口や客観的な顔タイプを表さない。投入された画像セットだけの近似として説明する。
- Do not add outputs that imply identity recognition, attractiveness scoring, personal value, or insulting labels.
- 本人識別、魅力度判定、人格推定、人への侮辱ラベルにつながる出力を追加しない。
