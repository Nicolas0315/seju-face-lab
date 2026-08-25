# Seju Vector Model v1 検証記録 — 2026-08-25

## 判定

`Vector Model v1`はローカル実データでpromotion gateを通過した。採用アルゴリズムは`B1`（人物均等のrobust global center）。`C1`（tangent component model）は実装・評価済みだが、B1に対する改善の95%信頼区間が0をまたぎ、bootstrap安定性も事前閾値に達しなかったため採用していない。

この判定は「cleanなローカルSeju snapshotに対する相対近似」を支えるもので、本人識別、所属確率、魅力度、人気予測を支えるものではない。

## データゲート

| 項目 | 実測 |
|---|---:|
| eligible download rows | 259 |
| signature assets rejected | 34 |
| face candidates | 225 |
| accepted single-face observations | 224 |
| no-face rejections | 1 |
| subjects | 34 |
| detector coverage | 0.9956 |
| near-duplicate representatives | 224 |

source URL host、eligible、local file、SHA-256、signature assetを顔検出より先にfail closedで検査した。顔観測は1顔だけを受理し、neural vectorとgeometry vectorを別artifactへ保存した。

## LOSOとモデル選択

| 項目 | 実測 |
|---|---:|
| subject-disjoint folds | 34 |
| score median | 49.95 |
| bootstrap 95% interval | 31.15–66.31 |
| worst-decile score | 9.77 |
| C1−B1 improvement median | 0.00083 |
| C1−B1 95% interval | -0.00082–0.00142 |
| C1 center cosine median | 0.9262 |
| C1 center cosine 95% interval | 0.8648–0.9553 |
| C1 subspace similarity median | 0.5741 |
| C1 subspace 95% interval | 0.4723–0.6868 |

各foldでheld-out subjectを中心、成分、品質正規化、校正から除外した。共通promotion gateは合格したが、C1固有gateは不合格だったためB1へfallbackした。

## CUDA摂動試験

224枚に対して元画像と6種類の摂動をCUDA InsightFaceで評価した。全gateが合格し、全体実行時間は283.36秒だった。

| 摂動 | coverage | embedding cosine median | B1 degradation p90 |
|---|---:|---:|---:|
| defocus blur | 0.9866 | 0.9430 | 0.0404 |
| detector 320 | 0.9955 | 0.9583 | 0.0319 |
| downscale | 0.9911 | 0.9519 | 0.0328 |
| exposure shift | 1.0000 | 0.9746 | 0.0180 |
| JPEG recompression | 1.0000 | 0.9506 | 0.0408 |
| partial crop | 0.9866 | 0.9648 | 0.0350 |

閾値はcoverage 0.95以上、embedding cosine 0.80以上、B1 degradation p90 0.08以下。

## 実行検証

正規環境は`C:\Users\ogosh\work\seju-face-lab\.venv`。worktreeのコードを検証する際は`PYTHONPATH=src`を指定した。

```powershell
$env:PYTHONPATH = "$PWD\src"
C:\Users\ogosh\work\seju-face-lab\.venv\Scripts\python.exe -m unittest discover -s tests
C:\Users\ogosh\work\seju-face-lab\.venv\Scripts\python.exe -m compileall -q src tests scripts
C:\Users\ogosh\work\seju-face-lab\.venv\Scripts\ruff.exe check .
python scripts/verify_mermaid_blocks.py docs/seju-vector-system-overview-ja.md --out .tmp/mermaid
npx --yes @mermaid-js/mermaid-cli -i <diagram.mmd> -o <diagram.svg>
```

- repository test suite: 157 tests、exit 0（skipped 2）
- Japanese diagrams: 8 blocks extracted、8/8 rendered
- CUDA scoring smoke: B1、Seju近似96.97、support内

最後のスコアは学習側の参照画像1枚を通したCLI/CUDA smokeであり、独立した精度証拠ではない。個別スコアはローカルartifactだけに保持する。

## Git安全監査

`git ls-files -- 'data/**' 'outputs/**' '*.npy' '*.npz'`で追跡対象は`data/README.md`のみ。raw画像、埋め込み、NPZ、生成画像、個別結果はコミットしていない。

## 未実施の別ゲート

- 架空の3人目候補画像の実生成
- 候補画像のsingle-face QA、support、Pareto多様性、blind human review
- ブラウザへの掲載・外部公開
- 独立した将来snapshotや別backendとのagreement試験

候補plannerとblind review bundleの実装はあるが、上記を実行していないため「人気が出る顔」や候補優劣の結論は出していない。
