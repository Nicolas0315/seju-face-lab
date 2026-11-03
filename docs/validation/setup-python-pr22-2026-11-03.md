# actions/setup-python v7.1.0 検証記録 — 2026-11-03

## 対象と記録の根拠

[PR #22](https://github.com/Nicolas0315/seju-face-lab/pull/22)のAction更新を確認した。
検証対象headは`0f8590c1e965d0656f629fd8eb6af5adce3746c0`、
base/mainは`1baa62598146534968aeb27154144654bcbd9477`。
差分は`.github/workflows/ci.yml`の`actions/setup-python`固定SHAをv7.0.0からv7.1.0へ更新する1行のみ。

`CONTRIBUTING.md`の「検証ログはdocs/に残す」に従い、重要な結果だけを記録する。
今回は日次triageや研究パイプラインの実行ではないため、`STATE.md`と`loop-run-log.md`は更新しない。
既存の研究日付・数値・判定は、このCI確認の結果で置き換えない。

## Actionの出所と互換性

GitHub APIで公式`actions/setup-python`の`refs/tags/v7.1.0`が
`6d4fd22b20465ccc0f7354476588e8df7c9ca0e0`のcommitを直接指すことを確認した。
固定SHAの[action.yml](https://github.com/actions/setup-python/blob/6d4fd22b20465ccc0f7354476588e8df7c9ca0e0/action.yml)は
既存の`python-version: "3.12"`入力を維持し、runtimeは`node24`。
[README](https://github.com/actions/setup-python/blob/6d4fd22b20465ccc0f7354476588e8df7c9ca0e0/README.md)の最小runner要件2.327.1に対し、
実際のtest runnerは2.338.2で要件を満たした。

## 実際に確認したCI

両runのeventは`pull_request`、`headSha`は上記headと完全一致し、結果はsuccess。
両checkoutログは、上記headを上記baseへ統合した
`4fff854d9f87332c347232cd82da91d2ffb5297f`の実行を示した。

[test run 41294655589](https://github.com/Nicolas0315/seju-face-lab/actions/runs/41294655589)（2026-11-01）:
Windows Server 2025 / runner 2.338.2で指定SHAを取得し、CPython 3.12.10のsetupに成功。
`ruff check .`はAll checks passed、`python -m compileall -q src tests scripts`は成功、
`python -m unittest discover -s tests`は166 tests / 10.249秒 / OK (skipped=3)。

[sast run 41294655596](https://github.com/Nicolas0315/seju-face-lab/actions/runs/41294655596)（2026-11-01）:
`BASE_SHA`は上記baseと一致し、Semgrep差分スキャンは1ファイル / 82 rules / 0 findings。
このworkflowにgitleaks実行ステップはなく、包括的なセキュリティ監査の結果とは扱わない。

## 確認範囲と未実施

上記Action更新のローカル全差分は`git diff --check`成功。
gitleaks 8.30.1で上記baseから検証対象headまでの1コミットを検査し、no leaks found。
ローカルのアプリケーションテストは再実行せず、実際のWindows CI結果を確認した。
Claude reviewはusage残高不足でスキップされており、実施済みレビューとして扱わない。
`AGENTS.md`が参照するWindows側の`C:\Users\ogosh\AGENTS.md`は、このMacから未確認。

この記録のCI結果は検証対象headに限る。記録追加後の新headは別途、必須`test`と`sast`の結果およびcheckout baseを確認する。
