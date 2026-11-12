# v110 SOURCE REFACTOR

[日本語の変更説明・モジュールマップ](README_V110_REFACTOR_JA.md) / [検証結果](../validation/VALIDATION_V110.md)

# KCIA v109 — long-response SFT context windows

長文SFTの `SFT response exceeds max_length` を修正し、PPO混合学習中でも応答を複数のコンテキスト窓へ分割して全トークンを学習できるようにしました。モデル/tokenizerの実効最大長も考慮します。詳細は [`README_V109_LONG_SFT_CONTEXT_JA.md`](README_V109_LONG_SFT_CONTEXT_JA.md) を参照してください。

> **v108 MPS Acceleration:** `README_V108_MPS_ACCELERATION_JA.md` — GUI GPU selection, PPO MPS sampling, GAE speed path and isolated `mps-ppo-smoke`.

**v107:** 学習中のPT・SFT・PPOを独立してplotするよう変更しました。詳細は [README_V107_OBJECTIVE_ACCURACY_PLOTS_JA.md](README_V107_OBJECTIVE_ACCURACY_PLOTS_JA.md) を参照。

## v106 (2026-11-04) — SFT/PPO HF dataset schema guard

Hugging Face SFT/PPO streams are validated before loading heavyweight Reward Model / Critic checkpoints; standard column aliases are mapped conservatively, and GUI has a Validate columns action. See `README_V106_HF_DATASET_SCHEMA_JA.md`.

> **v104**: HF Criticを大型weight取得前に検査し、GUI新規ルームのHF Critic初期headをNEWに変更。詳細: `README_V104_HF_CRITIC_EARLY_GUARD_JA.md`。既存署名付きルームの設定は変更しません。

> **v102 (HF Critic head initialisation)**: `README_V102_HF_CRITIC_BOOTSTRAP_JA.md` — pretrained scalar head を持たないHFモデルからPPOのCritic headを新規初期化できる明示的モードを追加。

# KCIA v99 — PPO・SFT・混合学習の安定化

新しいPPO/SFTのオプトイン機能: [README_V99_TRAINING_STABILITY_JA.md](README_V99_TRAINING_STABILITY_JA.md) を参照してください。v98の既存の署名済みルーム設定との互換性を維持しています。

> **v98 (2026-11-04):** carry付きfast-resumeの保存中にSQLiteを別スレッドから参照して `sqlite3.ProgrammingError` となる問題を修正。詳細は [README_V98_SQLITE_THREAD_FIX_JA.md](README_V98_SQLITE_THREAD_FIX_JA.md)。

> v97: 単独ピア世代確定のクラッシュ復旧・保留詳細表示 — `README_V97_SOLO_CERT_RECOVERY_JA.md` を参照。

> **v96 (2026-11-04):** 旧世代carryがあるとfast-resumeが全履歴再生へ戻る問題を修正し、dual-lane cursor付き統合weightからの復元と受信専用/保留中の定期保存に対応しました。詳細は [README_V96_DUAL_LANE_FAST_RESUME_JA.md](README_V96_DUAL_LANE_FAST_RESUME_JA.md)。

> **v95 (2026-11-04):** 1/1署名の自動世代移行がV89 HOLDで止まる際のエクスポート再試行と段階診断を修正しました。詳細は `README_V95_SOLO_EPOCH_PRESSURE_JA.md`。

# KCIA v94: epoch-scoped event identity fix

**Important:** See [README_V94_EPOCH_SCOPED_EVENT_IDS_JA.md](README_V94_EPOCH_SCOPED_EVENT_IDS_JA.md) for the fix to `conflicting event ID ... journal_sha256=missing` after a certified generation rollover. v94 keeps current-generation and old carry event IDs separate and prevents old carry from rewriting new-generation LoHA local factors. No reset is required.

> **macOS / KCIA v92 MPS safety:** macOS auto-device now uses CPU by default after repeated native Metal command-buffer aborts. To explicitly try unsupported experimental MPS, launch with `KCIA_EXPERIMENTAL_MPS=1 python kcia.py`. Run `python kcia.py mps-smoke` for an isolated probe. See `README_V92_MPS_CRASH_CONTAINMENT_JA.md`. No room or journal migration required.

> **v90:** GUI新規ルームの認証済み世代移行は1/1・2/2・2/3委員会に対応。`README_V90_SMALL_COMMITTEE_JA.md`を参照してください。既存の署名済み委員会は人数変更しません。

> **v89:** `durable update log is full` の原因確認には `python kcia.py journal-status --journal PATH`。詳細は `README_V89_EPOCH_PRESSURE_JA.md`。

# KCIA — 軽量化ソース配布版（v85 fast-resume ベース）

本配布版は、現在の学習・GUI・P2P・LoHA・Critic・認証付き世代移行に必要なコード、テスト、サーバーソースを維持し、**明らかに不要なアーカイブと旧実装**を省いたものです。Pythonの動作を軽くするための高速化とは異なり、主にソース配布物と保守対象の整理です。


## ディスク容量の安全対策（DiskGuard）

WebViewによるジャーナル上限の無条件解除を修正し、標準上限4GiB・予備空き容量1GiB、送信用snapshot専用ストアと自動GC、Mac/APFSのOptimizer CoW、WAL短縮、グラフ/ログの容量上限、オフライン回収コマンドを追加しています。**容量不足に備えて安全停止する仕組みであり、署名付き現役ジャーナルを無条件削除するものではありません。**

```bash
python kcia.py disk-report --journal /path/to/events.sqlite3
python kcia.py disk-reclaim --journal /path/to/events.sqlite3  # KCIA停止後
```

詳しくは [../guides/README_DISK_GUARD_JA.md](../guides/README_DISK_GUARD_JA.md) を参照してください。

## 実行

依存ライブラリは用途に応じて `requirements-patches.txt`、`requirements-gui.txt`、`requirements-eval.txt` 等を参照してください。

- WebView GUI: `python kcia.py` または `python -m patchlib.cli gui`
- 学習CLI: `python -m patchlib.cli train-live --help`
- GUI（旧Tkオプション）: `python -m patchlib.cli gui-desktop`
- メタデータ/その他コマンド: `python -m patchlib.cli --help`
- Nimシグナリングサーバー: `src/main.nim` からビルド (`scripts/check_nim.sh` は Nim が必要)

最新版のfast-resumeの詳細は [CHANGES_V85_FAST_RESUME_JA.md](CHANGES_V85_FAST_RESUME_JA.md) を参照してください。**現役世代の署名ジャーナルが容量を超えても自動的に消えるわけではありません。** 認証済み世代移行の完了後に限定して安全に回収します。

## デバッグ用：保存データの全初期化

KCIAがGUIで保存した全ローカル状態（weight・ジャーナル・Optimizer・チェックポイント・ルーム定義・peer identity・ログ等）を初期化できます。まずプレビューを確認し、KCIAの全プロセスを停止してから実行してください。

```bash
python kcia.py reset-local-data
python kcia.py reset-local-data --yes
```

CLI既定名の保存物まで削除する場合は `--include-cli-defaults` を追加します。Hugging Face共通キャッシュとユーザーの外部保存先は対象外です。詳細・注意事項は [../guides/RESET_LOCAL_DATA_JA.md](../guides/RESET_LOCAL_DATA_JA.md) に記載しています。

## 保持した機能・ソース

`patchlib/` の本番インポート経路、`web/` のGUI素材、`src/*.nim`、`deploy/`、`sandbox/`、`examples/`、`scripts/`、`tools/` および現在のユニットテストを収録。`kcia-rooms/` の部屋メタデータは破棄せず同梱しています。

古い名前を持つモジュールでも、現行のコードからimportされるものは削除していません。特に `behavior_checkpoint_v81.py` はv82実装の依存先であり保持しています。

## 整理対象

- ZIP展開時のmacOSメタデータ (`__MACOSX`, `._*`)
- ソースに反映済みの過去の `CHANGES_V*.patch`、過去の `PYTEST_V*`・`BENCHMARK_V*`・`VALIDATION_V*` 等
- 旧版の多数の進捗README。主要な操作資料と最新の修正記録は保持
- 既存のNimソースから再生成できるmacOS arm64ビルド済み実行ファイル `src/main`
- 最新v78/v82と無関係の旧実装 `distributed_critic_v77.py`, `trajectory_gossip_v81.py`
- 現行ランタイムに接続されていない試作 `epoch_sharded_rcu_v56.py` および対応する単独テスト
- 同内容のSVG重複 (`kcia.svg` と `web/kcia.svg`)

各ファイルを削除した根拠は `CLEANUP_MANIFEST.tsv` に記録。消した履歴が必要になった場合は元のv85 ZIPから復元してください。旧形式の通信や状態をサポートするために必要な**ライブ経路の互換コードは削除していません**。

## 検証

```
python -m compileall -q patchlib tests scripts
PYTHONPATH=.:tests pytest -q tests/test_fast_resume_v85.py tests/test_elastic.py tests/test_factor_ledger_v62.py tests/test_epoch_trainer_boot_v52.py
```

実機WebRTCの起動、巨大モデル、分散切替は別途検証してください。

補足: GUI関連のテスト2件で実装変更後も残っていた旧CSS変数の表記・背景色の期待値を、現在のコードに合わせて更新しました（GUI本体の描画は変更していません）。

## v88 自動認証済み世代移行（新規GUIルーム）

新しくGUIで作成するルームは、認証済みの1〜3ピアから所有者署名付きの委員会を1回だけ確定します。
定足数は1/1・2/2・2/3です。既存委員会は自動変更しません。
詳細は `README_V90_SMALL_COMMITTEE_JA.md` を参照してください。

## v93 イベントID衝突修正

`patchlib.core.PatchError: conflicting event ID` のローカル採番競合・未登録自己ID受信対策については `README_V93_EVENT_ID_COLLISION_JA.md` を参照してください。

## v101 HF Reward/Critic physical-disk preflight

When PPO selects an external Hugging Face Reward Model or Value Critic, KCIA checks
missing checkpoint bytes and disk headroom *before training begins*, and reports
ENOSPC explicitly instead of mislabeling it an unsupported model. See
`README_V101_HF_DISK_PREFLIGHT_JA.md` for recovery commands and limitations.


## v103 — PPO診断

`python kcia.py ppo-doctor` でオフラインの小型PPO+LoRA更新を検証できます。
PPO初期化・報酬ゼロ・勾配・KLの診断ログを追加しました。既存学習プロトコルは不変です。
`README_V103_PPO_DIAGNOSTICS_JA.md` を参照してください。
