# KCIA v111 — ソースマップ

## 入口

- `kcia.py` — デスクトップGUIと診断コマンドの入口
- `patchlib/cli.py` — CLIサブコマンドの入口。実処理は別モジュールへ委譲
- `src/main.nim` — Nimシグナリングサーバー
- `web/desktop_webview.html`, `web/desktop_webview.js` — 現行のデスクトップUI

## Pythonパッケージ

| 関心 | 中核 | 関連モジュール |
| --- | --- | --- |
| モデルを生成/読み込み | `model_loading_v110.py` | `hf_load_v85.py`, `model_contract_v64.py`, `core.py` |
| バッチ作成とPT/SFT/PPO | `local_training_v110.py` | `mixed_learning_v65.py`, `mixed_sources_v110.py`, `mixed_kernels_v110.py` |
| PPOの軌跡/評価 | `ppo_pipeline_v72.py` | `reward_model_v67.py`, `hf_value_critic_v69.py`, `distributed_critic_v78.py` |
| UIでの設定・実行 | `webview_controller.py` | `webview_lifecycle_v110.py`, `webview_validation_v110.py`, `webview_gui.py` |
| P2Pと署名済み更新 | `elasticnet.py` | `elastic.py`, `elastic_control_v110.py`, `trajectory_gossip_v82.py` |
| 世代確定と復元 | `epoch_autonomous_v56.py` | `epoch_carry_v46.py`, `epoch_lifecycle_v51.py`, `fast_resume_v85.py` |
| MPSの実行 | `mps_runtime_v91.py` | `mps_policy_v92.py`, `mps_acceleration_v108.py` |

**バージョン付きモジュール名は、Python importの互換性を守るため変更していません。** 新しい機能の呼び出し方は入口のファサードから追ってください。

## オフラインPT比較（v112）

`benchmarks/bench_serial_pt_v112.py` はGUIに依存しないCLI入口、`patchlib/serial_pt_benchmark_v112.py` はシリアルな1/NピアPT比較です。1つのモデルを使い回し、ローカル因子・Optimizerをディスクに退避します。本番のP2P署名・Gossipとは別の同期FedSum実験です。

## ディレクトリ

- `patchlib/`: 本番ランタイムとレガシー互換性が必要な公開モジュール
- `web/`: 現行WebViewとNimサーバーのWeb素材。旧HTMLもサーバー/テストから参照されます
- `src/`: Nimサーバーのソース
- `scripts/`: 運用・調査用の実行スクリプト
- `tools/`, `benchmarks/`: 開発時の診断・速度評価
- `examples/`: データセット・学習ポリシー例
- `tests/`: 回帰テスト
- `docs/guides/`: 操作説明
- `docs/releases/`: バージョン別変更履歴
- `docs/validation/`: 検証記録

## 安全な削除ポリシー

1. 本番のimport、CLIの文字列指定、GUIサブプロセス起動、NimのstaticReadを確認。
2. `tests/`, `scripts/`, `tools/`, `benchmarks/` による参照も確認。
3. `webview_diagnostic.py` は単独 `python -m` 実行ツール、`kcia_eval_worker.py` は文字列指定で起動、`training_sim.py` は開発ベンチマーク用。importされにくくても未使用ではありません。
4. `src/*.nim`、旧Webページ、互換性モジュールを表面的な参照数のみで削除しない。
5. 生成キャッシュ、過去のパッチ生成器、ソースZIPに混入した実行時ルーム情報は除外する。ただし既存インストールの生きたデータには触れない。

### 残る課題

`elasticnet.train_elastic()` は巨大な非同期状態機械です。より深い分割は、複数ピア・churn・異世代参加・クラッシュ復旧の統合テストを整備してから行うべきです。今回はAPI、署名、DB、学習式は維持しています。
