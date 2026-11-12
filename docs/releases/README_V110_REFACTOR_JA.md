# KCIA v110 — 可読性・保守性リファクタリング

## 目的

v109 の学習挙動、署名済みルームのポリシー、モデルとジャーナルの復元契約を維持しながら、肥大化した Python ソースを責務単位に整理しました。これは **構造の変更** であり、新しい PPO の数値アルゴリズムやネットワークプロトコルへの移行ではありません。

## 主要な変更

| 従来のファイル | v109 | v110 の入口 | 分離先 |
| --- | ---: | ---: | --- |
| `patchlib/cli.py` | 1715 行 | 約 960 行 | `model_loading_v110.py`, `local_training_v110.py`, `cli_arguments_v110.py` |
| `patchlib/mixed_learning_v65.py` | 1090 行 | 約 660 行 | `mixed_policy_v110.py`, `mixed_kernels_v110.py`, `mixed_sources_v110.py`, `mixed_errors_v110.py` |
| `patchlib/webview_controller.py` | 1414 行 | 約 1000 行 | `webview_lifecycle_v110.py`, `webview_validation_v110.py` |
| `patchlib/elasticnet.py` | 4332 行 | 約 4075 行 | `elastic_control_v110.py` |

関数名、既存の `patchlib.cli.load_model / training_batches / run_step`、`patchlib.mixed_learning_v65.spec / validate_args / MixedObjectiveTrainer`、`patchlib.webview_controller.WebviewController.save / start`、`patchlib.elasticnet.encode_control / decode_control / train_elastic` は維持しています。

### 責務ごとの構造

- **モデル準備**: `model_loading_v110.py` に Hugging Face モデルの固定 revision 読込、LoHA/LoRA 付与、ディスクオフロードをまとめました。コマンドパーサから分離しています。
- **ローカル学習**: `local_training_v110.py` にデータバッチ生成と PT/SFT/PPO の勾配処理をまとめました。署名付きイベントの送受信には触れません。
- **学習ポリシー**: `mixed_policy_v110.py` が共同学習の署名対象設定と入力検証を担当します。JSON のキー・デフォルト値・順序を変えていません。
- **学習の数値関数**: `mixed_kernels_v110.py` にロジット抽出、KV-cache、コンテキスト分割、Advantage 正規化、Value Head などを分離しました。
- **データソース**: `mixed_sources_v110.py` がローカル JSONL のストリーミング読み込みを担当します。
- **GUI の永続化・起動**: `webview_lifecycle_v110.py` にルーム保存、プロトコル・ジャーナル・デバイス検査、学習起動を移しました。GUI コントローラは薄い委譲メソッドを残します。
- **通信制御**: `elastic_control_v110.py` に署名付き制御フレーム、frontier、NEED、snapshot 供給選択などを分離しました。WebRTC 接続とモデルの状態管理は元の `elasticnet.py` に残します。

### 無駄な処理の整理

- 分離後に参照されなくなった、複数の標準ライブラリ import を削除。
- 従来の `if sharded_base_manifest: pass` だけの空分岐を、同条件を保持したガード条件に変更。
- Critic の AdamW モーメント `exp_avg / exp_avg_sq` をパラメータのデバイスに戻す重複処理を、1つの共通ヘルパーに統合。
- 読み込み失敗時・ランダム初期化・同期済みルーム用の古いコードは「参照が少ない」という理由で削除していません。将来の復旧経路になり得るためです。

## 境界条件（壊してはいけないもの）

1. **署名とモデル ID**: v65–v99 の共同学習ポリシー、v3 制御フレームのエンコードを変更していません。`tests/test_v110_modular_refactor.py` で v109 の SHA256 と比較しています。
2. **SQLite のスレッド所有権**: ジャーナルはイベントループ側、モデル更新はモデルワーカー側という境界を維持。SQLite 接続をワーカーへ受け渡していません。
3. **再起動と鍵**: 署名済みイベント、世代、チェックポイントを削除しません。`reset-local-data` は不要です。
4. **依存関係の差し替え**: `cli.training_batches()` のバッチファイルローダー、GUI のプロフィール割当を元モジュールから注入し、既存テストのモック／プラグインとの互換性を維持しています。
5. **MPS**: v108 の専用ワーカー・CPU 高精度演算の境界を変えていません。Mac の Metal 実機検証はこの環境では実施していません。

## 既知の残課題

`elasticnet.py` の `train_elastic()` は依然 **約 3200 行の単一 async 関数**です。多くの `nonlocal` 状態、イベントの順序、長寿命タスク、世代移行とジャーナル書き込みが結合しています。単純な自動関数抽出では状態機械の境界が崩れるため、今回は一括分割していません。

次段階は、(1) モデル更新の担当、(2) ネットワーク受信と署名検証、(3) 世代認証・checkpoint/GC、(4) GUI とテレメトリ、の状態を明示的なコンテキストクラスへ移し、単独・複数ピア・途中復帰・異世代同期の E2E 試験を揃えてから分割するのが適切です。

またこのリファクタリングは既存の P2P/MPS 不具合を修正するリリースではありません。可読性の改善と実稼働での安定性向上は別途測定が必要です。

## テスト実行

```bash
python -m compileall -q patchlib
PYTHONPATH=.:tests pytest -q tests/test_v110_modular_refactor.py
PYTHONPATH=.:tests pytest -q tests/test_mixed_learning_v65.py tests/test_gui_mixed_critic_v66.py tests/test_reconnect_sampling.py
```

ベースの v109 から継承する古い GUI CSS のスナップショット検査や、外部モデルのモックと現行 HF 呼び出し ABI が食い違うテストについては、`../validation/VALIDATION_V110.md` に区別して記録します。
