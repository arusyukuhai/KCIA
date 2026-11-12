# Kagami v85 — 統合済みweightからの高速resumeと認証済み世代GC

## 重要：できることと、まだできないこと

この改修は、**現役世代の署名付きジャーナルを無条件に消すものではありません**。現役ジャーナルを削除すると、未同期ピアのNEED要求、イベント重複排除、署名検証、世代間carryが失われます。すべてのPeerで安全に無限のジャーナル削除を実現するには、委員会が承認した次世代チェックポイントへ切り替える必要があります。既存の認証付き世代GCを保護したまま、ローカルのフル再生を省略する復元キャッシュを追加しました。

## 変更箇所

- `patchlib/fast_resume_v85.py`（新規）: 署名済みイベントを既に統合した**Committed baseのみ**を、分割・ハッシュ検証可能なシャードとして永続化。保存時には現在のLoHA/LoRAのlocal/anchor座標も独立したsafetensorsに格納。ローカル学習の未コミット因子をbaseに重複加算しない。
- `patchlib/shards.py`: `committed_base_only`エクスポートを追加。通常のモデル配布・評価で使用するeffective snapshot動作は変更しない。
- `patchlib/elastic.py`: SQLite変更監査カウンタ・スキーマバージョン・最後の行ID・総件数を組み合わせて復元基点を検証。保全された基点の後ろだけ署名付きイベントを再生。古いイベントの索引はP2Pのため保持。
- `patchlib/elasticnet.py`: 起動時に統合weightをハッシュ検証してロードし、追加された署名済み更新だけ復元。32件ごとのローカル公開更新と正常終了時に再取得可能な基点を自動保存。バックグラウンドスレッドでのファイル書き出し中は、モデル更新のcommitをロックして一貫性を保証。PPO paired recoveryとoptimizer snapshotの既存の意味を維持。
- `patchlib/fast_resume_v85.py`: 新しい基点が正常にアトミック公開された後に古いプライベートキャッシュ（不要シャード・古いmanifest・古いfactorファイル）を自動回収。2 GiBの空き領域を追加で確保できない場合などは、この任意の速度最適化をスキップし、署名付きジャーナルは保護。
- `patchlib/epoch_trainer_boot_v52.py`/`patchlib/live_generation_adoption.py`: 既存の厳格な認証済み世代GC後に、削除済み旧ジャーナルのtombstoneを正しく判定して再起動可能にする。空のcarryと証明できた場合に限り、古いプライベートキャッシュとoptimizer snapshotsもGC。現役の古いジャーナル／未認証の旧世代は削除禁止。
- `patchlib/cli.py`: `--fast-resume-every 32`（既定）、`0`で無効。`--generation-v2-trainer`が**明示的に有効**な場合だけ、既存の固定委員会による世代チェックポイント提案間隔を256ローカル公開更新に自動設定（`--generation-v2-propose-every 0`で停止可能）。委員会なしの一方的な世代変更はしない。

## 挙動

1. 通常の学習では、署名イベントを追加するたびにglobalのbase weightへ加算するという従来の意味を保つ。
2. 学習済みweightを含む復元基点を定期的にシリアライズ。出力は `<journal>.fast-resume/` に保存する。
3. 再起動時、チェックポイントの内容ハッシュ・SQLite indexの監査を検証。イベント件数が一致すれば**署名イベントの再生は0件**。クラッシュ後に新しいイベントだけが存在すれば、その**差分だけ**再生する。
4. 復元キャッシュが破損・失効・不足なら、署名イベントを保持した元のジャーナルから通常復元。途中までロードして破損した場合にはフェイルクローズして不正なweightで学習再開しない。
5. 独立した認証済み世代チェックポイントへの正常な切替が証明された場合に限って、既存のGC手続きが旧ジャーナルを消せる。P2P同期相手の必要履歴について安全性を優先する。

## 注意点・未対応

- 現役世代に対する署名履歴の**完全自動・無制限プルーニングは未実装**。未認証の削除はP2P整合性を壊すため、意図的に行っていません。
- 署名付き履歴は現役世代が続く限り増えます。ディスクフルの根本解消には、本番運用で認証済みv2世代移行と旧世代GCが実際に進むことが必要です。委員会が進捗しない場合も古い現役ログは削除しません。
- weightの統合キャッシュにはおおむね**モデル1個分**のディスク容量が別途必要で、保存にもI/O時間を要します。高速化は「履歴再生が増え続ける」問題の改善であり、ロードするモデルの大きさに対する時間を0にはできません。
- factor-ledgerのvirtual baseを使った実模型の復元試験は通過。既に稼働中の **paged COW** については現在のキャッシュ生成器が安全に読み出す仕組みを持たないため、間違ったweightを保存せずチェックポイントをスキップします。
- 認証済み世代のcarryアーカイブが**非空**な場合、現行v85は安全のため従来のcarry+署名履歴再生を採用します。別途carry frontierの証明が必要です。
- Power-lossによりキャッシュのストレージ同期が保証されない場合、チェックサム検証で拒否してジャーナルから再生します。ジャーナル自体がFULLモードでない場合、電源断耐久性は既存設定に依存します。

## 動作検証

`PYTHONPATH=.:tests pytest -q tests/test_fast_resume_v85.py tests/test_elastic.py tests/test_factor_ledger_v62.py tests/test_durable_io_v12.py tests/test_epoch_runtime_v43.py tests/test_epoch_trainer_boot_v52.py tests/test_epoch_carry_v46.py tests/test_mixed_learning_v65.py tests/test_paired_checkpoint_v75.py tests/test_paired_recovery_v76.py tests/test_hot_swap_v45.py`

**84 tests passed**（小型CPUモデル上）。300B級モデル・実P2P数千端末・本番運用の耐久性は未検証です。

## 使用方法

既存の`train-elastic`実行で、デフォルトの `--fast-resume-every 32` が有効。速度を優先する／モデルのI/O負荷が高い環境は `--fast-resume-every 128` などに変更。キャッシュ保存を無効化する場合は `--fast-resume-every 0`。

**旧ジャーナルを手動削除しないでください**。認証済み世代GCによってのみ安全に削除されます。再起動ログには `KAGAMI FAST RESUME loaded materialized committed base` と、署名付きsuffix再生件数が表示されます。
