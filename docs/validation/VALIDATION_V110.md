# Kagami v110 — 検証結果と既知の制限

実行環境: Linux / Python 3.13 / PyTorch CPU。Apple Silicon の MPS と、実ユーザーの HF 数 GiB チェックポイント・ジャーナルは利用していません。

## 1. 署名・API の互換性

- v109 の「PT+SFT+PPO 共有ポリシー」を canonical sorted JSON とした SHA-256 が一致。
- v109 の署名制御フレーム `encode_control({'type':'heartbeat'})` の SHA-256 が一致。
- Python AST による比較で、**移動した 44 個の関数・クラスの構文木が元の定義と一致**。互換性ラッパーと重複排除は別途テスト。
- `cli.load_model / training_batches / run_step`、`mixed_learning_v65.spec / validate_args / MixedObjectiveTrainer`、`WebviewController.save / start`、`elasticnet.train_elastic / encode_control` の公開 API を維持。
- 元ZIPから実質的に変更していないファイルは424個。変更対象は CLI/学習/GUI/elastic 制御の既存ファイルと README、追加は専用モジュールとテスト・監査資料。

## 2. 回帰試験

- `python -m compileall -q patchlib` : 成功
- `PYTHONPATH=.:tests python -m patchlib.cli --help` : 成功
- `PYTHONPATH=.:tests HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest -q tests --ignore=tests/test_live_gui_random_tinystories.py --deselect=<v109ですでに失敗する20ケース>` : **1102 passed / 5 skipped / 20 deselected**（68.89 秒）
- 新設 `tests/test_v110_modular_refactor.py` の6ケースは全件成功。上記1102件に含まれます。

### 元 v109 から存在していた失敗

全テストの初回実行では **1094 passed / 5 skipped / 22 failed**。22件のうち2件は今回のモジュール分割で現れた mock 差し替え箇所の違いであり、依存性注入で修正・該当テストを再通過させています。

残る20件はv109のソースを別ディレクトリへ展開し、**全20ケースを同じ環境で再実行して同様に失敗することを確認**しました。内容は主として旧GUI外観文字列と現HTMLの不整合、旧モックとHF loader API の差、特殊なテスト用モデルパスなどです。リファクタリングで元から失敗していたテストを通過したことにはしていません。

TinyStories初期化用の別テストファイルは、HF キャッシュ／通信や旧モック前提に依存するため、本回帰コマンドから除外しています。これらもv109の一部テストで元から失敗することを確認しました。

## 3. 改修範囲の明確化

- **変更なし**: 署名付きイベントの format、shared objective spec、SQLite journal schema、世代 roll-over の state machine、PPO/SFT/PT のloss係数、GUI HTML/JS、過去の weight・鍵。
- **変更あり（ローカル非署名）**: 重複した AdamW state device move の関数化、profile・batch loader の明示的な依存性注入、空の `pass` 分岐の削除。
- **MPS 実機未検証**: Metal クラッシュ、macOS のネイティブスレッド、実環境での学習性能の改善は主張しません。
- **残存する技術的負債**: `elasticnet.train_elastic()` は約3200行。安易な削除・一括抽出は行っていません。独立した複数ピアE2Eを増やしてから次段で作業すべきです。

## 4. 安全な導入

元の v109 のソースを Git commit 等でバックアップし、v110 をコードのみ差し替えてください。既存ルーム、journal、秘密鍵、checkpoint はそのまま保持し、`reset-local-data` は実行不要です。運用中の全ピアを一度に更新する必要はありませんが、コードを更新するときは学習を安全に停止してから、既存の resume 手順で再開してください。
