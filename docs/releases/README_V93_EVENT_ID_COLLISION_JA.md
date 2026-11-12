# Kagami v93: `conflicting event ID` 修正

## 変更内容

- `patchlib/elastic.py` の `ElasticEngine.seal_local_responsive()` に async のローカルシール専用ロックを追加。複数 coroutine が `MAX(seq)+1` を同時取得して、違う payload に同じ `(origin,seq)` を割り振ることを防止。
- `patchlib/continual.py` の `ContinualElasticEngine.seal_local_responsive()` にも同じロックを適用。ReLoRA / LoHA の因子・anchor 更新をシール処理が終わるまで保護。
- `patchlib/async_ingest.py` では P2P の通常更新・aggregate 共に、本人の `(origin,seq)` として**未登録の**リモート更新をジャーナルへ新規登録することを禁止。すでに登録済みの正しい本人更新のループバックは重複として受理。
- `patchlib/elastic.py` のエラーを改善。衝突時に origin、seq、既存ハッシュ、到着ハッシュ、genesis、local_origin を表示。記録の改ざん・削除・自動ID再採番はしない。
- 重複登録を後から検出した場合に、同じ weight delta を二重適用しないよう、`journal.append*()` の `False` を尊重。

## これは何を修正し、何を修正しないか

ソースコード中では、未確定の次シーケンス番号を先に読み、その後シリアライズと非同期の journal 保存が行われていました。これらの間に同じIDを持つ別のイベントがコミットされると `conflicting event ID` になります。v93ではローカルシールを直列化し、P2P側が未登録の本人IDイベントを新規受理しないようにしました。

ただし、このユーザーの **既存のジャーナルそのものは提供されていない**ため、今回の衝突の唯一の原因がその競合だと断定することはできません。誤った本人キーの複製、バックアップ復元、世代切替の不整合等が残る場合は、詳細なエラーで**安全に停止**します。意図的な署名イベントの二重発行を黙って修復することはしません。

## 導入

1. Kagami GUIと全学習プロセスを終了。
2. `events.sqlite3` と関連 sidecar、`.identity.pem`、checkpoint、carry、fast-resume のバックアップを確保。絶対に個別に手動削除しない。
3. v93のZIPを展開し、ソースコードを更新。保存データの初期化は不要。
4. 既存のルームと保存データから学習を再開。
5. 再発時は、**新しい完全な `conflicting event ID:` 行と、その直前の数十行のログ**を収集。`python kagami.py journal-status --journal /path/to/events.sqlite3` の診断も有用。

同じ `.identity.pem` をコピーして別の保存ジャーナルで起動したピアは、同一IDの更新を独立生成しないようにすること。意図的に2ピアを動かす場合は、各ピアに別々のidentityとjournalを使用。

## テスト

`tests/test_local_event_identity_v93.py` で以下を検証：

- 同時に2つのLoHA/FedSumシールを行っても連番が重ならない。
- リレー経由の未登録自己IDイベント（通常/aggregate）を保存せず拒否。
- 正常な自己イベントのループバックは重複扱い。
- 既存のID不整合を自動上書きせず、具体的に報告。

CLI、MPS、安全なディスクGC、P2Pの既存認証・epochプロトコルの形式は変更していません。
