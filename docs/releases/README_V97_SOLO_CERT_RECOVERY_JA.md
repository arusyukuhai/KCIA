# Kagami v97: 1/1 epoch candidate が `candidate-offered` で停止する問題の対策

## 今回提供されたログ

- signed log: **80.2%**（安全停止が動作中）
- 参加ピア: **online=1 / neighbors=0**
- 表示: `V95 EPOCH PRESSURE HOLD ... stage=candidate-offered`
- 12:57〜13:09にわたり表示が変わらない。

`candidate-offered` は **候補エクスポートが成功した**ことを示すだけで、本人の署名、証明書確定、weightへの切替や旧ジャーナルGCを証明するものではありません。このログのみでは最後に進んだ段階を断定できません。

## コード上で判明した復旧漏れ

`AutonomousPagedEpoch._remember_vote()` は `votes` テーブルに署名voteを **先に** 永続保存し、そのあと `CompactEpochLedger.certificate()` / `accept()` で認証済み証明書を確定します。両者は別々のSQLite DBなので、途中で異常終了すれば **署名voteだけが残り、証明書が未確定**になる可能性があります。

旧実装の `ready_for_vote()` は、ローカルvoteの保存を検知すると「もうvote済み」としてスキップします。そのため再起動しても証明書確定を再試行できず、ジャーナルの保留が続き得ます。

## v97の修正

1. `AutonomousPagedEpoch.reconcile_quorum()` を追加。次の未確定世代に対する保存済みvoteを毎回 **実際の署名で再検証**し、定足数を満たしていれば証明書確定を再試行します。署名の捏造・ID振り直し・weightの書き換え・未認証GCは一切しません。
2. HOLD画面の `stage=` を実際の保存状況に基づいて表示します（例: `eligible-for-local-vote`, `local-vote-worker-running=...`, `quorum-votes-persisted-certificate-recovery`, `certified-awaiting-activation`）。vote数、必要署名数、ページ数、受領証の有無、最後の検証エラーも表示します。
3. 巨大なcheckpointの検証が失敗した場合に0.5秒ごとの再ハッシュを繰り返さず、5〜120秒間隔で再試行します。認証を弱めません。
4. `journal-status` の読み取り専用表示に、最新候補のvote数、ページ数、受領証とマニフェストの存在確認を追加しました。

## 導入とデータ保護

1. KagamiのGUIと学習プロセスをすべて停止する。
2. **現在のルーム保存領域（鍵・journal・carry・checkpoint・`.live-epoch`を含む）をまるごとバックアップ**する。HDD/SSDの空き容量を確保する。
3. v97のソースを適用し、同じルームでGUIを再起動する。**`reset-local-data` は使わない**。
4. 次の読み取り専用コマンドで現在の状態を確認する。

```bash
python kagami.py journal-status --journal '/Users/al/Library/Application Support/Kagami/.fedpara-gui/rooms/gcaouHt6-VW-jo_kBI7BBdjIkELOV8Cz/peers/peer-0240df31ec25b1d1/events.sqlite3'
```

## 未解決の可能性

- `local-vote-worker-running=...` が長時間続く場合は、チェックポイントの実データ・署名ジャーナルを読み取り検証するI/O負荷、ファイル/SQLiteのロック、実モデル環境が原因の可能性があります。v97では実行時間を明示しますが、検証の省略はしません。
- `last_vote_error=` があれば、そのエラーを原因として修復する必要があります。署名した候補とは異なるweightを自動採用しません。
- `certified-awaiting-activation` の場合は認証済みの候補から実際の世代切替・carryを完了する処理をさらに調査する必要があります。
- ユーザー実機の4GiB相当のデータでは未検証です。旧ジャーナルを無条件には削除しません。
- v97は **証明書の取り残しと診断不足**を修正しますが、`candidate-offered` に至るすべての停止原因を解決したと断定するものではありません。

## 検証

新規 `tests/test_v97_quorum_recovery.py` で「署名vote保存直後・証明書確定前のクラッシュ」を再現し、同じ署名から安全に復旧すること、署名不足を自動補完しないこと、診断経路を検証。関連91件のテストに成功（macOS実機・長時間の複数ピア運転は未実施）。
