> **この文書は旧仕様（v88/v89）です。** 少数ピア対応の変更は `README_V90_SMALL_COMMITTEE_JA.md` を参照してください。

# KCIA v89 — durable update log 4 GiB 上限対策

## 何が起きているか

`durable update log is full (signed payload=3.999 GiB, incoming=0.010 GiB, configured cap=4.000 GiB)` は物理ディスク満杯ではなく、KCIAの論理ジャーナル上限エラーです。従来v88では、3署名ピア委員会・2署名の認証済み世代移行が不成立のままログを4GiBまで増やし、`PatchError` で停止し得ました。すでに4GiBに達した既存ジャーナルは、v89へ置き換える**だけ**では小さくなりません。

## v89の変更

1. 認証済み世代移行の先行提案を、上限の65%から35%へ早めました。
2. 新しいGUIルームの自動委員会モードでは、80%に達すると**ローカル学習の追加更新を一時停止**し、WebRTC通信、認証済み証明書の取得、旧世代切替、既存の認証済みGCを継続します。条件成立後は再開します。署名済み履歴は勝手に削除しません。受信更新がさらに多い場合は上限に到達することがあります。
3. `python kcia.py journal-status --journal PATH` を追加。ルームの世代管理、署名委員会、提案、投票、確定証明書、アクティブ世代ポインタの**保存済みメタデータ**を読み取り専用で表示します。出力は問題切り分け専用で、署名検証や削除承認の代用ではありません。
4. 上限エラー本文に `journal-status` を案内するよう修正。

## 手順（まずデータをバックアップ）

学習・GUIを停止し、イベントDB、`.fplog`、`-wal`、`-shm`、`.live-epoch`、関連モデルとOptimizer・鍵・carryを**まとめて**バックアップしてください。単独ファイルのコピーでは整合性が保証されません。署名履歴を手で消したり、SQLiteのイベント行をDELETEしたりしないでください。

```
# リポジトリで、GUIが使用している正確なジャーナルを探す
find .kcia-gui/rooms -name events.sqlite3 -type f

# GUIの実際のジャーナルパスを指定する
python kcia.py journal-status --journal .kcia-gui/rooms/ROOM/peers/PROFILE/events.sqlite3
python kcia.py disk-report --journal .kcia-gui/rooms/ROOM/peers/PROFILE/events.sqlite3
```

- `GUI pinned generation owner: False` → v88以前から使っている既存ルーム等で、自動世代管理の対象外。**既存の署名ルームのマニフェストを後付けで改変しない**こと。
- `owner-signed 3-peer roster: False` → ルーム作成者を含め、独立した3ピアが同時に認証できていません。少なくとも初期委員会構成時は3ピアの接続が必要。
- `certificates: 0` → 完全なチェックポイントと2/3の独立した署名がまだ成立していません。各ピアのログで `COMPACT EPOCH CANDIDATE`、`compact committee verification deferred` を確認。
- `certificates` があって `active epoch: None` → 切替・prewarm・carry中の障害をログで確認。
- `active epoch` があって旧ログが残る → `V88 COMPACT GC` / `certified old journal GC safely deferred` の理由を確認。

## すでに上限に達している場合

空き容量があり、3ピアでの認証済み世代切替が成立可能なら、GUIの Journal GiB を一時的に 8 などへ上げてから、v89で起動してください。これは**一時的な猶予**であって解決保証ではありません。空き容量がなければ増やさないでください。3ピアの委員会を用意できなければ、署名プロトコル上、認証済みのまま履歴を削除することはできません。デバッグ専用で履歴を破棄してよい場合に限り、停止後 `reset-local-data --yes` で初期化して新しいルームを作成してください。

## 限界

- v89は証明書・モデル・Optimizerの完全自動**成立を保証しません**。本当に2/3の署名が必要です。旧ルームの自動移行、認証抜きの履歴切捨て、既存の4GiBジャーナルの無条件圧縮は実装していません。
- 新しい容量抑制は、無証明で削除しないため、委員会不成立ではローカル学習が**一時停止したまま**になる可能性があります。
- テストはローカル中心。3台以上の実機での長時間学習は未検証です。
