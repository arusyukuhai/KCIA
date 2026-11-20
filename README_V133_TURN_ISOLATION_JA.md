# KCIA v133 — TURN/TCP relay-only WAN pilot

## 実機v132ログから確定できること

- Colabプロセスが `role=mac` かつ `device=cpu` で動いていた。正しくは `--role colab` + GPUランタイム。
- Mac/Colabとも P2Pログが `ICE TURN transport=auto` だった。以前試験したTURN/TCPを実際の学習では指定していなかった。
- ICE/DTLS/DataChannel/署名認証の成立は確認済み。ただし、スナップショット送信は約2.4 MiBで停止し、ICE consent expirationで切れる。
- SNAPSHOT PRIORITY HOLD は接続の間だけ効く。切断後に学習イベントが増えると、再接続時のマニフェストが変わり、シャード再利用率も下がり得る。

## 変更

1. WAN pilot launcherは既定で `KCIA_TURN_TRANSPORT=tcp` と `KCIA_FORCE_TURN_RELAY=1` を子プロセスに設定する（明示環境変数で上書き可能）。`P2P ICE TURN transport=tcp relay_only=True` とログに表示する。
2. 中核の `PatchNode` は `KCIA_FORCE_TURN_RELAY` が有効のときのみ、送信SDPをTURN relay candidateに限定する。一般のP2Pでの挙動は変わらない。
3. WAN pilotのシャード既定を1 MiBへ縮小。切断前に1個でも完全検証しやすくする（部分シャードのバイト単位再開ではない）。
4. Colabノートブックでも TURN/TCP relay-only を明示設定する。
5. 受信者が署名済みスナップショットをACKするまで（切断・再接続を挟んでも）送信者の新規optimizer stepを一時停止する。パイロット専用で最長15分。これにより再接続ごとにマニフェストが変わることを抑え、検証済みシャードの再利用を成立させやすくする。

## Mac

```
python scripts/start_wan_pt.py run --role mac --room-id 'ROOM_ID' --revision 'REVISION' 2>&1 | tee wan_mac_v133.log
```

## Colab（GPUを有効化）

```
python scripts/start_wan_pt.py run --role colab --room-id 'ROOM_ID' --revision 'REVISION' 2>&1 | tee wan_colab_v133.log
```

両端のログ先頭に `KCIA WAN ICE route: tcp relay_only=1` と、接続時に `P2P ICE TURN transport=tcp relay_only=True` が必要。Colabでは `KCIA COMPUTE DEVICE actor=cuda` も確認。

元の `wan_state` / 鍵 / journal は削除しない。Colabの旧 `wan_state/mac` は過去の署名履歴として残るが、`--role colab` は別の状態フォルダで新しく起動するため、その古い学習イベントを継承するわけではない。

## 残る課題

この版は経路と実験設定の取り違えを防止するための診断修正であり、ConoHa/Colab実WANで転送が成功した証明ではない。接続が維持できても、初期スナップショット約326 MBや巨大な署名イベントの通信量、切断後にソースイベントが増える問題は別途設計改善が必要。
