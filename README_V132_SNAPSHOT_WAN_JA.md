# KCIA v132 — WANモデル同期の安定化（Mac / Colab）

## 実機ログから分かったこと

- 異なるMac・Colab上で WebRTC ICE / DTLS / DataChannel / 署名付きHelloの確立まで確認済み。
- その直後、Colabから約326 MBの完全モデルをMacに同期しようとしたが、4つのシャードが残ったまま ICE consent expired で切断。
- Colabの署名付き更新は **1件約158 MB** と大きい。初期同期中にも別DataChannelから大量送信すると、SCTP/TURNの帯域や送信時間を争う。
- これは同期待ちに対する対処であり、実WANの安定接続・全量同期・協調学習の成功はまだ保証していない。

## v132で行った変更

1. `kcia-state` の送信量を **既定1.0 MiB/s** に協調的に制限。16 KiBフレームごとにイベントループへ処理権を渡し、ICE/STUNのタイマーが動く機会を確保。
2. `kcia-patch` の大きな署名済み更新も **既定1.0 MiB/s** で制限。スナップショット中はパッチ送信をフレーム境界で一時停止して、別DataChannelからの大容量転送が競合しないようにした。署名・内容は変更しない。
3. 初回Mac/Colabパイロットのスナップショットを **2 MiBシャード** に変更。接続が切れた場合、検証済みシャードだけディスク上で再利用し、部分シャードのみ取り直す。内部の署名/ハッシュ検証は変更しない。
4. `SNAPSHOT TX` / `SNAPSHOT RX` の進捗ログを約5秒ごとに出す。受信サイズ、検証済み/再利用シャード数、SCTPバッファ量を観測できる。
5. パイロットのみ `KCIA_SNAPSHOT_PRIORITY=1` を既定で有効化。送信中のColabは次のローカル学習ステップを一時停止し、ジャーナル肥大とネットワーク競合を抑える。完了後は再開。

**互換性**: WebRTCのオンワイヤ形式、ルームID、ジャーナル、署名鍵、モデルRevisionは変更なし。VPSのNim/coturnを更新する必要はない。Mac/Colabの両側をv132に揃えることを推奨。

## 起動

両端でv132 ZIPを展開し、依存関係をインストールする。`wan_state/`がある場合、**上書き・削除しない**。

Mac:

```bash
python -m pip install -r requirements-patches.txt
python scripts/start_wan_pt.py prepare   # 完全に新しい実験をする場合のみ
KCIA_TURN_TRANSPORT=tcp KCIA_SNAPSHOT_RATE_MIB=1.0 KCIA_PATCH_RATE_MIB=1.0 \
  python scripts/start_wan_pt.py run --role mac \
  --room-id 'ROOM_ID' --revision 'REVISION' \
  2>&1 | tee wan_mac_v132.log
```

Colab (GPUランタイムでv132をアップロードして展開後):

```python
import os, subprocess, sys
os.environ['KCIA_TURN_TRANSPORT']='tcp'
os.environ['KCIA_SNAPSHOT_RATE_MIB']='1.0'
os.environ['KCIA_PATCH_RATE_MIB']='1.0'
subprocess.run([sys.executable, 'scripts/start_wan_pt.py', 'run', '--role','colab',
  '--room-id','ROOM_ID','--revision','REVISION'], check=True)
```

同じRoom IDと40文字のrevisionを両端で指定する。学習状態・署名鍵は別々のまま。初回のローカル学習結果を捨てたくない場合は以前のroom/revisionを使用するが、古いイベントが多いと境界検証に時間がかかる。Colabのランタイム消去時に `wan_state/`を失わないよう保管。

## 進捗と成功判定

```text
SNAPSHOT TX sent_mib=... buffered_kib=... rate_limit_mib_s=1
SNAPSHOT RX received_mib=... verified_shards=... reused_shards=... outstanding=...
SNAPSHOT DOWNLOADED peer=...
SNAPSHOT READY: ...
JOIN READY: ...
elastic ... applied=... channels=1 authed=1 synced=2
```

`ICE=completed` と `SIGNED HELLO OK` だけでは**同期完了ではない**。`SNAPSHOT READY` / `JOIN READY` のあと、両端で `applied` が増えてはじめて署名付き更新の協調を評価できる。初回約326 MBなら転送速度1 MiB/sで**理論上5分強**、実際はさらにオーバーヘッドがかかる（バックグラウンド実行ではなくユーザー端末で進む）。

もし45秒前後で `Consent to send expired` が再発したら、原因は帯域競合以外の TURN/ICE keepalive、SCTP輻輳制御、コンピュート負荷等に残っている。ログの `SNAPSHOT TX/RX` に加えてcoturnのログも採取する。速度を上げる場合は両端で `KCIA_SNAPSHOT_RATE_MIB=2.0` 等（0.125～32）を試せるが、まずは1.0で検証。

## 注意

- 既存データ・署名鍵・ジャーナルは絶対に削除しない。
- `KCIA_SNAPSHOT_PRIORITY=0` でパイロットの学習ステップ一時停止を解除できるが、今回の実機テストでは非推奨。
- 「検証済みのシャードを再利用する」もので、切断したシャードの途中からバイト単位で続けるわけではない。
- `KCIA_PATCH_RATE_MIB` も大量のイベント交換時の安定性優先。150 MB級イベントが多数ある場合は同期に時間がかかる。
- このビルド環境には`aiortc`がなく、実WAN・ConoHa TURN経由の接続テストは未実施。ローカルでの転送・切断復帰テストを実施済み。

## 回帰テスト

```bash
python -m pytest -q tests/test_v132_snapshot_priority.py tests/test_live_snapshot.py \
  tests/test_shards.py tests/test_snapshot_reconnect_regression.py \
  tests/test_snapshot_request_delivery.py
```
