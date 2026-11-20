# KCIA v134 — WAN学習優先・低頻度通信版

## 変更の理由（実機ログに基づく）

- `JOIN READY` が記録され、署名付き更新を `applied=2` まで受信できた。WSS/ICE/TURN・署名認証は機能している。
- その一方で、Macは `SNAPSHOT PRIORITY HOLD` の間に学習が停止し、約326 MBの初期モデル転送に時間を使った。
- Colabの署名付き更新1イベントは約158 MB、Macは約11.5 MB。これを4ミニバッチごとに発行すると、計算より通信・ジャーナル増大が支配的になる。

## 新しいWANパイロット既定値

| 項目 | 旧WAN v133 | v134 | 理由 |
| --- | --- | --- | --- |
| `--local-steps` | 4 | **32** | 約8倍のローカル処理を1イベントへ集約 |
| `--reinit-every` | 20イベント | **20イベント** | 既存ルームのgenesis/署名履歴と互換性を維持 |
| `KCIA_SNAPSHOT_PRIORITY` | 1 | **0** | 初期モデル配信中もローカル学習を継続 |
| `KCIA_TURN_TRANSPORT` | tcp | **auto** | 実機で接続成功した設定に戻す |
| `KCIA_FORCE_TURN_RELAY` | 1 | **0** | 不要なrelay限定を解除 |

**重要**: 4×20は通信間隔ではない。4個のローカル *minibatches* で1つの署名付きイベントを発行し、20イベントごとにReLoRAリセットしていた。既定の32×20では理論上640 minibatches（CUDAのgrad_accum 4では約160 optimizer updates）ごとにリセットされる。ローカル勾配蓄積、明示seal、データ/ロスター変化によって実際のイベント間隔は変わり得る。

**通信データ量**：Colabの158 MB/イベントが変わらないと仮定すると、同じ640 minibatchesの署名イベントは旧4設定で160件→新32で20件（理論上、通信とジャーナル負荷を約87.5%削減）。実測ではパッチの変化や中断/追いつき転送により異なる。これは**イベント数を減らす**対策であり、1件158 MBを圧縮する修正ではない。

## Mac起動（既存ルーム/ジャーナルを維持）

```bash
KCIA_TURN_TRANSPORT=auto KCIA_FORCE_TURN_RELAY=0 \
python scripts/start_wan_pt.py run --role mac \
  --room-id 'ROOM_ID' --revision 'REVISION' \
  --local-steps 32 --snapshot-priority training \
  2>&1 | tee wan_mac_v134.log
```

## Colab起動（GPUランタイム使用）

```bash
KCIA_TURN_TRANSPORT=auto KCIA_FORCE_TURN_RELAY=0 \
python scripts/start_wan_pt.py run --role colab \
  --room-id 'ROOM_ID' --revision 'REVISION' \
  --local-steps 32 --snapshot-priority training \
  2>&1 | tee wan_colab_v134.log
```

32がまだ通信過多なら、Colabのみ `--local-steps 64` にして比較可能（ただし更新が古くなる時間・学習収束への影響を評価）。`--local-steps` の変更だけならroom genesisを変更しないため、同じルームで比較可能。**`--reinit-every` はルーム側で固定される重要な識別子であり、既存ルームでは変更しない。**

## スナップショット優先モードについて

新しい `--snapshot-priority training` は旧v133の長時間全停止をやめる。モデル転送はハッシュ/署名付きのある時点の状態として扱われ、その後の署名イベントをsuffixとして同期する既存の仕組みを使う。ただし、**同期中に作られたイベントは後で送る必要があり、WAN負荷はゼロにはならない**。接続不安定なら `--snapshot-priority snapshot` へ戻して安全性を優先できる。

中核の署名プロトコル・シャード形式・公開Nimサーバー・TURN設定・モデル・LoHAの物理rankは変更なし。ソースアップデートで既存`wan_state/mac`,`wan_state/colab`を消さない。モデル/署名鍵/ジャーナルは**端末間でコピーしない**。

## まず確認する項目

ログに `KCIA TRAIN cadence local_minibatches_per_event=32 reinit_every_events=20 ... snapshot_priority=training` が出る。`elastic steps` に `publish_target=... grad_accum=... local_minibatches_total=...` を追加。両端で`applied`が増えること、`emitted`が旧版より緩やかに増えること、lossが再開することを観察する。短いお試し文だけで品質改善は結論できない。

**未検証事項**: ConoHa+Colab+Macの実WAN学習をこの生成環境から再実行することはできない。配布物はローカルの単体/回帰テストで検証する。
