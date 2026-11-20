# KCIA v142 — 学習を止めないための補助ファイル耐障害修正

**v141からの差分**。署名方式、ジャーナル形式、LoHA、PT・SFT・PPOの学習内容、GUI/WAN既定URLは変更していません。実運用の保証ではありません。

## 実際に再現して修正した不具合

1. **バンディット補助キャッシュで学習起動が落ちる**：`*.wan-bandit-v139.json` が `[]`、`null`、数値、または不正な `history` を含むと、 `obj.get` 等で `AttributeError`/`TypeError` が発生する場合がありました。ルートと履歴行の型を検証し、破損キャッシュを無視します。署名済み履歴は保持します。
2. **診断値のNaNで学習が落ちる**：`live_telemetry.LocalTelemetry.record` は処理時間や通信MiBの `NaN/Infinity` が `json.dumps(...allow_nan=False)` へ渡ると `ValueError` が発生していました。補助診断値を有限数値へ正規化し、統計出力が学習を停止しないようにします。Loss自体が非有限なら `null` で記録します。
3. **末尾の破損レコードでグラフ停止**：最新1件を読む処理が、末尾に無効な完成済み行があると常に空配列を返していました。最大読み取り量の上限を保ちつつ末尾から正常行を探します。GUI再開時のステップオフセットも正常行を用います。
4. **長期履歴の内部にJSON `null` 等があるとグラフAPIが例外**：バイナリ検索の途中でオブジェクト以外のJSONを読み `AttributeError` が出る場合がありました。JSONの型を検証し、破損レコードは読み飛ばします。

## テスト

- `tests/test_v142_reliability.py` を新設。破損保存状態、NaN/Infinity、末尾・中間の壊れた行、グラフ復元を再現するテストを追加。
- `tests/test_local_telemetry.py` の古い想定（壊れた末尾行があると全履歴が空になる）を、最後の正常行を保持する期待値に変更。
- WAN・署名ジャーナル・GUI・スナップショット・WebRTC再接続まで193件の関連回帰テストを通過。

## 更新

GUIおよび学習ワーカーを終了し、v141のリポジトリで以下を実行します。

```bash
cd ~/Documents/GitHub/Fedpara
git apply --check ~/Downloads/KCIA_v142_from_v141_reliability.patch
git apply ~/Downloads/KCIA_v142_from_v141_reliability.patch
python -m patchlib.cli gui
```

現在の `wan_state`、`.kcia-gui`、`*.sqlite3`、秘密鍵、チェックポイントは削除しません。

## 未解決・未検証

実機Mac/Colabの長時間WAN学習、通信経路の断続的障害とバンディットの汎化性能はこの環境では未検証です。`wan-bandit-v139.json` は学習を加速するための補助統計であり、破棄するとそのピアの自動通信間隔の学習済み判断は初期化されます。
