# KCIA v122 — 学習設定モーダル・旧フォント復帰・異ランクLoHA送信修正

## 不具合の原因

1. **`missing heterogeneous rank protocol`**：v116で `UniversalAdapter.export()` にだけ異ランク宣言を付けた一方、普段の継続学習に使われる `ContinualElasticEngine._make_wire()` は `adapter._metadata()` を直接呼んでいました。低ランクの送信イベントから `rank_protocol` が欠け、受信・自己検証で拒否されていました。
2. **更新量のランク依存バグ**：`_factor_tile()` と `FrozenDelta` が相手の実ランクではなくローカルのrankでLoHAの尺度を計算する経路があり、異なるrankの参加者間で重みがずれる可能性がありました。
3. **学習設定フォームが画面内へはみ出す**：v119は既存の大きな `mixedPanel` を `NEW MODEL` 画面の末尾にappendしていました。固定高のウィンドウと画面スクロール抑止によりフォーム下部に届かなくなっていました。
4. **フォントが変わった**：v120/v121でフォント取得処理と優先順位を変更し、v118までのMyriad Condensed優先が失われました。

## 修正点

- `adapter._metadata()` が低ランクLoHAの `rank_protocol="heterogeneous-loha-v1"` を生成するため、`export()` 以外の署名経路でもプロトコル情報が付きます。検証強度は緩めていません。古い不正な署名イベントを無断で改変・再署名しません。
- LoHAの因子再構成経路で、実際に検証した送信因子のrank幅を使います。
- `NEW MODEL`の「学習方式・SFT/PPOデータセットを設定」ボタンから**独立したモーダル**を開きます。設定欄だけが縦スクロールし、上部タイトルと下部の「設定を反映」「閉じる」は常時表示します。背景クリック・Escで閉じることができます。Hubピッカーの入れ子にも対応します。
- CSSのMyriad Pro Condensed → Noto Sans JPの優先順位をv118相当に復元しました。外部フォントを利用できる環境では、ページ読み込み後に非同期でスタイルシートを要求します。ネットワークが使えなければmacOSのローカルMyriad/Noto/Hiraginoなどで代替表示します。**フォントそのものは本ZIPに同梱していません**。
- 新規ルームの送信間隔4ステップ・LoHA再初期化間隔20送信、その他の学習・署名ジャーナル形式は維持しています。

## テスト

- Chromium（1400×880、1060×650、800×510）でモーダルの開閉・スクロール・Esc・操作ボタンの固定位置を確認。
- rank 2の署名付き継続FedSum更新をrank 8ピアが受信し、rank 4ピアがジャーナル再生できることを実テスト。rankの異なる因子タイルと全量復元が一致することを確認。
- GUI操作・署名付き異ランク・ReLoRA・Factor Ledger等の関連54件のテスト成功。macOS WKWebViewとVPS/Colabの実ネットワークは未検証。

## 更新

KCIAの学習を停止してから、**コードのみ**v122へ更新してください。既存の署名鍵、重み、Optimizer、ジャーナルを消さないでください。

```sh
python -m pip install -r requirements-gui.txt -r requirements-patches.txt
python kcia.py
```

GUIがネイティブ側で開かない場合、`python kcia.py --gui-safe` でOSタイトルバーを用いた復旧起動ができます。

古いクライアントが自動生成してしまった、`rank_protocol` 欠損の低ランク署名イベントについては**検証を迂回せず拒否**します。解決しない場合はログとイベントIDを保存し、後から安全にジャーナル整合性を調査できるようにしてください。ライブ通信の相手もv122へ更新する必要があります。
