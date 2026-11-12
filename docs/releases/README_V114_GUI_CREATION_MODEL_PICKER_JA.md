# KCIA v114 — Create Model 修正 / Hugging Face モデルピッカー / KCIA SVG

## 原因と修正

v113 の `web/desktop_webview.js` は、まだ初期化されていない `const safe` をネットワークボタンのイベント登録時に参照していました。JavaScriptが `ReferenceError: Cannot access 'safe' before initialization` で停止し、`Create Model` をはじめ、その後のGUI操作が登録されませんでした。`safe` を関数宣言に変更し、ブラウザで実際の起動・ボタンクリックを検証しました。

## 変更内容

1. **Create Model**: ローカルmanifestの保存後に即座に MODELS へ切り替え、サーバー登録は別途行います。VPSが停止していてもローカル保存を失敗扱いしません。登録に失敗したらトースト通知し、既存ルームの登録をGUIから再試行できます。保存中はボタンを無効化し、失敗したら理由をフォームに残します。
2. **モデルピッカー**: `NEW MODEL` → `モデルを検索` でHubの公開モデルをタスク別に検索できます。`Browse models` は英語表示時のラベル。選択するとモデルID・40桁コミットSHA（取得できない場合はmain）を反映し、重みコピー方式では pretrained に切り替えます。これは検索メタデータであって、その端末でのロード・学習成功やモデルのライセンスは保証しません。
3. **ロゴ**: ユーザー提供SVGを改変せず `web/assets/kcia.svg` に同梱。サイドバーでは同一パスデータのSVGをインライン配置（WebViewのCSP/オフライン表示を維持）。
4. **内部ID**: 新規ルームIDの入力は引き続き不要。既存のルーム・暗号鍵・ジャーナル/署名イベントは消去・移行しません。

## 実行

```bash
python -m pip install -r requirements-patches.txt
python -m pip install -r requirements-gui.txt
python kcia.py
```

GUI: `NEW MODEL` → `モデルを検索`（または手入力） → 必要ならデータセット選択 → `モデルを作成`。初期状態はTinyStoriesサンプルのモデル・データセットが入力されています。

ネットワーク設定: `NETWORK` → `INTERNET` → `wss://signal.<your-domain>/ws` → 保存・接続試験。ルームを作成してから登録されます。認証アプリトークンは不要ですが、VPS側のTURN_SECRET・Cloudflare Tunnelトークンはサーバー内で引き続き必要です。

## テスト範囲

- `node --check web/desktop_webview.js` / `python -m compileall -q patchlib`（構文）。
- `python -m pytest -q tests/test_v114_create_model_and_picker.py tests/test_hub_discovery_v70.py tests/test_public_gui_network_v113.py tests/test_webview_gui.py`。
- `python tests/browser_create_model_v114.py`（ChromiumとPlaywrightがある環境）。CSS・JSをまとめた実ページでモデルピッカー、モデル選択、`Create Model`、一覧表示まで操作します。Python側のRPCはモックです。
- 実際のCloudflare/TURN/VPSとMac/Colab間のネットワーク通信、WebView固有のウィンドウ実行は、環境情報がないため未検証です。

アップデート前にはコードとローカル状態をバックアップしてください。既存のルーム・鍵・ジャーナルを含むデータディレクトリを新しいZIPで上書き・削除しないでください。
