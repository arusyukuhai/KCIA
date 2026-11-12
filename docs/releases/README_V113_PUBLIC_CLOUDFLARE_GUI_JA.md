# KCIA v113 — 公開 Cloudflare Relay & 実通信GUI

## 機能

- `APP_TOKEN` は Nim シグナリング認証、Compose、VPS設定生成、WANノートブック、利用手順から撤廃。
- `ROOM_PUBLISH_TOKEN` による公開モデル登録の運営者制限も撤廃。匿名クライアントによる新規公開登録を許可。
- Cloudflare Tunnel は WSS/HTTP のみ。TURN UDP/TCP は VPS `coturn` に直接接続。短期 HMAC 認証を継続して静的 TURN password を配布しない。
- GUI `NETWORK` に LOCAL / INTERNET と WSS接続設定・永続化・疎通確認（WSS ping/pong + HTTP directory）。
- GUI `NEW MODEL` は room ID の入力不要。IDは作成時に裏側で自動生成し、同じモデルを選んで再開する場合は保存済IDを使用。
- モデル作成時にサーバーへ自動登録。ネットワーク不通時はローカル作成を保ち、GUIに明示する。
- 招待URLに `ws` と `room` を格納することで、非公開ルームをGUI同士で共有（このURLは機密扱い）。
- 元の `train-live` 起動・署名・P2Pルーム制約・ジャーナル形式は変更しない。

## テスト

- Pythonテスト: 75件通過（関連領域を選択実行）
- GUI JS構文チェック: `node --check web/desktop_webview.js` 通過
- Colabノートブック: 各PythonコードセルのAST構文検査通過
- 既存テスト `tests/webview_click_smoke.js` は簡易JSモックに `document.querySelectorAll()` がないため単独実行できず（今回のUIに限らない既存テストハーネスの制限）
- Dockerビルド、Nimコンパイル、Cloudflare実接続、Mac/Colab間のTURNデータ通信・LLM学習は未実施（実環境が必要）。

## 公開前の運用事項

- Cloudflare 側で `/ws` と `/api/rooms` へのアクセス急増を監視・制限する。既存 Nim の WebSocketごとのメッセージ数・バイト数上限はそのまま有効。
- `coturn` のクォータと UDPリレー用ポート範囲はPoC向け。公開TURN接続はVPS帯域費用の発生源になるため、帯域・エラー率・割当数を監視する。
- public directory は 256ルーム、room metadata は合計 4096ルームの上限があり、現状では長期放置された公開ルームの自動整理は未実装。公開サービスの長期運用前に TTL/GCとabuse policyが必要。
- 公開ディレクトリ掲載の部屋は誰でも列挙可能。非公開ルームの招待URLは room ID を含むため公表しない。
- `KCIA_SIGNALING_URL=wss://.../ws` をGUI初期起動時の接続先として任意に指定できる。GUIから明示的に保存した設定があればそちらを優先する。
