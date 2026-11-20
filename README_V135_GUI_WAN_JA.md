# KCIA v135 — WAN-first GUI（kcia.tech / v134 Training First 統合）

## 起動

```bash
python -m pip install -r requirements-gui.txt
python -m patchlib.cli gui
```

ブラウザ内のWebRTCデモではなく、KCIAのデスクトップGUI（`web/desktop_webview.html`）で有効になります。Mac/Linux上でPyWebView対応のデスクトップセッションが必要です。

## ネットワーク

- 通常運用の既定は **WAN**。実際の接続先は **`wss://signal.kcia.tech/ws`** （`kcia.tech`の実機で検証済みのシグナリング用サブドメイン）。`wss://kcia.tech/ws` の利用可否は確認されていないため使用しません。
- `MODELS` 画面では公開ディレクトリの照会をローカル一覧より先に発行します。WANのルームを優先表示し、手元で保存された参加済み・非公開ルームも合流します。公開リスト自体はサーバーが公開を許可したルームのみ表示します。
- `NETWORK`の `LOCAL · デバッグ専用` を明示選択すれば `ws://127.0.0.1:8080/ws` に切り替えます。GUIはローカルのMummy/Nimサーバーを自動起動しません。
- 過去に保存された **v1 のデフォルト localhost 接続** は、WANへ移行します。v135で明示したローカルデバッグ選択はv2設定として再起動後も保持されます。独自のWSS設定も保持します。
- 「接続テスト」は実際のWSS ping と公開ディレクトリ照会です。サーバー未到達時はGUIを停止せず、エラーをNETWORKに表示します。存在しない公開ルームを捏造しません。

## GUI TRAINING設定

`TRAINING`画面でワーカー起動前に変更できます。

| 項目 | WAN既定 | 設定可能 |
|---|---|---|
| 公開間隔 | **32ローカルミニバッチ/署名更新** | カスタム 4〜256（4の倍数）、従来の自動調整 |
| 自動資源調整 | 維持 | モデル/メモリによる初期ポリシーは維持 |
| 自動更新間隔の下限 | 32 | カスタム値を下限にする。自動調整に戻すと下限を解除 |
| ReLoRA再初期化 | **20イベント** | 既存ルームの署名済み定義が優先 |
| スナップショット中 | **学習継続** | 学習停止（従来の安全優先） |
| TURNトランスポート | **auto** | tcp / udp |
| ICE候補 | **auto、relay限定OFF** | TURN限定モード（診断用） |
| 状態シャード | 1 MiB | GUI版固定（CLI単独は従来仕様） |

UIに表示される `Steps / publication` は**ワーカーの署名イベント生成間隔**です。ReLoRAの20イベントは別のルーム設定です。

GUI操作時、`--local-steps` / `--min-publish-steps` / `--snapshot-priority` / `--turn-transport` / `--no-relay-only` が `patchlib.cli train-live` に渡ります。`--min-publish-steps` は自動資源調整によるローカル同期間隔の不用意な短縮を防ぎます。ピア署名、イベント形式、公開ルームマニフェスト、既存のチェックポイント形式は変更しません。

GUIのトレーニング設定は **`.kcia-gui/training-preferences.json`** にモード別の機密情報なしで保存します。保存は原子的に行い、ディレクトリは0700・ファイルは0600、値の妥当性を検査して復元します。学習中に編集しても既存ワーカーへ即時反映せず、次回起動で有効になります。

## 既存データに関する注意

- GUIのルーム・署名鍵・ジャーナルは `~/.kcia/...` 等のGUI管理データルート内の `.kcia-gui/` に格納されます。実際のルートは環境依存です。
- 以前のCLI実験で使った **`wan_state/` は自動的にはGUIへ移行しません。** 既存ルームをGUIで使う場合、モデル定義を公開ディレクトリまたは参加リンクから受信し、GUI自身の署名ピアとして参加します。GUIのイベント履歴とCLIのイベント履歴を無検証で統合しません。
- 過去の署名ジャーナル、鍵、`wan_state/` を削除しないでください。モデル・ジェネシス・データセット条件が異なれば、署名済みルームへ強制参加させずエラーにします。
- VPSの既存 `signal.kcia.tech` / `turn.kcia.tech` は変更不要です。DNSやCloudflare側の設定変更もありません。

## テスト

```bash
python -m pytest -q tests/test_v135_gui_wan_defaults.py tests/test_public_gui_network_v113.py tests/test_webview_gui.py
node --check web/desktop_webview.js
```

上記テストはネットワーク通信やモデル学習をエミュレート/モックしたオフライン試験です。Mac・Colab双方のGUI実機で長時間WAN学習できることの証明ではありません。
