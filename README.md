> **KCIA v142 — バンディット保存状態・GUIグラフ・診断値のクラッシュ耐性修正**。詳細は [README_V142_RELIABILITY_JA.md](README_V142_RELIABILITY_JA.md)。

# KCIA — P2P分散LLM学習（v142）

KCIAはWebRTC / WSSシグナリングを利用する分散PT・SFT・PPOの研究実装です。
LoHA/LoRAによる局所学習、署名付きGossip、ジャーナルと再接続復旧に対応しています。
**研究用ソフトウェアであり、WANの長期安定性・収束を保証するものではありません。**

## まず起動する

```bash
python -m pip install -r requirements-patches.txt -r requirements-gui.txt
python -m patchlib.cli gui
```

- 通常のGUIは **WAN / kcia.tech** が既定値です。接続先は実機検証済みの `wss://signal.kcia.tech/ws`。
- **MODELS** は公開WANルームを優先し、参加済み・端末内のルームとは区別します。
- **TRAINING** はモデル規模と実測更新サイズから**32～1024ミニバッチ／署名付き更新**を自動推定（最小32、上限は端末設定に準じる）、ReLoRA再初期化はルームの既定20イベント。
- `LOCAL` / `ws://127.0.0.1:8080/ws` は明示的に選ぶデバッグ接続（次回起動ではWANへ戻ります）。
- コントロールパネルの変更はローカル端末側のみであり、署名された共有ルーム設定を変更しません。

[学習方式・適応間隔の設定](README_V138_ADAPTIVE_OBJECTIVES_JA.md)、[WANバンディット](README_V139_ONLINE_COST_BANDIT_JA.md)を参照してください。

GUIが動かない場合は [GUIトラブルシューティング](docs/guides/README_GUI_TROUBLESHOOTING.md) を参照してください。

## VPS / Colab / Mac実機学習

[WANセットアップ](README_WAN_PILOT_JA.md) / [今すぐ学習](README_TRAIN_NOW_JA.md) / [VPS・Cloudflare構成](deploy/vps/README_CLOUDFLARE_JA.md) / [Colab notebook](notebooks/KCIA_Cloudflare_Mac_Colab.ipynb)

WAN制御は`signal.kcia.tech`の公開WSSと`turn.kcia.tech`のcoturnで構成します。
モデルと署名鍵・学習済み重みはシグナリングサーバーへアップロードしません。

## コード構成

| 機能 | 入口 |
|---|---|
| GUI / JS / OSウィンドウ | `patchlib/webview_gui.py`, `web/desktop_webview.js` |
| GUI設定・一覧・学習起動 | `patchlib/webview_controller.py`, `patchlib/webview_lifecycle_v110.py` |
| ローカルグラフ・履歴 | `patchlib/live_telemetry.py` |
| WEBRTC / ICE / TURN | `patchlib/p2p.py`, `patchlib/roomid.py` |
| 署名付き分散学習 | `patchlib/elasticnet.py`, `patchlib/shardp2p.py` |
| CLI / Wan PT | `patchlib/cli.py`, `scripts/start_wan_pt.py` |
| Nimシグナリング | `src/main.nim` |

[詳細なソースマップ](docs/SOURCE_LAYOUT_JA.md) / [ドキュメント索引](docs/INDEX.md) / [旧リリース記録](docs/releases/RELEASE_HISTORY_PRE_V137.md)

## 検証

```bash
python tools/audit_release.py
python -m pytest -q tests/test_v142_reliability.py tests/test_v141_bandit_churn_cleanup.py tests/test_local_telemetry.py
node --check web/desktop_webview.js
```

## 更新時の注意

**アップデート前に現在のソースをバックアップし、KCIAワーカーを終了してください。**
実行状態 `wan_state/`・`.kcia-gui/`・署名鍵・`*.sqlite3`・チェックポイントは削除しません。
既存のCLI状態とGUI状態は別の保存先を使用しており、自動的な統合は行いません。
