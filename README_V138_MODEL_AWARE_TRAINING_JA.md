# KCIA v138 — モデル規模連動WAN通信／モデル作成時の学習方式選択

## 変更概要

1. WANの学習更新発行間隔をモデル規模に応じて長くする。初期値はHugging Face configのモデル構造または明示されたパラメータ数から概算し、学習エンジン起動後は実際のモデルパラメータ数と**署名付き更新のシリアライズ済みバイト数**で再評価する。参加ピアごとのローカル助言設定なので、署名済みモデル・Genesis・ルームID・ReLoRAのイベント数は変わらない。
2. 新規モデル作成の「SFT・PPO・Preferenceのデータセットと詳細を設定」ダイアログで、PT、SFT、PPO、Preference-SFTを個別にON/OFFにできる。重み0で停止するだけでなく、無効なSFT/PPOソース情報をワーカーに渡さない。**少なくともPT/SFT/PPOを1つ有効にする**。
3. 既存の参照モデル不要のペアワイズ選好最適化（SimPO系、DPOではない）を、ローカルJSONLに加えて、**Hugging Faceの固定40桁Revisionのデータセット**でも実行できるよう拡張した。SFTデータセットのprompt/response（chosen）/rejected列を明示できる。必要な列がない場合は重量モデルのロード前に拒否。
4. PTデータセットのローカル選択をピアごとに保持できる（署名済みルームmanifestには入らない）。同時起動する別GUIピアの設定は混同しない。

## WANの自動送信間隔

MODELSまたはNEW MODELでモデルを選び、TRAININGの「送信間隔」を **WAN · モデル規模と更新サイズで自動調整** にする（既定）。

| 概算パラメータ数 | 推奨間隔の例（ミニバッチ／署名付き更新） |
|---:|---:|
| 約140M | 32 |
| 約250M | 32 |
| 約1.7B | 128 |
| 約8.7B | 256 |
| 約86B | 1024 |

表の数値はGUIの概算例であり、実際の更新バイト数やルーム状況により長くなる。端末が大規模モデルをメモリ内に収容できることを保証しない。実機計測の `KCIA WAN AUTO CADENCE` ログと `publish_target` を確認すること。

- `カスタム（手動）` を選ぶと4～1024の範囲で指定できる。
- `従来の自動調整（デバッグ）` も残す。WANの自動推定は可変でも、ReLoRAの `reinit_every` はルーム側の固定イベント数を維持する。
- 低帯域の実機・接続切断時は長い間隔が学習の品質／遅延にどう影響するか測定する。長期の収束は未検証。

## モデル作成時の学習方式

**NEW MODEL** → **SFT・PPO・Preferenceのデータセットと詳細を設定** → PT/SFT/PPO/Preference-SFTを個別切り替え → 対応するローカルJSONLまたは固定RevisionのHFデータセットを選ぶ → 保存。混合方式は現時点で `causal-lm` に対応する。

| 方式 | 必須条件 | 詳細 |
|---|---|---|
| PT | 既定のルームまたはピアごとのPTデータセット | 自己回帰事前学習 |
| SFT | 正解出力付きHFまたはJSONL | Response-masked CE + 既存のラベルスムージング |
| PPO | プロンプト・報酬参照または外部Reward Model、Critic | 既存のClipped PPO、各種安定化設定 |
| Preference-SFT | SFTをON、`prompt` / `response`(chosen) / `rejected` | 参照モデル不要のペアワイズ選好損失（SimPO系） |

OFFの方式に関係するローカルデータセットは読み込まない。PPO報酬／Critic／Preference係数など**学習を再現する数値設定は署名済みルーム方針として固定**されるため、後で変えるには新ルームを作る。各ピアのデータセットパスやHFコーパスは個別に保持し、公開ルームのmanifestへ埋め込まない。

### コマンド

```bash
python -m pip install -r requirements-patches.txt -r requirements-gui.txt
python -m patchlib.cli gui
```

既存の `wan_state/`、`.kcia-gui/`、署名鍵・SQLite・学習済みチェックポイントは消さない。VPS/coturn/Cloudflareは変更不要。

### 検証

```bash
python -m pytest -q tests/test_v138_cadence_objectives.py tests/test_v138_cadence_objectives_sources.py tests/test_v137_gui_audit.py
node --check web/desktop_webview.js
```

実機でのWAN収束・追加方式の精度比較は未実施。学習方式の追加は既存の数学的損失と実装に基づくものであり、DPO/RLHFの新規完全実装を意味しない。
