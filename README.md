> **KCIA v132: WANスナップショット安定化** — 詳細は [README_V132_SNAPSHOT_WAN_JA.md](README_V132_SNAPSHOT_WAN_JA.md) を参照。

# KCIA v130 — Mac + Colab live training pilot

[今すぐ学習を開始](README_TRAIN_NOW_JA.md) / [WAN接続準備](README_WAN_PILOT_JA.md)

# KCIA v129 — WAN pilot preparation

VPS + Cloudflare + Mac + Colab の実機接続は [README_WAN_PILOT_JA.md](README_WAN_PILOT_JA.md) から開始してください。v128の学習プロトコルは変更していません。

# KCIA v127 — macOS Adobe Fonts origin / diagnostics

[修正内容](README_V127_JA.md)

# KCIA v126 — Adobe Fonts load race and diagnostics

[修正内容](README_V126_JA.md)

# KCIA v125 — Adobe Fonts loading diagnosis and CSP fix

[修正内容](README_V125_JA.md)

# KCIA v124 — 完全ブランド統一

最新の変更: [README_V124_JA.md](README_V124_JA.md)。起動は `python kcia.py`。新規の招待リンクは `kcia://join?...` のみで、旧 `kagami://` は非対応です。

# KCIA v123 — Adobe Fonts Myriad Pro Semi Condensed

[フォント変更について](README_V123_JA.md)

# v121 — フォント・全ローカルリセット・GUI起動時の学習継続防止

[変更内容・完全初期化手順](README_V121_JA.md)

# KCIA v120 — 公開ルーム学習設定とGUI修正

**変更内容:** [README_V120_JA.md](README_V120_JA.md) を参照してください。新規ルームでは SFT/PPO の Hugging Face データセット参照と学習ポリシーを保存し、他のピアで復元します。データセットにアクセスできないピアは PT のみで参加する設定を選べます。新規ルームのReLoRA間隔は署名付き送信20回（公開ごとのローカルステップ既定4。旧ルームは既存設定を維持）。Nim サーバーもマニフェストの最大サイズを16KiBに対応させたため、VPS側のサーバーも更新してください。

# KCIA v118 — KCIA ロゴの表示上の左揃え修正

旧v117ではSVGの要素ボックスだけを左揃えにしていましたが、幅195pxに対して高さ49pxを上限にしたため、SVGの`preserveAspectRatio`既定動作で図形がボックス内で中央寄せされていました。v118では元SVGの縦横比を保つ表示幅138px・高さ自動に修正し、実際の図形左端をサイドバー本文の左端に合わせます。元のSVGパスデータは変更していません。

学習・通信・GUI機能はv117から変更していません。

# KCIA v117 — GUI responsiveness / window drag hotfix

- `Create Model`, network controls and other UI flows are retained from v116.
- Frameless native window actions now return a queued acknowledgement, and a separate
  serialized worker executes the actual OS calls after the WebView RPC has returned.
- Window control actions no longer wait for model/network initialization.
- The entire empty title strip and KCIA logo area are native drag handles. SVG
  hit-testing is disabled inside the drag handle; buttons/inputs remain clickable.
- The KCIA logo is explicitly aligned to the **left** of the sidebar.
- No stored model state, optimizer journals, peer IDs or signatures are migrated.

Known verification limit: Mac/WebKit's native window drag behavior must still be
confirmed on a Mac (this build environment has Linux Chromium, not Cocoa).

---

# v116 physical heterogeneous LoHA + frameless GUI

[設計・使用方法](docs/releases/README_V116_HETEROGENEOUS_LOHA_JA.md)

# KCIA v114 — Create Model 修正・モデルピッカー・KCIAロゴ

- GUIの初期化停止を修正: `safe` が初期化前に参照され、`Create Model` を含む後続ボタンの登録が止まっていた問題を解消。
- `NEW MODEL` にHugging Faceベースモデルの検索・選択ダイアログを追加。タスク別候補・名前検索に対応し、Hubが返すコミットSHAをリビジョンへ反映。
- `Create Model` はローカル保存完了後にすぐMODELSへ進める。Cloudflare/VPS停止中でもローカル保存を成功として扱い、サーバー登録は別途実行して失敗を明示。
- ボタンの保存中状態・失敗理由を表示。ネットワーク登録再試行が既存モデルを誤って新規作成しないよう修正。
- サイドバーのインラインロゴを添付されたKCIA SVGに正確に置換（原本は `web/assets/kcia.svg` に同梱）。
- room IDの自動生成、匿名パブリックシグナリング、既存のモデル/データの形式は変更なし。移行時の履歴・鍵・チェックポイントの削除は不要。

使い方・検証: [v114の日本語リリースノート](docs/releases/README_V114_GUI_CREATION_MODEL_PICKER_JA.md)。

---

# KCIA v113 — Public Cloudflare signaling and live GUI

- `APP_TOKEN` と `ROOM_PUBLISH_TOKEN` を廃止。公開シグナリング、モデル公開登録、リモート参加にユーザートークンは不要。
- `NETWORK` タブで `INTERNET` / `LOCAL` 接続先を保存、WebSocket ping/pong で疎通試験。
- `NEW MODEL` はルームID自動生成。作成時に自動で公開/非公開登録し、招待リンクでリモート参加。
- `TURN_SECRET` と Cloudflare Tunnel token はサーバー専用で存続。CloudflareプロキシはTURN UDPを中継しない。
- 学習は元の `train-live` WebRTC DataChannel / 署名付き Gossip エンジンを使用。VPS/Colab/Macの実機確認は別途必要。

手順: [deploy/vps/README_CLOUDFLARE_JA.md](deploy/vps/README_CLOUDFLARE_JA.md)

---

# VPS + Cloudflare Tunnel + Mac/Colab test package

[日本語セットアップ手順](deploy/vps/README_CLOUDFLARE_JA.md) / [Colab notebook](notebooks/KCIA_Cloudflare_Mac_Colab.ipynb)

# KCIA — 分散型LLM学習プラットフォーム

KCIAは、LoHA/LoRAによるモデル学習、P2P Gossip、PT・SFT・PPO、署名付き更新、世代移行、再起動後の学習復旧に対応する実験的な実装です。

**v112.2はGUI非依存の直列PT比較ベンチマークを、ピアごとの同一計算量と仮想ネットワーク遅延に対応させます。既存の学習アルゴリズム、署名形式、データベース形式は変更しません。**

## VPS（Cloudflare Tunnel/WSS + Nimシグナリング + coturn）・Mac / Colab 実機テスト

[導入手順（日本語）](deploy/vps/README_CLOUDFLARE_JA.md) / [Colabノートブック](notebooks/KCIA_Cloudflare_Mac_Colab.ipynb)。
この構成はv112.2の学習プロトコルを維持し、サーバーにモデルや鍵・重み・ジャーナルを置きません。

## 起動

```bash
python kcia.py                     # WebView GUI
python -m patchlib.cli --help       # 学習・診断・P2P CLI
python kcia.py ppo-doctor         # 小型CPU PPO更新の自己診断
python kcia.py mps-ppo-smoke      # Mac MPS の隔離実行テスト
```

依存ライブラリは目的に応じて `requirements-patches.txt`、`requirements-gui.txt`、`requirements-eval.txt` から導入してください。Nimシグナリング実装は `src/` にあります。

## 1ピア vs 複数ピアのPT比較（低メモリ・直列）

```bash
python -m pip install matplotlib  # PNGグラフを出す場合（PyTorchは別途必要）
python -m benchmarks.bench_serial_pt_v112 --peers 1,2,4,8 --rounds 30 --steps-per-round 8 \
  --rtt-ms 80 --jitter-ms 12 --bandwidth-mbps 20 --server-bandwidth-mbps 100
```

常駐モデルは1つだけ、仮想ピアのLoHA/AdamW状態は一時ディスクへ切替。これは**同期FedSumのオフライン検証**であり、実際のP2P通信性能を証明するものではありません。**`--steps-per-round`は各ピアのstep数です。** 標準グラフの横軸は仮想並列経過秒で、ネットワークRTT・共有帯域・ジッター・再送・同期バリアを反映します。詳細は [`docs/releases/README_V112_2_SCALED_WORK_LATENCY_JA.md`](docs/releases/README_V112_2_SCALED_WORK_LATENCY_JA.md)。

## コードの探し方

| 目的 | 最初に読む場所 |
| --- | --- |
| GUIの起動とルーム管理 | `kcia.py`, `patchlib/webview_controller.py`, `patchlib/webview_lifecycle_v110.py` |
| CLI引数と学習実行 | `patchlib/cli.py`, `patchlib/cli_arguments_v110.py`, `patchlib/local_training_v110.py` |
| PT / SFT / PPO | `patchlib/mixed_learning_v65.py`, `patchlib/mixed_kernels_v110.py`, `patchlib/mixed_sources_v110.py` |
| Hugging Faceモデルの準備 | `patchlib/model_loading_v110.py`, `patchlib/hf_load_v85.py` |
| P2P・署名・ジャーナル・世代管理 | `patchlib/elasticnet.py`, `patchlib/elastic.py`, `patchlib/elastic_control_v110.py` |
| MPSと診断 | `patchlib/mps_runtime_v91.py`, `patchlib/mps_acceleration_v108.py` |
| 保存・復旧・fast-resume | `patchlib/fast_resume_v85.py`, `patchlib/epoch_carry_v46.py` |

詳細なモジュール地図は [`docs/SOURCE_LAYOUT_JA.md`](docs/SOURCE_LAYOUT_JA.md)、利用ガイドと過去の変更記録は [`docs/INDEX.md`](docs/INDEX.md) を参照してください。

## データ保護と移行

**実行中にソースを上書きせず、更新前にコードと実データをバックアップしてください。** 配布用ZIPからは実行時に生成された `kcia-rooms/*.json` を取り除きましたが、**既存のインストールにあるルーム・鍵・署名ジャーナル・weightを削除してはいけません**。更新のための `reset-local-data` も不要です。

Macの `auto` デバイスは過去のMetalクラッシュ対策のためCPU安全モードが基本です。MPSは明示的に設定し、自己診断と実機確認を経てから使ってください。

## 検証

```bash
python -m compileall -q patchlib tests scripts
PYTHONPATH=.:tests pytest -q tests/test_v110_modular_refactor.py
python tools/audit_release.py
```

テスト結果と既知の制約は [`docs/validation/VALIDATION_V111.md`](docs/validation/VALIDATION_V111.md) を参照してください。

### v112.1 — PT plot x-axis

Serial PT benchmark primary plots now use **tokens processed per peer** (total / peers) for ideal parallel-work comparison; parallel wall time is **not measured**. Equal-total-tokens plots are retained with `-total-tokens.png` suffix. See `docs/releases/README_V112_1_PARALLEL_AXIS_JA.md`.

### v112.2 — 同一ローカル計算量 + 仮想ネットワーク

**v112.1以前の「総計算量固定」とは実験条件が異なります。** 各ピアのステップ数を揃え、総トークン数をピア数に比例して増やします。旧形式のグラフ名と異なり、標準グラフはRTT・帯域・バリア込みの仮想秒を横軸にします。
