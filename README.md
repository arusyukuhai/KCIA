# VPS + Cloudflare Tunnel + Mac/Colab test package

[日本語セットアップ手順](deploy/vps/README_CLOUDFLARE_JA.md) / [Colab notebook](notebooks/Kagami_Cloudflare_Mac_Colab.ipynb)

# Kagami — 分散型LLM学習プラットフォーム

Kagamiは、LoHA/LoRAによるモデル学習、P2P Gossip、PT・SFT・PPO、署名付き更新、世代移行、再起動後の学習復旧に対応する実験的な実装です。

**v112.2はGUI非依存の直列PT比較ベンチマークを、ピアごとの同一計算量と仮想ネットワーク遅延に対応させます。既存の学習アルゴリズム、署名形式、データベース形式は変更しません。**

## VPS（Cloudflare Tunnel/WSS + Nimシグナリング + coturn）・Mac / Colab 実機テスト

[導入手順（日本語）](deploy/vps/README_CLOUDFLARE_JA.md) / [Colabノートブック](notebooks/Kagami_Cloudflare_Mac_Colab.ipynb)。
この構成はv112.2の学習プロトコルを維持し、サーバーにモデルや鍵・重み・ジャーナルを置きません。

## 起動

```bash
python kagami.py                     # WebView GUI
python -m patchlib.cli --help       # 学習・診断・P2P CLI
python kagami.py ppo-doctor         # 小型CPU PPO更新の自己診断
python kagami.py mps-ppo-smoke      # Mac MPS の隔離実行テスト
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
| GUIの起動とルーム管理 | `kagami.py`, `patchlib/webview_controller.py`, `patchlib/webview_lifecycle_v110.py` |
| CLI引数と学習実行 | `patchlib/cli.py`, `patchlib/cli_arguments_v110.py`, `patchlib/local_training_v110.py` |
| PT / SFT / PPO | `patchlib/mixed_learning_v65.py`, `patchlib/mixed_kernels_v110.py`, `patchlib/mixed_sources_v110.py` |
| Hugging Faceモデルの準備 | `patchlib/model_loading_v110.py`, `patchlib/hf_load_v85.py` |
| P2P・署名・ジャーナル・世代管理 | `patchlib/elasticnet.py`, `patchlib/elastic.py`, `patchlib/elastic_control_v110.py` |
| MPSと診断 | `patchlib/mps_runtime_v91.py`, `patchlib/mps_acceleration_v108.py` |
| 保存・復旧・fast-resume | `patchlib/fast_resume_v85.py`, `patchlib/epoch_carry_v46.py` |

詳細なモジュール地図は [`docs/SOURCE_LAYOUT_JA.md`](docs/SOURCE_LAYOUT_JA.md)、利用ガイドと過去の変更記録は [`docs/INDEX.md`](docs/INDEX.md) を参照してください。

## データ保護と移行

**実行中にソースを上書きせず、更新前にコードと実データをバックアップしてください。** 配布用ZIPからは実行時に生成された `fedpara-rooms/*.json` を取り除きましたが、**既存のインストールにあるルーム・鍵・署名ジャーナル・weightを削除してはいけません**。更新のための `reset-local-data` も不要です。

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
