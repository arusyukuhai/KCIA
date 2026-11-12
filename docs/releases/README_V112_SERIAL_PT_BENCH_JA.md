# KCIA v112 — 省メモリの直列PT比較ベンチマーク

## 目的

**1ピア**と**複数ピア**の causal language modeling（PT）の学習品質を、限られたメモリで比較します。GUIには組み込みません。既存のルーム・署名済みイベント・SQLite・チェックポイントには触れません。

1つの `TinyCausalLM`（自己注意付きの小さな自己回帰LM）と、1つの `UniversalAdapter` + AdamWだけをメモリ上に保持します。LoHA/LoRAとAdamWの状態をピアごとに一時ファイルに保存し、順番に読み替えます。全員のローカル更新を一ラウンドの最後に **FedSum**（有効重み差分の総和、ピア数で割らない）として共有ベースに適用します。アンカーは「前回コミット済みの因子」を表し、過去の更新を二重適用しません。

これは**直列・同期型のオフラインFedSumシミュレーション**です。KCIA本番の非同期Gossip、WebRTC接続、署名付きイベント・ジャーナル、ネットワーク遅延、ピア切断からの復帰は実行・検証しません。したがって本ベンチマーク単体で「本番P2Pが正常に収束した」とは主張できません。

## インストールと実行

KCIAルートで：

```bash
python -m pip install torch matplotlib
python -m benchmarks.bench_serial_pt_v112 \
  --peers 1,2,4,8 \
  --rounds 30 \
  --steps-per-round 8 \
  --batch-size 4 \
  --output-dir benchmark-results/serial-pt-v112
```

`--steps-per-round` は**全ピア合計**であり、ピア数で割り切れるように設定します。上の例では1ピアが8step、2ピアが各4step、4ピアが各2step、8ピアが各1stepを順番に実行します。**全条件で同一の合計ステップ数・合計トークン数・ミニバッチ列**を使用します。1ピアに対して余分な計算量を与えることも、複数ピアのトークン数を倍増させることもありません。

最初は小さなテストで動作確認を推奨します：

```bash
python -m benchmarks.bench_serial_pt_v112 \
  --peers 1,2,4 --rounds 3 --steps-per-round 4 \
  --width 16 --context 8 --train-sequences 64 --valid-sequences 16 \
  --output-dir benchmark-results/v112-smoke
```

標準データは、ダウンロード不要の**合成次トークン予測タスク**です。自然言語の品質は示しません。手元のテキスト（8MiB以下）からUTF-8の**バイト列**を学習する場合：

```bash
python -m benchmarks.bench_serial_pt_v112 \
  --peers 1,2,4 --steps-per-round 8 --rounds 30 \
  --text-file ./my_corpus.txt \
  --output-dir benchmark-results/serial-pt-text
```

`--text-file` はローカルテキストを先頭80%の訓練領域・末尾20%の検証領域に分割します。これは**バイト単位のPT**であり、Hugging Faceトークナイザや任意の商用LLMをロードするものではありません。テストに使うデータは機密情報を含まないようにしてください。

## メモリとディスク

- LM、Adapter、Optimizerオブジェクトは**全体で各1つ**。ピア数に応じて常駐するモデル数は増えません。
- ピアごとの因子・AdamWモーメントは `tempfile.TemporaryDirectory` に置き、成功時も例外時も実行終了時に削除します。実行中のディスク容量はピア数と因子サイズに比例します。
- 一時ファイルの保存先を選ぶ場合は `--scratch-dir ~/large-volume`（事前に作成済みのフォルダ）。Macで空き容量が少ない場合は利用してください。
- CPUがデフォルト。MPS実行は実験的で、環境変数 `KCIA_EXPERIMENTAL_MPS=1` と `--device mps` を明示的に指定してください。Mac実機での検証は未完了です。
- 入力データテンソルの理論サイズが128MiBを超える設定は事前に拒否します。モデルのメモリおよびPyTorchのワークスペースは別途必要です。

## 出力

`--output-dir` に次を保存します：

- `pt-heldout-loss.png` — 同一総トークン数に対する検証Cross-Entropy
- `pt-heldout-accuracy.png` — 同一総トークン数に対する検証Top-1正解率
- `trace.csv` — ラウンド別の学習CE、検証CE、検証Accuracy、計算時間、直列経過時間
- `summary.csv` — ピア数別の最終品質、計算時間、一時ファイルI/O時間、同一ミニバッチ検証ハッシュ
- `result.json` — 設定・入力データ情報・公平性条件・制約

グラフの出力には `matplotlib` が必要です。未導入でもCSV/JSONは生成します。

### 解釈上の注意

* 比較対象は **1ピアのLoHA/PT vs nピアの同期FedSum/LoHA/PT** です。すべて同じ凍結ベースモデルから開始します。
* 複数ピアの「8ステップ」は **合計8ステップ** です。各ピアが独立に8ステップずつ実行する比較ではありません。
* 複数ピアの各Optimizerのモーメントは独立しています。1ピアでは8連続ステップですが、4ピアなら2連続ステップでいったんFedSumするため、学習曲線が異なることは正常です。
* 計測した直列wall timeは**実際のローカル実行時間**です。`/peers` にした「仮想的並列時間」を実測値として表示しません。
* 検証Accuracyは次トークンTop-1の**未知サンプル上の精度**であり、PPO報酬や会話品質評価ではありません。
* `run_serial_pt_benchmark` はオフラインの数値合算です。本番の署名・ネットワーク・世代復旧を保証しません。KCIA本番とは異なる同期タイミングを用います。

## 回帰テスト

```bash
PYTHONPATH=.:tests python -m pytest -q tests/test_v112_serial_pt.py
```

ソース／GUIから独立したベンチマークのため、`kcia.py`、`web/`、CLI本体、署名付きプロトコルは変更していません。実際のローカル学習データを削除するコマンドもありません。
