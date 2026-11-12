# KCIA v112.2 — 同一ローカルPT計算量 × 仮想ネットワーク遅延

**GUIは使わず、実際の並列ピア・ソケットも作りません。** 小メモリMacで、1つの実物の`TinyCausalLM + LoHA/LoRA + AdamW`を共有し、仮想ピアの因子とOptimizer状態をディスクへ順番に退避して、1・2・4・8ピアのPTを比較します。

## 重要な変更：計算予算の定義

`--steps-per-round` は **「1ピアあたり・1ラウンドあたり」** のOptimizer step数に変更しました。

| ピア数 | 1ピアあたりのstep数 / round | 全体step数 / round |
|---:|---:|---:|
| 1 | 8 | 8 |
| 2 | 8 | 16 |
| 4 | 8 | 32 |
| 8 | 8 | 64 |

たとえば`rounds=30, steps-per-round=8, batch-size=4, context=16`なら、どの条件でも各ピアは240 step・15,360トークンを学習し、全体の計算量はピア数に比例します。1ピア目のミニバッチ順は全実験で同一です。新規ピアは同じ訓練データプールから独立にシャッフルしたミニバッチを取得するため、**全条件で総ミニバッチ列が同じとは主張しません**。初期ベース重み、ホールドアウトデータは固定です。

## 使い方

```bash
python -m pip install matplotlib  # torch は別途必要
python -m benchmarks.bench_serial_pt_v112 \
  --peers 1,2,4,8 \
  --rounds 30 --steps-per-round 8 \
  --batch-size 4 \
  --rtt-ms 80 --jitter-ms 12 \
  --bandwidth-mbps 20 --server-bandwidth-mbps 100 \
  --loss-rate 0.01 --retry-rto-ms 250 --max-retries 3 \
  --output-dir benchmark-results/pt-v1122
```

`--device cpu`が標準であり、Mac MPSは`KCIA_EXPERIMENTAL_MPS=1`を明示した場合のみ実験的に選択できます。`--text-file ./corpus.txt`で8MiB以下のローカルUTF-8テキストをバイト単位に学習できます。

ネットワークのみを毎回同条件で比較したいときは、実測計算時間の代わりに固定の1ステップ実行時間を指定できます。このときFedSum統合時間は標準で仮想0秒になります（指定する場合は `--aggregation-ms 5` など）。これにより実測のCPUスケジューリング変動が仮想時刻に混入しません。

```bash
python -m benchmarks.bench_serial_pt_v112 \
  --peers 1,2,4 --rounds 8 --steps-per-round 4 \
  --compute-ms-per-step 30 \
  --rtt-ms 100 --jitter-ms 10 --output-dir benchmark-results/what-if
```

## 仮想ネットワークの仕組み

シード固定の離散イベントシミュレーションです。**実際にはsleepもネットワーク接続も行いません。** 1つの同期FedSumラウンドで以下を再現します。

1. 中央集約ノードがモデル更新を**全ピアに個別配信**。共通の下り回線の送信キューと各ピアの個別下り回線容量を反映。
2. 各ピアは配信完了後にローカルPTを実行。`--compute-ms-per-step`がなければ、そのピアの**実測optimizerステップ時間と有効差分生成時間**を仮想計算時間として再生するため、物理実行はあくまで直列です。
3. 更新済みの有効重み差分を中央ノードへ**個別アップロード**。共通の上りキューと各ピアの個別上り回線を反映。
4. 全ピアの更新が届くまで同期バリア待ち。FedSumで有効重み差分を平均せず1回だけ加算し、次ラウンドへ進みます。

各メッセージに`RTT/2`の片道伝搬遅延と正規分布のジッター（負値は0へクリップ）を加えます。帯域制限は**Mbps (megabits/sec)**であり、`--payload-bytes`が未指定なら、実際に更新する**dense effective-delta のbyte数**を推定して使用します。送信には`--message-overhead-bytes`も加算します。`--loss-rate`はメッセージ単位の損失確率で、再送は`--retry-rto-ms`後に別の回線枠を消費します。`--max-retries`を超える場合はベンチマークを失敗として停止します。

1ピア構成はネットワーク送受信なしのローカル学習を表すため仮想通信遅延0です。

### 重要なモデル化の限界

- **これは同期型・中央集約（star topology）のモデルであり、KCIAの本番非同期Gossip、WebRTC、NAT、署名、P2P接続をシミュレートしたことにはなりません。**
- 同期FedSumの**最終的なPT精度はRTTを変更しても同じ**です。レイテンシは「その精度までの仮想的経過時間」を変えます。遅延によるstale-gradient、異なるバージョン同士の混合を評価するには別途非同期実験が必要です。
- 回線の共有キュー、RTT、ジッター、簡略化した損失再送は再現しますが、TCP輻輳制御、実パケット分割、実回線の輻輳、シリアライズ負荷、署名計算は未実装です。
- 小型モデルで計測したステップ時間を実機の巨大LLMへ外挿するのは不適切です。`--payload-bytes`はあくまで通信の仮定を切り替えるもので、実際の通信を発生させません。
- 実測の直列wall clockには一時ファイルのI/Oと検証処理が含まれますが、仮想並列時間には**それらの直列実行由来のオーバーヘッドを含めません**。検証処理も経過時間には含めません。
- 参加ピアごとのモデルは独立に常駐せず、**モデル・Optimizerオブジェクトはそれぞれ1つ**です。因子・AdamWモーメントのみディスク保存します。

## 出力

`--output-dir`に`trace.csv`, `summary.csv`, `result.json`と以下のグラフを保存します。

| プロット | 横軸 | 用途 |
|---|---|---|
| `pt-heldout-loss.png`, `pt-heldout-accuracy.png` | **仮想並列経過秒** | 通信・同期を含めた収束までの時間を比較（標準） |
| `*-ideal-compute.png` | **最長ピアの累積計算秒** | RTTやバリア待ちを除いて比較 |
| `*-local-steps.png` | **各ピアの累積ローカルstep** | 同一ローカル計算量を比較 |
| `*-total-tokens.png` | **全ピア合計の学習トークン数** | 計算効率（ピア数に比例して増加）を比較 |

`trace.csv`/`summary.csv`には仮想時刻、累積critical compute、ネットワーク・同期待ち、再送回数、メッセージ数、通信byte数、実測の直列wall timeを**別フィールド**で出力します。パラメータと制約は`result.json`へ保存します。

## テスト

```bash
PYTHONPATH=.:tests python -m pytest -q \
  tests/test_v112_serial_pt.py tests/test_v1122_virtual_network.py
```

既存ルーム・ジャーナル・鍵やGUIを変更しません。すべての学習条件で同じ初期重みと固定されたholdoutを使用します。ソース更新前にインストールと実行データのバックアップをおすすめします。
