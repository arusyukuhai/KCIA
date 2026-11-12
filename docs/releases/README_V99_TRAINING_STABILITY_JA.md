# Kagami v99 — PPO / SFT / PT mixed training enhancements

v98 を基盤とした、混合学習の**オプトイン**改善です。通常のトレーニング実装・署名イベント・ルームID・分散 Critic・fast-resume の永続形式を変更しません。

## PPO の改善

- `--ppo-adv-norm rms`：Advantage のRMSで割り、1つの短いtrajectoryで符号や有益な平均値を打ち消しません。従来の中心化 `legacy` は互換性のため標準のままです。
- `--ppo-target-kl 0.02`：既に再利用しているrolloutの平均 proxy KL（chosen action の log-ratio の二乗平均 / 2）が基準を超えたら、**同じローカルモデルロック内で一回だけ**再サンプルしてから更新。log-ratioが20超などの異常は従来通り停止。
- `--ppo-adaptive-kl --ppo-kl-horizon 128`：実際に再利用されたrolloutから観測したproxy KLに基づくKL係数の緩やかな更新（0への発散や爆発を抑える範囲制限）。Fresh rollout時は更新しません。ローカル状態のため再起動後に係数は初期値へ戻ります。
- `--ppo-entropy-coef 0.001`：探索性を残すentropy bonusをactor lossへ加算（正確にはentropyに負の係数）。
- `--ppo-critic-grad-norm 1.0`：ローカルおよび受信trajectory Criticで勾配をクリップ。非finiteのまま重みを更新しません。
- PPOメトリクスに `clipfrac`、`ratio_mean`、`adv_abs_mean`、`kl_coef` を追加。

上記のPPOのproxy KLは、完全な分布全体のKLでも固定参照モデルに対するKLでもありません。P2Pで異なる方策間の理論的収束や報酬改善を保証するものではありません。

## PT / SFT / PPO の混合最適化

- `--mixed-actor-grad-norm 1.0`：PT、SFT、PPO の勾配を累積**した後**、Adapter全体に対してグローバルnorm clip。個々のlossに別々のclipは掛けず、混合比率を変えない設計です。train/elastic/live のactor optimizer stepに反映。
- `--sft-label-smoothing 0.02`：応答部分だけのCEにlabel smoothing。元のpromptマスクを維持。出力logitsが取得できるモデルに限ります。
- `--sft-pref-coef 0.2`：ローカルSFT JSONLに `rejected` 列が存在するとき、好ましい `response` と好ましくない `rejected` の長さ正規化log probability差をsigmoid lossとして追加。参照モデルは不要です。**SimPO論文に着想を得たSFT-anchored hybrid**であり、オリジナルSimPOの再現実装ではありません。
- `--sft-pref-beta 2.0` と `--sft-pref-margin 0.5`：選好比較のlog-odds係数と目標margin。

選好学習のJSONL例：

```json
{"prompt":"Explain the reasoning.","response":"Here is a verifiable explanation ...", "rejected":"An unsupported assertion ..."}
```

1レコードには必ず`prompt`,`response`,`rejected`の非空文字列を含めてください。`--sft-pref-coef` > 0 は**ローカルJSONLのSFTデータ**のみ対応（HF SFT streamingを用いる場合は使用不可）。chosen/rejectedとも自動でEOSを付け、同じprompt contextを使って比較します。

## 推奨する最初の実験設定

```bash
# 既存の train-elastic コマンドの末尾に追加（新しいルームで）
--ppo-adv-norm rms \
--ppo-target-kl 0.02 \
--ppo-adaptive-kl \
--ppo-kl-horizon 128 \
--ppo-entropy-coef 0.001 \
--ppo-critic-grad-norm 1.0 \
--mixed-actor-grad-norm 1.0
```

選好ペアがある場合にのみ、`--sft-pref-coef 0.2 --sft-pref-beta 2.0 --sft-pref-margin 0.5` を追加。GUIからも混合学習の詳細欄に設定できます。係数・勾配閾値は、モデル・トークン数・Rewardスケールに応じて実測調整してください。

## 互換性・制約

- 新設定をすべてデフォルトのままにした場合、`mix-policy` の署名済みプロトコルJSONはv98と同じです。
- **1つでも新しい非デフォルトの設定を使うと、生成されるimmutable mixed policyも変わる**ため、既存の学習ルームに新しい目的を後付けすることはできません。新規ルームを作ってください。すべてのピアを同じバージョン・同じ新ハイパーパラメータに揃えてください。
- 既存の署名付きjournalやweightをリセット/改変せず、継続学習用の旧設定はデフォルトで維持します。
- 選好学習はchosenとrejectedの2つの追加forwardを必要とし、大規模モデルでは負荷が高まります。GPU/CPU/MPSの実機性能や長期間P2P動作を検証したわけではありません。
- adaptive KL係数は現在のプロセス内ローカル状態で、checkpointに追加永続化しません。正確な再開を要する実験は固定KLを使うか、この制約を明記してください。
- PT自体のアルゴリズムは既存のcausal LM cross-entropyを維持。`mixed-actor-grad-norm` はPTにも効きますが、新しい事前学習アルゴリズムやGRPO/DAPO学習器は実装していません。

## 検証

Linux CPU / PyTorch環境で既存のmixed training / HF critic / distributed critic / event epoch / fast-resume等の重点回帰テストを実行。詳細は `../validation/VALIDATION_V99.md`。Mac MPSでの実機学習、巨大分散ネットワーク、Reward Model精度の改善は未検証です。
