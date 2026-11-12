# Kagami v107 — PT / SFT / PPO 学習指標を独立plot

v106をベースにした**観測機能のみ**の改修です。LoHA、Optimizer、P2P署名、ジャーナル、世代切替、学習ルームの署名付きプロトコルを変更しません。既存ルームの履歴を削除する必要はありません。

## TRAINING のグラフ

TRAINING画面にスクロール可能な学習グラフを追加しました。上部のフィルタで「全指標 / PT / SFT / PPO / Loss / Top-1」を切り替えられます。表示期間は従来どおり5分から全期間まで選択できます。

| グラフ | 計測対象 | 精度上の注意 |
| --- | --- | --- |
| PT token Top-1 | PTのミニバッチにおける正解次トークンのTop-1率。`-100`は除外 | 訓練データ、validationではない |
| SFT response Top-1 | 応答部分だけの正解次トークンTop-1。promptは除外 | 訓練データ、validationではない |
| SFT preference win | `logp(chosen)>logp(rejected)` のペア比率 | 一回の学習ペアによる訓練指標であり、外部評価ではない |
| PPO reference match | 生成トークンと正解文の**位置一致数/長い方の長さ** | `token_match`/`exact`等、正解文がある場合にのみ算出。意味的正しさではない |
| PPO reward | 実際に得られた報酬スコア | **正解率ではない**。HF Reward Modelではモデルの生スコア |
| PPO approximate KL | PPOの近似KL | 小さい方が必ず優秀という意味ではない |
| PPO Critic loss | 価値関数の学習損失 | **正解率ではない** |

PT、SFT Top-1は追加の巨大なforward passをせず、既存forward結果から10ローカルステップに1回計算します。未対応のタスク/出力形状、正解ラベルなし、PT/SFT/PPOを無効にした場合には指標は欠損値（グラフに点を追加しない）になります。PPOは得られた軌跡の報酬を使用し、正解文がないHF報酬モードではreference-matchは空欄です。

記録先は従来の各ピアの**ローカルtelemetry JSONL**で、`pt_accuracy`, `sft_accuracy`, `sft_preference_accuracy`, `ppo_reference_match`, `ppo_reward`, `ppo_kl`, `ppo_critic_loss` キーを追加しました。既存の `accuracy` / `loss` キー、GUIのhistory APIはそのままです。古い履歴には新しいキーがないため、**v107導入後のデータからのみ手法別グラフが表示**されます。実際のデータを偽造して過去の値を埋めることはしません。既存履歴は消しません。

## 導入

1. KagamiのGUIと学習プロセスを停止し、ルーム保存データをバックアップします。
2. v107のソースへ更新します。既存の環境設定/保存データは残します。
3. GUIのTRAINING画面に入り、PT/SFT/PPOの各グラフを選択します。
4. 学習方式が無効な場合や計測前には `NO MEASUREMENTS YET`、正解文のないPPOではreference plotに `NO LABELED REFERENCE` と出ます。

v107はグラフ追加であり、PPO、HF Critic、MPS、データセット自体の既存障害を修復するパッチではありません。

## 検証

`PYTHONPATH=. pytest -q tests/test_v107_objective_metric_plots.py` で新規回帰テストを実行できます。ミニバッチ正解率のshift/responseマスク、報酬を正解率と偽らないこと、実測値のJSONL保存、GUIに各グラフが存在することを検査します。macOS実機のWebView描画確認は実施していません。
