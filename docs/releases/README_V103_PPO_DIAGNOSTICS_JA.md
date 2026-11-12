# Kagami v103 — PPO 実行系の切り分け・観測性改善

## 重要：原因未確定

v102のあと「PPOがうまくいかない」という報告に新しいトレースバックがないため、具体的な実機エラーを再現できたという主張はしません。本版は **既存ルームの学習プロトコルを変えずに、失敗地点・ゼロ報酬・Actor勾配の有無を観測するための診断版**です。v102までのHFモデル相性・Metalネイティブクラッシュ・ディスク容量不足等を自動解決するものではありません。

## 追加したコマンド

    python kagami.py ppo-doctor

- 一時ディレクトリ上にミニデータセットを作成し、小型PyTorch causal LM + LoRA + PPO + ローカルMLP Criticを4回学習します。
- Actorのパラメータが実際に更新されたことを確認します。HFからのダウンロード、ユーザーの署名付きジャーナル読み書き、ルーム接続はありません。
- `--steps 1..20` で短い実行も可能です。
- `--device mps` は、既存のMac安全ポリシーに従って `KAGAMI_EXPERIMENTAL_MPS=1` を設定した場合のみ実行できます。Metalの致命的ネイティブアボートまでは捕捉できないため、既存の学習・保存状態があるプロセスとは分離してください。MPS実機動作は本リリースでは未検証です。
- 合格しても実際のHF Reward Model、HF Value Critic、データセット、P2P署名経路、長時間収束が正しいことを意味しません。

## 学習中の診断出力

- `KAGAMI PPO INIT phase=reward-model`：HF Reward Modelの読み込み開始
- `KAGAMI PPO INIT phase=value-critic`：HF Criticの読み込み開始。`head=pretrained|new`も表示
- `KAGAMI PPO INIT phase=ready`：CriticのOptimizer構築完了
- `KAGAMI PPO DIAG`：定期的に `reward`, `actor_loss`, `critic_loss`, `kl`, `clipfrac`, `actor_accum_grad_norm`, `zero_reward_streak`, `rollouts`, `optimizer_pending` を出します。
- `KAGAMI PPO LEARNING WARNING`：referenceベースのPPOで補助損失がなく、ゼロ報酬が10、50、100本続いたことを表示。

`actor_accum_grad_norm` はPT/SFTも同時に実行している場合、その既存勾配を含むため「PPO単独の勾配」とはみなせません。また勾配が非ゼロでもOptimizerのweight更新や汎化性能向上の証明ではありません。

## 次の故障の切り分け

1. `ppo-doctor` が失敗：ローカルのPPO/LoRA/トーチ環境の問題（最終エラー行を共有）
2. Doctor成功、HF Reward/Value初期化が失敗：チェックポイント形式・容量・依存ライブラリ・head設定を確認
3. 初期化後に `DATA WAIT stage=model-step` が続く：大きなHFモデルの推論負荷やHFキャッシュなどが候補。処理中の位置をログから絞る
4. `KAGAMI PPO DIAG` で `reward=0` が続く：reward関数と実際のデータを照合。`token_match` は大語彙では0を返しやすい。必要に応じて新規ルームでSFTや明示的なreference補助損失を検討
5. `critic_loss`が非有限・`clipfrac`高止まり：Critic/Actorの更新率、PPO stale rollout、KL及びscaleの問題が候補
6. `EPOCH PRESSURE HOLD`：PPOではなく署名付きジャーナル・世代認証の停止を優先して調査

本版はログの観測値を増やしますが、**PPOの性能改善、MPSの完全対応、未知のエラー解消は保証しません**。
