# Kagami v100 — PPO実動作修正（v99ベース）

## 問題と修正

1. **MPS未対応のfloat64演算（分散Critic）**
   - `patchlib/offpolicy_critic_v79.py`: trajectory importance sampling の64bit計算を、MPSの場合に限りCPUへ移して実行します。Critic値はautogradの接続を維持して転送するため、GPUのCriticパラメータまで勾配が戻ります。戻り値はMPS対応float32です。CPU/CUDAの従来経路は維持します。
   - `patchlib/mixed_learning_v65.py`: 受信したPPO軌跡の旧方策確率をMPSでfloat64テンソル化しないよう変更。
   - `patchlib/behavior_checkpoint_v81.py`, `behavior_checkpoint_v82.py`: 署名済み行動確率の照合はCPU float64で比較（各軌跡は最大128要素）。Model自体は指定デバイスに留まります。
   - MPS生成時の `torch.multinomial` をCPU float32へ移し、GPUへの未対応sampling演算を避けます。Actor forward/backwardは元デバイスのままです。

2. **報酬0が続く場合のActor学習信号不足**
   - `ppo_reward=token_match/exact` は、大きな語彙や短い生成で0報酬が続きやすくなります。従来のPPO推定量そのものは維持し、明示的に有効化した場合だけ正解文を教師強制する補助損失を加えます。
   - 設定: `--ppo-reference-aux-coef 0.1` など。0（既定）では従来と完全同じ目的関数です。`hf_model` 報酬では指定できません。
   - この補助損失は参考テキストへ学習するSFT成分です。**純粋なPPOの報酬関数を改善したものではありません**。必要性は評価データで検証してください。
   - 新規署名ルームでは係数をprotocolに固定します。既存のルームに異なる値のまま参加することはできません。

3. **勾配が流れないときの可視化**
   - `run_step` が `actor_grad_norm` をPPO統計に記録し、PPO専用で勾配が0の場合は警告を1回出します。NaN/Inf勾配はoptimizer適用前に拒否します。
   - 元のPT/SFT/PPOの混合勾配・Critic更新方式を変更しません。

## 検証

- CPUでPPO関連98テスト成功、MPS実機専用テスト1件はLinux上のためスキップ。
- 新規テストは完全一致報酬が常に0の状況でも、明示的な補助目的を入れればActorのreference cross entropyが減ること、LoRAアダプタが実際に更新されることを確認します。
- 数値的なIS補正の係数とCritic勾配をCPU上の厳密計算と比較。
- GUI経由でオプションがCLIへ渡り、既存ルームの既定プロトコルが変わらないことを確認。

## 導入・試験

既存データをバックアップしてv99から更新。新しい補助係数を使うときは新規ルームを作成してください。

純粋PPOを維持したい場合は `--ppo-reference-aux-coef 0`（既定）。
正解文付きPPOをまず安定して学習させたい場合は `--ppo-reference-aux-coef 0.1 --ppo-adv-norm rms` を実験してください。

Mac上でPPOを検証する際は、通常GUIはv92以降のCPU安全モードを維持します。MPSを使うには明示的な実験モードが必要です。MPSのMetalコマンドバッファクラッシュまでは今回の修正で解決を保証できません。まずGPUを使わない構成で試し、MPSは実機で別途検証してください。

```bash
PYTHONPATH=. python -m pytest -q tests/test_v100_ppo_runtime_reliability.py
```

**注意**: ディスク、ジャーナル、署名済み履歴、fast-resumeに変更はありません。モデルが非因果型、Tokenizerに互換性がない、報酬モデルが無効、Criticのcheckpointが壊れている、Metalドライバがabortする等のケースはこのパッチだけでは直りません。
