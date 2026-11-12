# Kagami v109 — 長文SFT/モデル実効コンテキスト対応

## 対象の実機エラー

```
[transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (3659 > 2048)
patchlib.mixed_learning_v65.MixedLearningError: SFT response exceeds max_length; would lose supervision
```

このエラーはPPO/MPSの計算の失敗ではなく、**混合学習中のSFT回答が1回のforwardへ収まらない**ことで起こる。v108はプロンプトのみ末尾切り詰めできたが、responseが設定上限を超えるとモデルに入力する前に失敗していた。

## v109での変更

- **長いSFT回答を連続したtoken窓へ分割**。各ローカル学習ステップで一つの窓を処理する。これにより1回のforward/backwardの最大入力長を制限し、MPSの一時メモリ利用を抑える。
- **回答の全トークンを教師信号として保持**。プロンプトおよび既学習の回答前半はコンテキスト専用（labels=-100）として重複学習せず、次の窓で新たに出現した回答トークンだけにcausal CEを適用する。末尾EOSも対象。短い回答では従来と同じ単一窓を使用。
- **モデルとtokenizerの実際の最大長を考慮**。`args.max_length`、`tokenizer.model_max_length`（実際の値がある場合）、モデルの`max_position_embeddings / n_positions / n_ctx`の最小値を使用。tokenizerの巨大な不明長sentinelは無視。例えば設定4096・モデル2048なら2048以下でforward。
- **PPO生成プロンプトも同じ実効上限で切り詰め**、生成トークン用空きを確保。PPOのteacher-forcing補助損失にも上限を適用。
- **tokenizerの長文エンコード警告を抑制**。長文をtokenizerで一度完全にトークン化してから、モデルforwardに適した小さな窓へ分割するのは意図どおりの処理。`verbose=False`を受け取れない独自tokenizerは互換性経路を使用。
- 既存の署名付きルームプロトコル、署名ジャーナル、P2Pメッセージ、Optimizer/checkpoint形式は変更しない。`reset-local-data` は不要。

## 学習の意味と制限

- 長文の全tokenが**複数回の学習ステップにわたって**処理される。たとえば3659tokenの回答が1ステップだけで全部学習されるわけではない。学習途中に停止すると、未処理の回答窓は次の学習開始時に再取得されるデータセット順序次第では同じ続きからは再開されない（v109は窓カーソルを永続化しない）。
- 窓を分割するとattentionは各窓内のトークンに制限される。前窓の全履歴を参照できるわけではない。長文応答の**全文監督**は維持されるが、無制限の文脈長を実現するものではない。
- v99で追加した**選好対のsequence-level SimPO**は短い回答に従来どおり対応する。長いchosen/rejected対については、安易な部分比較が本来の損失を変えてしまうため、v109では引き続き明確な例外で止める。今回の長文自動分割は通常のSFT（`sft_pref_coef=0`）のための修正。
- `max_length`やモデル設定が異常に短すぎる場合、またはモデルが位置埋め込み長を正しく公開していない場合は別途設定の調整が必要。MPSネイティブ長時間運転は未検証。

## 動作確認

```bash
python -m compileall -q patchlib
PYTHONPATH=.:tests pytest -q \
  tests/test_v109_long_sft_context.py \
  tests/test_mixed_learning_v65.py \
  tests/test_v99_mixed_optimization.py \
  tests/test_v107_objective_metric_plots.py \
  tests/test_v108_mps_acceleration.py \
  tests/test_ppo_correctness_performance_v73.py \
  tests/test_hf_mixed_datasets_v68.py \
  tests/test_v106_hf_schema_guard.py \
  tests/test_v100_ppo_runtime_reliability.py \
  tests/test_v103_ppo_diagnostics.py \
  tests/test_ppo_pipeline_v72.py \
  tests/test_mps_worker_v91.py \
  tests/test_mps_policy_v92.py \
  tests/test_fast_resume_v85.py \
  tests/test_fast_resume_v96.py
```

`tests/test_v109_long_sft_context.py`では長さ3659の回答にEOSを付け、**2048以下の窓だけ**で全tokenを正確に1度ずつ教師信号に使うことを実証。長いPPO promptの上限、短い選好損失、SFT+PPO同時学習でもforward長を超えないことをテスト。
