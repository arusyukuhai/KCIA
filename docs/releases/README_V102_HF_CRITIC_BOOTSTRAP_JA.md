# Kagami v102 — HF Critic が scalar pretrained head を持たず停止する問題への対応

## 何が起きていたか

v101 までの `critic_kind=hf` は、Hugging Face チェックポイントから **学習済みの単一スカラー value head（num_labels=1）** がロードできる場合だけ起動します。分類ラベルが 2 個以上のモデル、または causal LM を HF Critic に選ぶと、巨大なウェイトのダウンロード後であっても `HFValueError: HF Critic must expose a PRETRAINED single scalar value head (num_labels=1)` で失敗します。

このエラーは Reward Model と違い **Critic** に起因します。Reward Model のスカラー出力を流用したり、未学習の Value head を学習済みと見なす修正は行っていません。

## 使用方法（新規ルーム）

GUI → TRAINING → Mixed PT/SFT/PPO → VALUE CRITIC:

- 手軽さ優先: **Architecture = MLP**。別の HF Critic を読み込まずに PPO を学習できます。
- HF backbone 優先: **Architecture = Hugging Face** にした上で **Value head initialization = NEW: initialize a fresh 1-output head and train it with PPO** を選択します。Model ID と pin された 40桁SHAは引き続き必要です。

CLI で指定する場合は、既存の `--critic-kind hf --critic-hf-repo ... --critic-hf-revision <40桁SHA>` に加えて以下を設定します。

```sh
--critic-hf-head-init new
```

### 新モードの実装

- `AutoModelForSequenceClassification`（または `AutoModelForTokenClassification`）の **num_labels=1** を指定し、形状が異なる分類headだけを置き換えられるようにします。
- `output_loading_info` を検査し、**Transformer backbone の重みが欠損または不一致なら停止**。不完全なバックボーンを黙って訓練しません。
- 取得した単一 `nn.Linear` head の重みとバイアスをゼロで初期化し、**headだけをPPO value lossで学習**、backboneは凍結します。
- 保存対象はこれまでと同じ Critic の head/AdamW。通常の `critic-hf-v69.safetensors` に保存され、同じ pinned checkpoint と同じ room の場合にだけ復元します。
- 現行ルームの学習則と異なるため、新モードは **新規ルームでのみ有効化**してください。既存の学習済み head モードは従来どおりで、プロトコル識別子も変更していません。
- 今回のスカラーhead初期化が必ず収束する、一般の言語モデルで有効な価値推定器になる、といった性能保証はありません。PPOが正しい報酬信号を得られなければ勾配は発生しません。

### セキュリティ・対応範囲

`trust_remote_code=False`、immutable SHA、safetensorsを維持。HF model のカスタムhead、多段head、未対応のアーキテクチャを無理に汎用化しません。`missing_keys` のうち新しい単一head以外が欠けた場合は拒否します。GPUバックエンド固有のMetalクラッシュへの対策ではありません。

## テスト

関連テスト: 108成功 / 1スキップ。新規テストで、通常の2ラベルHF Criticを厳格モードが拒否すること、新規モードで `num_labels=1` の明示指定が渡ること、欠損したbackboneを拒否すること、PPOの価値損失からheadに勾配が流れ、weightが変更され、save→resumeでも同一のweightに戻ることを確認しています。GUI設定→CLI→ルームの学習ポリシーも検証済みです。

実際の Hugging Face 大規模モデルの取得と Apple MPS の実機テストは未検証です。
