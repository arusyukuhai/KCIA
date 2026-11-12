# Kagami v105: PPO Critic GUI起動前の互換性チェック

## 今回のエラーの意味

`OpenHands/openhands-critic-4b-v1.0` の固定リビジョン（80369c1468c5...）の `config.json` は `id2label` を27個宣言します。これはPPOが予測する単一スカラー状態価値 `V(s)` の学習済みheadとは異なります。このモデルの公開カードもエージェント軌跡の複数項目評価（rubric）向けと説明しています。モデルカード本文のラベル数表記とconfig.jsonの実際の数は異なることがあるので、Kagamiのhead互換性判定は実際の固定されたconfig.jsonに従います。

v104のCLI preflightは正しく不一致を検出しましたが、GUI側のチェックはワーカー開始後だったため、同じ設定での失敗を繰り返せました。

## v105変更

- GUIから開始するとき、混合PT/SFT/PPO設定を正規化・署名ポリシー照合した後、**未学習でない学習済みスカラーHF Critic** を選択し、PPOデータが設定されている場合に限り、Hubの小さな `config.json` をチェックします。
- `id2label != 1` または `num_labels != 1` が明示されていると、**新しいimmutable mixed objective policyを保存する前**、ワーカー起動・大型weight取得の前にエラーにします。
- チェックに失敗しても既存ジャーナル、署名済みポリシー、weight、Optimizerやモデルのweightは消しません。
- `Value Critic = Local MLP` または新規ルームで意図的に `Head initialization = NEW` を選んだ場合は従来どおり動作します。NEWは未学習headを追加する操作であり、元の27ラベルrubric headを学習済み状態価値に変換する操作ではありません。
- ネットワーク不調等でconfigを取得できなければ既存のワーカー側チェックへ委ねます。実際の学習成功を保証するものではありません。

## PPOを実際に開始するには

**推奨:** 新しいルームを作成 → TRAININGでPT/SFT/PPOを選択 → VALUE CRITIC=Local MLP → PPOデータとRewardの適切な設定を入れる → 学習開始。HF Criticの追加ダウンロードはありません。Reward Modelは別設定です。

HF Value Criticにこだわる場合は、PPO用の単一状態価値headを既に持つチェックポイントを選択するか、NEW scalar headを新規ルームで明示指定し、Value Lossで学習させてください。特にOpenHandsの4B rubricモデルはPPOのValue Criticとして適切とは限らず、ストレージとRAMを大きく消費します。

**署名済みの既存ルームのCritic方式は書換不可**です。新ルームを作成しても自動的に旧ルームの学習weightが引き継がれるわけではありません。旧ルームのジャーナル、ベースモデル、チェックポイントは削除しないでください。

## 検証

- v105 GUI起動の新規3テストを含む対象65件成功、MPS依存1件スキップ。
- `python kagami.py ppo-doctor` はLinux CPUで4ステップ成功し、LoRA weightが更新されました。
- Apple Silicon/MPSの実機およびユーザーの実データでは未検証です。
