# KCIA v104 — HF Critic の繰り返しエラーを起動前に検出

## 症状

v103ログで `HFValueError: HF Critic must have a single scalar value head (num_labels=1)`。
`HF Critic` の旧 `pretrained` モードに通常の 2クラス分類モデルを指定したため、重いweightをダウンロードし終えた後に初期化失敗するケースです。

## v104修正

1. **新しいGUIルームのみ**、HF Criticのhead初期化のUI既定値を `new` に設定。通常の2クラス分類チェックポイントでも `num_labels=1` の新しい未学習headを構築し、PPOで訓練する既存v102実装を利用します（実効性はデータ・設定次第）。既存の `mix_gui_v66.DEFAULT`、CLI既定値、保存済み設定、署名付きルームプロトコルは**変更しません**。新しいルームの設定を明示的に保存した場合にのみ、`new-scalar-zero-v102` のhead identityが適用されます。
2. **従来pretrainedモード**のHF Criticは、重いweightに先立ち、固定revisionの軽量 `config.json` を取得して `id2label` と `num_labels` を読みます。明示的に2以上（または0）が宣言されていたら `PRE-FLIGHT` で停止し、ダウンロードを開始しません。この検査はReward Model読み込みより前に実行します。
3. ネットワーク障害でconfigを取得できない、または明示的なラベル数がない場合は後続の元の厳格なHF weight検査を残します。**完全にダウンロード失敗を防げるわけではありません。**

## 現在のルームへの対処

既存ルームが `pretrained` HF Critic を署名付き学習プロトコルで固定している場合、同じルームのCriticを自動的に `new` / `mlp` に切り替えることはできません。署名プロトコルが変化するからです。

- HF Criticをもう一つ取得したくない場合: 新規ルームの TRAINING → VALUE CRITIC → `MLP`。
- HF Criticのバックボーンを使う場合: 新規ルームで `Hugging Face` ＋ `NEW ... train fresh 1-output head`。GUIの新規設定ではこれが初期選択になりました。
- 事前学習済みのscalar value head付きチェックポイントを確実に指定できる場合だけ `pretrained (legacy)` を選択します。

実行前に既存のルーム保存データをバックアップしてください。`reset-local-data` は不要。新規ルームは署名付き学習履歴を自動移行しないので、元のルームは消さないでください。

## 対応範囲

今回の実装は **Criticのhead不一致と無駄なダウンロードへの対策**です。PPOの収束率、MPSのネイティブMetal安定性、数GiB級のHFモデルの性能を保証しません。実際のmacOS/MPS環境の大型HFモデルは未検証です。

## 検証

`tests/test_v104_critic_preflight.py`: 複数ラベルのconfigでの早期拒否、DL対象が `config.json` だけであること、MLP / `new` のスキップ、single scalarの受理、メタデータ取得失敗時に偽陽性を出さないこと、Critic preflightがReward Model初期化より先に起こること、署名付きpretrainedとnewプロトコルの区別を検証。関連テスト94件成功、1件スキップ。macOS実機のGPU実行は未検証。
