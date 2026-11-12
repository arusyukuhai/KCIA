# KCIA v106 — HF SFT/PPO データセット列対応・学習開始前検証

## 事象

`HFMixedDatasetError: Missing Hub column 'prompt'` はHF CriticやMPSのエラーではない。HF Reward ModelとValue Criticをロードして `KCIA PPO INIT phase=ready` に達した後、SFTデータセットの先頭行を読んだ時点で、選択した列名と実際のHubデータセットのスキーマが異なっていた。

`prompt/response` はKCIAの標準入力形式であり、すべてのHFデータセットが同じ列名を持つわけではない。

## 変更点

- Hub SFT/PPOの先頭行を **Reward ModelとHF Criticの巨大なweightをロードする前**にストリームで読み、値と列対応を検証する。
- 先頭行はメモリ上で1行保持して実際の学習に渡す。最初の学習例を捨てない。
- SFTでは`instruction/output`, `question/answer`, `prompt/completion`, `query/response`など一意に確定する組を標準`prompt/response`へ自動マッピング。PPOでも同様に`prompt/reference`へマッピング。
- 標準的な `messages: [{role, content}, ...]` を使うSFT会話形式では、最後のassistantの回答をターゲット、残りをpromptとして分ける（最後の回答をpromptに含めない）。
- 明示指定済みの列名が不一致、複数の正当な候補が存在、未対応の形式、値が文字列でない場合は自動補完しない。利用可能な列名と修正手段を表示して停止。
- 正解参照が必要なtoken-match/exact PPOで`reference`が見つからなければReward Modelロード前に停止。外部HF Reward ModelではPPOプロンプトのみも許可。
- GUIのHF SFT/PPOの`Validate columns`で学習開始前に先頭行を試読可能。ユーザーデータ本文ではなく列名と採用されたマッピングだけを表示する。`Pin + inspect`は旧来どおりメタデータ確認。
- 確定されたマッピングはそのデータセットストリーム中に固定し、次行の欠損はエラー（空文字で学習しない）。
- 既存ルームの署名付き学習プロトコルや永続化形式は変更しない。Hubの列設定は各ピアのローカル設定。

## GUI手順

1. TRAINING → SUPERVISED FINE-TUNING → Hugging Face を開く。
2. 選択したデータセットの40桁revision、config、splitを確認する。
3. `Validate columns`を押す。実際の先頭行の列名と確定マッピングが表示される。必要なら`列とsplitを確認・調整`から明示的に設定する。
4. PPO側も選択している場合は同様に検証する。`HF Reward Model`を使うときはPPOにreference列がなくてもよい。
5. エラーが消えてから学習を再実行する。現在のルームをリセットする必要はない。

## CLIでの明示的設定例

列名が `instruction` と `output` のSFTデータセットの場合:

```bash
--sft-hf-prompt-column instruction --sft-hf-response-column output
```

`question` と `answer` の場合:

```bash
--sft-hf-prompt-column question --sft-hf-response-column answer
```

実際の列名が不明な場合、推測で設定しないこと。データセットのschema/GUI検証出力を確認する。

## 検証上の限界

HF Hub上の実際のユーザー指定データセットIDはログに含まれておらず、こちらではそのスキーマを確認していない。ネットワーク経由の特定Hubリポジトリの読込や、Mac実機での大型Actor/Reward Model/Critic同時学習は未検証。v106は列名の不一致を早期に防ぎ、いくつかの典型形式を安全に認識する修正であり、あらゆる形式の自動変換やPPO収束は保証しない。
