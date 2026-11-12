# Kagami v108 — Apple MPS対応・PPO/PT/SFT高速化

v107をベースにした、**各ピアのローカル実行デバイス**改善です。署名付き学習目的、モデルID、P2PイベントID、ジャーナル/fast-resume形式、optimizer/checkpoint形式を変更しません。

## 起動方法

まず既存のKagamiを停止して、ルームのジャーナル・鍵・チェックポイントをバックアップしてください。**`reset-local-data` は実行しないでください。**

1. ターミナルで `python kagami.py mps-smoke` を実行すると、軽量なMetal probeを独立プロセスで実行します。
2. `python kagami.py mps-ppo-smoke` を実行すると、**別の子プロセス＋専用MPSモデルスレッド**で、小型ActorのLoRA/PPOを4ステップ学習し、重みが変わったかと1ステップの所要時間を表示します。HFモデルは取得せず、既存のユーザージャーナルは変更しません。
3. 両方成功した場合のみ、GUIの `TRAINING SETTINGS → Compute device (this peer only)` にて `Apple MPS · GPU (experimental)` を**明示的に**選択してから学習開始してください。`AUTO` はmacOS上では従来どおりCPU安全モードです。
4. Reward ModelとHF CriticをGPUに配置したい場合は、各々の `Inference device` / `Critic device` で `Apple MPS` を選択します。CPUで動作させても、ActorのPT/SFT/PPOはMPSで学習できます。`Local MLP` Criticは常にActorのデバイスで処理されます。
5. ログの `KAGAMI COMPUTE DEVICE actor=mps ...` を確認します。`actor=cpu` と表示される場合、GPU学習ではありません。

CLIで直接起動する場合は起動前に環境変数を指定する必要があります:

```bash
KAGAMI_EXPERIMENTAL_MPS=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
  python -m patchlib.cli train-elastic [既存の引数...] --device mps
```

GUIから明示的にMPSを選択した場合のみ、学習**子プロセス**に `KAGAMI_EXPERIMENTAL_MPS=1` を設定します。システム全体や他ルームの設定を変えません。未対応MPS演算には `PYTORCH_ENABLE_MPS_FALLBACK=1` をデフォルト設定（既に環境変数を指定している場合はそのまま尊重）。**float64はCPU上で計算する必要があり、フォールバック環境変数だけでは対処できません。**

## 高速化の内訳

- **生成トークンのサンプリング:** これまでMPSでは**毎トークン**語彙全体（例えば数万〜十数万要素）の確率ベクトルをCPUにコピーしていました。v108はまずMPSネイティブ `torch.multinomial` を使い、GPU上のまま抽出します。対応していないPyTorch/MPS演算に限ってCPUへフォールバック。非対応時や明示的に試験したい場合、`KAGAMI_MPS_NATIVE_SAMPLING=0` で従来のCPUサンプリングを使えます。
- **GAEの高速化:** 最大128要素の逐次再帰で多数のMPSカーネルを起動する代わりに、old-valuesの小ベクトルを1回CPUへ移してGAEを計算、計算結果をまとめてMPSへ返します。AdvantageとReturnは引き続きMPS上でPPOのbackwardに利用します。ループの演算順・terminal reward・discountの意味を変えません。
- **スレッド統一:** v91の専用GPUワーカーでActorの初期化・PPO・weight/optimizer更新を実行し、モデル操作の境界で同期します。安全上、タスクごとの同期を省いていません。
- **モデルをGPUに常駐:** Reward Modelの `CPU offload between scores` をオフにすると、モデルがGPUに収まる場合は頻繁なCPU↔MPS転送を省けます。ただしGPUメモリを大きく消費します。
- **PPOのActorはMPS**、検証のFP64 importance weights / 署名整合性チェックはCPU。Local MLP Criticを選ぶと、別の4B HF Criticで生成トークンの各prefixを評価する高コストなforwardを避けられます。

これらは**高速化を狙った実装**ですが、Apple GPUを持たないテスト環境では実際の処理速度改善を測定できていません。小さいモデルではGPU転送・起動コストのためCPUの方が速い場合もあります。

## MPSを使う際の安全性・制限

- v91までに報告された `commit an already committed command buffer` などの**ネイティブMetal abort**はPythonの例外処理で捕捉できません。今回の変更でもMacでの実機安定性は未証明です。単独の診断プロセスが成功しても、4Bモデルの長時間運転が成功するとは限りません。
- `mps-ppo-smoke` はクラッシュしてもGUIを巻き込みませんが、**通常の学習ワーカープロセス自体**は異常終了する可能性があります。事前にジャーナル/チェックポイントをバックアップしてください。
- Apple Siliconはunified memoryのため、Actor/Reward/Criticの3モデルを同時にGPU常駐させるとRAM容量を容易に超えます。**最初はActor=MPS、Critic=Local MLP、Reward=CPU**。十分なメモリと安定性を確認してからReward=MPSを試してください。
- 古いroomの署名付き学習規約に書かれた目的関数は変わりません。GPU/CPUの選択はピアごとのローカル実行設定です。新しいトレーニング目的を必要とせず、既存ルームをリセットする必要はありません。
- 既存の `torch.mps.synchronize()`によるモデルタスク境界は維持しました。安全性と速度のトレードオフがあり、強制的な非同期GPUキュー共有はしていません。

## 診断

MPS上での `PPO` 更新動作を分離確認: `python kagami.py mps-ppo-smoke`

通常CPUの参考値: `python kagami.py ppo-doctor --device cpu --steps 4`

ステップごとに `seconds=`、最後に `avg_step_seconds=` を出します。比較は同一マシンで実施してください。小型モデルの速度結果を巨大HF Actor/RM/Criticの実測と混同しないでください。

**実施した検証:** 165件の関連回帰テスト成功、5件は環境やApple GPUがないためスキップ。`node --check`、`python -m compileall`、CPUのPPO doctor更新テスト通過。macOS実機MPSでの試験は未実施。
