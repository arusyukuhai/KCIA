# Kagami v92: Metalコマンドバッファクラッシュの隔離

## 現状
v91 の単一 GPU ワーカー化後にも macOS Metal が `commit an already committed command buffer` で異常終了した。
これは Python 例外として回復できない native assertion。Kagami 側だけでは PyTorch/OS 内部の不具合を確実に修正できない。

## v92 の変更

- **macOSの自動デバイス選択をCPU優先**に変更。GUIは引き続き `--device auto` を渡し、起動後のCLIでCPUを選ぶ。リスクを隠したままGPUを選択しない。
- Actorだけでなく、Reward Model/Criticの明示的なMPS指定も安全モード中は拒否する。
- Eval worker と `device_planner` の自動選択もCPUに統一する。
- v91で初期モデル作成がイベントループ側だった問題を修正。実験モードでのモデル作成も単一GPUワーカーに移動。
- `python kagami.py mps-smoke` により、GUIを巻き込まずに独立プロセスでMPS動作を短時間検査できる。
- 学習、チェックポイント、ジャーナル、署名、ルーム定義を削除・変更する操作は一切ない。

## 起動手順

1. 実行中の学習・GUIプロセスを終了する（ジャーナルとweightは消さない）。
2. 通常の `python kagami.py` でGUIを開く。**macOSでは安全のためCPU学習を選択**する。
3. GPUの簡易検査には `python kagami.py mps-smoke` を実行する（これだけは隔離した子プロセスでGPUを使う）。
4. 実験的にGPUを使用したい場合のみ、GUI起動前に `KAGAMI_EXPERIMENTAL_MPS=1 python kagami.py` を実行する。
   CLIでも同じ環境変数を設定した後 `--device mps` を指定できる。

**注意**: mps-smokeが成功しても長時間学習でのMPSクラッシュ解消は保証しない。
GPU利用時の native abort は、チェックポイント以前の非コミット学習状態を失わせ得る。

Python 3.13.15 / PyTorch の組み合わせ、macOS バージョン、機種などで動作が変化する。
`python -c "import torch,platform;print(torch.__version__, platform.mac_ver())"` で環境を確認し、
PyTorchのMPS修正を含むバージョンで隔離再検証することを推奨。
