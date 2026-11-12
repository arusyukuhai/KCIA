# KCIA v91 — macOS Metal/MPS 実行競合対策

対象: v90 のソースをベースに、MPS 向けの学習・Optimizer更新・P2P反映・チェックポイント操作を安全側へ変更した版。

## なぜ修正したか

ユーザー報告の例:

```
failed assertion _status < MTLCommandBufferStatusCommitted
in -[IOGPUMetalCommandBuffer setCurrentCommandEncoder:]
... resource_tracker: There appear to be 1 leaked semaphore objects ...
```

Metalで既にcommitされたコマンドバッファにエンコーダを追加しようとしたことを示すアサーション。KCIAでは従来、学習、Optimizer step、署名済みP2P更新の適用、weightの出力・読み込みが `asyncio.to_thread()` の不特定ワーカーに渡され、各ワーカーのMPSキューが完了する前に次の処理に移る可能性があった。これは再現可能な潜在的競合である。ただし実機クラッシュの唯一の原因とは断定できない。

`resource_tracker` のセマフォ警告は異常終了の後処理による可能性があり、このパッチでは乱暴なセマフォunlinkは行わない。

## 実装変更

1. `patchlib/mps_runtime_v91.py` に MPS専用 `ThreadPoolExecutor(max_workers=1)` を追加。GPUに触るモデル操作は同じワーカーに直列投入し、各操作の完了・例外発生時に `torch.mps.synchronize()` を挟む。CPU処理の実行方法は従来どおり。
2. `run_step_offloop` のストリーミングデータ取得を一般ワーカーに分離。HTTP待ちの間にMPSワーカーが塞がれないようにする。
3. MPSバッチ転送で `non_blocking=True` を避ける（CUDA転送は従来どおり）。
4. 学習、Optimizer step、署名イベント適用（単体・バッチ）、carry、署名イベント作成、統合weightの保存・反映、working/paired checkpointのGPUコピー、世代移行のshadow model等をMPSワーカーに集約。
5. `--device cpu`でも別配置のHF Reward ModelやCriticがMPSを使う構成を検出する。
6. 初回のモデルロード直後、MPSワーカーへの引継ぎ前にGPUキューを同期する。

GPUコマンドを多く発行する場合、各操作の同期・逐次化によって処理速度が低下する可能性がある。今回はクラッシュ再発の抑制を優先。

## 導入

1. GUIと学習プロセスをすべて停止し、現在の `rooms` ディレクトリ・ジャーナル・学習済みモデルと識別鍵をバックアップする。
2. 本ZIPを新しいディレクトリに展開し、既存の保存領域は削除せずに通常の方法で起動する。原則、同一ルームの全ピアで配布コードを統一する。
3. Macで **既存ルームのまま** 少数ステップを試し、`torch.mps.synchronize()` まわりの停止・金属例外が出ないか確認する。従来のデータを初期化する必要はない。
4. 再現する場合、`PYTHONFAULTHANDLER=1 python kcia.py` で実行し、クラッシュ直前の学習ログと `python -c 'import torch,platform;print(torch.__version__,platform.mac_ver(),torch.backends.mps.is_available())'` の結果を保管する。
5. CPUに切り替えて再現するか比較する。CPUでは起きずMPSだけで起きるなら、MPSバックエンド側の問題という仮説が強くなる。PyTorch/macOSの組み合わせを変更して検証する場合は、まず仮想環境を複製する。

**注意**: 既存v90の署名済みジャーナル、epoch/carry、GC認証、コミッティー人数ルールは変更していない。このパッチはディスク不足対策ではない。

## 検証範囲

- Linux / CPU PyTorch環境で、MPSワーカーの直列実行・例外時同期・contextvars伝搬を模擬したテスト、およびP2P・保存復元関連の重点回帰テスト: 95件成功、1件スキップ。
- Python構文チェック・CLI help表示成功。
- **Apple Silicon / macOS / MPSを使った実機再現試験は未実施。** このためMetalドライバ/PyTorch側のアサーションが完全に消える保証はない。
