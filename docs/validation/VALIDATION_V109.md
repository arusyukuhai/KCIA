# v109 検証記録

- 改修ベース: v108 MPS PPO fast
- ソース修正: `patchlib/mixed_learning_v65.py`
- 新規テスト: `tests/test_v109_long_sft_context.py`
- `python -m compileall -q patchlib` 通過。
- 実行可能なCPU回帰テスト: 110 passed, 2 skipped (詳細はコマンド実行結果に基づく)。
- 長さ3659応答・2048 token実効制限の分割、全文回答監督トークン一致、MPS/PPO関連のCPUテスト通過。
- 未検証: macOS Apple MPS実機の大規模モデル・metal native crash、実ユーザーのSFT datasetそのもの。選好対の長文SimPO対応はスコープ外。
