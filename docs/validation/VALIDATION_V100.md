# v100 実行記録
- Base: `kagami_v99_training_optimization.zip`
- Corrected: MPS float64 in Critic importance weights, MPS float64 in behavior probability verification (v81/v82), MPS generation sampling, imported PPO rollout dtype.
- Optional grounded reference guidance with protocol-pinned coefficient; v99 defaults preserved.
- Tests: 98 passed, 1 skipped (requires Mac with MPS) in selected PPO+reward+critic+GUI+compatibility suite.
- Python syntax: compileall.
- JS syntax: node --check desktop_webview.js.
- No real macOS/Metal GPU available in test container; GPU-specific runtime remains unverified.
