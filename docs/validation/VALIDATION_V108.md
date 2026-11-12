# v108 Test notes
- Source base: v107.
- New GPU command selection is LOCAL ONLY (no signed protocol changes).
- MPS optional: cannot be fully validated on the Linux CPU-only test host.
- Tests: 165 passed, 5 skipped in focused regression (PPO, HF, MPS policy, worker, WebView, fast-resume, off-policy, critic epochs).
- Additional CPU PPO Doctor: 3 independent optimizer steps, nonzero LoRA weight delta.
- JS: Node --check web/desktop_webview.js succeeded.
- Python compileall succeeded.
- No assertion of actual Mac speedup or long-run GPU stability.
