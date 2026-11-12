# Kagami v99 validation

- Base: `kagami_v98_sqlite_thread_fix.zip` (v98 source).
- Command: `PYTHONPATH=. pytest -q tests/test_v99_mixed_optimization.py tests/test_mixed_learning_v65.py tests/test_gui_mixed_critic_v66.py tests/test_async_mixed_v6.py tests/test_hf_value_critic_v69.py tests/test_hf_mixed_datasets_v68.py tests/test_ppo_pipeline_v72.py tests/test_ppo_correctness_performance_v73.py tests/test_offpolicy_critic_v79.py tests/test_critic_epochs_v78.py tests/test_fast_resume_v85.py tests/test_fast_resume_v96.py tests/test_epoch_identity_scope_v94.py tests/test_epoch_export_retry_v95.py`
- Result: **105 passed** (Linux CPU + torch 2.10.0+cpu).
- `python -m patchlib.cli train-elastic --help`: parsed and showed new v99 options.
- `node --check web/desktop_webview.js`: passed.
- `python -m compileall -q patchlib`: passed.
- Actual GUI boot: not tested in container because `pywebview` is not installed. Existing GUI tests and settings bridge pass.
- Apple MPS/GPU actual training, long-duration P2P convergence and benchmark gains: not tested.

New tests: advantage modes, pairwise CE+preference gradients, invalid option validation, moderate KL-stale rollout recollection, adaptive KL update only on reused rollout, stable legacy room policy, pinned v99 room policy, GUI→CLI options, gradient clipping and fail-closed nonfinite gradients.

Known limitations: new options require new immutable mixed room; adaptive KL is local state and resets on process restart; SimPO-inspired SFT-preference mode currently uses local JSONL only and increases forward cost; very stale PPO histories are still rejected rather than silently accepted.
