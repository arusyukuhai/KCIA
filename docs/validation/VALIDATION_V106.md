# Kagami v106 validation

- Source base: Kagami v105 (ppo_critic_startup_guard)
- Static: Python compileall; Node.js `node --check web/desktop_webview.js`
- New tests: tests/test_v106_hf_schema_guard.py
- Covered: alias mapping for SFT/PPO, no-loss first-row peeking, correct chat completion isolation, fail-closed ambiguous mappings, missing follow-up row, no reference for ground-truth PPO, SFT preflight before Reward/Critic loading, prefetch errors preserved, GUI probe returns field names only.
- Other runs: tests/test_hf_mixed_datasets_v68.py, test_ppo_pipeline_v72.py, test_audit_v74.py, test_mixed_learning_v65.py, test_gui_mixed_critic_v66.py, test_v103_ppo_diagnostics.py, test_v105_gui_critic_guard.py, test_v100_ppo_runtime_reliability.py, test_v99_mixed_optimization.py, test_hf_disk_preflight_v101.py, test_v102_hf_scalar_bootstrap.py, test_v104_critic_preflight.py, test_hf_reward_v67.py, test_hf_value_critic_v69.py.
- Mac MPS, actual HF network streaming and long real-model PPO learning: not tested here.
