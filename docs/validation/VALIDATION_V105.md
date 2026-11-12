# v105 Validation

Base: v104 ZIP (`kagami_v104_hf_critic_preflight.zip`).

Changed:
- `patchlib/mix_gui_v66.py` — preflight_gui_critic: signed-room-safe metadata guard.
- `patchlib/webview_controller.py` — call preflight before persisting mixed policy and before launching trainer.
- `tests/test_v105_gui_critic_guard.py` — reproduction and regression checks for 27-label, MLP, NEW, disabled-PPO paths.

Commands validated:
```
python -m pytest -q tests/test_v105_gui_critic_guard.py tests/test_v104_critic_preflight.py tests/test_v103_ppo_diagnostics.py tests/test_v102_hf_scalar_bootstrap.py tests/test_hf_value_critic_v69.py tests/test_hf_disk_preflight_v101.py tests/test_gui_mixed_critic_v66.py tests/test_webview_gui.py tests/test_v100_ppo_runtime_reliability.py tests/test_ppo_pipeline_v72.py tests/test_ppo_correctness_performance_v73.py
# 65 passed, 1 skipped
python kagami.py ppo-doctor
# PASS adapter_weight_change_l1 > 0 on Linux CPU; not an HF/model/MPS guarantee.
```

No tests use actual 4B OpenHands weights. Checkpoint `80369c1468c5...` was corroborated via pinned HF config metadata (27 id2label; Qwen3ForCausalLM architecture) and the public model card (agent rubric model, not PPO V(s)).
