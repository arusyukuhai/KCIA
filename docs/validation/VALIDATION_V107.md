# v107 validation

- `PYTHONPATH=. pytest -q tests/test_v107_objective_metric_plots.py tests/test_mixed_learning_v65.py tests/test_ppo_correctness_performance_v73.py tests/test_ppo_pipeline_v72.py tests/test_v99_mixed_optimization.py tests/test_v100_ppo_runtime_reliability.py tests/test_v103_ppo_diagnostics.py tests/test_v106_hf_schema_guard.py tests/test_kcia_v36.py::test_graph_history_preserved_across_restarts_and_long_ranges tests/test_kcia_v36.py::test_model_select_recovers_old_history_without_running_worker tests/test_hf_mixed_datasets_v68.py tests/test_gui_mixed_critic_v66.py tests/test_v102_hf_scalar_bootstrap.py tests/test_v104_critic_preflight.py tests/test_v105_gui_critic_guard.py --disable-warnings --maxfail=1`: **94 passed, 1 skipped**.
- `node --check web/desktop_webview.js`: successful.
- `python -m compileall -q patchlib`: successful.
- Not a complete end-to-end HF dataset/Mac MPS/pywebview test; graphics rendering and responsiveness on user hardware not confirmed.
- Two older legacy typography assertion tests (`test_kcia_v36.py::test_font_and_ui_are_csp_permitted_and_no_courier` and `test_kcia_v31.py::test_kcia_screen_and_localization_contract`) also fail unchanged on unmodified v106 due to old exact font strings. Not counted in 94 pass set.
