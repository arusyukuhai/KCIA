# v104 validation

- Critical root cause: v103 used `critic_hf_head_init=pretrained` when loading a non-scalar Hugging Face classification checkpoint, failing only after large download.
- Verified head metadata preflight uses pinned config.json only and precedes HF Reward Model initialization.
- New GUI room defaults to `new` (explicitly signed into the new room); legacy CLI, existing signed rooms and persisted settings are not changed.
- 94 focused unit/integration tests passed, 1 skipped (environment dependent).
- No physical macOS MPS or remote huge model training executed; early guard is not proof of convergence.
