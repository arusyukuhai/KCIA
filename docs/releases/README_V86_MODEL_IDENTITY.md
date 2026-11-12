# Kagami v86 — Unrecognized model diagnosis & safe config identity

## What v86 fixes

The user-reported error `Unrecognized model in ... Should have a model_type key in its config.json` originates from the Transformers `AutoConfig` architecture resolver. Versions v83–v85 could not safely solve a checkpoint that lacks *both* an explicit `model_type` and an identifiable `architectures` declaration. v86 does not pretend that every such checkpoint can be trained.

- Loader errors now identify **the exact model/tokenizer ID, revision, and local config path** when known. For an under-specified config, the error includes its declared `model_type` and `architectures` fields, so the source can be repaired rather than repeatedly masking the exception.
- **No more fallback to model-name heuristics** for missing or corrupt local `config.json`. This prevents accidental construction of a different architecture simply because the folder name contains `gpt2` etc.
- A local **PEFT adapter directory** (typically with `adapter_config.json`) and a **dataset directory** are explicitly distinguished from a complete base model checkpoint. Point the model field at its actual base model; do not use the adapter itself as a base checkpoint.
- The **GUI compatibility endpoint** inspects the actual model config (downloading its small JSON when uncached) before marking an architecture as a trainable candidate; it returns a diagnostic status instead of hiding an identity error.
- Tokenizer and auxiliary model loaders now preserve the failing ID even if Transformers unexpectedly tries to resolve `AutoConfig` again *after* a verified config was supplied.
- `hf-diagnose` exposes `raw_model_type`, `config_path`, `architecture_identity_verified`, and a PEFT/dataset warning. It does not inspect or run checkpoint weights.

## How to diagnose your exact failure

From inside the extracted `kagami-v86` directory, substitute the same ID/revision that appears after **`Unrecognized model in`**:

```bash
python -m patchlib.cli hf-diagnose --model <ORG/MODEL-OR-LOCAL-PATH> --revision main
```

Check `model_type`, `architectures`, `source_kind`, `notes`, and `config_path`.

- If the path contains `adapter_config.json`, it is likely an adapter; use its independently verified `base_model_name_or_path` as the base model instead.
- If `config.json` lacks `model_type` but declares a single standard `architectures` entry, v86 recovers it via the installed Transformers built-in mapping. No original files are changed.
- If *both* are missing, the model type cannot be determined safely. Find the true source architecture, or set the correct type in a local copy **only after verification**:

```bash
python scripts/fix_hf_model_type_v83.py /path/to/local/base --model-type <VERIFIED-TYPE>
python scripts/fix_hf_model_type_v83.py /path/to/local/base --model-type <VERIFIED-TYPE> --write
```

Do not set `gpt2` (or another type) arbitrarily. For a signed P2P room, changing a pinned config requires an authenticated new model revision/genesis; never silently rewrite the old one. `--hf-task causal-lm` is the prediction task, **not** the model architecture.

### Important limits

This environment has no installed Transformers package or access to the failing user's specific model/traceback; the exact checkpoint has **not** been verified to load or train. This release improves **source identification, diagnostics, and compatibility reporting**, not universal architecture support. It preserves LoHA, signed checkpoints and distributed PPO protocols.

## Tests

`PYTHONPATH=. python -m pytest -q tests/test_hf_identity_v86.py tests/test_hf_load_v85.py tests/test_hf_config_v83.py tests/test_training_compat_v84.py tests/test_hf_compat_v63.py tests/test_hf_reward_v67.py tests/test_hf_value_critic_v69.py tests/test_webview_gui.py tests/test_hf_datasets.py`

Result in this environment: **85 passed**. All real-model Hugging Face end-to-end checks remain unperformed.
