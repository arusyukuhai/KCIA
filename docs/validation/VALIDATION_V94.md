# Kagami v94 validation

- Source base: v93 `kagami_v93_event_id_fix.zip`.
- Regression proof (old v93): `test_exact_prior_error_missing_journal_digest_is_not_generated_by_replayed_carry` raises `PatchError: conflicting event ID: ... journal_sha256=missing ... local_origin=True`.
- Identical test (v94): passes.
- Related test selection: 85 passed in local Linux/CPU environment, including six new v94 tests.
- Python compileall: passed.
- Not tested: user's actual production journal, macOS/MPS, multi-day federated run.
- No signed data, checkpoint, or protocol schema modification performed.
