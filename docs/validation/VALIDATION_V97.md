# v97 regression test log

- 91 focused tests passed (`pytest -q` across epoch, carry, and solo-committee test files).
- New tests: saved vote followed by failed certificate commit; reopen; recover from verified original signatures; no quorum synthesis when votes absent; hold stage instrumentation.
- `python -m compileall -q patchlib`: passed.
- Not tested: the user’s actual 3+ GiB journal / macOS Metal or production multiple devices.
- Source journal, identities, and P2P signatures remain unchanged.
