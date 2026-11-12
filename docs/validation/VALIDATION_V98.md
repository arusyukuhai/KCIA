# v98 test results

- Demonstrated reproduction of the reported `sqlite3.ProgrammingError` on untouched v97 using a newly added test that invokes `save()` through `model_task()` in a different thread, with a real SQLite carry archive.
- Verified v98 fix for both `asyncio.to_thread` (CPU) and simulated serialized MPS executor (Linux).
- Focused 75-case tests for materialized fast resume, signed carry, certified epoch GC, solo recovery, MPS isolation, Optimizer and journal pressure all passed.
- SQLite `check_same_thread=True` remains the default. Nothing deleted from the signed journal or carry history.
- No actual macOS MPS run or multi-GiB production-journal test was performed.
