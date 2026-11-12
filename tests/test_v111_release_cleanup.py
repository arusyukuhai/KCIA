"""v111 source distribution rules; no state, no unwanted build outputs."""
import json
from pathlib import Path
from tools.audit_release import audit

ROOT = Path(__file__).resolve().parents[1]


def test_source_dist_contains_no_live_user_state():
    assert not list((ROOT / 'fedpara-rooms').glob('*.json'))
    assert not (ROOT / '_patch_v100.py').exists()
    assert not (ROOT / 'CLEANUP_MANIFEST.tsv').exists()


def test_docs_index_and_historical_guides_have_live_links():
    outcome = audit(ROOT)
    # During this test run Python generates __pycache__. Those are ignored by
    # the zip packager, NOT by the source cleanliness audit.
    failures = [x for x in outcome['problems'] if not x.startswith('generated cache in release:')]
    assert failures == []


def test_runtime_modules_and_compatibility_tools_retained():
    inventory = list((ROOT / 'patchlib').glob('*.py'))
    # v111 retained 147; v112 and later may add independently tested modules.
    assert len(inventory) >= 147
    for path in ('patchlib/fast_resume_v85.py', 'patchlib/elastic.py',
                 'patchlib/mixed_learning_v65.py', 'patchlib/elasticnet.py',
                 'patchlib/kagami_eval_worker.py', 'patchlib/webview_diagnostic.py',
                 'patchlib/training_sim.py', 'src/main.nim', 'web/patches.html'):
        assert (ROOT / path).is_file(), path


def test_pruning_audit_is_explicit_and_reversible():
    info = json.loads((ROOT / 'docs/validation/FILE_AUDIT_V111.json').read_text())
    assert len(info['removed']) == 8
    assert info['production_python_modules_preserved'] == 147
    assert info['live_user_state_modified'] is False
    assert info['api_or_runtime_source_deleted'] is False
