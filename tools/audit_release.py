"""Offline read-only audit for release cleanliness and documentation links.

Usage: python tools/audit_release.py [--json]

This does NOT delete files or assume unimported modules are dead: dynamic subprocess
imports, command-line entrypoints, Nim staticRead and standalone tools are real uses.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r'(?<!!)\[[^]]*\]\(([^)]+)\)')
REQUIRED = (
    'kcia.py', 'patchlib/cli.py', 'patchlib/elasticnet.py',
    'patchlib/mixed_learning_v65.py', 'patchlib/fast_resume_v85.py',
    'patchlib/kcia_eval_worker.py', 'patchlib/webview_diagnostic.py',
    'patchlib/training_sim.py', 'web/desktop_webview.html',
    'web/desktop_webview.js', 'web/index.html', 'web/patches.html',
    'src/main.nim', 'docs/INDEX.md', 'docs/SOURCE_LAYOUT_JA.md',
)
OBSOLETE = (
    'CLEANUP_MANIFEST.tsv', 'CLEANUP_SUMMARY.json', '_patch_v100.py',
    'V94_SOURCE_CHANGES.diff', 'V98_PATCH.diff', 'README_CUSTOM_STATE.txt',
    'web/dashboard.html',
)


def audit(root: Path = ROOT) -> dict:
    problems: list[str] = []
    for rel in REQUIRED:
        if not (root / rel).is_file():
            problems.append(f'missing required runtime or documentation: {rel}')
    for rel in OBSOLETE:
        if (root / rel).exists():
            problems.append(f'obsolete build artifact in release: {rel}')
    for path in root.rglob('*'):
        if '__pycache__' in path.parts or '.pytest_cache' in path.parts:
            problems.append(f'generated cache in release: {path.relative_to(root)}')
        if path.is_file() and path.parent == root / 'kcia-rooms':
            problems.append(f'private/runtime room data in source release: {path.relative_to(root)}')
    docs = sorted([root / 'README.md', *((root / 'docs').rglob('*.md'))])
    for file in docs:
        if not file.exists():
            continue
        for raw in LINK.findall(file.read_text(encoding='utf-8')):
            dest = raw.strip().split('#', 1)[0].split('?', 1)[0]
            if not dest or '://' in dest or dest.startswith(('mailto:', '/')):
                continue
            # Github anchors and remote URLs do not correspond to local paths.
            if not (file.parent / dest).exists():
                problems.append(f'broken docs link: {file.relative_to(root)} -> {raw}')
    source_count = len(list((root / 'patchlib').glob('*.py')))
    if source_count < 147:
        problems.append(f'expected runtime modules appear missing (found {source_count}/147)')
    return {
        'status': 'PASS' if not problems else 'FAIL',
        'python_runtime_modules': source_count,
        'top_level_file_count': sum(p.is_file() for p in root.iterdir()),
        'documentation_pages': len(docs),
        'problems': sorted(set(problems)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    result = audit()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"KCIA source audit: {result['status']} | "
              f"{result['python_runtime_modules']} Python runtime modules | "
              f"{result['top_level_file_count']} root files")
        for p in result['problems']:
            print(' -', p)
    return 0 if not result['problems'] else 1


if __name__ == '__main__':
    sys.exit(main())
