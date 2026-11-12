"""Kagami desktop application launcher.

Usage: python kagami.py
       python kagami.py reset-local-data [--yes] [--root DIR] [--include-cli-defaults]
       python kagami.py disk-report [--journal PATH]
       python kagami.py disk-reclaim [--journal PATH]
       python kagami.py journal-status [--journal PATH] [--manifest PATH]
       python kagami.py mps-smoke
       python kagami.py mps-ppo-smoke
       python kagami.py ppo-doctor [--device cpu|mps]
All legacy CLI commands remain available through python -m patchlib.cli.
"""
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ('disk-report','disk-reclaim'):
        requested = sys.argv[1]
        import argparse
        parser = argparse.ArgumentParser(description=('Read-only Kagami disk usage report' if requested == 'disk-report' else 'Safely reclaim disposable Kagami caches after stopping training'))
        parser.add_argument('--journal', default='elastic-events.sqlite3')
        args = parser.parse_args(sys.argv[2:])
        from patchlib.disk_health_v86 import report, reclaim
        (report if requested == 'disk-report' else reclaim)(args.journal)
        raise SystemExit(0)
    if len(sys.argv) > 1 and sys.argv[1] == 'journal-status':
        import argparse
        parser = argparse.ArgumentParser(description='Read-only certified epoch / journal diagnostic')
        parser.add_argument('--journal', default='elastic-events.sqlite3')
        parser.add_argument('--manifest')
        args = parser.parse_args(sys.argv[2:])
        from patchlib.journal_recovery_status_v89 import report
        report(args.journal,args.manifest)
        raise SystemExit(0)
    if len(sys.argv) > 1 and sys.argv[1] == 'mps-smoke':
        from patchlib.mps_smoke_v92 import main as probe_main
        raise SystemExit(probe_main())
    if len(sys.argv) > 1 and sys.argv[1] == 'mps-ppo-smoke':
        from patchlib.mps_ppo_smoke_v108 import main as probe_ppo_main
        raise SystemExit(probe_ppo_main())
    if len(sys.argv) > 1 and sys.argv[1] == 'ppo-doctor':
        from patchlib.ppo_doctor_v103 import main as doctor_main
        raise SystemExit(doctor_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == 'reset-local-data':
        from patchlib.reset_local_data import main as reset_main
        raise SystemExit(reset_main(sys.argv[2:]))
    from patchlib.webview_gui import run_webview
    run_webview()
