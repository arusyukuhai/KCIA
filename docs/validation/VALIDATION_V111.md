# Kagami v111 — 配布ファイル整理の検証

- 基準: `kagami_v110_clean_architecture.zip`（v109からのv110構造分割済み）
- 変更点: ドキュメントの配置、配布物の古い生成物の除外、最上位READMEの整理、説明用docstringと空行の整形、オフライン監査ツールとテスト
- 実稼働の署名付きジャーナル、ルーム、weight、秘密鍵、チェックポイントには触れず、元のソースZIPを展開した作業コピーに対してだけ処理
- リポジトリの全Pythonモジュール147個を保持。旧コードの通常/動的参照を監査し、実行時コードを削除しない方針
- 旧README全文は `../releases/README_V110_FULL_HISTORY.md`、ファイルの削除/移動理由は `FILE_AUDIT_V111.json`

## テスト結果

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest -q -p no:cacheprovider tests --ignore=tests/test_live_gui_random_tinystories.py --deselect=...` → **1,107 passed, 5 skipped, 20 deselected**（69.20秒）。
- 除外した20件は元のv109で同条件下に失敗することがv110作成時に確認済み。今回新たに除外したテストはない。
- `tests/test_v111_release_cleanup.py`（実行時データ混入防止、docsリンク、147モジュール維持）および退役HTML検査は成功。
- `python -m patchlib.cli --help`、`node --check web/desktop_webview.js` 成功。
- v110にあった147個の本番Pythonモジュールはすべて存在。143個はバイト完全一致、残り4個はPython AST上の実行コードが同一（説明文・空行の調整のみ）。
- `kagami.py`、`src/main.nim`、`web/desktop_webview.html/js`、`patchlib/elastic.py`、`patchlib/fast_resume_v85.py` はv110とバイト同一。
- オフラインの `tools/audit_release.py` を実行し、配布に不要な生成キャッシュ・個人ルーム定義・文書リンク切れがないことを確認。
- v111のソースZIPのCRC検査に成功。Git形式の差分パッチをv110の独立コピーに適用し、全ファイルのバイト一致を確認。

## 制限事項

- 実機macOS MPSとWebView、巨大Hugging Faceモデル、P2P 4096ピアでの実走はこの環境では未検証。
- v110以前から失敗する既知の旧GUIスナップショット等のテストについて、v111で全て解消したとは主張しない。
- `elasticnet.train_elastic` は巨大な状態機械のまま。大幅な制御フロー改変は本リリースに含まない。
