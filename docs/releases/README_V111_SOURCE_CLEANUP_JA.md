# KCIA v111 — 配布ファイル整理 / docs再編

## 変更したこと

- v110にあったルート直下のリリースノートと検証ログを `docs/releases`, `docs/validation`, `docs/guides` へ移動。歴史資料は捨てず、READMEから探せるようにした。
- v85の古いCLEANUP資料、適用済みv94/v98差分、作成専用 `_patch_v100.py`、不要な短文のREADMEポインタを配布から除外した。
- 現行サーバー／GUIから参照されず旧HTML外観テストにのみ残っていた `web/dashboard.html` を廃止。対応テストを実際のWebアセット構成に更新した。
- ユーザー/端末固有の `kcia-rooms/*.json` をソース配布から除外した。既存の利用者のファイルを削除する変更ではない。
- 実行経路と公開importに使われる147個のPythonモジュールは削除せず維持。静的importと文字列参照の両方を監査。
- バージョンを重ねて肥大化したトップレベルREADMEを、起動・ソースマップ・安全対策に絞った読みやすい入口へ書き換えた。旧README全文は `docs/releases/README_V110_FULL_HISTORY.md` に保存。
- CLIなどの大きな空行塊を整形。`patchlib/scale_sim.py` 内の消失済み参考文書への参照を修正。
- `.gitignore` とオフラインで実行できるソース配布監査スクリプトを追加。

## 変更していないもの

`patchlib` の公開API、署名付きイベントとルームポリシー、SQLite schema、学習の数値処理、LoHA、PPO、MPS、世代移行、鍵、既存のジャーナルとチェックポイント。インストール先で `rm -rf kcia-rooms` のような操作はしないでください。

## 削除の限界

`kcia_eval_worker.py`, `training_sim.py`, `webview_diagnostic.py` は通常の静的importが少ないですが、別プロセス起動・ベンチマーク・診断から参照されます。未使用ではないため保持しました。全ての条件で使われない関数を証明するのは静的解析のみでは不可能で、無理な削除はしていません。

## 検証

詳細は `../validation/VALIDATION_V111.md` と `../validation/FILE_AUDIT_V111.json` を参照してください。
