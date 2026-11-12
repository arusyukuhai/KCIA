# Kagami v96 — 認証済み世代移行後の高速RESUME（event replay削減）

## 根本原因

v95までは `patchlib/elasticnet.py` の起動復元で、carry archive に1件でも旧世代更新があれば、**既存のfast-resume統合weightが有効であっても利用を完全にスキップ**していました。その結果、`carry_archive.replay(engine)` → `engine.replay()` で全イベントを再度復元していました。

また、v85の統合weightは「ローカル更新 N 件ごと」と「正常終了時」にしか保存されませんでした。受信専用ピアや80%ジャーナル容量保留中は、最新の統合weightが生成されないまま長期化し得ました。

## v96の変更

1. **世代またぎのdual-lane checkpoint。** 現世代イベントジャーナルと旧世代carry archiveの両方について、同一の `inbound_commit_lock` 下でcursor（件数、最終rowid/ID、変更監査情報）を記録し、統合済みweight・LoHA/LoRA factor・anchorと紐づけます。アーカイブ側は認証済みの世代/manifest/cutoffs由来の独立genesisに固定されます。
2. **再起動時は両cursorを検証後にcacheを使用。** 未反映イベントが0件なら両journalとも再生0件です。一方だけ増えている場合はその差分だけ検証・適用します。両方に未反映更新がある場合は、二つのレーン間の適用順序を永続記録できていないため安全側へ戻り、全履歴を再生します。
3. **非ローカル更新に対応した定期保存。** 受信イベントのみ増えた場合や、容量保留中も、未封印のローカルoptimizer stepがない安全な境界で必要に応じて統合weightを保存します。通常は最大5分間隔、容量保留中は最短2分間隔で試行し、変更がなければ保存しません。`--fast-resume-every 0` は従来どおり無効化です。
4. **認証済みの古い世代の境界検証も加速。** SQLiteの変更監査と最終rowidが一致するときのみ、前回検証したcarry/旧世代のハッシュ付きprefix witnessを利用して、起動時のO(履歴件数)メタデータ再走査を省略します。証明書・署名認証を置換するものではありません。
5. **診断ログの改善。** キャッシュが存在しない、モデルIDが違う、世代のcarry cursorが古い、両laneが更新された等を明示します。

## 導入

既存のjournal, carry, checkpoint, 署名鍵を**バックアップ**した上で、すべてのKagamiプロセスを終了し、v96へ更新してください。`reset-local-data` の実行は不要です。

- **最初のv96起動**：v95のキャッシュにはcarry cursorが含まれていないため、carryがあるルームでは一度だけ従来の署名履歴復元が必要です。その後、学習の安全な境界でdual-lane checkpointが自動保存されます。アプリを異常終了させず、ログに `KAGAMI MATERIALIZED COMMITTED BASE` が出たことを確認します。
- **その後の起動**：ログに `KAGAMI FAST RESUME loaded ...`, `KAGAMI FAST RESUME carry restored suffix=0` および `cached base + 0 signed suffix updates` が出ると、署名イベントの再生0件です。
- `KAGAMI FAST RESUME: ... full replay` が出る場合、表示された理由（両レーンの進行、cursor不一致、容量不足等）を確認してください。

### 必要な空き容量

通常の統合weight cacheはモデル約1個分の追加ディスク容量と安全な空き領域を要します。空き容量不足で保存が延期された場合、再起動を高速化するための復元点は作成できません。既存の安全なジャーナルは削除されません。

**制約：v61 paged COWの仮想weightを持つ構成ではfast-resumeの保存を引き続き拒否します。** それを無理にエクスポートすると未反映のページを失う危険があります。全モデルで無条件にreplayが0になる保証はありません。GPU/MPSでの実機の長時間試験も未実施です。

### 安全性

認証済みチェックポイントと旧世代carryの意味を変更しておらず、署名付き更新の削除・再採番・上書きは行いません。双方向suffixの適用順序不明、変更監査不一致、ファイル破損、checkpoint失敗などの場合は署名履歴を再生するか、安全に停止します。

## 実装対象

- `patchlib/fast_resume_v85.py`
- `patchlib/epoch_carry_v46.py`
- `patchlib/epoch_trainer_boot_v52.py`
- `patchlib/compact_journal_gc_v88.py`
- `patchlib/elasticnet.py`
- `patchlib/cli.py`
- `tests/test_fast_resume_v96.py`

