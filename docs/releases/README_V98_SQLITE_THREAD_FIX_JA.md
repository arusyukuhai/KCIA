# Kagami v98 — carry付きfast-resumeのSQLiteスレッド不整合を修正

## 症状

`save_fast_resume_cache()` から `model_task()` へ `fast_resume_v85.save()` をオフロードする場合、次のエラーで通常学習が終了する問題がありました。

```
sqlite3.ProgrammingError: SQLite objects created in a thread can only be used in that same thread.
...
patchlib/fast_resume_v85.py:193: carry_archive.journal.fast_resume_cursor()
patchlib/elastic.py:248: self.db.execute('PRAGMA schema_version')
```

## 原因

KagamiはSQLiteジャーナルをイベントループスレッドで開きますが、CPUの場合は`asyncio.to_thread()`、MPSの場合は専用executorを使ってモデル・ディスクI/O処理をワーカースレッドに移しています。v96/v97では `elasticnet.save_fast_resume_cache()` がすでに両ジャーナルのカーソルをイベントループスレッドで採取しているにもかかわらず、`fast_resume_v85.save()` が **carry側だけ再びSQLiteに問い合わせていました**。

Python sqlite3の標準スレッド制約に違反します。MetalやイベントIDの衝突ではありません。

## 修正内容

- `patchlib/fast_resume_v85.py`: 保存ワーカー内部の `carry_archive.journal.fast_resume_cursor()` 呼び出しを除去。受け渡されたスナップショットが辞書であることを確認し、モデル/ファイルの保存を行う。
- `patchlib/elasticnet.py`: `inbound_commit_lock` 中、 **保存の前と後の両方** でメイン・carryジャーナルのカーソルを同じイベントループスレッドから取得・照合。旧イベント履歴、署名、モデルの永続形式は変更しない。
- optionalなfast-resume保存でSQLiteのスレッド制約例外が再び発生しても、署名イベントの学習コミットまで巻き込んでクラッシュさせず、警告として記録する。

**`check_same_thread=False` には変更していません**。SQLを不用意に共有スレッドで実行する回避策は採らず、接続を元スレッドへ閉じ込めています。

## 検証

- v97のコードに新しい回帰テストを追加すると、ユーザーのログと同じ `sqlite3.ProgrammingError` を再現。
- v98では同じ回帰テストがCPUワーカー・MPS専用ワーカー模擬の両経路で成功。
- fast-resume、carry、ジャーナル圧迫、単独ピア世代移行、MPS・Optimizerの重点テスト **75件成功**。
- Python構文チェック済み。**macOS/MPS実機での長時間学習とユーザーの実際のDBは未検証**。

## 導入

1. GUIと学習プロセスを停止。
2. 既存のジャーナル・carry・署名鍵・復元キャッシュをバックアップ。
3. v98のコードに更新して同じルームで再開。`reset-local-data` は不要。
4. `KAGAMI MATERIALIZED COMMITTED BASE` が出れば統合weightの保存が完了したことを示します。次回のRESUME時に `KAGAMI FAST RESUME loaded materialized committed base` を確認してください。

**注意:** 以前の版で作ったfast-resumeのスナップショットがなければ初回は全イベントの復元が必要になる場合があります。容量不足・paged COW・両ジャーナルの同時進行など、既存の安全なfallback条件は引き続き残ります。v98は旧ジャーナルの無条件削除や世代証明書の検証条件緩和は行いません。
