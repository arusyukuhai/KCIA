# KCIA v94 — 世代間のイベントID衝突・carry復元修正

## 修正した再現可能な問題

KCIA v93では、旧世代の署名イベントを `CrossEpochCarryArchive.replay()` / `accept()` で現在のモデルに反映すると、`ElasticEngine.applied.hot` に `(origin, seq)` が登録されていました。しかし、旧世代のイベントは**旧世代の別のcarryジャーナル**に永続保存され、現世代の `EventJournal` には存在しません。そのため新世代で `(同じorigin, 同じseq)` を発行すると `journal_sha256=missing` で衝突と誤判定されます。

今回、v93の未修正版を使った独立テストで以下を再現しました。

`PatchError: conflicting event ID: origin=alice seq=1 journal_sha256=missing ... local_origin=True`

v94の同じテストは成功します。入力はEd25519署名付きの旧世代イベントと実際のSQLite＋sidecarジャーナルです。

## 変更点

- `EventInfo` に署名済みメタデータから検証された `genesis` 情報を保持。イベント署名、ワイヤ形式、DBスキーマは変更しません。
- 現世代の `applied` インデックスへは**現世代イベントのみ**記録。旧世代のイベントは、もともと専用のcarryジャーナル `(origin,seq)` と世代境界で重複排除されるため、現世代のID空間に混ぜない。
- 旧世代の**本人**イベントがcarryされても、現世代のLoHA/LoRAローカル因子・anchorを上書きしたり、過去世代の `reinit_after` に従って現世代のReLoRA因子を再初期化したりしない。検証済みのモデル差分だけをグローバルweightへ反映。
- 通常の現在世代イベントに対する署名検証、ID衝突拒否、重複適用拒否、ジャーナル上限、世代GC、MPS安全モードは維持。

## 導入

1. すべてのKCIAプロセスを終了します。GUIで停止後、学習プロセスが残っていないことを確認。
2. 既存の保存先（ルーム、ピア識別鍵、`events.sqlite3` と `.fplog`/`-wal`/`-shm`、carry、認証済みcheckpoint、fast-resume、Optimizer状態）を**一式バックアップ**します。
3. このZIPのコードでKCIAを起動します。**reset-local-dataは不要**です。署名済みイベントも自動で削除・再採番しません。
4. 起動後、以前失敗していたsealが実行できること、journalが現在のイベントを保存し、carryが旧イベントを保持することを確認します。

### 注意

- ユーザーの実際のジャーナル/実機Macを受け取っての検証ではありません。v93コードから再現した同型エラーに対してのみ、修正前に失敗し修正後に成功することを確認しています。
- もし既存のイベントIDに本当に**同一世代内の異なる署名済みpayload**が存在すれば、引き続き衝突として拒否します。それは自動上書きしてはいけない状態です。
- 過去に異常終了したセッションのweightやOptimizerが部分的に保存されている可能性は別途あります。v94は既存の復旧検証を回避しません。

## テスト

- 新規 `tests/test_epoch_identity_scope_v94.py`：6テスト。旧・新世代同一ID、自己carry、再起動時carry＋現世代ジャーナル再生、非同期P2P受信、ReLoRA境界、正確な既存エラーメッセージの再現。
- v93ベースの関連テストと併せて **85件成功**（CPU、短時間、実ネットワークなし）。
- 構文検査成功。Metal/MPS実機での長期学習は未検証。
