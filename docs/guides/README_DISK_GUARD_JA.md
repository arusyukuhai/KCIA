# Kagami DiskGuard — ディスク容量不足の対策

前回の軽量化・reset-local-data・fast-resume実装の上に、**Kagamiの保存経路を直接修正**しました。

## 確認した原因

- Python CLI の `--max-journal-gb` は従来 0（無制限）。それだけでなく、**WebView JavaScript の `trainingOptions()` が `max_journal_gb:0` を固定送信**しており、バックエンドの初期値だけを直しても GUI から無制限で起動する。
- P2Pのフルモデル送信では `*.snapshots/objects` に、受信・認証済み世代データと送信用の使い捨てデータが混在。使い捨て古い送信データを安全にGCしにくい。
- SQLite WALのPASSIVEチェックポイントはWALを必ず切り詰めるわけではない。
- Disk AdamWは、MacのAPFSでもLinuxのreflink命令を試すだけで、結果として巨額のストリーミングコピーになる場合があった。さらに moment ファイルの `truncate` は論理サイズだけ確保するため、後で書き込んだ際にENOSPC/SIGBUSとなり得た。
- `*.working.safetensors.moments-*` にクラッシュで孤立した古い復元用ディレクトリが残り得る。
- GUIログやグラフ履歴は無制限に増加し得る。GUIが独立して作る `models/<peer>` / `checkpoints/<peer>` も、ジャーナルだけの容量調査では見落とす。

## 実装した保護

1. GUIのJournal GiBを**明示的な数値設定**に変更。新規のCLI/GUIの初期値は **4GiB**。0は意図的な無制限指定として残すが推奨しない。既存学習のジャーナル自体は変換・削除しない。
2. 署名イベントを追記する前に、パケット長とSQLiteの追加容量＋空き容量予備（初期値1024MiB）をチェック。残量不足では `KAGAMI DISK PRESSURE` と理由を出して拒否し、イベントの一部だけをコミットしない。Optimizer stepに入る前も定期チェック。上限到達時の学習停止は**意図したフェイルセーフ**。
3. P2P**送信用**フルsnapshotを専用の `*.snapshots/offers/{manifests,store}` へ分離。受信済み状態・世代移行のオブジェクトは従来の `*.snapshots/objects` に保持。送信終了を確認できるときに限り、新専用領域で最新1件を残して過去のマニフェストと不要なシャードを自動GC。転送中のものは消さない。
4. フルsnapshotエクスポート前に推定サイズの空き容量を確認。不足したら送信用snapshotだけ延期し、署名ジャーナルの予備容量を消費しない。
5. WALが128MiB以上に増えた場合、PASSIVEに加えてSQLite API経由で `TRUNCATE` チェックポイントを短いタイムアウトで試す。使用中のWALを直接削除しない。
6. Disk AdamWのCoW複製としてmacOS `clonefile` / Linux `FICLONE`をサポート（実ファイルシステムで機能を試験）。CoW非対応ならコピー用の空き容量を確認。実稼働の新しいmmap momentファイルは空間を先に物理予約し、遅延SIGBUSを回避する。
7. 学習開始時、ジャーナルのロックを取得してから、**現在のworking snapshotが参照していない**孤立した旧optimizerバックアップを回収。
8. グラフ履歴は最大約16MiB、超えたら直近約8MiBへ圧縮（古いグラフ履歴のみ失う）。GUIログは約32MiBで切り詰め（デバッグログのみ）。署名履歴とは無関係。
9. `fast-resume`の古い生成キャッシュを、現行の検証可能なポインタに基づいて再掃除する。現行チェックポイントは消さない。

## コマンド

まず学習中でも使える読み取り専用診断：

```bash
python kagami.py disk-report --journal /path/to/events.sqlite3
```

GUIのジャーナルは通常 `.fedpara-gui/rooms/<room>/peers/<profile>/events.sqlite3` にあります。上のコマンドはこのレイアウトの場合、`models/<profile>`、`checkpoints/<profile>`、Critic、グラフ履歴も分類表示します。サイズは**論理サイズ**（APFSのCoW領域などでは実消費と異なる可能性）。HF共通キャッシュ・独自`--shard-store`・TMPDIRは外部領域なので自動集計しません。

Kagamiを**完全停止してから**、安全に不要領域を回収：

```bash
python kagami.py disk-reclaim --journal /path/to/events.sqlite3
```

書き込みロックが保持されていれば拒否します。既存の署名ジャーナル、モデルの確定checkpoint、認証済み世代は削除しません。新方式で作成した送信用キャッシュと、孤立したoptimizerバックアップだけを対象にします。

CLI `train-live` の新設定：

```bash
--max-journal-gb 4 --min-free-disk-mib 1024
```

`--max-journal-gb 0` は無制限を明示指定、`--min-free-disk-mib 0` は空き容量ガードを無効化。どちらもディスク逼迫時には非推奨です。

## 重要な残る制約

- **現役世代の署名ジャーナルのサイズを完全な定数に保ちつつ、すべての長期切断端末との無損失同期を保証することはできません。** 過去履歴の完全削除には認証済み世代切替と同期状態の証明が必要です。容量上限に達した場合は安全に停止します。これを「無制限に永久学習できるようになった」とは呼びません。
- 旧版の `*.snapshots/objects` は、受信済み・世代保存データが混在するため**無条件に削除しません**。v86導入後の送信用だけが新しい専用領域へ書かれます。初回起動で旧形式を勝手に掃除しません。
- `models/` と `checkpoints/` の認証済み・外部参照あり得るweight、Hugging Faceモデルやdatasets、OSが管理するキャッシュも勝手に消しません。実機で空き容量がほぼ0なら、まず不要な個人ファイル・未使用の別モデルキャッシュ等を移動して空きを作ってください。
- 大規模モデルで **モデル本体と復元キャッシュ等を同時に保持する余裕がない**場合、追加のディスク、またはアーキテクチャ上の保存方法の再設計が必要です。今回のバージョンは複数GB級の実機・長期間・数千ピアでの完全検証をしていません。
- Python/OS/外部ライブラリが予備容量を使うこともあります。ガードは「一切のENOSPCを数学的に不可能にする」ものではなく、把握できるKagamiの書き込みを事前に拒否する防御です。

## テスト

`tests/test_disk_safety_v86.py` に、署名済みイベントの非部分追記、送信用シャードGCと復元、独立したP2Pストア経由の受信、孤立moments掃除、SQLite管理のWAL切り詰め、容量内訳、テレメトリー上限、オフライン掃除のロックを含めました。

ローカル環境では108件の関連回帰テストを通過。Hugging Faceへの接続を要する一部の既存テストはネットワーク未接続のため未完了です。macOS/APFS実機の `clonefile`、大規模モデル、複数端末を使った長期運転は未検証です。
