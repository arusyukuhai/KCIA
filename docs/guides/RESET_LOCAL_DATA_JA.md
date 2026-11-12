# Kagami ローカル保存データの完全初期化（デバッグ用）

**実行前に全てのKagami GUIと学習プロセスを終了してください。** この操作は取り消せません。学習済みweight、LoHA/LoRA差分、署名済みP2Pジャーナル、復元チェックポイント、Optimizer状態、ルーム定義、ピア識別鍵、グラフ、ログ、およびGUIに登録したカスタムコードが失われます。別のピアが保持する情報は削除できません。

プロジェクト直下で、まず削除対象を確認します。

```bash
python kagami.py reset-local-data
```

出力に表示される `Kagami storage root:` が期待する保存先であることを確認したら、実際に削除します。

```bash
python kagami.py reset-local-data --yes
```

Kagamiのデータを独自の場所に置いている場合は、`KAGAMI_DATA_DIR` を設定するか、その保存ルート（直下に `.fedpara-gui` があるディレクトリ）を指定します。

```bash
python kagami.py reset-local-data --root "/path/to/data-root"
python kagami.py reset-local-data --root "/path/to/data-root" --yes
```

GUI保存先の `.fedpara-gui` 以下のみがデフォルトの削除対象です。CLIでデフォルト設定の `train-elastic` を実行した場合、カレントディレクトリに生成した **既定名** の `elastic-events.sqlite3*` 関連ファイルと `live-checkpoints` も追加するには、`--include-cli-defaults` を指定してください。このオプションを使う際は、学習CLIの実行に用いたディレクトリでコマンドを実行してください。

```bash
python kagami.py reset-local-data --include-cli-defaults
python kagami.py reset-local-data --include-cli-defaults --yes
```

**削除しないもの：** Hugging Face共通キャッシュ、プロジェクトのソースコード、`--journal` / `--checkpoint-dir` / `--disk-optimizer-dir` / `--streaming-scratch-dir` 等で**明示指定した別の保存先**、ユーザーのデータセット、他の端末やネットワーク上のデータ。これらは誤削除防止のため自動探索・削除しません。また `~/.config/kagami/data-root.txt` は学習データではなく保存先の案内設定なので残します。

**安全装置：** デフォルトは削除せず一覧を表示するdry-run。`--yes` が必要です。GUI保存先自体がシンボリックリンクなら拒否し、内部のリンク先は辿りません。GUIピアロックが保持されている場合は削除を中断します。ただしCLI起動を完全に検知できるわけではないため、必ず先に学習を停止してください。停止したピアがネットワークに再接続すると、他ピアから学習状態が再同期される場合があります。完全に新しい実験を行う場合は新規ルームIDを使用してください。

標準ライブラリのみで動作するため、PyTorchやGUIライブラリを入れずに `python -m patchlib.reset_local_data --help` として直接実行することもできます。
