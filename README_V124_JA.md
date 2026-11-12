# KCIA v124 — 完全リブランド / Myriad Pro Condensed / SVG ナビゲーション

- プロジェクト表示名、起動スクリプト、CLI、GUI、WebSocket/招待スキームは KCIA に統一。
- 旧 `kagami://` 招待リンクは受け付けません。新形式は `kcia://join?...` です。
- 起動: `python kcia.py`、復旧起動: `python kcia.py --gui-safe`、テスト用初期化: `python kcia.py reset-local-data`。
- 欧文書体: Adobe Fonts `https://use.typekit.net/wzz8rse.css` の `myriad-pro-condensed` を優先。日本語は Noto Sans JP / macOS標準フォントへフォールバック。
- サイドバー6アイコンを文字記号からinline SVGへ変更。
- アプリ設定キー、環境変数、既定データ保存先とDockerボリューム名もKCIAへ変更しています。**旧Kagami版とはデータ保存先、環境変数、招待リンク、プロトコル識別子の互換性を保証しません**。
- 旧ローカルデータは自動移行・削除しません。必要なら別途バックアップした上で手動移行してください。公開VPS/各クライアントのバージョンは同時に揃えてください。
- Adobe Fontsのキットに対象ファミリーが含まれない、またはインターネットに接続できない場合、ローカル代替書体になります。
