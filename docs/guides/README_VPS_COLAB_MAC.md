# VPS / Mac / Colab 実機接続ガイド（更新版）

Cloudflare Tunnel に変更しました。Caddy を使う旧案は廃止しました。

**現行手順は [Cloudflare Tunnel 版](../../deploy/vps/README_CLOUDFLARE_JA.md) を使用してください。**

学習CLIの引数は `python -m patchlib.cli train-live --help` で実機のバージョンと突き合わせてください。ネットワーク疎通に合格した後、小さなモデル・共有したモデルrevision・同一room manifest・各ホスト固有のidentity/journalを用意してPT試験を開始します。
