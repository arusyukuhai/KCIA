# Kagami v101 — PPO / HF Reward Model の物理ディスク容量エラー対策

## 発生したエラー

`reward_model_v67.py -> load_hf_model -> transformers.from_pretrained -> huggingface_hub.snapshot_download -> hf_xet.xet_get` で `File reconstruction error: IO Error: No space left on device (os error 28)`。

これは Kagami の署名付きジャーナル上限とは別の **macOS の物理ストレージ不足**。従来は `RewardModelError: incompatible scalar head` のように見える汎用エラーで包んでしまっていた。

## v101 の対策

1. PPO が HF Reward Model / HF Value Critic を使用する場合、学習CLIを開始する直前に `snapshot_download(dry_run=True)` で **実際にダウンロードが必要なファイルサイズを事前見積り**。ここでファイルのダウンロードは行わない。
2. HF Hub・Xet・OS一時領域の各ボリュームを確認。保守的な `2 × 未キャッシュ容量 + 1GiB` の必要容量を下回る場合は、モデルやジャーナルを変更する前に `HFCacheSpaceError` で停止。
3. Hugging Face のバージョンやネットワーク状態によってdry-runができない場合は、最低限の1GiBリザーブ検査だけを行い、その旨を明示する。**この場合は完全に容量不足を防げない**。
4. ダウンロード中にOSレベルのENOSPC/EDQUOTが起きても、Reward/Criticの非対応モデルと誤表示しない。
5. PPOのロールアウト生成時にdeviceが文字列`'cpu'`の場合の`.type`エラーを修正。
6. 既存のPPO学習則、署名付き履歴、LoHA、fast-resume、キャッシュ保存形式は変更しない。

**重要:** 空き容量が実際にない場合、ソフトウェア側の変更だけでは巨大な外部モデルをインストールできません。今回のパッチは「後から壊れる」ことを防止し、原因を明確にする修正です。既存のHFキャッシュを勝手に消さず、モデルを自動で別の報酬方式へ切り替えません。

## 現在のMacを復旧するには

すべてのKagami学習プロセスを終了してから：

```bash
# ボリュームの物理空き容量
df -h ~
# Hugging Face の2種類の主要キャッシュ
DU_HF="$HOME/.cache/huggingface"
du -sh "$DU_HF/hub" "$DU_HF/xet" 2>/dev/null
# HF の現在のキャッシュを確認
hf cache list --sort size
# 中断された不完全ダウンロード・使われなくなったリビジョンの削除予定を確認
hf cache prune --dry-run
```

対象を確認してから、安全な不要キャッシュを削除する場合：

```bash
hf cache prune
```

特定モデルの旧キャッシュは `hf cache rm <repo_id> --dry-run` で必ず対象を確認してから手動で実行してください。`rm -rf ~/.cache/huggingface` のような全消去は推奨しません。使用中の学習や他のツールがそのキャッシュを必要とする場合があります。

### 外付けSSDを使う場合

容量の十分あるボリュームを指定して **Kagami起動前に** 設定します。

```bash
mkdir -p '/Volumes/ExternalSSD/huggingface'
HF_HOME='/Volumes/ExternalSSD/huggingface' python kagami.py
```

注意：環境変数の有効範囲はこの起動プロセスと子プロセスです。既存のキャッシュは移動されず、新しい保存先で再ダウンロードが必要になる場合があります。容量と空き容量を先に確認してください。GUIをFinderから起動する場合はこのShell環境変数を引き継げない場合があります。

## 検証

- 追加7件: Xetの入れ子ENOSPC、容量見積、キャッシュ済みファイル、メタ情報欠損、PPOの選択対象、Reward/Criticの例外型を検証。
- 関連回帰計95件成功、1件スキップ（macOS MPS実機依存）。
- 実際のHugging Faceからの巨大モデルダウンロードとmacOS実機での長時間の容量計測は未検証。
- dry-runのファイルサイズは精確なピーク一時容量を保証しない。2倍+1GiBは保守的な見積に過ぎない。
