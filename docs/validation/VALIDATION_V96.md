# KCIA v96 検証

- CPU/Linuxで関連テスト **113件成功**。
- 実署名付き旧世代carryを用い、統合済みweightから現在journal/carry両方のsuffix=0で再起動することを確認。
- carryのみ増えた場合はcarryの署名済みsuffixのみ再生。
- 両journalが同時に伸びた場合、checkpointロード開始前に安全な全履歴fallbackを行うことを確認。
- carry/journalのUPDATE/DELETEを検知した場合はcacheを採用しない。
- 確定済みcarry handoffのSQLite witnessによる再ハッシュ省略、および変更後の失効を確認。
- 旧世代のfrozen source cut witnessの保存/再使用と、書き込み発生後の失効を確認。
- Python compileall（patchlib）成功。
- 未検証：macOS/MPS長時間実機、数GiB/数千ピアの本番実測、v61 paged COWによるfast-resume。
