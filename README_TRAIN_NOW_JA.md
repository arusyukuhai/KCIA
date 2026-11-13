# KCIA v130 — Mac ↔ Colab で最初の実PTを開始

この版は v129 を基準とし、**学習アルゴリズムと署名プロトコルは不変更**です。CloudflareのBrowser Integrity CheckがPython標準User-Agentを遮断する構成に備え、実通信経路（ルーム発見/登録、WebRTC、GUI ping、公開一覧）で `KCIA-WAN-Client/1.0` を送ります。事前診断もUA対応済みです。

**重要なv129手順訂正**: `train-live --room-id` では `--signing-key` を付けないこと。自動ルームモードが `<journal>.identity.pem` に鍵を生成します。`--auto-tune` が明示的 `--device cpu` をMPSへ置換し得るため、初回Mac用のCPU試験では `--no-auto-tune` を指定します。`scripts/start_wan_pt.py` が自動的に行います。

## 前提

- VPSの `signal.kcia.tech` WSS診断が成功し、TURN資格情報が配布されている。
- MacとColabは**同じv130のソース**、Python 3.11〜3.13と `requirements-patches.txt` を使う。
- ColabはGPUランタイムを選ぶ。Macは初回CPUで、MPSは独立した新しい試験ルームで後から試す。
- MacとColabの通信を直接実証する署名付きWebRTCテストは、可能なら学習前に `scripts/check_wan_p2p.py` で実施。

## 1. Mac側

```bash
python -m pip install -U -r requirements-patches.txt
python scripts/wan_preflight.py --role mac --ws wss://signal.kcia.tech/ws --turn-host turn.kcia.tech --network
python scripts/start_wan_pt.py prepare
```

`prepare` が出力した `room_id` と `revision` （40桁のSHA）だけをColabにも共有。これらを以下コマンドの値へ入れる。`room_id`は参加資格に相当するため、不特定多数に公開しない。

```bash
python scripts/start_wan_pt.py run --role mac \
  --room-id 'PASTE_ROOM_ID' --revision 'PASTE_40_CHAR_SHA' \
  2>&1 | tee wan_mac.log
```

※コマンドは記載のコードから直接実行。`--rounds 0` が既定で、手動でCtrl-Cするまで継続。ログの保存は端末側に任せる。初回のモデルダウンロード・初期状態作成には空きディスクとネット接続が必要。

## 2. Colab側

付属 `notebooks/KCIA_Cloudflare_Mac_Colab.ipynb` を開いてGPUランタイムを選び、**v130 ZIPをアップロード**。パッケージと学習依存関係のインストール後、同じ `room_id` と `revision` を学習セルに入力。Macとは**別のデータ/ジャーナル/署名鍵**を自動生成する。

既にP2Pハンドシェイクが成功しているなら該当セルを省略し、最後のPTセルへ進めてよい。Colabを中断した場合はセッション中の `wan_state/colab` が消える可能性がある。復旧試験をする場合は安全な別保存領域に保護してから再起動する。

## 3. 成功判定

- 両端で `bootstrap: ... semantic_sha256` の値が一致。
- 各端末の物理LoHA rankがローカル予算に合わせて選択された（CPUは上限8、CUDAは上限32）。
- ルーム参加者が2人になり、WebRTC/署名付き更新の送受信ログが両端で観測される。
- 学習lossの数値が出力され、ジャーナルが蓄積される。短い合成サンプルではlossの継続低下を保証できない。
- ネットワーク・モデル取得・署名・genesisが一致しない場合は、既存ジャーナルを削除せずログを確認する。

## 4. 既知の制約

- 同梱のMac/Colabのローカルテキストは動作確認用の極小サンプルで、モデル性能の改善を評価するデータではない。
- WebRTCハンドシェイクと署名付きGossipの実WAN通信はこの配布環境では未検証。VPSへ直接SSHする機能はない。
- Cloudflare側がKCIAクライアントUAまでブロックする場合は、APIホスト向けのBrowser Integrity Check除外を設定。ただしレート制限やDDoS対策は残す。
- 既存のroom/journal/optimizerを操作・消去しない。新ルーム・新ジャーナルで初回テスト。
