# KCIA v129 / VPS + Cloudflare + Mac + Colab 実機パイロット

この版は **v128 の学習形式と署名形式を変更しない** デプロイ・検証パッケージです。Nim シグナリング用イメージの **KCIA 保存ディレクトリ/実行ファイル不一致を修正**し、ローカル／外部疎通用 preflight を追加しています。

> **未実施**: VPSへのSSH接続、Cloudflareアカウント設定、Docker実ビルド、MacのMPS/pywebview、ColabのGPU、実WAN通信。コード同梱とオフラインテストだけで疎通成功とはみなしません。

## 構成

```
Mac (KCIA CLI / GUI, MPS or CPU) ── WSS ─┐
                                         ├─ Cloudflare Tunnel ─ cloudflared ─ signal:8080 (Nim/Mummy)
Colab (KCIA CLI, CUDA) ──────────── WSS ─┘
         └─ ICE / DTLS / SCTP ──── peer / TURN VPS:3478 UDP/TCP + UDP:49160-49200
```

**VPSは機械学習せず、ルーム発見・SDP交換・TURNリレーだけを担当**。WebRTCの学習差分は原則ピア間です。TURNには短期資格情報を発行するためのサーバー内部 `TURN_SECRET` が必要ですが、参加者の `APP_TOKEN` は不要です。

## Phase 0: VPS・Cloudflare の設定

必要なもの: Docker Engine + `docker compose` v2 が使えるLinux VPS、Cloudflare管理ドメイン、VPSグローバルIPv4、SSH。最初は帯域費用を監視しながら **2ピア限定のパイロットとして運用**してください。サービスは公開認証なしのため、上流のWAF・レート制限、運用者の監視は不可欠です。

Cloudflare Dashboard `Networking → Tunnels` で remotely-managed Tunnel を作成し、**公開ホスト名** `signal.example.com` → **サービス** `http://signal:8080` に設定。アクセス認証・ブラウザチャレンジを `/ws` に適用しないでください（CLIから使えなくなります）。Cloudflare DNSで `turn.example.com` のAレコードをVPSのIPv4に向け **DNS only（灰色の雲）** とします。`signal` と `turn` は別ホスト名にしてください。

VPSで（**ここでは仮のドメインとIPを必ず実値に置き換える**）:

```bash
python3 deploy/vps/setup_env.py \
  --signal-host signal.example.com \
  --turn-host turn.example.com \
  --public-ip YOUR_PUBLIC_IPV4
# deploy/vps/.env に Cloudflare Dashboard の Tunnel token を自分で設定
chmod 600 deploy/vps/.env
./deploy/vps/start.sh

docker compose --env-file deploy/vps/.env -f deploy/vps/compose.yml ps
# ログを他人へ送る場合、tokenが含まれていないか確認
curl -fsS https://signal.example.com/healthz
```

VPSファイアウォール／クラウドファイアウォールは、**3478/udp, 3478/tcp, 49160-49200/udp** を許可（SSHは管理者のIPだけ）。8080はDocker内部公開のみ。Cloudflare Tunnelからはアウトバウンド7844（TCP/UDP）が使える必要があります。TURNのDNS-only AレコードはIPを公開します。

**失敗したら**: `docker compose ... logs --tail=80 signal cloudflared coturn`、CloudflareのTunnel Healthy状態、`.env`権限、DNSを確認。最初にNimイメージが正しくビルド・起動することが必須です。

## Phase 1: VPS上のWSSを確認（MacかColab）

MacでZIPを展開し、Python 3.11〜3.13の仮想環境から:

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-patches.txt
python -m pip install -r deploy/vps/requirements-transport.txt

python scripts/wan_preflight.py --role mac --require-deps
python scripts/wan_preflight.py --role mac \
  --ws wss://signal.example.com/ws --turn-host turn.example.com --network
```

`PASS: HTTPS health...`, `PASS: TURN credentials advertised` を確認。これは **WSS接続とSDP中継まで**であり、WebRTCが成立した証拠ではありません。

## Phase 2: Mac ↔ Colab の署名付きWebRTC

Macで公開鍵を表示（秘密鍵は `private/mac.pem` に保存、共有禁止）:

```bash
python scripts/check_wan_p2p.py identity --key private/mac.pem
python -c 'from patchlib.roomid import create_room_id; print(create_room_id())'
```

Colabで `notebooks/KCIA_Cloudflare_Mac_Colab.ipynb` を開き、このZIPをアップロードして、通信依存関係をインストール。**Colabの公開鍵**をMacへ、**Macの公開鍵と生成したroom ID**をColabへ渡す。どちらも秘密鍵は渡さない。

MacとColabの両方で **同じroom IDと互いの公開鍵**を使って実行。Mac側:

```bash
python scripts/check_wan_p2p.py connect \
  --key private/mac.pem --ws wss://signal.example.com/ws \
  --room-id 'COLAB_WITH_MAC_SHARED_ROOM' \
  --remote-public-key 'COLAB_PUBLIC_KEY_BASE64' --timeout 180
```

Colabはノートブックの `connect` セルを実行。**両側に `PASS: cross-host ICE/DTLS/SCTP and signed Ed25519 peer handshake`** が出ることを確認。つながらない場合、双方を `--turn-only` モードにして中継接続のみを試す。TURN専用で失敗するならcoturnのリレーポート、パブリックIP、クライアントのUDP制限を調べる。

## Phase 3: PT-onlyで小規模実学習

まず通信試験とは **別の新しいルームID** を作り、Mac/Colabで共有します。モデルのID・**確定した40桁のrevision SHA**・ルーム最大rank・学習設定は両方完全に合わせる。モデルは `HuggingFaceTB/SmolLM2-135M` を例にします。変更時は両ピアを必ず揃えてください。

```bash
python -c 'from huggingface_hub import model_info; print(model_info("HuggingFaceTB/SmolLM2-135M").sha)'
python -c 'from patchlib.roomid import create_room_id; print(create_room_id())'
```

Mac:

```bash
mkdir -p wan_state/mac
export KCIA_WS='wss://signal.example.com/ws'
export KCIA_TRAIN_ROOM='PASTE_NEW_SHARED_TRAIN_ROOM'
export KCIA_MODEL_SHA='PASTE_SAME_40_CHAR_SHA'
python -m patchlib.cli train-live \
  --ws "$KCIA_WS" --room-id "$KCIA_TRAIN_ROOM" \
  --model HuggingFaceTB/SmolLM2-135M --revision "$KCIA_MODEL_SHA" \
  --auto-state --state-base wan_state/mac/genesis.fstate \
  --text-file examples/wan_sample.txt --signing-key private/mac.pem \
  --kind loha --rank 32 --local-loha-rank auto \
  --reinit-every 20 --local-steps 4 --device cpu \
  --batch-size 1 --max-length 64 --rounds 12 \
  --journal wan_state/mac/events.sqlite3 2>&1 | tee wan_state/mac/run.log
```

Colabはノートブック内のオプション学習セルを使い、**必ず共有rankを32に変更**して実行。 `--local-loha-rank auto` を追加し、`--device cuda` にします。 `wan_state/colab` とColab側の署名鍵を使い、**Macの状態ファイルをコピーしない**こと。rank32はルーム共通の上限で、MacのCPU側は小さい物理LoHAランクを選び、ColabはGPUに合わせて決めます。

`bootstrap: ... semantic_sha256` が一致し、双方が `active` / `SWARM` / 署名付き更新の送受信を記録するか確認。**各ピアで異なるlocal rankを使用していること、合流後も状態・署名検証が一致すること**を観測します。`--rounds 12` は各ピアのローカル送信イベント数上限です。接続のタイミングが遅いと先に終了し得るため、両側でほぼ同時に起動するか、診断用に `--rounds 0` で手動停止してください。

初回はMacで **CPU** を推奨（既知のMPSクラッシュと区別するため）。CPUが成功した後、**新しいルームと別のジャーナル**で `--device mps` の試験を行う。PPO/SFTとGUIからの参加検証は、まずPTで両端の通信と更新の整合性を確認してから実施します。

## Phase 4: GUIの実通信試験（CLIが成功してから）

Macで `python -m pip install -r requirements-gui.txt` 後 `python kcia.py --gui-safe` でまず安定性確認。GUIの `NETWORK` で `INTERNET / Cloudflare` と `wss://signal.example.com/ws` を設定し、接続テスト。GUIから新規ルーム作成・公開/非公開参加の操作を確認する。既存のCLI試験用ルームの状態や秘密鍵を上書きしないこと。CLI成功後にフレームレスモード `python kcia.py` も試す。

### 成功基準

- VPS: `signal / cloudflared / coturn` が起動、`/healthz` が200、ルーム登録が継続保存できる。
- WSS: 2接続のjoin + SDP中継、TURN資格情報が払い出される。
- Mac/Colab: 双方でICE/DTLS/SCTP + Ed25519 handshakeがPASS。
- PT: 両側のモデルrevisionとgenesis SHAが同じ、peer数2、更新が相互到達、異ランクの署名が受理される。
- 再開: 明示的に開始するまで学習しない。ジャーナル・鍵を保持したまま同じルームに再参加できる。
- GUI: 通常モード/復旧モードの操作を比較、押せなくなるならその時のボタンとログを記録。

### 注意

`APP_TOKEN`は不要ですが、**Cloudflare Tunnel token と TURN_SECRET はVPSの秘密**です。サーバーが匿名公開のため、STUN/TURNの無制限運用を推奨しません。公開前にCloudflare側でrate limit/WAFを設定し、Coturnの帯域使用量・割当・通信料金を監視してください。

Colabランタイムは再起動でローカル鍵・状態が消えることがあります。継続試験には安全な鍵／状態のバックアップを各端末で管理してください。Macは旧版から残った学習ワーカーを停止して、VPS・Mac・Colabの**同じv129配布物**を使ってください。
