# KCIA v112.2 — Cloudflare Tunnel / VPS / Mac + Colab 検証

## アーキテクチャ（Caddy / nginx 不要）

```text
Mac (PyTorch + aiortc) ── WSS ──┐
                                ├── Cloudflare edge ─ cloudflared (VPS) ── Mummy :8080
Colab (PyTorch + aiortc) ─ WSS ──┘
      ╲                                ※ signal は Docker 内からしか到達不可
       ╲─ ICE / DTLS / SCTP ─────────── Mac <> Colab (直接P2P)
                    ╲
                     └── NAT越え失敗時：VPS coturn :3478 (UDP/TCP), relay UDP 49160-49200
```

**重要**: 普通の Cloudflare DNS/CDN/Tunnel は汎用 UDP TURN relay をプロキシしない。TURN サーバー用の DNS は **DNS only (灰色の雲)** にする。Cloudflare Realtime TURN / Spectrum は別サービスであり、この構成では使っていない。シグナリングだけ Cloudflare Tunnel (オレンジの雲)。TURN 用の VPS IP は DNS から見える。Cloudflare Tunnel が代わりに TURN 通信を隠すことはできない。

**VPS で LLM を学習しない**。VPS は Nim Mummy のルーム管理・SDP/ICE シグナリング・coturn のみ。更新・重み・署名イベントは DataChannel 側で送る。全員がトークンなしでシグナリングとモデル公開登録を利用できる。個人端末の秘密署名鍵はサーバーへ送らない。

## 1. Cloudflare で公開ホスト名を作る

* Cloudflare で管理するドメインを使用。
* `Networking > Tunnels` から **remotely-managed** Tunnel を新規作成。名前は `kcia-prototype` など。**Tunnel token (`eyJ...`)** を控え、第三者に公開しない。
* `Published application` のホスト名: `signal.example.com`、**Service URL: `http://signal:8080`**。`cloudflared` と `signal` は同じ Docker Compose bridge にいるため、 `localhost` ではなく service 名 `signal` を使う。HTTP WebSockets を許可する。Cloudflare の challenge / Access login を `/ws` に適用しないこと（Python aiortc クライアントにはログイン画面を処理できない）。Cloudflare側で利用状況・濫用対策を設定する。非公開roomのIDはランダムに生成し、参加リンクを知る人だけに共有する。
* Cloudflare DNS に別の A レコード `turn.example.com` → VPS のグローバル IPv4。**Proxy status: DNS only** を必ず選ぶ。
* `signal.example.com` と `turn.example.com` は必ず別のホスト名にする。

## 2. VPS (Ubuntu/Debian Linux + Docker Compose v2)

`Docker Engine` および `docker compose` plugin が必要。Docker 起動時ユーザーは Docker を実行できる必要がある。少なくとも 1vCPU / 1GiB RAM 程度の開発用 VPS を目安とする（保証値ではない）。

ソースを VPS に転送し、そのソースのルートで実行:

```bash
python3 deploy/vps/setup_env.py \
  --signal-host signal.example.com \
  --turn-host turn.example.com \
  --public-ip 198.51.100.1  # 実際のパブリック IPv4 に置換。予約文書アドレスは使用不可
```

`deploy/vps/.env` がパーミッション `0600` で生成される。`CF_TUNNEL_TOKEN=PASTE_...` を Cloudflare Tunnel の実 token に置換する。TURN用の内部鍵 `TURN_SECRET` は自動生成される。この `.env` は**絶対に公開 ZIP/リポジトリに入れない**。Mac/Colab への秘密ファイルの転送は不要。

VPS/クラウドの両方のファイアウォールで以下だけをインバウンド許可する（SSH は別途管理者のアクセス元のみ許可）:

```text
3478/udp        coturn STUN + TURN
3478/tcp        coturn TURN TCP control (Colab 等で UDP が塞がれている場合の接続候補)
49160-49200/udp  coturn TURN media relay
```

`8080/tcp`, `80/tcp`, `443/tcp` を開ける必要はない。`cloudflared` には Cloudflare 側へのアウトバウンド TCP/UDP `7844` または通信環境に応じた通信許可が必要。

```bash
./deploy/vps/start.sh
# 確認
# (秘密がコマンド出力に含まれ得るため compose config の全出力を公開しない)
docker compose --env-file deploy/vps/.env -f deploy/vps/compose.yml ps
docker compose --env-file deploy/vps/.env -f deploy/vps/compose.yml logs --tail=50 cloudflared signal coturn
```

Cloudflare Tunnel が Healthy になり、`https://signal.example.com/healthz` に `{"ok":true}` が返ることを確認（`/healthz` は公開する）。ルーム一覧 `/api/rooms` は原則未公開ルームを列挙しないが、公開を選んだルームは列挙される。

## 3. Mac (Python 3.11–3.13; Python 環境とディスク空き容量に注意)

ソースのルート:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r deploy/vps/requirements-transport.txt

# 通信サーバーの単独検査（モデルをロードしない）:
python scripts/check_signaling_vps.py \
  --ws wss://signal.example.com/ws

# Mac のピア署名鍵（秘密鍵は Mac のみ、0600。消さない）:
python scripts/check_wan_p2p.py identity --key ./private/mac.pem
```

出力 `Public key (share this): ...` だけを Colab 側へ渡す。`private/mac.pem` は絶対に渡さない。

## 4. Colab

この ZIP に含む `notebooks/KCIA_Cloudflare_Mac_Colab.ipynb` を Colab で開く。最初のセルで KCIA の ZIP をアップロードして依存関係をインストールし、Colab のピア署名鍵を作って **Colab の公開鍵のみ** Mac 側へ渡す。Colab セッション再作成時には鍵が変わるので、リモート公開鍵を更新する。

両者で **同じランダムな room ID** を使う。例えば Mac で:

```bash
python -c 'from patchlib.roomid import create_room_id; print(create_room_id())'
```

Mac から先に `connect` コマンドを実行し、120 秒以内に Colab の最後のセルを実行する（逆でもよい）:

```bash
python scripts/check_wan_p2p.py connect \
  --key ./private/mac.pem \
  --ws wss://signal.example.com/ws \
  \
  --room-id '共有したランダムroom-id' \
  --remote-public-key 'Colabに表示された公開鍵' \
  --timeout 180
```

両側に `PASS: cross-host ICE/DTLS/SCTP and signed Ed25519 peer handshake` が表示されれば接続が成立している。`--turn-only` を両側に追加すると **TURN relay 候補だけ**をSDPに載せた検証ができる。パケットが Cloudflare Tunnel を経由して転送されたことはこの試験では意味せず、TURN 経由の接続候補だけが残る。

## GUIでの実通信（Mac）

`python -m pip install -r requirements-gui.txt -r requirements-patches.txt` の後、`python kcia.py` で起動する。

1. `NETWORK` → 接続モード `INTERNET · Cloudflare / VPS` を選択する。
2. シグナリングURLを `wss://signal.example.com/ws` にし、`設定保存・接続テスト` を押す。HTTPディレクトリに加えて **実際の WSS ping/pong** を試験する。
3. `NEW MODEL` に移動しモデル・データセットを決めて `モデルを作成` を押す。ルームIDは非表示で自動生成され、公開/非公開に合わせてリモート登録される。接続できなければローカル設定は保持される。
4. `MODELS` で選択→ `TRAINING` → `開始`。既存の `train-live` が指定のWSSに接続し、署名付きWebRTC Gossipを開始する。
5. ほかのGUIは公開モデル一覧から選ぶ。非公開モデルは `参加用リンク` をコピーして、相手のGUIの `参加リンクを読み込む` で登録する。IDを手入力する必要はない。

接続テストは **WSS接続のみ**で、他ピアとの ICE / TURN 接続・実学習を保証しない。学習稼働中は接続先変更をブロックする。ColabはGUIでなくNotebook / CLIで同じルームに接続する。

## 5. フェーズを分ける

1. `check_signaling_vps.py`: HTTP + 公開 WSS + SDP 中継、**WebRTC 成否は検証しない**。
2. `check_wan_p2p.py connect`: 別ホスト 2 台の ICE / DTLS / SCTP + Ed25519 署名ハンドシェイク、**LLM 学習はしない**。
3. (次段階) `python -m patchlib.cli train-live --help`、GUI、同一 room manifest / モデル / tokenizer / Genesis を揃えて実分散学習。v112.2 の CUDA と MPS 同期・再開実績はこの ZIP だけで保証されないので、小さなモデル・小ステップで進める。

## 既知の制約・セキュリティ

* 本構成で実機接続成功を確認したことにはならない。SSH/VPS、Cloudflare アカウント、ドメイン、実機ネットワークはこの配布物には含まれない。Docker build や実 Colab/Mac の疎通はユーザー環境で実施が必要。
* Cloudflare の WAF / CAPTCHA / Access 認証 / WebSocket timeout により WSS が妨げられる場合がある。クライアントは切断時に再接続するが、大規模・長時間の無停止稼働保証はしない。
* TURN UDP 3478, relay UDP 49160-49200 の到達性が Colab 側 egress に依存する。TCP 3478 の候補は用意するが、UDP メディアが厳しいネットワークでの成功は保証しない。ブラウザ対応の TURN/TLS 443 はこの構成に含まれない。
* 公開サーバーのWSSは認証不要。CloudflareのWAFレート制限、VPSのFD/帯域監視、TURNのクォータ・短寿命credentials、メトリクス監視を必須とする。無制限の公開 TURN リレーや本番品質を保証する設定ではない。
* TURN を 1 VPS・41 relay ports で運用する試験用構成なので 4096 peers 規模向けではない。帯域使用・費用監視・送受信上限・失敗時挙動を段階的に確認する。
* 秘密鍵 `private/*.pem`、自動保存 journals/weights を ZIP やクラウドに含めない。

Cloudflare 文書: https://developers.cloudflare.com/tunnel/get-started/ 、https://developers.cloudflare.com/cloudflare-one/faq/cloudflare-tunnels-faq/ 、https://developers.cloudflare.com/dns/proxy-status/limitations/

## 6. 署名済みGossipの実学習パイロット（通信試験 PASS 後に実行）

MacとColabの**両側**にフルZIPを展開し、`requirements-patches.txt` も入れる（Colabでは必要なら `%pip install -r requirements-patches.txt`）。GPUやMPSをいきなり指定せず Mac では CPU から始める。以下は **PT-only の小規模サンプル**。

まず **同じ** `HuggingFaceTB/SmolLM2-135M` の **固定 commit SHA** を得る。どちらか一方で:

```bash
python -c 'from huggingface_hub import model_info; print(model_info("HuggingFaceTB/SmolLM2-135M").sha)'
```

両側で `HF_SHA`, `TRAIN_ROOM` を揃える。ただし **秘密鍵・journal・state-base の保存先は別々**とする。学習用ルームは通信診断のルームと別のランダムIDを使う。

```bash
export HF_SHA='ここに上で得た40桁のcommit SHA'
export TRAIN_ROOM="$(python -c 'import secrets;print(secrets.token_urlsafe(24))')"
# 共有するのは TRAIN_ROOM の文字列のみ。Colabは Mac の同じ値を設定する。
mkdir -p ./wan_state/mac
python -m patchlib.cli train-live \
  --ws wss://signal.example.com/ws --room-id "$TRAIN_ROOM" \
  --model HuggingFaceTB/SmolLM2-135M --revision "$HF_SHA" \
  --auto-state --state-base ./wan_state/mac/genesis.fstate \
  --text-file examples/wan_sample.txt \
  --signing-key ./private/mac.pem \
  --kind loha --rank 16 --reinit-every 20 \
  --device cpu --batch-size 1 --max-length 64 --local-steps 4 \
  --journal ./wan_state/mac/events.sqlite3 --rounds 0
```

Colabでは **別の** `--state-base ./wan_state/colab/genesis.fstate`, `--journal ./wan_state/colab/events.sqlite3`, `--signing-key ./private/colab.pem` を使い、`--device cuda` (GPUランタイム使用時) にする。それ以外の共有設定は同一。`TRAIN_ROOM` も同一値を指定。Colabノートブック最初のセルで展開した場合、作業ディレクトリは `/content/kcia`。学習用のクライアント認証トークンは不要。

既存ジャーナルやルームを消さず、各側で `bootstrap: saved ... semantic_sha256=...` とログに出る *semantic SHA* が一致することを確認。相互認証・オンラインピア数2・送信／受信イベント・lossの進行を見て、小数回の更新のあと終了する。`--rounds 0` は継続動作なので実験後は **Ctrl+C** で終了すること。CPU/MPS/CUDA別の数値差や、resume後の動作は別途検証する。
