# KCIA historical root release notes (through v136)
These are archived engineering notes, not the current launch defaults. For v137 behavior see the top-level README.md and README_V137_AUDIT_JA.md.


## Archived README_V119_JA.md

> この文書はv119時点の記録です。v120の既定値（公開4・再初期化20）とGUI修正は README_V120_JA.md (historical link: README_V120_JA.md) を参照してください。

# KCIA v119 — room learning intent, GUI spacing and refresh policy

- NEW MODEL now contains PT/SFT/PPO objective settings and a public HF-revision-pinned dataset selection. The signed room manifest includes the compact immutable `training` section when enabled. Local file paths never leave the peer.
- Mixed policy is bound at model creation and restored by new participants. A peer without the necessary datasets may disable SFT/PPO and train PT only, retaining the shared signed objective policy. Note: distributed Critic arrangements may not accept a PT-only participant as a Critic worker; no silent policy rewrites occur.
- LoHA/LoRA reinit cadence is separate from local publication cadence, but NEW rooms default to 4 signed publications rather than 20. `optimization.reinit_every` is immutable per new room, preventing peers from disagreeing. Legacy rooms retain their existing settings.
- GUI spacing restored for KCIA logo (aligned with sidebar content), training metrics, chart toolbar and New Model objective controls.

Caveats: First-party Mac WebView and WAN WebRTC end-to-end testing need actual devices. HF streaming datasets require reachability/auth and exact commit SHAs. Failed runtime dataset loading requires PT-only selection; creating a model does not download or verify all HF data.


## Archived README_V120_JA.md

# KCIA v120 — GUI起動・応答性の修正と学習間隔の復元

## 初期値

- 公開間隔: **4ローカル学習ステップ**（基準値。実際の公開頻度はオンラインピア数などから自動調整されます）
- LoHA再初期化間隔: **20回の署名付き公開**。0で無効化。数値は新規作成ルームの初期値です
- 既存のルームの学習ポリシー、署名、ジャーナル、学習済み重みには変更を加えていません。作成済みルームは元の再初期化間隔を引き継ぎます

## GUIの安定性

- 初回描画とクリックをHugging Face / Cloudflare / PyTorch端末検出の完了から分離
- ローカルのモデル一覧を先に描画し、公開モデル一覧の取得失敗で待たない
- GPU / CPU端末能力の検出はNEW MODEL・TRAINING画面で初めて実行
- TRAINING画面のグラフ（最大9枚）を毎回全描画せず、最大4秒に1回程度＋表示変更時に更新。スクロール位置外のグラフは遅延描画
- Python状態確認はロックが長時間占有された場合に直近の値を返し、読み取りが連続してUIを塞がない
- 外部フォントの自動ダウンロードを起動時から削除（インストール済みフォント、OSフォントにフォールバック）
- ブリッジの一時的な起動失敗には自動再試行。表示中の再接続ボタンも利用可能

## 起動方法

```bash
python -m pip install -U -r requirements-gui.txt
python kcia.py
```

どうしてもウィンドウをクリックできない場合、アプリを終了して以下を実行してください。

```bash
python kcia.py --gui-safe
```

`--gui-safe`は、OS標準タイトルバーを使ってフレームレスのネイティブドラッグ処理を迂回する緊急復旧モードです。データ保存先・学習アルゴリズム・通信プロトコルは通常モードと共通です。**クリック不能の根本原因がmacOSのWebKit/Cocoa側にある場合の切り分け・救済手段**でもあります。フレームレスGUIは通常モードのままです。

詳細な症状を残すには、終了・再起動時に以下を指定してください。

```bash
KCIA_GUI_DEBUG=1 PYWEBVIEW_LOG=debug python kcia.py
```

## テストの境界

- Chromium/Playwrightによるクリック操作、ハングした通信・端末検出の再現、1,600点のチャート再描画制限とPythonユニットテストを実施
- 実際のmacOS WKWebViewでのフリーズ再現と原因確定は、このLinux環境では実施できません。macOSでの改善は現時点では未検証です
- ネットワーク・ルーム保存・分散LoHAはv119の実装を維持

## アップデートの注意

必ず実行中のKCIAを終了し、旧ZIP内のコードと保存済みデータをバックアップしてからソースを更新してください。`.kcia-gui`、鍵、署名付きジャーナル、重みファイルは消さず、同じ永続データ保存先を使ってください。


## Archived README_V121_JA.md

# KCIA v121 — フォント修正、フルローカルリセット、無断学習の防止

## フォント

v120は起動時の外部フォント通信を止めましたが、macOSに必ず備わる日本語フォントへのフォールバックが不十分でした。v121は日本語UIにNoto Sans JP → Hiragino Kaku Gothic ProN / Hiragino Sans → Yu Gothic / Meiryo → OS system-uiの順でフォールバックし、数字主体の表示はMyriad Pro Condensedを優先します。アプリ起動時にフォントのネットワーク通信は行いません。特定フォントのインストールは必須ではありません。

## GUIから全ローカル学習データを消去

サイドバー最下部「全ローカルデータ消去 / RESET」から、**削除対象をプレビュー → DELETE KCIA と入力 → アプリ終了後の削除**を行います。ボタンを押しただけでは消えません。実行中の学習・ベンチマーク・EVALがある場合は拒否し、他のGUIが同一保存先で開いている場合もロックで中断します。消去に失敗した場合はアプリ起動元ターミナルにエラーが表示されます。

対象はGUIで管理する `.kcia-gui` 以下（全ルーム、学習済み重み、LoHA/LoRA、optimizer、秘密鍵、署名済み更新、ジャーナル、グラフ、ログ、GUI設定）。オプションをチェックすると**この配布物のソースルート直下**にある既定名 `elastic-events.sqlite3*` および `live-checkpoints` も対象に追加します。削除対象はプレビューに個別表示されます。

意図しないデータ損失を避けるため、共有Hugging Faceキャッシュ・モデル本体のダウンロード、独自指定した外部CLIジャーナル、任意のローカルデータセット、VPSや他のピアのデータは削除しません。公開ルームを消したい場合はVPS側に別管理が必要です。新たに同じネットワークへ再参加すると外部ピアから共有状態が届く場合があります。完全に新しい実験には新規ルームを作成してください。

### CLIからの一括削除（GUIが開けない場合）

すべてのGUIと学習ワーカーを停止してから以下を実行します。最初は**dry-run**です。

```bash
python kcia.py reset-local-data
python kcia.py reset-local-data --yes
```

GUI以外の既定名CLIジャーナルも含める場合は `--include-cli-defaults` を両方に追加します。プロジェクト外の保存先には `--root` を指定できます。

## GUI再起動後、明示操作なしに学習しない

GUIの起動処理は、最後に開いていたルームを「選択」するだけで、学習は開始しません。v121からGUIで起動した `train-live` ワーカーはGUIの親PIDを監視し、GUIが異常終了したときもSIGTERMによる既存の学習停止処理を要求します。正常終了時も既存のチェックポイント保存を行います。**単独で起動したCLI学習**には影響を与えません。

**注意：** この変更はv121から起動したワーカーだけに適用されます。v120以前に立ち上げた残存ワーカーを自動的に停止することはできません。手動で実行中の `patchlib.cli train-live` を確認し、必要ならSIGTERMで正常終了させてください。PIDを確認せず一括killしないでください。モデルを書き込み中のワーカーは正常終了に時間がかかることがあります。

## 確認

Pythonのローカルリセット・プロセス監視・GUI操作テストとChromiumでのダイアログ操作（入力前の実行不可、確認後のクローズ指示、ページロード時の学習起動ゼロ）を実施しました。macOS WKWebViewでの起動、実機クラッシュ後の大型モデル保存、VPS/Colab通信はここでは未検証です。


## Archived README_V122_JA.md

# KCIA v122 — 学習設定モーダル・旧フォント復帰・異ランクLoHA送信修正

## 不具合の原因

1. **`missing heterogeneous rank protocol`**：v116で `UniversalAdapter.export()` にだけ異ランク宣言を付けた一方、普段の継続学習に使われる `ContinualElasticEngine._make_wire()` は `adapter._metadata()` を直接呼んでいました。低ランクの送信イベントから `rank_protocol` が欠け、受信・自己検証で拒否されていました。
2. **更新量のランク依存バグ**：`_factor_tile()` と `FrozenDelta` が相手の実ランクではなくローカルのrankでLoHAの尺度を計算する経路があり、異なるrankの参加者間で重みがずれる可能性がありました。
3. **学習設定フォームが画面内へはみ出す**：v119は既存の大きな `mixedPanel` を `NEW MODEL` 画面の末尾にappendしていました。固定高のウィンドウと画面スクロール抑止によりフォーム下部に届かなくなっていました。
4. **フォントが変わった**：v120/v121でフォント取得処理と優先順位を変更し、v118までのMyriad Condensed優先が失われました。

## 修正点

- `adapter._metadata()` が低ランクLoHAの `rank_protocol="heterogeneous-loha-v1"` を生成するため、`export()` 以外の署名経路でもプロトコル情報が付きます。検証強度は緩めていません。古い不正な署名イベントを無断で改変・再署名しません。
- LoHAの因子再構成経路で、実際に検証した送信因子のrank幅を使います。
- `NEW MODEL`の「学習方式・SFT/PPOデータセットを設定」ボタンから**独立したモーダル**を開きます。設定欄だけが縦スクロールし、上部タイトルと下部の「設定を反映」「閉じる」は常時表示します。背景クリック・Escで閉じることができます。Hubピッカーの入れ子にも対応します。
- CSSのMyriad Pro Condensed → Noto Sans JPの優先順位をv118相当に復元しました。外部フォントを利用できる環境では、ページ読み込み後に非同期でスタイルシートを要求します。ネットワークが使えなければmacOSのローカルMyriad/Noto/Hiraginoなどで代替表示します。**フォントそのものは本ZIPに同梱していません**。
- 新規ルームの送信間隔4ステップ・LoHA再初期化間隔20送信、その他の学習・署名ジャーナル形式は維持しています。

## テスト

- Chromium（1400×880、1060×650、800×510）でモーダルの開閉・スクロール・Esc・操作ボタンの固定位置を確認。
- rank 2の署名付き継続FedSum更新をrank 8ピアが受信し、rank 4ピアがジャーナル再生できることを実テスト。rankの異なる因子タイルと全量復元が一致することを確認。
- GUI操作・署名付き異ランク・ReLoRA・Factor Ledger等の関連54件のテスト成功。macOS WKWebViewとVPS/Colabの実ネットワークは未検証。

## 更新

KCIAの学習を停止してから、**コードのみ**v122へ更新してください。既存の署名鍵、重み、Optimizer、ジャーナルを消さないでください。

```sh
python -m pip install -r requirements-gui.txt -r requirements-patches.txt
python kcia.py
```

GUIがネイティブ側で開かない場合、`python kcia.py --gui-safe` でOSタイトルバーを用いた復旧起動ができます。

古いクライアントが自動生成してしまった、`rank_protocol` 欠損の低ランク署名イベントについては**検証を迂回せず拒否**します。解決しない場合はログとイベントIDを保存し、後から安全にジャーナル整合性を調査できるようにしてください。ライブ通信の相手もv122へ更新する必要があります。


## Archived README_V123_JA.md

# KCIA v123 — 欧文フォント指定の変更

- Adobe Fonts stylesheet: `https://use.typekit.net/wzz8rse.css`
- 欧文優先フォント: `"myriad-pro-condensed"`
- 日本語フォールバック: `"Noto Sans JP"`、`"Hiragino Kaku Gothic ProN"`、`"Hiragino Sans"`、`"Yu Gothic"`、`"Meiryo"`
- GUIのロードはAdobe Fontsの応答を待ちません。通信できない場合はインストール済みフォントにフォールバックします。
- Content Security Policyには `use.typekit.net` をスタイルとフォントの許可元として追加し、旧cdnfonts参照を削除しました。
- 学習アルゴリズム、ルーム、署名、通信、保存形式は変更していません。


## Archived README_V124_JA.md

# KCIA v124 — 完全リブランド / Myriad Pro Condensed / SVG ナビゲーション

- プロジェクト表示名、起動スクリプト、CLI、GUI、WebSocket/招待スキームは KCIA に統一。
- 旧 `kagami://` 招待リンクは受け付けません。新形式は `kcia://join?...` です。
- 起動: `python kcia.py`、復旧起動: `python kcia.py --gui-safe`、テスト用初期化: `python kcia.py reset-local-data`。
- 欧文書体: Adobe Fonts `https://use.typekit.net/wzz8rse.css` の `myriad-pro-condensed` を優先。日本語は Noto Sans JP / macOS標準フォントへフォールバック。
- サイドバー6アイコンを文字記号からinline SVGへ変更。
- アプリ設定キー、環境変数、既定データ保存先とDockerボリューム名もKCIAへ変更しています。**旧Kagami版とはデータ保存先、環境変数、招待リンク、プロトコル識別子の互換性を保証しません**。
- 旧ローカルデータは自動移行・削除しません。必要なら別途バックアップした上で手動移行してください。公開VPS/各クライアントのバージョンは同時に揃えてください。
- Adobe Fontsのキットに対象ファミリーが含まれない、またはインターネットに接続できない場合、ローカル代替書体になります。


## Archived README_V125_JA.md

# KCIA v125 — Adobe Fonts Condensed を確実に診断できるよう修正

- **欧文**: 指定のAdobe Fonts Web Project `https://use.typekit.net/wzz8rse.css` から `myriad-pro-condensed` を使用。CSSでの指定に加えて、`document.fonts.load()` が**実際のロード済みFontFaceを返した場合だけ** `FONT ✓` を表示します。
- Adobe Fontsの正式なCSP要件に合わせ、`font-src data:` と `img-src https://p.typekit.net` を追加。
- 日本語: `Noto Sans JP` 優先。文字がWebフォントにない場合のシステムフォールバックは維持。
- フォントCSSは非同期取得とし、GUI起動をネットワーク待ちにしません。
- GUI右上の `FONT …` をクリックすると再読み込みします。`FONT !` にマウスを載せるとエラー理由が表示されます。
- `FONT !` のうち `Adobe CSS loaded, but ... was not in its font faces` という表示なら、Adobe FontsのWeb Projectに `myriad-pro-condensed` のregular（font-weight 400）を追加し、変更を公開してください。
- ネットワークがないときやAdobe Fonts側で必要なウェイトが指定されていないときは、インストール済みの `Myriad Pro Condensed` またはNoto Sans JP等にフォールバックします。
- ライセンス上、Adobe Fontsのフォントファイルは同梱せず、提供元から読み込みます。
- 学習/通信プロトコルとローカル学習データには変更ありません。

注: この実行環境からAdobeの実配信CSSをダウンロードできないため、Web Project `wzz8rse` に対象書体が実際に登録されているか、MacのWebViewで読み込み成功するかまでは断言できません。GUIの表示で原因を確認できます。


## Archived README_V126_JA.md

# KCIA v126 — Adobe Fonts 遅延読み込みとウェイト診断修正

## 現象

v125の `FONT !` は、「CSS取得失敗」「キットに指定書体が無い」「400ウェイトが未登録」「WebViewの非同期スタイル適用より先に検査した」等の原因を一括表示していました。**現行のMac上で何が起きたかはホバー時のメッセージを見ないと確定できません**。

## 修正

- HTML後付けCSSはもともと初期画面表示をブロックしないため、`media=print` の一時設定を廃止。Safari/WKWebViewのスタイル反映が1フレーム遅れて「フォントが無い」と誤判定する競合を防止。
- FontFaceSetへの登録を数回調べ直し、必要な`@font-face`を確認してから実ロードを検査。
- Adobe CSSが読めない場合、CSSは読めたが指定ファミリが無い場合、通常ウェイト400が無い場合を別々に表示。
- `FONT ✓` → 要求した書体ファイルを実際にロード確認済み / `FONT WGT?` → 書体は読めたが400ウェイトが見つからない / `FONT KIT?` → CSSは取得できたが指定書体が登録されていない / `FONT CDN!` → CSS・フォントデータ取得失敗 / `FONT API!` → WebViewがFont Loading API非対応。
- `FONT ...` ボタンへのホバーで原因を確認でき、クリックで再試行できます。JSを確認できる環境なら `window.__kciaFontStatus` が詳細状態です。
- フォント配信先は理事様が指定した `https://use.typekit.net/wzz8rse.css` のまま。フォントファイルはライセンス上同梱していません。

Adobe側のWeb Projectに本当に対象ファミリが登録されているか、macOSのWKWebView実機で読み込めるかはこの環境からは断定できません。**キット未登録の場合、この変更だけでは解決しない**ので、Adobe Fonts Web ProjectでMyriad Pro Condensedの必要なウェイトを登録してください。


## Archived README_V127_JA.md

# KCIA v127 — Adobe Fonts接続診断とWKWebViewの配信方式

## 状況

v126 の `FONT CDN!` は Adobe Fonts CSS (https://use.typekit.net/wzz8rse.css) の取得失敗か、指定フォントファイルの取得失敗を意味します。**原因がネットワーク側なのかWebView側なのかは表示だけでは確定できません**。

## 変更点

- macOSの実pywebviewでは、HTML文字列を直接WKWebViewにロードする方式に代わり、**127.0.0.1 のランダムな一時URLでKCIAのHTMLだけを配信**します。通常のHTTPページのoriginを得て、WebKit固有の外部フォント読込制限を回避できるか検証します。
- サーバーは起動中だけ存在し、ウィンドウ終了時に停止します。`GET /<ランダム値>` 以外は404、POSTは405、Hostヘッダー不一致は421。ローカルHTML以外を配信しません。学習API・保存データ・フォントファイルは配信しません。
- 既存のpywebview Python APIブリッジはそのまま利用します。Nim/P2P通信と学習処理は変更しません。
- 本番でも配信先は公式Adobe FontsのCSS。フォントファイルのコピー・同梱・プロキシ配信はしません。
- `FONT` ボタンのツールチップにページoriginとCSS/フォント段階を記録します。

## Macでの確認

1. `curl -IL https://use.typekit.net/wzz8rse.css` を実行。200系が得られなければネットワーク、VPN、DNS、プロキシ、Adobe CDNの状態を確認してください。
2. `python kcia.py` で起動。`FONT ✓`になれば読み込み成功です。
3. `FONT CDN!` の場合、マウスでボタンのツールチップを確認。`CSS could not be downloaded` ならCSS側、`font file could not be loaded` ならフォントバイナリ側。ページoriginも表示します。
4. 診断する場合: `KCIA_GUI_DEBUG=1 python kcia.py`。従来方式へ戻す場合: `KCIA_GUI_HTTP_ORIGIN=0 python kcia.py`。非MacでもHTTP版を試す場合: `KCIA_GUI_HTTP_ORIGIN=1 python kcia.py`。

**注意:** 実Mac上のWKWebViewとユーザー所有Adobe Fonts Web Projectを使用したCDN到達性は、この実行環境からは検証できません。通信自体が失敗している場合はHTTP起点に切り替えても解決しません。


## Archived README_V128_JA.md

# KCIA v128 — Adobe Fonts の正しいファミリー名へ修正

Adobe Fonts の Web Project `https://use.typekit.net/wzz8rse.css` で使われているファミリー名に合わせ、欧文フォントを `myriad-pro-cond` に変更しました。

- CSS（テキスト・入力欄・ラベルなど）と JavaScript の `FONT` ロード診断が同じ名前を使用します。
- v127 の `myriad-pro-condensed` は Adobe Fonts の CSS ファミリー名と一致しないため、フォールバックや `FONT KIT?` の原因になり得ました。
- Web Project にないウェイト、CDNやWebView側のネットワーク障害などがある場合は引き続き診断が必要です。
- 学習処理、通信形式、保存済みデータやルーム設定は変更していません。

起動: `python kcia.py`。


## Archived README_V132_SNAPSHOT_WAN_JA.md

# KCIA v132 — WANモデル同期の安定化（Mac / Colab）

## 実機ログから分かったこと

- 異なるMac・Colab上で WebRTC ICE / DTLS / DataChannel / 署名付きHelloの確立まで確認済み。
- その直後、Colabから約326 MBの完全モデルをMacに同期しようとしたが、4つのシャードが残ったまま ICE consent expired で切断。
- Colabの署名付き更新は **1件約158 MB** と大きい。初期同期中にも別DataChannelから大量送信すると、SCTP/TURNの帯域や送信時間を争う。
- これは同期待ちに対する対処であり、実WANの安定接続・全量同期・協調学習の成功はまだ保証していない。

## v132で行った変更

1. `kcia-state` の送信量を **既定1.0 MiB/s** に協調的に制限。16 KiBフレームごとにイベントループへ処理権を渡し、ICE/STUNのタイマーが動く機会を確保。
2. `kcia-patch` の大きな署名済み更新も **既定1.0 MiB/s** で制限。スナップショット中はパッチ送信をフレーム境界で一時停止して、別DataChannelからの大容量転送が競合しないようにした。署名・内容は変更しない。
3. 初回Mac/Colabパイロットのスナップショットを **2 MiBシャード** に変更。接続が切れた場合、検証済みシャードだけディスク上で再利用し、部分シャードのみ取り直す。内部の署名/ハッシュ検証は変更しない。
4. `SNAPSHOT TX` / `SNAPSHOT RX` の進捗ログを約5秒ごとに出す。受信サイズ、検証済み/再利用シャード数、SCTPバッファ量を観測できる。
5. パイロットのみ `KCIA_SNAPSHOT_PRIORITY=1` を既定で有効化。送信中のColabは次のローカル学習ステップを一時停止し、ジャーナル肥大とネットワーク競合を抑える。完了後は再開。

**互換性**: WebRTCのオンワイヤ形式、ルームID、ジャーナル、署名鍵、モデルRevisionは変更なし。VPSのNim/coturnを更新する必要はない。Mac/Colabの両側をv132に揃えることを推奨。

## 起動

両端でv132 ZIPを展開し、依存関係をインストールする。`wan_state/`がある場合、**上書き・削除しない**。

Mac:

```bash
python -m pip install -r requirements-patches.txt
python scripts/start_wan_pt.py prepare   # 完全に新しい実験をする場合のみ
KCIA_TURN_TRANSPORT=tcp KCIA_SNAPSHOT_RATE_MIB=1.0 KCIA_PATCH_RATE_MIB=1.0 \
  python scripts/start_wan_pt.py run --role mac \
  --room-id 'ROOM_ID' --revision 'REVISION' \
  2>&1 | tee wan_mac_v132.log
```

Colab (GPUランタイムでv132をアップロードして展開後):

```python
import os, subprocess, sys
os.environ['KCIA_TURN_TRANSPORT']='tcp'
os.environ['KCIA_SNAPSHOT_RATE_MIB']='1.0'
os.environ['KCIA_PATCH_RATE_MIB']='1.0'
subprocess.run([sys.executable, 'scripts/start_wan_pt.py', 'run', '--role','colab',
  '--room-id','ROOM_ID','--revision','REVISION'], check=True)
```

同じRoom IDと40文字のrevisionを両端で指定する。学習状態・署名鍵は別々のまま。初回のローカル学習結果を捨てたくない場合は以前のroom/revisionを使用するが、古いイベントが多いと境界検証に時間がかかる。Colabのランタイム消去時に `wan_state/`を失わないよう保管。

## 進捗と成功判定

```text
SNAPSHOT TX sent_mib=... buffered_kib=... rate_limit_mib_s=1
SNAPSHOT RX received_mib=... verified_shards=... reused_shards=... outstanding=...
SNAPSHOT DOWNLOADED peer=...
SNAPSHOT READY: ...
JOIN READY: ...
elastic ... applied=... channels=1 authed=1 synced=2
```

`ICE=completed` と `SIGNED HELLO OK` だけでは**同期完了ではない**。`SNAPSHOT READY` / `JOIN READY` のあと、両端で `applied` が増えてはじめて署名付き更新の協調を評価できる。初回約326 MBなら転送速度1 MiB/sで**理論上5分強**、実際はさらにオーバーヘッドがかかる（バックグラウンド実行ではなくユーザー端末で進む）。

もし45秒前後で `Consent to send expired` が再発したら、原因は帯域競合以外の TURN/ICE keepalive、SCTP輻輳制御、コンピュート負荷等に残っている。ログの `SNAPSHOT TX/RX` に加えてcoturnのログも採取する。速度を上げる場合は両端で `KCIA_SNAPSHOT_RATE_MIB=2.0` 等（0.125～32）を試せるが、まずは1.0で検証。

## 注意

- 既存データ・署名鍵・ジャーナルは絶対に削除しない。
- `KCIA_SNAPSHOT_PRIORITY=0` でパイロットの学習ステップ一時停止を解除できるが、今回の実機テストでは非推奨。
- 「検証済みのシャードを再利用する」もので、切断したシャードの途中からバイト単位で続けるわけではない。
- `KCIA_PATCH_RATE_MIB` も大量のイベント交換時の安定性優先。150 MB級イベントが多数ある場合は同期に時間がかかる。
- このビルド環境には`aiortc`がなく、実WAN・ConoHa TURN経由の接続テストは未実施。ローカルでの転送・切断復帰テストを実施済み。

## 回帰テスト

```bash
python -m pytest -q tests/test_v132_snapshot_priority.py tests/test_live_snapshot.py \
  tests/test_shards.py tests/test_snapshot_reconnect_regression.py \
  tests/test_snapshot_request_delivery.py
```


## Archived README_V133_TURN_ISOLATION_JA.md

# KCIA v133 — TURN/TCP relay-only WAN pilot

## 実機v132ログから確定できること

- Colabプロセスが `role=mac` かつ `device=cpu` で動いていた。正しくは `--role colab` + GPUランタイム。
- Mac/Colabとも P2Pログが `ICE TURN transport=auto` だった。以前試験したTURN/TCPを実際の学習では指定していなかった。
- ICE/DTLS/DataChannel/署名認証の成立は確認済み。ただし、スナップショット送信は約2.4 MiBで停止し、ICE consent expirationで切れる。
- SNAPSHOT PRIORITY HOLD は接続の間だけ効く。切断後に学習イベントが増えると、再接続時のマニフェストが変わり、シャード再利用率も下がり得る。

## 変更

1. WAN pilot launcherは既定で `KCIA_TURN_TRANSPORT=tcp` と `KCIA_FORCE_TURN_RELAY=1` を子プロセスに設定する（明示環境変数で上書き可能）。`P2P ICE TURN transport=tcp relay_only=True` とログに表示する。
2. 中核の `PatchNode` は `KCIA_FORCE_TURN_RELAY` が有効のときのみ、送信SDPをTURN relay candidateに限定する。一般のP2Pでの挙動は変わらない。
3. WAN pilotのシャード既定を1 MiBへ縮小。切断前に1個でも完全検証しやすくする（部分シャードのバイト単位再開ではない）。
4. Colabノートブックでも TURN/TCP relay-only を明示設定する。
5. 受信者が署名済みスナップショットをACKするまで（切断・再接続を挟んでも）送信者の新規optimizer stepを一時停止する。パイロット専用で最長15分。これにより再接続ごとにマニフェストが変わることを抑え、検証済みシャードの再利用を成立させやすくする。

## Mac

```
python scripts/start_wan_pt.py run --role mac --room-id 'ROOM_ID' --revision 'REVISION' 2>&1 | tee wan_mac_v133.log
```

## Colab（GPUを有効化）

```
python scripts/start_wan_pt.py run --role colab --room-id 'ROOM_ID' --revision 'REVISION' 2>&1 | tee wan_colab_v133.log
```

両端のログ先頭に `KCIA WAN ICE route: tcp relay_only=1` と、接続時に `P2P ICE TURN transport=tcp relay_only=True` が必要。Colabでは `KCIA COMPUTE DEVICE actor=cuda` も確認。

元の `wan_state` / 鍵 / journal は削除しない。Colabの旧 `wan_state/mac` は過去の署名履歴として残るが、`--role colab` は別の状態フォルダで新しく起動するため、その古い学習イベントを継承するわけではない。

## 残る課題

この版は経路と実験設定の取り違えを防止するための診断修正であり、ConoHa/Colab実WANで転送が成功した証明ではない。接続が維持できても、初期スナップショット約326 MBや巨大な署名イベントの通信量、切断後にソースイベントが増える問題は別途設計改善が必要。


## Archived README_V134_TRAINING_FIRST_JA.md

# KCIA v134 — WAN学習優先・低頻度通信版

## 変更の理由（実機ログに基づく）

- `JOIN READY` が記録され、署名付き更新を `applied=2` まで受信できた。WSS/ICE/TURN・署名認証は機能している。
- その一方で、Macは `SNAPSHOT PRIORITY HOLD` の間に学習が停止し、約326 MBの初期モデル転送に時間を使った。
- Colabの署名付き更新1イベントは約158 MB、Macは約11.5 MB。これを4ミニバッチごとに発行すると、計算より通信・ジャーナル増大が支配的になる。

## 新しいWANパイロット既定値

| 項目 | 旧WAN v133 | v134 | 理由 |
| --- | --- | --- | --- |
| `--local-steps` | 4 | **32** | 約8倍のローカル処理を1イベントへ集約 |
| `--reinit-every` | 20イベント | **20イベント** | 既存ルームのgenesis/署名履歴と互換性を維持 |
| `KCIA_SNAPSHOT_PRIORITY` | 1 | **0** | 初期モデル配信中もローカル学習を継続 |
| `KCIA_TURN_TRANSPORT` | tcp | **auto** | 実機で接続成功した設定に戻す |
| `KCIA_FORCE_TURN_RELAY` | 1 | **0** | 不要なrelay限定を解除 |

**重要**: 4×20は通信間隔ではない。4個のローカル *minibatches* で1つの署名付きイベントを発行し、20イベントごとにReLoRAリセットしていた。既定の32×20では理論上640 minibatches（CUDAのgrad_accum 4では約160 optimizer updates）ごとにリセットされる。ローカル勾配蓄積、明示seal、データ/ロスター変化によって実際のイベント間隔は変わり得る。

**通信データ量**：Colabの158 MB/イベントが変わらないと仮定すると、同じ640 minibatchesの署名イベントは旧4設定で160件→新32で20件（理論上、通信とジャーナル負荷を約87.5%削減）。実測ではパッチの変化や中断/追いつき転送により異なる。これは**イベント数を減らす**対策であり、1件158 MBを圧縮する修正ではない。

## Mac起動（既存ルーム/ジャーナルを維持）

```bash
KCIA_TURN_TRANSPORT=auto KCIA_FORCE_TURN_RELAY=0 \
python scripts/start_wan_pt.py run --role mac \
  --room-id 'ROOM_ID' --revision 'REVISION' \
  --local-steps 32 --snapshot-priority training \
  2>&1 | tee wan_mac_v134.log
```

## Colab起動（GPUランタイム使用）

```bash
KCIA_TURN_TRANSPORT=auto KCIA_FORCE_TURN_RELAY=0 \
python scripts/start_wan_pt.py run --role colab \
  --room-id 'ROOM_ID' --revision 'REVISION' \
  --local-steps 32 --snapshot-priority training \
  2>&1 | tee wan_colab_v134.log
```

32がまだ通信過多なら、Colabのみ `--local-steps 64` にして比較可能（ただし更新が古くなる時間・学習収束への影響を評価）。`--local-steps` の変更だけならroom genesisを変更しないため、同じルームで比較可能。**`--reinit-every` はルーム側で固定される重要な識別子であり、既存ルームでは変更しない。**

## スナップショット優先モードについて

新しい `--snapshot-priority training` は旧v133の長時間全停止をやめる。モデル転送はハッシュ/署名付きのある時点の状態として扱われ、その後の署名イベントをsuffixとして同期する既存の仕組みを使う。ただし、**同期中に作られたイベントは後で送る必要があり、WAN負荷はゼロにはならない**。接続不安定なら `--snapshot-priority snapshot` へ戻して安全性を優先できる。

中核の署名プロトコル・シャード形式・公開Nimサーバー・TURN設定・モデル・LoHAの物理rankは変更なし。ソースアップデートで既存`wan_state/mac`,`wan_state/colab`を消さない。モデル/署名鍵/ジャーナルは**端末間でコピーしない**。

## まず確認する項目

ログに `KCIA TRAIN cadence local_minibatches_per_event=32 reinit_every_events=20 ... snapshot_priority=training` が出る。`elastic steps` に `publish_target=... grad_accum=... local_minibatches_total=...` を追加。両端で`applied`が増えること、`emitted`が旧版より緩やかに増えること、lossが再開することを観察する。短いお試し文だけで品質改善は結論できない。

**未検証事項**: ConoHa+Colab+Macの実WAN学習をこの生成環境から再実行することはできない。配布物はローカルの単体/回帰テストで検証する。


## Archived README_V135_GUI_WAN_JA.md

# KCIA v135 — WAN-first GUI（kcia.tech / v134 Training First 統合）

## 起動

```bash
python -m pip install -r requirements-gui.txt
python -m patchlib.cli gui
```

ブラウザ内のWebRTCデモではなく、KCIAのデスクトップGUI（`web/desktop_webview.html`）で有効になります。Mac/Linux上でPyWebView対応のデスクトップセッションが必要です。

## ネットワーク

- 通常運用の既定は **WAN**。実際の接続先は **`wss://signal.kcia.tech/ws`** （`kcia.tech`の実機で検証済みのシグナリング用サブドメイン）。`wss://kcia.tech/ws` の利用可否は確認されていないため使用しません。
- `MODELS` 画面では公開ディレクトリの照会をローカル一覧より先に発行します。WANのルームを優先表示し、手元で保存された参加済み・非公開ルームも合流します。公開リスト自体はサーバーが公開を許可したルームのみ表示します。
- `NETWORK`の `LOCAL · デバッグ専用` を明示選択すれば `ws://127.0.0.1:8080/ws` に切り替えます。GUIはローカルのMummy/Nimサーバーを自動起動しません。
- 過去に保存された **v1 のデフォルト localhost 接続** は、WANへ移行します。v135で明示したローカルデバッグ選択はv2設定として再起動後も保持されます。独自のWSS設定も保持します。
- 「接続テスト」は実際のWSS ping と公開ディレクトリ照会です。サーバー未到達時はGUIを停止せず、エラーをNETWORKに表示します。存在しない公開ルームを捏造しません。

## GUI TRAINING設定

`TRAINING`画面でワーカー起動前に変更できます。

| 項目 | WAN既定 | 設定可能 |
|---|---|---|
| 公開間隔 | **32ローカルミニバッチ/署名更新** | カスタム 4〜256（4の倍数）、従来の自動調整 |
| 自動資源調整 | 維持 | モデル/メモリによる初期ポリシーは維持 |
| 自動更新間隔の下限 | 32 | カスタム値を下限にする。自動調整に戻すと下限を解除 |
| ReLoRA再初期化 | **20イベント** | 既存ルームの署名済み定義が優先 |
| スナップショット中 | **学習継続** | 学習停止（従来の安全優先） |
| TURNトランスポート | **auto** | tcp / udp |
| ICE候補 | **auto、relay限定OFF** | TURN限定モード（診断用） |
| 状態シャード | 1 MiB | GUI版固定（CLI単独は従来仕様） |

UIに表示される `Steps / publication` は**ワーカーの署名イベント生成間隔**です。ReLoRAの20イベントは別のルーム設定です。

GUI操作時、`--local-steps` / `--min-publish-steps` / `--snapshot-priority` / `--turn-transport` / `--no-relay-only` が `patchlib.cli train-live` に渡ります。`--min-publish-steps` は自動資源調整によるローカル同期間隔の不用意な短縮を防ぎます。ピア署名、イベント形式、公開ルームマニフェスト、既存のチェックポイント形式は変更しません。

GUIのトレーニング設定は **`.kcia-gui/training-preferences.json`** にモード別の機密情報なしで保存します。保存は原子的に行い、ディレクトリは0700・ファイルは0600、値の妥当性を検査して復元します。学習中に編集しても既存ワーカーへ即時反映せず、次回起動で有効になります。

## 既存データに関する注意

- GUIのルーム・署名鍵・ジャーナルは `~/.kcia/...` 等のGUI管理データルート内の `.kcia-gui/` に格納されます。実際のルートは環境依存です。
- 以前のCLI実験で使った **`wan_state/` は自動的にはGUIへ移行しません。** 既存ルームをGUIで使う場合、モデル定義を公開ディレクトリまたは参加リンクから受信し、GUI自身の署名ピアとして参加します。GUIのイベント履歴とCLIのイベント履歴を無検証で統合しません。
- 過去の署名ジャーナル、鍵、`wan_state/` を削除しないでください。モデル・ジェネシス・データセット条件が異なれば、署名済みルームへ強制参加させずエラーにします。
- VPSの既存 `signal.kcia.tech` / `turn.kcia.tech` は変更不要です。DNSやCloudflare側の設定変更もありません。

## テスト

```bash
python -m pytest -q tests/test_v135_gui_wan_defaults.py tests/test_public_gui_network_v113.py tests/test_webview_gui.py
node --check web/desktop_webview.js
```

上記テストはネットワーク通信やモデル学習をエミュレート/モックしたオフライン試験です。Mac・Colab双方のGUI実機で長時間WAN学習できることの証明ではありません。


## Archived README_V136_GUI_WAN_FAST_GRAPHS_JA.md

# KCIA v136 — WANを正しい既定動作へ / GUIグラフ応答改善

## v135から修正した問題

1. `MODELS` のWANフィルターが `x.local && networkMode==='internet'` を含んでいたため、**単なるローカルキャッシュもWANとして表示**していた。この条件を撤去し、公開サーバー一覧から受け取ったルーム、またはサーバーから登録ACKを得ていたルームのみをWANルーム欄に表示する。
2. v135以前の `.kcia-gui/network.json` にローカル `ws://127.0.0.1:8080/ws` が保存されていると、GUIの起動時にそのままローカルへ戻ってしまった。**GUI起動時はWANが標準**となるよう、旧設定から移行する。LOCALデバッグはGUIで明示的に選択できるが、次回起動の標準設定には保存しない。カスタムWSS接続先は引き続き保持する。
3. 新規モデルの「公開範囲」を **公開**（WAN登録）既定にした。非公開のWANルームも引き続き作成できる。公開を選ぶとモデルID、ルームID、データセット要約等の公開メタデータがディレクトリへ掲載される。機密のある場合は「非公開」を選ぶこと。
4. WSS `room-register` の**成功応答を受けたときだけ**、そのルームに `wan-registration.json` (0600) を記録。非公開ルームは通常の公開ディレクトリには出ないため、この登録履歴に基づきWAN登録済みと区別する。ローカルで生成しただけのモデルはWANとは偽装表示しない。サーバー現在状態とは異なることがあるため、表示名は「WAN登録済み（確認履歴）」にしている。
5. `LOCAL-FIRST` と旧版番号の残骸を削除。NETWORK図の中央にある `LOCAL` は「接続元端末」という意味だが、WANでないと誤解されるため `SELF` とした。
6. グラフの再描画上限 **4秒 → 約0.8秒**、GUIステータスポーリング **2.5秒 → 0.9秒**。最新データが1秒間隔で記録されれば概ね1秒前後で描画される。古い履歴は12秒ごとに再取得し、SVGの描画点を360個以内に間引いてレンダリング負荷を抑制する。正確な原本は従来のテレメトリJSONLに保持する。
7. 起動時に前回選択した「WAN未登録のローカルルーム」を勝手に復元してWAN公開モデルを覆う挙動も止めた。

## 注意事項

- `kcia.tech` はネットワークのブランド名。実接続先は `wss://signal.kcia.tech/ws` で変えていない。VPSの再構築は不要。
- **公開WANディレクトリには公開設定のルームだけが表示されます**。非公開ルームを他の端末から参照するには招待リンクが必要です。
- v135以前にネットワーク登録済みの非公開ルームは、旧クライアントがACK履歴を保存していなかったため、v136で1回「ネットワーク登録を再試行」するか、招待リンクから参加してください。古い履歴から登録済みを推測・捏造しません。
- GUIでのモデル登録と計算・モデル状態の保存がローカルファイルなのはP2P設計上正常です。**ローカル保存≠ローカル接続**。WANシグナリング・P2Pデータ通信とは別です。
- v135までの `.kcia-gui`、署名鍵、ジャーナル、`wan_state` は削除しません。CLIの `wan_state` をGUIへ移行する機能も追加していません。
- グラフはGUIがローカルワーカーから記録する実際のテレメトリを表示。ネットワーク越しの全ピア分のLossを自動的に集約したものではありません。

## 実行

v135が適用されたソースに差分パッチを適用するか、v136完全版を展開。

```bash
python -m patchlib.cli gui
```

GUIの `NETWORK` は `WAN · kcia.tech`、接続先は `wss://signal.kcia.tech/ws`。`MODELS` のWANルームには公開サーバーが実際に返したルームと、登録成功履歴を持つ非公開ルームが並びます。サーバーがオフラインなら、エラーを表示します。

## ローカル検証

```bash
python -m pytest -q tests/test_v136_gui_wan_graph.py tests/test_v135_gui_wan_defaults.py \
  tests/test_public_gui_network_v113.py tests/test_webview_gui.py
node --check web/desktop_webview.js
```

実際のMac WebViewレンダリング性能や、ライブWANディレクトリへの実登録はこのオフライン環境からは検証していません。


## Archived root README.md (v136)

> **KCIA v136**: WAN既定値の復旧・モデル登録状態区別・高速グラフについては README_V136_GUI_WAN_FAST_GRAPHS_JA.md (historical link: README_V136_GUI_WAN_FAST_GRAPHS_JA.md) を参照。

> **KCIA v135: WAN-first GUI / kcia.tech / Training First** — README_V135_GUI_WAN_JA.md (historical link: README_V135_GUI_WAN_JA.md)

> **KCIA v134 — WAN学習優先・32 minibatches/event**: README_V134_TRAINING_FIRST_JA.md (historical link: README_V134_TRAINING_FIRST_JA.md)

> **KCIA v132: WANスナップショット安定化** — 詳細は README_V132_SNAPSHOT_WAN_JA.md (historical link: README_V132_SNAPSHOT_WAN_JA.md) を参照。

# KCIA v130 — Mac + Colab live training pilot

今すぐ学習を開始 (historical link: README_TRAIN_NOW_JA.md) / WAN接続準備 (historical link: README_WAN_PILOT_JA.md)

# KCIA v129 — WAN pilot preparation

VPS + Cloudflare + Mac + Colab の実機接続は README_WAN_PILOT_JA.md (historical link: README_WAN_PILOT_JA.md) から開始してください。v128の学習プロトコルは変更していません。

# KCIA v127 — macOS Adobe Fonts origin / diagnostics

修正内容 (historical link: README_V127_JA.md)

# KCIA v126 — Adobe Fonts load race and diagnostics

修正内容 (historical link: README_V126_JA.md)

# KCIA v125 — Adobe Fonts loading diagnosis and CSP fix

修正内容 (historical link: README_V125_JA.md)

# KCIA v124 — 完全ブランド統一

最新の変更: README_V124_JA.md (historical link: README_V124_JA.md)。起動は `python kcia.py`。新規の招待リンクは `kcia://join?...` のみで、旧 `kagami://` は非対応です。

# KCIA v123 — Adobe Fonts Myriad Pro Semi Condensed

フォント変更について (historical link: README_V123_JA.md)

# v121 — フォント・全ローカルリセット・GUI起動時の学習継続防止

変更内容・完全初期化手順 (historical link: README_V121_JA.md)

# KCIA v120 — 公開ルーム学習設定とGUI修正

**変更内容:** README_V120_JA.md (historical link: README_V120_JA.md) を参照してください。新規ルームでは SFT/PPO の Hugging Face データセット参照と学習ポリシーを保存し、他のピアで復元します。データセットにアクセスできないピアは PT のみで参加する設定を選べます。新規ルームのReLoRA間隔は署名付き送信20回（公開ごとのローカルステップ既定4。旧ルームは既存設定を維持）。Nim サーバーもマニフェストの最大サイズを16KiBに対応させたため、VPS側のサーバーも更新してください。

# KCIA v118 — KCIA ロゴの表示上の左揃え修正

旧v117ではSVGの要素ボックスだけを左揃えにしていましたが、幅195pxに対して高さ49pxを上限にしたため、SVGの`preserveAspectRatio`既定動作で図形がボックス内で中央寄せされていました。v118では元SVGの縦横比を保つ表示幅138px・高さ自動に修正し、実際の図形左端をサイドバー本文の左端に合わせます。元のSVGパスデータは変更していません。

学習・通信・GUI機能はv117から変更していません。

# KCIA v117 — GUI responsiveness / window drag hotfix

- `Create Model`, network controls and other UI flows are retained from v116.
- Frameless native window actions now return a queued acknowledgement, and a separate
  serialized worker executes the actual OS calls after the WebView RPC has returned.
- Window control actions no longer wait for model/network initialization.
- The entire empty title strip and KCIA logo area are native drag handles. SVG
  hit-testing is disabled inside the drag handle; buttons/inputs remain clickable.
- The KCIA logo is explicitly aligned to the **left** of the sidebar.
- No stored model state, optimizer journals, peer IDs or signatures are migrated.

Known verification limit: Mac/WebKit's native window drag behavior must still be
confirmed on a Mac (this build environment has Linux Chromium, not Cocoa).

---

# v116 physical heterogeneous LoHA + frameless GUI

設計・使用方法 (historical link: docs/releases/README_V116_HETEROGENEOUS_LOHA_JA.md)

# KCIA v114 — Create Model 修正・モデルピッカー・KCIAロゴ

- GUIの初期化停止を修正: `safe` が初期化前に参照され、`Create Model` を含む後続ボタンの登録が止まっていた問題を解消。
- `NEW MODEL` にHugging Faceベースモデルの検索・選択ダイアログを追加。タスク別候補・名前検索に対応し、Hubが返すコミットSHAをリビジョンへ反映。
- `Create Model` はローカル保存完了後にすぐMODELSへ進める。Cloudflare/VPS停止中でもローカル保存を成功として扱い、サーバー登録は別途実行して失敗を明示。
- ボタンの保存中状態・失敗理由を表示。ネットワーク登録再試行が既存モデルを誤って新規作成しないよう修正。
- サイドバーのインラインロゴを添付されたKCIA SVGに正確に置換（原本は `web/assets/kcia.svg` に同梱）。
- room IDの自動生成、匿名パブリックシグナリング、既存のモデル/データの形式は変更なし。移行時の履歴・鍵・チェックポイントの削除は不要。

使い方・検証: v114の日本語リリースノート (historical link: docs/releases/README_V114_GUI_CREATION_MODEL_PICKER_JA.md)。

---

# KCIA v113 — Public Cloudflare signaling and live GUI

- `APP_TOKEN` と `ROOM_PUBLISH_TOKEN` を廃止。公開シグナリング、モデル公開登録、リモート参加にユーザートークンは不要。
- `NETWORK` タブで `INTERNET` / `LOCAL` 接続先を保存、WebSocket ping/pong で疎通試験。
- `NEW MODEL` はルームID自動生成。作成時に自動で公開/非公開登録し、招待リンクでリモート参加。
- `TURN_SECRET` と Cloudflare Tunnel token はサーバー専用で存続。CloudflareプロキシはTURN UDPを中継しない。
- 学習は元の `train-live` WebRTC DataChannel / 署名付き Gossip エンジンを使用。VPS/Colab/Macの実機確認は別途必要。

手順: deploy/vps/README_CLOUDFLARE_JA.md (historical link: deploy/vps/README_CLOUDFLARE_JA.md)

---

# VPS + Cloudflare Tunnel + Mac/Colab test package

日本語セットアップ手順 (historical link: deploy/vps/README_CLOUDFLARE_JA.md) / Colab notebook (historical link: notebooks/KCIA_Cloudflare_Mac_Colab.ipynb)

# KCIA — 分散型LLM学習プラットフォーム

KCIAは、LoHA/LoRAによるモデル学習、P2P Gossip、PT・SFT・PPO、署名付き更新、世代移行、再起動後の学習復旧に対応する実験的な実装です。

**v112.2はGUI非依存の直列PT比較ベンチマークを、ピアごとの同一計算量と仮想ネットワーク遅延に対応させます。既存の学習アルゴリズム、署名形式、データベース形式は変更しません。**

## VPS（Cloudflare Tunnel/WSS + Nimシグナリング + coturn）・Mac / Colab 実機テスト

導入手順（日本語） (historical link: deploy/vps/README_CLOUDFLARE_JA.md) / Colabノートブック (historical link: notebooks/KCIA_Cloudflare_Mac_Colab.ipynb)。
この構成はv112.2の学習プロトコルを維持し、サーバーにモデルや鍵・重み・ジャーナルを置きません。

## 起動

```bash
python kcia.py                     # WebView GUI
python -m patchlib.cli --help       # 学習・診断・P2P CLI
python kcia.py ppo-doctor         # 小型CPU PPO更新の自己診断
python kcia.py mps-ppo-smoke      # Mac MPS の隔離実行テスト
```

依存ライブラリは目的に応じて `requirements-patches.txt`、`requirements-gui.txt`、`requirements-eval.txt` から導入してください。Nimシグナリング実装は `src/` にあります。

## 1ピア vs 複数ピアのPT比較（低メモリ・直列）

```bash
python -m pip install matplotlib  # PNGグラフを出す場合（PyTorchは別途必要）
python -m benchmarks.bench_serial_pt_v112 --peers 1,2,4,8 --rounds 30 --steps-per-round 8 \
  --rtt-ms 80 --jitter-ms 12 --bandwidth-mbps 20 --server-bandwidth-mbps 100
```

常駐モデルは1つだけ、仮想ピアのLoHA/AdamW状態は一時ディスクへ切替。これは**同期FedSumのオフライン検証**であり、実際のP2P通信性能を証明するものではありません。**`--steps-per-round`は各ピアのstep数です。** 標準グラフの横軸は仮想並列経過秒で、ネットワークRTT・共有帯域・ジッター・再送・同期バリアを反映します。詳細は `docs/releases/README_V112_2_SCALED_WORK_LATENCY_JA.md` (historical link: docs/releases/README_V112_2_SCALED_WORK_LATENCY_JA.md)。

## コードの探し方

| 目的 | 最初に読む場所 |
| --- | --- |
| GUIの起動とルーム管理 | `kcia.py`, `patchlib/webview_controller.py`, `patchlib/webview_lifecycle_v110.py` |
| CLI引数と学習実行 | `patchlib/cli.py`, `patchlib/cli_arguments_v110.py`, `patchlib/local_training_v110.py` |
| PT / SFT / PPO | `patchlib/mixed_learning_v65.py`, `patchlib/mixed_kernels_v110.py`, `patchlib/mixed_sources_v110.py` |
| Hugging Faceモデルの準備 | `patchlib/model_loading_v110.py`, `patchlib/hf_load_v85.py` |
| P2P・署名・ジャーナル・世代管理 | `patchlib/elasticnet.py`, `patchlib/elastic.py`, `patchlib/elastic_control_v110.py` |
| MPSと診断 | `patchlib/mps_runtime_v91.py`, `patchlib/mps_acceleration_v108.py` |
| 保存・復旧・fast-resume | `patchlib/fast_resume_v85.py`, `patchlib/epoch_carry_v46.py` |

詳細なモジュール地図は `docs/SOURCE_LAYOUT_JA.md` (historical link: docs/SOURCE_LAYOUT_JA.md)、利用ガイドと過去の変更記録は `docs/INDEX.md` (historical link: docs/INDEX.md) を参照してください。

## データ保護と移行

**実行中にソースを上書きせず、更新前にコードと実データをバックアップしてください。** 配布用ZIPからは実行時に生成された `kcia-rooms/*.json` を取り除きましたが、**既存のインストールにあるルーム・鍵・署名ジャーナル・weightを削除してはいけません**。更新のための `reset-local-data` も不要です。

Macの `auto` デバイスは過去のMetalクラッシュ対策のためCPU安全モードが基本です。MPSは明示的に設定し、自己診断と実機確認を経てから使ってください。

## 検証

```bash
python -m compileall -q patchlib tests scripts
PYTHONPATH=.:tests pytest -q tests/test_v110_modular_refactor.py
python tools/audit_release.py
```

テスト結果と既知の制約は `docs/validation/VALIDATION_V111.md` (historical link: docs/validation/VALIDATION_V111.md) を参照してください。

### v112.1 — PT plot x-axis

Serial PT benchmark primary plots now use **tokens processed per peer** (total / peers) for ideal parallel-work comparison; parallel wall time is **not measured**. Equal-total-tokens plots are retained with `-total-tokens.png` suffix. See `docs/releases/README_V112_1_PARALLEL_AXIS_JA.md`.

### v112.2 — 同一ローカル計算量 + 仮想ネットワーク

**v112.1以前の「総計算量固定」とは実験条件が異なります。** 各ピアのステップ数を揃え、総トークン数をピア数に比例して増やします。旧形式のグラフ名と異なり、標準グラフはRTT・帯域・バリア込みの仮想秒を横軸にします。
