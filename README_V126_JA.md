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
