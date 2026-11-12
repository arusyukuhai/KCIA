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
