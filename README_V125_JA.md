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
