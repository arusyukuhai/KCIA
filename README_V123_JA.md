# KCIA v123 — 欧文フォント指定の変更

- Adobe Fonts stylesheet: `https://use.typekit.net/wzz8rse.css`
- 欧文優先フォント: `"myriad-pro-condensed"`
- 日本語フォールバック: `"Noto Sans JP"`、`"Hiragino Kaku Gothic ProN"`、`"Hiragino Sans"`、`"Yu Gothic"`、`"Meiryo"`
- GUIのロードはAdobe Fontsの応答を待ちません。通信できない場合はインストール済みフォントにフォールバックします。
- Content Security Policyには `use.typekit.net` をスタイルとフォントの許可元として追加し、旧cdnfonts参照を削除しました。
- 学習アルゴリズム、ルーム、署名、通信、保存形式は変更していません。
