# KCIA v128 — Adobe Fonts の正しいファミリー名へ修正

Adobe Fonts の Web Project `https://use.typekit.net/wzz8rse.css` で使われているファミリー名に合わせ、欧文フォントを `myriad-pro-cond` に変更しました。

- CSS（テキスト・入力欄・ラベルなど）と JavaScript の `FONT` ロード診断が同じ名前を使用します。
- v127 の `myriad-pro-condensed` は Adobe Fonts の CSS ファミリー名と一致しないため、フォールバックや `FONT KIT?` の原因になり得ました。
- Web Project にないウェイト、CDNやWebView側のネットワーク障害などがある場合は引き続き診断が必要です。
- 学習処理、通信形式、保存済みデータやルーム設定は変更していません。

起動: `python kcia.py`。
