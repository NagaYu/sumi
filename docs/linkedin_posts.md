# LinkedIn post drafts — Sumi

All figures below are the **template-independent (OOD)** measurements: 100 positive
and 100 hard-negative documents, same CPU, same thread count for every condition.

---

## 日本語版（日本向け・メイン）

日本語のPII検出で本当に厄介なのは「見つけられないこと」ではなく、「PIIでないものをPIIだと言い張ること」でした。

森、林、泉、大和、青木。長野、福島、千葉、山口。
これらは全部、普通名詞や地名であると同時に、ありふれた姓でもあります。

日本語NERを足したPresidioに、個人情報が1件も含まれない文書100件を流してみました。
83%の文書で誤検出が出ました。「長野県の気候は…」の長野を人名として、「型番 TX-2024-0355」を電話番号として拾ってしまう。

そこで墨（Sumi）を作りました。CPUで動く0.13Bの日本語PII検出器です。

同じ100文書での比較：

・誤検出率　0.830 → 0.000
・誤検出を5%まで許したときの検出率　0.01 → 0.98
・住所　0.08 → 1.00 ／ 金融口座　0.00 → 1.00
・4B級ローカルLLMの162倍のスループット、メモリは約1/6（1.0GB 対 6.0GB）

正直に書いておくと、氏名単体では 0.81 対 0.78 で差はごくわずかです。
効いているのは住所と番号類、そして何より「余計なものを拾わないこと」でした。

工夫したのはモデルよりもデータの側です。
「PIIに見えるがPIIではない」表現を8種類つくり分け、学習後にモデルがどの型で間違えたかを数えて、次の生成バッチをその型に偏らせる——という閉ループを回しています。

Presidioには1行で差し込めます。既存の匿名化パイプラインはそのまま動きます。

対応表をローカルに残したまま、マスク済みテキストだけを外部LLMに送り、返ってきた結果を復元する経路も入れました。元の値が送信内容に混ざった場合は、記録ではなく例外で送信そのものを止めます。

学習・評価データはすべて合成で、実在の個人情報は一切使っていません。
カード番号やマイナンバー様式の数字列は「形式は正しく、値は無効」になるよう生成しています。

なお、これは法令遵守を保証するものではありません。リスクを下げる道具であり、検出漏れは必ず起こります。

ブラウザ上で試せます（貼り付けたテキストは端末外に出ません）:
Demo: https://huggingface.co/spaces/NagaYu/sumi
Code: https://github.com/NagaYu/sumi
Model: https://huggingface.co/NagaYu/sumi-ja-pii
Dataset: https://huggingface.co/datasets/NagaYu/sumi-ja-pii-corpus

#自然言語処理 #機械学習 #個人情報保護 #プライバシー #オープンソース #NLP

---

## English version

The hard part of Japanese PII detection isn't recall. It's false positives.

森 (forest), 林 (woods), 泉 (spring), 長野, 福島, 千葉 — each of these is both an ordinary Japanese word or place name *and* a common surname.

I ran Presidio with a Japanese NER model over 100 documents containing no personal data whatsoever. It flagged something in 83% of them: prefecture names read as people, a part number "TX-2024-0355" read as a phone number.

So I built Sumi — a 0.13B Japanese PII detector that runs on CPU.

On the same 100 documents:

• False-positive rate: 0.830 → 0.000
• Recall at a 5% false-positive budget: 0.01 → 0.98
• Address 0.08 → 1.00, bank account 0.00 → 1.00
• 162× the throughput of a 4B local LLM, at a sixth of the memory (1.0 GB vs 6.0 GB)

Where it doesn't win, stated plainly: on personal names alone the margin is 0.81 vs 0.78. Small. The advantage is addresses, the ID and number families, and above all not crying wolf.

The interesting work was in the data, not the model. I generate eight kinds of deliberately confusable "looks like PII but isn't" material, then count which kinds the trained model still gets wrong and skew the next generation batch toward exactly those. A closed loop.

It drops into Presidio in one line, and existing anonymisation pipelines keep working.

There's also a reversible masking path: send masked text to an LLM, restore the response locally. If an original value would ever reach the wire, the send raises rather than logs.

Everything — training and evaluation — is synthetic. No real personal data anywhere. Card-shaped and My-Number-shaped strings are format-valid and deliberately value-invalid, so nothing usable can be generated.

It reduces risk. It is not a compliance guarantee, and misses will happen.

Try it in your browser — nothing you paste is uploaded anywhere:
Demo: https://huggingface.co/spaces/NagaYu/sumi
Code: https://github.com/NagaYu/sumi
Model: https://huggingface.co/NagaYu/sumi-ja-pii
Dataset: https://huggingface.co/datasets/NagaYu/sumi-ja-pii-corpus

#NLP #MachineLearning #Privacy #PII #OpenSource #Japanese

---

## 短縮版・日本語（フィードで流し読みされる想定）

「長野県の気候は」の"長野"を人名として拾う。
「型番 TX-2024-0355」を電話番号として拾う。

日本語NERを足したPresidioに、個人情報を1件も含まない文書100件を流したら、83%の文書で誤検出が出ました。日本語のPII検出で難しいのは、見つけることより「見つけすぎないこと」です。

CPUで動く0.13Bの日本語PII検出器「墨（Sumi）」を公開しました。

同じ100文書で、誤検出率 0.830 → 0.000。
誤検出を5%まで許したときの検出率は 0.01 → 0.98。
4B級ローカルLLMの162倍速、メモリは約1/6。

氏名単体では 0.81 対 0.78 で大差はありません。効いているのは住所・番号類と、誤検出の少なさです。

Presidioに1行で差し込めます。学習データは全て合成、実在の個人情報は不使用。法令遵守を保証するものではありません。

ブラウザ内で動くデモ（テキストは端末外に出ません）:
https://huggingface.co/spaces/NagaYu/sumi

#自然言語処理 #機械学習 #個人情報保護 #NLP

---

## Short English version

"長野" in "the climate of Nagano Prefecture" — flagged as a person.
"TX-2024-0355" — flagged as a phone number.

I ran Presidio with Japanese NER over 100 documents containing zero personal data. It fired on 83% of them. In Japanese PII detection, the hard part isn't finding things. It's not over-finding them.

Sumi is a 0.13B Japanese PII detector that runs on CPU. On those same 100 documents: false-positive rate 0.830 → 0.000, and recall at a 5% false-positive budget 0.01 → 0.98. It's 162× the throughput of a 4B local LLM at a sixth of the memory.

On names alone the margin is small (0.81 vs 0.78). The win is addresses, ID/number families, and precision.

One line to drop into Presidio. All training data synthetic — no real personal information. Not a compliance guarantee.

Runs in your browser, nothing uploaded:
https://huggingface.co/spaces/NagaYu/sumi

#NLP #MachineLearning #Privacy #OpenSource

---

## Posting notes

- LinkedIn truncates at roughly 200 characters, so the first two lines carry the
  post. Both long versions open on the counterintuitive finding rather than on the
  release announcement.
- Attach `figures/fig3_false_positives_ood.png` (the false-positive chart) — it is
  the single clearest image and matches the hook. `fig1_name_detection_ood.png` is
  the alternative if you prefer to lead with recall.
- Put the links in the first comment instead of the post body if you want more
  reach; LinkedIn suppresses posts with outbound links.
- The demo runs client-side, so "paste your own text, nothing is uploaded" is a
  true statement and a strong reason for people to click. Worth saying explicitly.
- The "where it doesn't win" line is deliberate. Naming the weak result is what
  makes the strong ones credible to a technical audience.
