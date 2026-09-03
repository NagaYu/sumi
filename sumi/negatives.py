"""HardNegativeGenerator — 「PIIに見えるがPIIでない」表現を意図的に作る差別化の中心。

Claim: 低誤検出 — Sumi の主戦場は「紛らわしい否定例での誤検出率」である。
本モジュールは、普通名詞と同形の姓・地名と同形の人名・企業名・敬称境界のゆれ・
電話番号に見える型番/注文番号/日時・住所に見える施設名などを構成的に生成し、
学習時にはモデルへ、評価時には誤検出率の測定対象として供給する。

さらに ``reweight_from_errors`` により、学習後のモデルが実際に間違えた型へ
次バッチの生成分布を寄せる閉ループを提供する。
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from sumi.types import Document, Span, normalize

NEGATIVE_KINDS: tuple[str, ...] = (
    "common_noun_surname",    # 森/林/泉/大和/青木 を普通名詞として使う
    "place_as_person",        # 地名と同形の人名 (長野/福島/千葉...) を地名として使う
    "company_as_person",      # 企業名と同形の人名 (本田技研/大和ハウス...)
    "honorific_boundary",     # 様/さん/氏/殿 の有無による境界のゆれ
    "phone_like_id",          # 電話番号に見える型番・注文番号・日時
    "address_like_facility",  # 住所に見える施設名
    "number_like_id",         # 会員番号/口座に見えるが違う数字列
    "date_like_nondob",       # 生年月日に見えるが違う日付
)


@dataclass
class NegativeItem:
    """1件の hard negative 表現。

    Claim: 低誤検出 — 「どの型の紛らわしさか」を surface と一緒に保持することで、
    誤検出を型別に集計して閉ループの入力にできる。
    """

    text: str
    kind: str
    note: str = ""


# --------------------------------------------------------------------------
# 素材 — すべて公開されている一般語・地名・法人名であり、個人情報を含まない
# --------------------------------------------------------------------------

# 普通名詞と同形の姓 (森/林/泉/大和/青木 ...) を、明らかに普通名詞として使う文脈
_COMMON_NOUN_SURNAME = [
    ("森", ["{x}の中を歩くと鳥の声が聞こえる。", "国有{x}の管理計画を見直す。", "鎮守の{x}が残っている。"]),
    ("林", ["{x}業の振興に関する予算を計上する。", "雑木{x}が広がっている。", "防風{x}を整備した。"]),
    ("泉", ["{x}が湧く場所として知られる。", "温{x}街の景観を保全する。", "源{x}の水質を検査した。"]),
    ("大和", ["{x}政権の成立について述べる。", "{x}言葉を用いた表現。", "{x}絵の技法を解説する。"]),
    ("青木", ["{x}の生垣を剪定した。", "{x}は日陰でもよく育つ低木である。"]),
    ("石田", ["{x}三成に関する史料は多い。", "{x}地区の圃場整備が完了した。"]),
    ("原", ["草{x}が一面に広がる。", "{x}材費が高騰している。", "{x}則として認められない。"]),
    ("谷", ["深い{x}に橋を架ける。", "{x}間の集落を調査した。"]),
    ("島", ["離{x}の医療体制を整える。", "{x}全体が国立公園に指定されている。"]),
    ("川", ["一級河{x}の氾濫を想定する。", "{x}沿いに遊歩道を整備した。"]),
    ("本田", ["{x}圃場の稲作を継続する。"]),
    ("東", ["{x}の空が明るくなってきた。", "{x}西南北の方位を確認する。"]),
    ("西", ["{x}日が強く差し込む。", "{x}洋医学の観点から検討する。"]),
    ("南", ["{x}向きの窓を設ける。", "{x}極観測隊が出発した。"]),
    ("北", ["{x}向きの斜面は雪が残る。", "{x}半球の気候を比較する。"]),
    ("上田", ["{x}と下田の区分を用いる。"]),
    ("小島", ["湾内の{x}に灯台がある。"]),
    ("松", ["{x}の木を植樹した。", "門{x}を飾る習慣がある。"]),
    ("藤", ["{x}の花が見頃を迎えた。", "{x}棚の下で休憩する。"]),
    ("柳", ["{x}が風に揺れている。", "川辺の{x}を剪定した。"]),
    ("岩", ["{x}盤の強度を調査する。", "巨{x}が転がっている。"]),
    ("辻", ["{x}に道標が立っている。"]),
    ("堀", ["城の外{x}を一周した。"]),
    ("関", ["{x}所の跡地を訪ねた。", "{x}係者以外立入禁止。"]),
    ("高", ["標{x}千メートルの地点。", "{x}温注意報が発表された。"]),
    ("新井", ["{x}戸を掘削した記録が残る。"]),
    ("花田", ["{x}植えの作業を行う。"]),
]

# 地名と同形の人名 (長野/福島/千葉/山口/宮崎/石川/岡山...) を **地名として** 使う
_PLACE_AS_PERSON = [
    "長野", "福島", "千葉", "山口", "宮崎", "石川", "岡山", "福井", "山形",
    "徳島", "香川", "大分", "佐賀", "奈良", "三重", "滋賀", "愛媛", "高知",
]
_PLACE_TEMPLATES = [
    "{x}県の気候は寒暖差が大きい。",
    "{x}の復興状況について報告する。",
    "{x}市内の交通量を調査した。",
    "来月、{x}支店で研修を実施する。",
    "{x}県産の農産物を取り扱っている。",
    "{x}方面行きの列車が遅延している。",
    "{x}地方の方言を記録した。",
    "本社を{x}へ移転する案が出ている。",
]

# 企業名と同形の人名
_COMPANY_AS_PERSON = [
    "本田技研工業", "大和ハウス工業", "松下電器産業", "伊藤忠商事", "住友商事",
    "三井物産", "青木あすなろ建設", "小林製薬", "森永製菓", "山崎製パン",
    "鹿島建設", "大林組", "竹中工務店", "西松建設", "高松コンストラクション",
    "石川島播磨重工業", "川崎重工業", "島津製作所", "村田製作所", "京セラ",
    "田中貴金属工業", "中村屋", "吉野家ホールディングス", "松井証券", "野村證券",
]
_COMPANY_TEMPLATES = [
    "{x}との取引条件を見直す。",
    "{x}が新製品を発表した。",
    "{x}の決算資料を参照のこと。",
    "本件は{x}へ再委託している。",
    "{x}株式会社宛に見積書を送付した。",
    "{x}の担当部署に確認する。",
]

# 敬称の有無による境界のゆれ。**人名でない「様」** を多く含める
_HONORIFIC_TRAPS = [
    "お客様各位におかれましては、平素より格別のご高配を賜り厚く御礼申し上げます。",
    "皆様のご協力に感謝申し上げます。",
    "神様に祈るような気持ちで結果を待った。",
    "王様と大臣の逸話を引用する。",
    "奥様向けの説明会を開催します。",
    "貴社御中宛に書類を送付いたしました。",
    "関係者各位殿へ通知する。",
    "ご担当者様までお問い合わせください。",
    "有難う存じます、と先方様よりご連絡がありました。",
    "この度は誠に恐れ入り様のない対応で失礼いたしました。",
    "殿様行列の再現行事が行われた。",
    "様式第三号により申請すること。",
    "同様の事案が過去にも発生している。",
    "多様な働き方を推進する。",
    "仕様書の改訂版を配布した。",
    "模様替えのため休館する。",
    "御中元の手配を進める。",
    "氏名欄は空欄のままで構いません。",
    "氏族制度について概説する。",
    "君主制の歴史を振り返る。",
    "諸君の健闘を祈る。",
    "先生方のご指導を仰ぐ。",
    "衛生管理者を選任した。",
    "部長会議の議事録を回覧する。",
    "課長職の職務範囲を明確化する。",
    "社長就任の挨拶が行われた。",
]

# 電話番号に見える型番・注文番号・日時・コード
_PHONE_LIKE_PATTERNS = [
    ("型番 TX-{a4}-{b4} の在庫を確認する。", "型番"),
    ("製品コード {a2}-{b4}-{c4} は生産終了となりました。", "製品コード"),
    ("注文番号 0120-{b4}-{c3} でお問い合わせください。", "注文番号"),
    ("受付整理番号 0800-{b3}-{c4} を控えてください。", "整理番号"),
    ("会議は {y}-{m2}-{d2} に開催する。", "日付"),
    ("契約期間は {y}-{m2}-{d2} から一年間とする。", "日付"),
    ("ISBN 978-4-{b4}-{c4}-1 を参照。", "ISBN"),
    ("追跡番号 {a4}-{b4}-{c4} で配送状況を照会できる。", "追跡番号"),
    ("図面番号 {a3}-{b4} を改訂した。", "図面番号"),
    ("会議室 {a3}-{b4} を予約した。", "会議室番号"),
    ("バージョン {a2}.{b2}.{c2} をリリースした。", "バージョン"),
    ("郵便物番号 {a4}-{b4}-{c4} は追跡対象外です。", "郵便物番号"),
    ("試験区分 {a3}-{b4}-{c4} の合格基準を示す。", "試験区分"),
    ("在庫コード {a2}-{b4}-{c4} を棚卸しした。", "在庫コード"),
    ("測定値は {a4}-{b4} の範囲に収まった。", "測定値"),
]

# 住所に見える施設名 (居住地ではない)
_ADDRESS_LIKE_FACILITY = [
    "新宿区民センター", "中央区役所", "東京駅八重洲口", "西新宿ビル",
    "北海道大学", "県立中央病院", "市立図書館", "川崎市produce会館",
    "横浜赤レンガ倉庫", "大阪城公園", "神戸港旅客ターミナル", "名古屋国際会議場",
    "福岡市総合体育館", "仙台市青葉区役所", "京都府警察本部", "千代田区立日比谷図書文化館",
    "県庁第二庁舎", "地方合同庁舎", "国立国会図書館", "東京国際フォーラム",
    "札幌市時計台", "広島平和記念公園", "沖縄県立博物館", "群馬県森林事務所",
    "港区スポーツセンター",
]
_FACILITY_TEMPLATES = [
    "会場は{x}です。",
    "{x}にて説明会を実施する。",
    "{x}の利用申請を行った。",
    "{x}周辺の駐車場は混雑が予想される。",
    "{x}まで徒歩十分の距離にある。",
]

# 会員番号/口座に見えるが違う数字列
_NUMBER_LIKE_TEMPLATES = [
    "議案第{n2}号について採決を行う。",
    "平成{n2}年法律第{n2b}号に基づき処理する。",
    "売上高は{big}円であった。",
    "座席番号{a2}-{n2}にご着席ください。",
    "本件の予算額は{big}円を計上している。",
    "第{n3}回定例会の議事日程を配布した。",
    "整理番号{n7}は欠番となっている。",
    "統計表の第{n2}表を参照のこと。",
    "在庫数量は{n5}個であった。",
    "延べ{n6}人が来場した。",
    "標高{n4}メートル地点で観測した。",
    "投票総数{n6}票のうち有効票は{n6b}票。",
]

# 生年月日に見えるが生年月日でない日付
_DATE_LIKE_TEMPLATES = [
    "契約日は{ymd}とする。",
    "次回会議は{ymd}に開催する。",
    "発行日{ymd}付で通知した。",
    "{ymd}をもって解散した。",
    "着工予定日は{ymd}である。",
    "{ymd}に施行された規則による。",
    "納品期限は{ymd}です。",
    "{ymd}開催の説明会に参加した。",
    "有効期限は{ymd}まで。",
    "{ymd}時点の残高を確認した。",
    "受付開始は{ymd}午前九時からとする。",
    "{ymd}の株主総会で承認された。",
]

# 否定例文書に混ぜる中立的なつなぎ文
_FILLER = [
    "以下のとおり報告します。",
    "詳細は別紙のとおりです。",
    "ご不明な点がございましたらお知らせください。",
    "引き続きよろしくお願いいたします。",
    "経過については追って共有します。",
    "本件に関する資料を添付しました。",
    "next の対応方針を整理する。",
    "現時点で特段の問題は生じていない。",
    "関係部署と調整のうえ進める。",
    "実施状況を定期的に確認する。",
]


class HardNegativeGenerator:
    """紛らわしい否定例を生成し、誤検出に応じて生成分布を更新する。

    Claim: 低誤検出 — 既存ツールが日本語で誤検出しやすい型を狙って供給し、
    学習で潰し、評価で測る。閉ループにより「まだ弱い型」へ資源を寄せる。
    """

    def __init__(self, seed: int = 0, weights: dict[str, float] | None = None) -> None:
        self.rng = random.Random(seed)
        n = len(NEGATIVE_KINDS)
        self.weights: dict[str, float] = dict(weights) if weights else {
            k: 1.0 / n for k in NEGATIVE_KINDS
        }
        self._normalize_weights()
        #: 生成した表面形 -> kind の索引 (classify_false_positive の一次情報源)
        self.surface_index: dict[str, str] = {}
        self._round = 0

    # -------------------------------------------------------------- weights
    def _normalize_weights(self) -> None:
        tot = sum(self.weights.values()) or 1.0
        self.weights = {k: v / tot for k, v in self.weights.items()}

    def _pick_kind(self) -> str:
        kinds = list(NEGATIVE_KINDS)
        w = [self.weights.get(k, 0.0) for k in kinds]
        if sum(w) <= 0:
            return self.rng.choice(kinds)
        return self.rng.choices(kinds, weights=w, k=1)[0]

    # ------------------------------------------------------------- sampling
    def sample(self, kind: str | None = None) -> NegativeItem:
        """1件の hard negative 文を生成する。

        Claim: 低誤検出 — 「PIIに見える非PII」を構成的に作ることで、
        誤検出率という指標に測定対象を与える。
        """
        kind = kind or self._pick_kind()
        if kind not in NEGATIVE_KINDS:
            raise ValueError(f"unknown negative kind: {kind!r}")
        text, note = getattr(self, f"_gen_{kind}")()
        text = normalize(text)
        for surf in self._surfaces_of(kind, text):
            self.surface_index.setdefault(surf, kind)
        return NegativeItem(text=text, kind=kind, note=note)

    def _surfaces_of(self, kind: str, text: str) -> list[str]:
        """索引に登録すべき「紛らわしい表面形」を取り出す。

        Claim: 低誤検出 — 後で誤検出スパンを型に割り当てるための対応表を作る。
        """
        out: list[str] = []
        for m in re.finditer(r"[0-9][0-9\-.]{4,}[0-9]", text):
            out.append(m.group(0))
        for m in re.finditer(r"[一-龥ァ-ヶА-я]{2,10}", text):
            out.append(m.group(0))
        return out

    # ----------------------------------------------------------- generators
    def _gen_common_noun_surname(self) -> tuple[str, str]:
        word, tmpls = self.rng.choice(_COMMON_NOUN_SURNAME)
        return self.rng.choice(tmpls).format(x=word), f"普通名詞としての「{word}」"

    def _gen_place_as_person(self) -> tuple[str, str]:
        p = self.rng.choice(_PLACE_AS_PERSON)
        return self.rng.choice(_PLACE_TEMPLATES).format(x=p), f"地名としての「{p}」"

    def _gen_company_as_person(self) -> tuple[str, str]:
        c = self.rng.choice(_COMPANY_AS_PERSON)
        return self.rng.choice(_COMPANY_TEMPLATES).format(x=c), f"法人名「{c}」"

    def _gen_honorific_boundary(self) -> tuple[str, str]:
        return self.rng.choice(_HONORIFIC_TRAPS), "人名でない敬称・様/氏/君"

    def _gen_phone_like_id(self) -> tuple[str, str]:
        r = self.rng
        tmpl, note = r.choice(_PHONE_LIKE_PATTERNS)
        vals = {
            "a2": f"{r.randint(10, 99)}", "a3": f"{r.randint(100, 999)}",
            "a4": f"{r.randint(1000, 9999)}",
            "b2": f"{r.randint(10, 99)}", "b3": f"{r.randint(100, 999)}",
            "b4": f"{r.randint(1000, 9999)}",
            "c2": f"{r.randint(10, 99)}", "c3": f"{r.randint(100, 999)}",
            "c4": f"{r.randint(1000, 9999)}",
            "y": f"{r.randint(2018, 2025)}", "m2": f"{r.randint(1, 12):02d}",
            "d2": f"{r.randint(1, 28):02d}",
        }
        return tmpl.format(**vals), f"{note}に見える数字列"

    def _gen_address_like_facility(self) -> tuple[str, str]:
        f = self.rng.choice(_ADDRESS_LIKE_FACILITY)
        return self.rng.choice(_FACILITY_TEMPLATES).format(x=f), f"施設名「{f}」"

    def _gen_number_like_id(self) -> tuple[str, str]:
        r = self.rng
        tmpl = r.choice(_NUMBER_LIKE_TEMPLATES)
        vals = {
            "a2": f"{r.randint(10, 99)}",
            "n2": f"{r.randint(1, 99)}", "n2b": f"{r.randint(1, 99)}",
            "n3": f"{r.randint(100, 999)}", "n4": f"{r.randint(1000, 9999)}",
            "n5": f"{r.randint(10000, 99999)}", "n6": f"{r.randint(100000, 999999)}",
            "n6b": f"{r.randint(100000, 999999)}", "n7": f"{r.randint(1000000, 9999999)}",
            "big": f"{r.randint(1, 999):,}{r.choice(['万', '億', ''])}",
        }
        return tmpl.format(**vals), "PII でない数字列"

    def _gen_date_like_nondob(self) -> tuple[str, str]:
        r = self.rng
        y, m, d = r.randint(2015, 2026), r.randint(1, 12), r.randint(1, 28)
        style = r.randint(0, 3)
        if style == 0:
            ymd = f"{y}年{m}月{d}日"
        elif style == 1:
            ymd = f"{y}/{m:02d}/{d:02d}"
        elif style == 2:
            ymd = f"令和{y - 2018}年{m}月{d}日"
        else:
            ymd = f"{y}-{m:02d}-{d:02d}"
        return r.choice(_DATE_LIKE_TEMPLATES).format(ymd=ymd), "生年月日でない日付"

    # ------------------------------------------------------------- injection
    def inject(self, doc: Document, k: int = 2) -> Document:
        """既存文書に否定例を差し込む。正解スパンは増やさず、座標を正しくずらす。

        Claim: 低誤検出 — 「本物のPIIと紛らわしい非PIIが同居する」現実的な文脈を作る。
        挿入位置以降の既存 gold スパンは挿入長ぶん平行移動させ、
        末尾で ``Document.validate()`` を通すことで座標破壊を機械的に防ぐ。
        """
        text = doc.text
        spans = list(doc.spans)
        kinds = list(doc.negative_kinds)

        for _ in range(max(0, k)):
            item = self.sample()
            # 行境界 (既存スパンを割らない位置) を候補にする
            candidates = [0, len(text)]
            for m in re.finditer(r"\n", text):
                candidates.append(m.start() + 1)
            for m in re.finditer(r"(?<=。)", text):
                candidates.append(m.start())
            safe = [
                p for p in sorted(set(candidates))
                if all(not (s.start < p < s.end) for s in spans)
            ]
            if not safe:
                break
            pos = self.rng.choice(safe)
            ins = item.text
            if pos > 0 and not text[pos - 1 : pos] == "\n":
                ins = ins if text[pos - 1 : pos] in "。\n" else ins
            if not ins.endswith(("\n", "。")):
                ins += ""
            # 挿入
            text = text[:pos] + ins + text[pos:]
            shift = len(ins)
            spans = [
                s.with_(start=s.start + shift, end=s.end + shift) if s.start >= pos else s
                for s in spans
            ]
            kinds.append(item.kind)

        out = Document(
            text=text, spans=spans, doc_id=doc.doc_id, subset=doc.subset,
            genre=doc.genre, source_license=doc.source_license,
            source_ref=doc.source_ref, negative_kinds=kinds, meta=dict(doc.meta),
        )
        out.validate()
        return out

    def build_negative_documents(
        self,
        n: int,
        *,
        base_items: Sequence[object] | None = None,
        subset: str = "negatives",
        min_sent: int = 3,
        max_sent: int = 8,
        id_prefix: str = "neg",
    ) -> list[Document]:
        """正解スパン0件の否定例文書を n 件作る (誤検出率の測定対象)。

        Claim: 低誤検出 — この部分集合で出た検出は **すべて誤検出** である。
        したがって誤検出率が定義でき、閾値固定時の検出率も算定できる。
        """
        docs: list[Document] = []
        bases = list(base_items or [])
        for i in range(n):
            n_sent = self.rng.randint(min_sent, max_sent)
            kinds: list[str] = []
            parts: list[str] = []
            for j in range(n_sent):
                if self.rng.random() < 0.72:
                    item = self.sample()
                    parts.append(item.text)
                    kinds.append(item.kind)
                elif bases and self.rng.random() < 0.5:
                    b = self.rng.choice(bases)
                    bt = getattr(b, "text", str(b))
                    sent = bt.split("\n")[0][:160]
                    if sent:
                        parts.append(sent if sent.endswith("。") else sent + "。")
                else:
                    parts.append(self.rng.choice(_FILLER))
            text = normalize("".join(parts))
            doc = Document(
                text=text, spans=[], doc_id=f"{id_prefix}-{i:05d}", subset=subset,
                genre="negatives", source_license="synthetic (CC0-1.0)",
                source_ref="", negative_kinds=kinds,
                meta={"n_sent": n_sent},
            )
            doc.validate()
            docs.append(doc)
        return docs

    # ------------------------------------------------------------ closed loop
    def reweight_from_errors(
        self, fp_counts: dict[str, int], *, strength: float = 1.0
    ) -> dict[str, float]:
        """誤検出の多い型へ生成分布を寄せる (閉ループ)。

        Claim: 低誤検出 — 学習後に残った誤りの型を測り、次バッチの生成を
        その型に偏らせることで、弱点を狙って潰す。

        更新式 (乗法更新 + 正規化 + クリップ):
            share_k = fp_k / (Σ fp + ε)
            w_k <- w_k * (1 + strength * share_k)
            w   <- w / Σw
            w_k <- clip(w_k, 0.25/K, 4.0/K) して再正規化
        クリップにより、どの型も枯渇せず (下限)、単一の型が支配もしない (上限)。
        """
        eps = 1e-9
        total = sum(max(0, v) for v in fp_counts.values())
        K = len(NEGATIVE_KINDS)
        new = dict(self.weights)
        for k in NEGATIVE_KINDS:
            share = max(0, fp_counts.get(k, 0)) / (total + eps)
            new[k] = new.get(k, 1.0 / K) * (1.0 + strength * share)
        tot = sum(new.values()) or 1.0
        new = {k: v / tot for k, v in new.items()}
        lo, hi = 0.25 / K, 4.0 / K
        new = {k: min(hi, max(lo, v)) for k, v in new.items()}
        tot = sum(new.values()) or 1.0
        self.weights = {k: v / tot for k, v in new.items()}
        self._round += 1
        return dict(self.weights)


# ---------------------------------------------------------------- classifier

_KIND_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("phone_like_id", re.compile(r"型番|製品コード|注文番号|整理番号|ISBN|追跡番号|図面番号|会議室|バージョン|在庫コード|試験区分|測定値|郵便物番号")),
    ("date_like_nondob", re.compile(r"契約日|次回会議|発行日|着工|施行|納品期限|有効期限|株主総会|説明会|時点の残高|受付開始|解散")),
    ("number_like_id", re.compile(r"議案第|法律第|売上高|座席番号|予算額|定例会|統計表|在庫数量|延べ|標高|投票総数|欠番")),
    ("address_like_facility", re.compile(r"センター|区役所|市役所|県庁|庁舎|図書館|病院|大学|会館|公園|ターミナル|体育館|博物館|警察|フォーラム|倉庫|時計台|事務所|駅")),
    ("company_as_person", re.compile(r"株式会社|工業|商事|製作所|製菓|製パン|建設|重工|物産|証券|建機|ホールディングス|工務店|コンストラクション|ハウス|電器|貴金属")),
    ("honorific_boundary", re.compile(r"各位|皆様|お客様|神様|王様|奥様|御中|担当者様|先方様|殿様|様式|同様|多様|仕様|模様|御中元|氏名欄|氏族|君主|諸君|先生方|衛生|部長会議|課長職|社長就任")),
    ("place_as_person", re.compile(r"県|市内|支店|県産|方面|地方|移転|復興")),
]


def classify_false_positive(span: Span, doc: Document) -> str:
    """誤検出スパンがどの否定例型に当たるかを推定する (閉ループの入力)。

    Claim: 低誤検出 — 誤りを型別に集計できて初めて、
    「どの紛らわしさに弱いか」を測り、生成を偏らせる閉ループが回る。

    優先順位:
      1. 生成時に作った表面形索引 (``generator.surface_index``) の完全一致
      2. スパン周辺の文脈語による正規表現判定
      3. 文書に付与された ``negative_kinds`` が1種類ならそれ
      4. ``"other"``
    """
    surf = span.text or span.slice_of(doc.text)
    idx = doc.meta.get("_surface_index") if isinstance(doc.meta, dict) else None
    if isinstance(idx, dict) and surf in idx:
        return idx[surf]

    a = max(0, span.start - 24)
    b = min(len(doc.text), span.end + 24)
    ctx = doc.text[a:b]
    for kind, pat in _KIND_PATTERNS:
        if pat.search(ctx):
            return kind

    # 普通名詞と同形の姓
    for word, _ in _COMMON_NOUN_SURNAME:
        if surf == word:
            return "common_noun_surname"
    if surf in _PLACE_AS_PERSON:
        return "place_as_person"

    kinds = list(dict.fromkeys(doc.negative_kinds or []))
    if len(kinds) == 1:
        return kinds[0]
    return "other"


def attach_surface_index(docs: Iterable[Document], gen: HardNegativeGenerator) -> None:
    """生成器の表面形索引を各文書の meta に埋め込む。

    Claim: 低誤検出 — 誤検出の型判定を推測ではなく生成時の真実に基づかせる。
    """
    idx = dict(gen.surface_index)
    for d in docs:
        d.meta["_surface_index"] = idx


# ------------------------------------------------------------------ selftest

def _selftest() -> None:
    """自己テスト。

    Claim: 低誤検出 — 否定例生成・座標保存・閉ループが壊れていないことを確認する。
    """
    from collections import Counter

    print("=" * 72)
    print("sumi.negatives 自己テスト")
    print("=" * 72)

    gen = HardNegativeGenerator(seed=0)
    docs = gen.build_negative_documents(200)
    assert all(len(d.spans) == 0 for d in docs), "否定例文書に gold span があってはならない"
    hist = Counter(k for d in docs for k in d.negative_kinds)
    print(f"否定例文書 {len(docs)} 件 / 混入 {sum(hist.values())} 表現 / 平均文字数 "
          f"{sum(len(d.text) for d in docs)//len(docs)}")
    print("-" * 46)
    for k in NEGATIVE_KINDS:
        c = hist.get(k, 0)
        print(f"  {k:24s} {c:4d} {'#' * int(40 * c / max(hist.values()))}")
    print("-" * 46)

    print("\n[例] 各型から1件ずつ")
    g2 = HardNegativeGenerator(seed=7)
    for k in NEGATIVE_KINDS:
        it = g2.sample(k)
        print(f"  {k:24s} {it.text}")

    # --- inject が gold 座標を壊さないこと ---
    print("\n[inject] 既存 gold スパンの座標保存")
    from sumi.synth import build_documents

    base = build_documents(12, seed=5)
    ok = 0
    for d in base:
        before = [(s.label.value, s.text) for s in d.sorted_spans()]
        nd = HardNegativeGenerator(seed=len(d.text)).inject(d, k=3)
        nd.validate()  # ここで text[start:end] == span.text が強制される
        after = [(s.label.value, s.text) for s in nd.sorted_spans()]
        assert before == after, f"{d.doc_id}: gold の中身が変わった"
        assert len(nd.text) > len(d.text), "挿入されていない"
        ok += 1
    print(f"  {ok}/{len(base)} 文書で validate() 通過・gold の中身が不変 ✓")
    sample = HardNegativeGenerator(seed=3).inject(base[0], k=2)
    print(f"  例: {base[0].doc_id} {len(base[0].text)}字 -> {len(sample.text)}字, "
          f"混入 {sample.negative_kinds}")

    # --- 誤検出の型判定 ---
    print("\n[classify] 誤検出の型判定")
    g3 = HardNegativeGenerator(seed=11)
    nd = g3.build_negative_documents(60)
    attach_surface_index(nd, g3)
    hits = Counter()
    for d in nd[:30]:
        m = re.search(r"[一-龥]{2,6}", d.text)
        if m:
            sp = Span(m.start(), m.end(), __import__("sumi.types", fromlist=["PIIType"]).PIIType.NAME,
                      m.group(0))
            hits[classify_false_positive(sp, d)] += 1
    print("  推定内訳:", dict(hits))
    assert sum(v for k, v in hits.items() if k != "other") > 0, "型判定が全く効いていない"

    # --- 閉ループ ---
    print("\n[閉ループ] reweight_from_errors")
    g4 = HardNegativeGenerator(seed=1)
    before_w = dict(g4.weights)
    fp = {"phone_like_id": 40, "common_noun_surname": 25, "place_as_person": 10,
          "honorific_boundary": 3, "company_as_person": 0, "address_like_facility": 0,
          "number_like_id": 2, "date_like_nondob": 0}
    after_w = g4.reweight_from_errors(fp, strength=2.0)
    print(f"  {'kind':24s} {'fp':>4s} {'before':>8s} {'after':>8s}  変化")
    for k in NEGATIVE_KINDS:
        d = after_w[k] - before_w[k]
        print(f"  {k:24s} {fp.get(k,0):4d} {before_w[k]:8.4f} {after_w[k]:8.4f}  {d:+.4f}")
    assert after_w["phone_like_id"] > before_w["phone_like_id"], "誤検出の多い型が増えていない"
    assert after_w["date_like_nondob"] < before_w["date_like_nondob"], "誤検出0の型が減っていない"
    assert abs(sum(after_w.values()) - 1.0) < 1e-9, "重みが正規化されていない"
    lo, hi = 0.25 / len(NEGATIVE_KINDS), 4.0 / len(NEGATIVE_KINDS)
    assert all(lo - 1e-9 <= v <= hi + 1e-9 for v in after_w.values()), "クリップが効いていない"

    # 分布が実際に偏るか
    g5 = HardNegativeGenerator(seed=2, weights=after_w)
    drawn = Counter(g5.sample().kind for _ in range(2000))
    print(f"  更新後2000サンプルの実分布 上位3: {drawn.most_common(3)}")
    assert drawn["phone_like_id"] > drawn["date_like_nondob"], "実サンプリングに反映されていない"

    print("\n" + "=" * 72)
    print("すべての自己テストに合格 (否定例生成 / 座標保存 / 型判定 / 閉ループ)")
    print("=" * 72)


if __name__ == "__main__":
    _selftest()
