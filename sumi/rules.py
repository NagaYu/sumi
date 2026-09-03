"""RuleLayer — 形式が確定している種別を規則で高精度に拾い、モデル出力と統合する。

Claim: 低誤検出 / 検出率 — 電話番号・メール・口座・番号体系のように書式が
決まっている種別は、文脈判断より正規表現のほうが精度も速度も上回る。
本モジュールはそれらを高信頼で拾い、**規則が確実な箇所は規則を優先し、
それ以外はモデルに委ねる** という明示的な優先順位で統合する。

重要な設計判断 (契約 規則2):
    チェックディジットの成否は ``meta["checksum_valid"]`` に **記録するだけ** で、
    検出可否の条件には **しない**。墨消しの目的は「それらしいものを残さないこと」
    であり、検査数字が合わないカード番号様式を見逃してよい理由はないため。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from sumi.types import (
    MODEL_DRIVEN,
    RULE_DETERMINISTIC,
    PIIType,
    Source,
    Span,
    normalize,
)

# ---------------------------------------------------------------- checksums


def luhn_ok(digits: str) -> bool:
    """Luhn チェック (クレジットカード様式) を検証する。

    Claim: 低誤検出 — 検出はしたうえで「値として有効か」を注記するために使う。
    合成データは形式のみ正しく値は無効に作ってあるので、通常 False になる。
    """
    ds = [int(c) for c in digits if c.isdigit()]
    if len(ds) < 12:
        return False
    total = 0
    for i, d in enumerate(reversed(ds)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def mynumber_check_ok(digits: str) -> bool:
    """マイナンバー様式12桁の検査用数字を検証する (総務省の算式)。

    Claim: 低誤検出 — 同上。検出のゲートには使わず、注記のみに用いる。

    算式: 下位11桁を C1..C11 (右から1始まり)、
          Pn = n+1 (1<=n<=6), Pn = n-5 (7<=n<=11)
          check = 11 - (Σ Pn*Cn mod 11)、11以上なら 0。
    """
    ds = [int(c) for c in digits if c.isdigit()]
    if len(ds) != 12:
        return False
    check = ds[-1]
    payload = ds[:-1]           # C11..C1 の並び (左が上位)
    total = 0
    for n in range(1, 12):      # n = 1..11 (右から)
        cn = payload[-n]
        pn = n + 1 if n <= 6 else n - 5
        total += pn * cn
    r = total % 11
    expect = 0 if r <= 1 else 11 - r
    return check == expect


# --------------------------------------------------------------- phone rules

#: 2桁市外局番 (0 を含む)
_AREA2 = {"03", "06"}
#: 3桁市外局番 (代表的なもの)
_AREA3 = {
    "011", "015", "017", "018", "019", "022", "023", "024", "025", "026", "027",
    "028", "029", "042", "043", "044", "045", "046", "047", "048", "049", "052",
    "053", "054", "055", "058", "059", "072", "073", "075", "076", "077", "078",
    "079", "082", "083", "084", "086", "087", "088", "089", "092", "093", "095",
    "096", "097", "098", "099",
}
_MOBILE = {"070", "080", "090"}
_IP = {"050"}
_TOLLFREE4 = {"0120", "0800", "0570"}


def _digits(s: str) -> str:
    return "".join(c for c in s if c.isdigit())


def is_valid_jp_phone(s: str) -> bool:
    """日本の番号計画に照らして電話番号として妥当かを判定する。

    Claim: 低誤検出 — 「電話番号に見える型番・注文番号・日時」を落とすための
    中核判定。総桁数 (固定 10 桁 / 携帯・IP・0800 は 11 桁) と
    先頭の番号帯を両方満たすものだけを通す。
    """
    d = _digits(s)
    if not d.startswith("0"):
        return False
    if len(d) not in (10, 11):
        return False
    if d[:3] in _MOBILE or d[:3] in _IP:
        return len(d) == 11
    if d[:4] in _TOLLFREE4:
        if d[:4] == "0120":
            return len(d) == 10
        if d[:4] == "0800":
            return len(d) == 11
        return len(d) == 10          # 0570
    if d[:2] in _AREA2:
        return len(d) == 10
    if d[:3] in _AREA3:
        return len(d) == 10
    # 4桁市外局番 (0xxx) — 総桁 10 桁のみ許容
    return len(d) == 10


_DATE_RE = re.compile(r"^(19|20)\d{2}[-/.](0?[1-9]|1[0-2])[-/.](0?[1-9]|[12]\d|3[01])$")


def _looks_like_date(s: str) -> bool:
    """``2024-01-15`` のような日付形状かどうか。

    Claim: 低誤検出 — 日付を電話番号として拾う典型的な誤りを塞ぐ。
    """
    return bool(_DATE_RE.match(s.strip()))


# ------------------------------------------------------------------ specs


@dataclass
class RuleSpec:
    """1つの検出規則。

    Claim: 低誤検出 — 規則を宣言的に持ち、``rule_id`` で誤りの出所を追跡できるようにする。

    Attributes:
        rule_id: 一意な規則名 (meta に記録され、誤り分析に使う)。
        label: 検出する PII 種別。
        pattern: 正規表現。捕捉は group(0) か名前付き group ``val`` を使う。
        confidence: 文脈語が無いときの基礎スコア。
        require_context: True なら文脈語が無い一致を破棄する。
        context: 加点する文脈語。
        negative_context: これが近傍にあれば **破棄** する語 (型番・注文番号など)。
        priority: 重なり解消時の優先度 (大きいほど強い)。
        validator: 追加検証関数 (文字列 -> bool)。
    """

    rule_id: str
    label: PIIType
    pattern: str
    confidence: float = 0.9
    require_context: bool = False
    context: tuple[str, ...] = ()
    negative_context: tuple[str, ...] = ()
    priority: int = 0
    validator: object = None
    _re: re.Pattern[str] | None = field(default=None, repr=False, compare=False)

    def compiled(self) -> re.Pattern[str]:
        """正規表現をコンパイルして返す (キャッシュ付き)。

        Claim: CPU速度 — 規則層は文書ごとに何度も走るため、コンパイルを1回に抑える。
        """
        if self._re is None:
            object.__setattr__(self, "_re", re.compile(self.pattern))
        return self._re


#: 数字列が他の英数字トークンに埋まっていないことを要求する共通の境界
_NB = r"(?<![0-9A-Za-z\-])"
_NA = r"(?![0-9A-Za-z\-])"

_PHONE_CTX = ("TEL", "Tel", "tel", "電話", "でんわ", "連絡先", "携帯", "FAX", "Fax",
              "お電話", "番号", "ご連絡", "直通", "代表")
_PHONE_NEG = ("型番", "製品コード", "注文番号", "整理番号", "追跡番号", "ISBN",
              "図面番号", "会議室", "在庫コード", "試験区分", "バージョン",
              "郵便物番号", "測定値", "商品コード", "管理番号", "伝票")

DEFAULT_SPECS: tuple[RuleSpec, ...] = (
    RuleSpec(
        rule_id="email.basic",
        label=PIIType.EMAIL,
        # 日本語句読点を飲み込まないよう ASCII に限定する
        pattern=r"(?<![A-Za-z0-9._%+\-])[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?)+",
        confidence=0.99, priority=90,
    ),
    RuleSpec(
        rule_id="phone.jp",
        label=PIIType.PHONE,
        pattern=(_NB + r"(?:\+81[\-\s]?)?0\d{1,4}[\-\(\)\s]?\d{1,4}[\-\)\s]?\d{3,4}" + _NA),
        confidence=0.80, context=_PHONE_CTX, negative_context=_PHONE_NEG,
        priority=80, validator=is_valid_jp_phone,
    ),
    RuleSpec(
        rule_id="postal.jp",
        label=PIIType.POSTAL_CODE,
        pattern=_NB + r"\d{3}-\d{4}" + _NA,
        confidence=0.55, context=("〒", "郵便番号", "住所", "所在地", "送付先", "ご住所"),
        require_context=True, priority=70,
    ),
    RuleSpec(
        rule_id="bank.code_branch_account",
        label=PIIType.BANK_ACCOUNT,
        pattern=_NB + r"\d{4}-\d{3}-\d{7}" + _NA,
        confidence=0.95, context=("口座", "振込", "支店", "普通", "当座", "引落", "銀行"),
        priority=85,
    ),
    RuleSpec(
        rule_id="bank.prose",
        label=PIIType.BANK_ACCOUNT,
        pattern=r"(?:普通|当座|貯蓄)\s*(?:預金)?\s*" + _NB + r"\d{7}" + _NA,
        confidence=0.92, priority=84,
    ),
    RuleSpec(
        rule_id="card.16digit",
        label=PIIType.CREDIT_CARD,
        pattern=_NB + r"(?:\d{4}[\-\s]\d{4}[\-\s]\d{4}[\-\s]\d{4}|\d{16})" + _NA,
        confidence=0.93, priority=88,
    ),
    RuleSpec(
        rule_id="card.amex",
        label=PIIType.CREDIT_CARD,
        pattern=_NB + r"3[47]\d{2}[\-\s]\d{6}[\-\s]\d{5}" + _NA,
        confidence=0.93, priority=88,
    ),
    RuleSpec(
        rule_id="mynumber.12digit",
        label=PIIType.MYNUMBER,
        pattern=_NB + r"(?:\d{4}[\-\s]\d{4}[\-\s]\d{4}|\d{12})" + _NA,
        confidence=0.70,
        # 文脈語に裸の「番号」を入れてはならない (追跡番号・郵便物番号に反応する)
        context=("マイナンバー", "個人番号", "通知カード", "本人確認", "マイナ"),
        negative_context=("追跡番号", "郵便物番号", "注文番号", "型番", "整理番号",
                          "伝票", "管理番号", "在庫", "図面番号", "試験区分",
                          "製品コード", "商品コード", "会議室", "ISBN"),
        require_context=True,
        priority=86,
    ),
    RuleSpec(
        rule_id="member.prefixed",
        label=PIIType.MEMBER_ID,
        pattern=r"(?<![A-Za-z0-9])[A-Z]{2,6}[\-]?\d{4,12}(?:[\-]\d{2,6})?(?![A-Za-z0-9])",
        confidence=0.60,
        context=("会員番号", "顧客番号", "契約番号", "受付番号", "お客様番号", "整理番号",
                 "会員ID", "顧客ID", "社員番号", "登録番号", "カード番号"),
        require_context=True, priority=60,
    ),
    RuleSpec(
        rule_id="member.numeric_ctx",
        label=PIIType.MEMBER_ID,
        pattern=_NB + r"\d{6,12}(?:-\d{2,6})?" + _NA,
        confidence=0.55,
        context=("会員番号", "顧客番号", "契約番号", "お客様番号", "社員番号", "会員ID"),
        require_context=True, priority=55,
    ),
)


class RuleLayer:
    """形式確定型の PII を正規表現で拾う層。

    Claim: 低誤検出 / CPU速度 — モデルを介さずに済む種別を規則で確定させることで、
    誤検出を抑えつつ、推論コストも下げる。
    """

    def __init__(
        self,
        *,
        types: Iterable[PIIType] | None = None,
        context_window: int = 12,
        specs: Sequence[RuleSpec] | None = None,
    ) -> None:
        self.context_window = context_window
        allowed = set(types) if types is not None else set(RULE_DETERMINISTIC)
        self.specs = [s for s in (specs or DEFAULT_SPECS) if s.label in allowed]

    # ------------------------------------------------------------- internals
    def _context_of(self, text: str, start: int, end: int) -> str:
        """一致箇所の前後の文脈 (加点用)。

        Claim: 低誤検出 — 「TEL」「口座」等の手掛かり語は前後どちらにも現れうる。
        """
        a = max(0, start - self.context_window)
        b = min(len(text), end + self.context_window)
        return text[a:b]

    def _left_context_of(self, text: str, start: int) -> str:
        """一致箇所の **左側だけ** の文脈 (否定語の判定用)。

        Claim: 低誤検出 — 「型番」「注文番号」「追跡番号」のようなラベルは
        値の **前** に置かれる。後方まで見ると、直後に無関係な型番が続くだけで
        正当な電話番号を落としてしまう (実際に落ちた)。左側限定にすることで、
        誤検出を抑えつつ取りこぼしを防ぐ。
        """
        a = max(0, start - self.context_window)
        return text[a:start]

    def _candidates(self, text: str) -> list[Span]:
        out: list[Span] = []
        for spec in self.specs:
            for m in spec.compiled().finditer(text):
                s, e = m.span()
                val = m.group(0)
                if spec.validator is not None and not spec.validator(val):
                    continue
                if spec.label is PIIType.PHONE and _looks_like_date(val):
                    continue
                ctx = self._context_of(text, s, e)
                left = self._left_context_of(text, s)
                if any(w in left for w in spec.negative_context):
                    continue
                hit = next((w for w in spec.context if w in ctx), None)
                if spec.require_context and hit is None:
                    continue
                score = min(0.99, spec.confidence + (0.15 if hit else 0.0))
                meta: dict = {
                    "rule_id": spec.rule_id,
                    "matched_context": hit,
                    "priority": spec.priority,
                }
                if spec.label is PIIType.CREDIT_CARD:
                    meta["checksum_valid"] = luhn_ok(val)
                elif spec.label is PIIType.MYNUMBER:
                    meta["checksum_valid"] = mynumber_check_ok(val)
                out.append(
                    Span(start=s, end=e, label=spec.label, text=val,
                         score=score, source=Source.RULE, meta=meta)
                )
        return out

    @staticmethod
    def _resolve(cands: list[Span]) -> list[Span]:
        """重なる候補を priority -> score -> 長さ の順で解消する。

        Claim: 低誤検出 — 同じ数字列に複数規則が当たったとき、
        重複を誤検出として二重計上しないようにする。
        """
        chosen: list[Span] = []
        ordered = sorted(
            cands,
            key=lambda s: (-int(s.meta.get("priority", 0)), -s.score, -(s.end - s.start), s.start),
        )
        for s in ordered:
            if all(not s.overlaps(c) for c in chosen):
                chosen.append(s)
        return sorted(chosen, key=lambda s: (s.start, s.end))

    # ---------------------------------------------------------------- public
    def detect(self, text: str) -> list[Span]:
        """規則で検出したスパンを返す (非重複・start 昇順)。

        Claim: 低誤検出 / 検出率 — 形式確定型を取りこぼさず、かつ
        紛らわしい数字列を文脈語と桁数規則で退ける。
        """
        return self._resolve(self._candidates(text))

    def explain(self, text: str) -> list[dict]:
        """各検出について、どの規則がなぜ当たったかを説明する。

        Claim: 低誤検出 — 誤検出の原因を規則単位で特定できるようにする。
        """
        rows = []
        for s in self.detect(text):
            rows.append(
                {
                    "text": s.text,
                    "label": s.label.value,
                    "rule_id": s.meta.get("rule_id"),
                    "score": round(s.score, 3),
                    "context": s.meta.get("matched_context"),
                    "checksum_valid": s.meta.get("checksum_valid"),
                    "start": s.start,
                    "end": s.end,
                }
            )
        return rows


# -------------------------------------------------------------------- merge


def merge_spans(
    model_spans: Sequence[Span],
    rule_spans: Sequence[Span],
    text: str,
    *,
    rule_types: Iterable[PIIType] = RULE_DETERMINISTIC,
    min_model_score: float = 0.0,
) -> list[Span]:
    """規則出力とモデル出力を、明示的な優先順位で統合する。

    Claim: 低誤検出 / 検出率 — 「規則が確実な箇所は規則を優先、それ以外はモデル」
    という方針を、暗黙のスコア比較ではなく **順序の決まった5段階** として実装する。
    この明示性そのものが成果物であり、README はこの手順を引用する。

    優先順位:
        1. ``rule_types`` に属する規則スパンは常に採用する。
        2. 採用済みの規則スパンと重なるモデルスパンは破棄する
           (同種別でも異種別でも、規則を優先)。
        3. 残ったモデルスパンのうち ``min_model_score`` 以上のものを採用する。
        4. モデル同士が重なる場合は score の高い方を残す。
        5. 結果は start 昇順・非重複。採用されたスパンには ``Source.MERGED`` を付け、
           規則由来には ``meta["from"]="rule"``、モデル由来には ``"model"`` を残す。
    """
    out: list[Span] = []
    rule_types = set(rule_types)

    # --- 1. 規則スパンを無条件に採用 ---
    accepted_rules: list[Span] = []
    for s in sorted(rule_spans, key=lambda s: (s.start, s.end)):
        if s.label not in rule_types:
            continue
        if any(s.overlaps(a) for a in accepted_rules):
            continue
        accepted_rules.append(s)
    for s in accepted_rules:
        meta = dict(s.meta)
        meta["from"] = "rule"
        out.append(s.with_(source=Source.MERGED, meta=meta))

    # --- 2. 規則と重なるモデルスパンを破棄 ---
    survivors = [m for m in model_spans if not any(m.overlaps(r) for r in accepted_rules)]

    # --- 3. スコア下限で足切り ---
    survivors = [m for m in survivors if m.score >= min_model_score]

    # --- 4. モデル同士の重なりは score 優先で解消 ---
    kept: list[Span] = []
    for m in sorted(survivors, key=lambda s: (-s.score, -(s.end - s.start), s.start)):
        if all(not m.overlaps(k) for k in kept):
            kept.append(m)
    for m in kept:
        meta = dict(m.meta)
        meta["from"] = "model"
        out.append(m.with_(source=Source.MERGED, meta=meta))

    # --- 5. 整列して返す ---
    return sorted(out, key=lambda s: (s.start, s.end))


# ------------------------------------------------------------------ selftest


_POSITIVE_CASES = [
    ("お電話 03-1234-5678 までご連絡ください。", PIIType.PHONE, "03-1234-5678"),
    ("携帯 090-1234-5678 に連絡した。", PIIType.PHONE, "090-1234-5678"),
    ("TEL 0463-12-3456", PIIType.PHONE, "0463-12-3456"),
    ("フリーダイヤル 0120-123-456 へ", PIIType.PHONE, "0120-123-456"),
    ("連絡先 0800-123-4567", PIIType.PHONE, "0800-123-4567"),
    ("IP電話 050-1234-5678", PIIType.PHONE, "050-1234-5678"),
    ("電話は 045-123-4567 です", PIIType.PHONE, "045-123-4567"),
    ("メールは taro.yamada@example.co.jp です。", PIIType.EMAIL, "taro.yamada@example.co.jp"),
    ("連絡先: a_b-c@example.test に送付。", PIIType.EMAIL, "a_b-c@example.test"),
    ("ご住所 〒160-0023 東京都", PIIType.POSTAL_CODE, "160-0023"),
    ("引落口座 0001-234-5678901 を登録", PIIType.BANK_ACCOUNT, "0001-234-5678901"),
    ("普通 1234567 名義人", PIIType.BANK_ACCOUNT, "普通 1234567"),
    ("カード番号 4111 1111 1111 1112", PIIType.CREDIT_CARD, "4111 1111 1111 1112"),
    ("マイナンバー 1234 5678 9012 を確認", PIIType.MYNUMBER, "1234 5678 9012"),
    ("個人番号 123456789013", PIIType.MYNUMBER, "123456789013"),
    ("会員番号 EMP32301241 の照会", PIIType.MEMBER_ID, "EMP32301241"),
    ("顧客番号 00012345678 について", PIIType.MEMBER_ID, "00012345678"),
]

_NEGATIVE_CASES = [
    "型番 TX-2024-0355 の在庫を確認する。",
    "製品コード 03-1234-5678 は生産終了となりました。",
    "注文番号 0120-8834-221 でお問い合わせください。",
    "受付整理番号 0800-123-4567 を控えてください。",
    "会議は 2024-01-15 に開催する。",
    "契約期間は 2021-02-18 から一年間とする。",
    "ISBN 978-4-1234-5678-1 を参照。",
    "追跡番号 1234-5678-9012 で配送状況を照会できる。",
    "図面番号 123-4567 を改訂した。",
    "会議室 101-2024 を予約した。",
    "バージョン 12.34.56 をリリースした。",
    "議案第12号について採決を行う。",
    "平成15年法律第57号に基づき処理する。",
    "売上高は1,234万円であった。",
    "延べ713984人が来場した。",
    "標高2345メートル地点で観測した。",
    "投票総数123456票のうち有効票は123400票。",
    "座席番号12-34にご着席ください。",
    "在庫数量は12345個であった。",
    "第123回定例会の議事日程を配布した。",
    "森の中を歩くと鳥の声が聞こえる。",
    "林業の振興に関する予算を計上する。",
    "長野県の気候は寒暖差が大きい。",
    "本田技研工業が新製品を発表した。",
    "お客様各位におかれましては、平素より格別のご高配を賜り。",
    "会場は仙台市青葉区役所です。",
    "発行日2023-03-10付で通知した。",
    "次回会議は2025/06/12に開催する。",
    "測定値は1234-5678 の範囲に収まった。",
    "統計表の第12表を参照のこと。",
    "郵便物番号 1234-5678-9012 は追跡対象外です。",
    "試験区分 123-4567-8901 の合格基準を示す。",
    "本件の予算額は567億円を計上している。",
    "有効期限は2026-03-31まで。",
    "着工予定日は2024/09/01である。",
    "バージョン 3.14.15 に更新した。",
    "国有林の管理計画を見直す。",
    "温泉街の景観を保全する。",
    "大和政権の成立について述べる。",
    "青木の生垣を剪定した。",
]


def _selftest() -> None:
    """自己テスト。

    Claim: 低誤検出 / 検出率 — 形式確定型を取りこぼさないことと、
    紛らわしい否定例で誤検出しないことを、同時に確認する。
    """
    print("=" * 74)
    print("sumi.rules 自己テスト")
    print("=" * 74)

    # --- チェックディジット ---
    print("\n[checksum] 手計算例との照合")
    assert luhn_ok("4111111111111111"), "Luhn 正例が通らない"
    assert not luhn_ok("4111111111111112"), "Luhn 誤例が通ってしまう"
    assert luhn_ok("5500 0000 0000 0004"), "区切りあり Luhn 正例"
    # マイナンバー: 検査数字を総当たりで1つだけ正解があることを確認
    base = "12345678901"
    valid = [d for d in range(10) if mynumber_check_ok(base + str(d))]
    assert len(valid) == 1, f"検査数字の正解が {len(valid)} 個 (1個であるべき)"
    print(f"  luhn_ok      : 4111111111111111 -> True / ...1112 -> False ✓")
    print(f"  mynumber_ok  : {base}? の正解は末尾 {valid[0]} のみ ✓")

    layer = RuleLayer()

    # --- 正例 ---
    print("\n[positive] 形式確定型の取りこぼし検査")
    miss = []
    for text, label, want in _POSITIVE_CASES:
        text = normalize(text)
        want = normalize(want)
        got = [s for s in layer.detect(text) if s.label is label]
        hit = any(want in s.text or s.text in want for s in got)
        mark = "✓" if hit else "✗"
        if not hit:
            miss.append((text, label.value, want, [(s.label.value, s.text) for s in layer.detect(text)]))
        print(f"  {mark} {label.value:12s} {want!r:32s} <- {text[:34]!r}")
    if miss:
        print("\n  取りこぼし:")
        for t, l, w, g in miss:
            print(f"    {l} {w!r} in {t!r} -> got {g}")

    # --- 否定例 ---
    print("\n[negative] 紛らわしい否定例での誤検出検査")
    fp_total = 0
    for text in _NEGATIVE_CASES:
        text = normalize(text)
        got = layer.detect(text)
        if got:
            fp_total += len(got)
            print(f"  ✗ FP {[(s.label.value, s.text, s.meta.get('rule_id')) for s in got]}"
                  f"  <- {text[:40]!r}")
    print(f"\n  否定例 {len(_NEGATIVE_CASES)} 件中の誤検出スパン数: {fp_total}")
    print(f"  規則層の否定例あたり誤検出率: {fp_total/len(_NEGATIVE_CASES):.3f} 件/文書")

    # --- 検出はするが checksum は無効、という設計の確認 ---
    print("\n[checksum を検出条件にしない] 設計確認")
    t = normalize("カード番号 4111 1111 1111 1112、マイナンバー 1234 5678 9012 です。")
    got = layer.detect(t)
    cc = [s for s in got if s.label is PIIType.CREDIT_CARD]
    mn = [s for s in got if s.label is PIIType.MYNUMBER]
    assert cc, "Luhn 不一致のカード番号様式が検出されていない (設計違反)"
    assert mn, "検査数字不一致のマイナンバー様式が検出されていない (設計違反)"
    print(f"  CREDIT_CARD 検出={bool(cc)} checksum_valid={cc[0].meta['checksum_valid']} (False でも検出する) ✓")
    print(f"  MYNUMBER    検出={bool(mn)} checksum_valid={mn[0].meta['checksum_valid']} (False でも検出する) ✓")

    # --- merge_spans ---
    print("\n[merge_spans] 明示的優先順位の確認")
    text = normalize("田中太郎様の電話は 090-1234-5678 です。")
    rule = layer.detect(text)
    ph = next(s for s in rule if s.label is PIIType.PHONE)
    model = [
        Span(0, 4, PIIType.NAME, "田中太郎", 0.97, Source.MODEL),
        # 規則と重なる誤ったモデル出力 (種別も違う) — 規則が勝つべき
        Span(ph.start, ph.end - 2, PIIType.MEMBER_ID, text[ph.start:ph.end - 2], 0.99, Source.MODEL),
        # モデル同士の重なり — score が高い方が残るべき
        Span(0, 2, PIIType.NAME, "田中", 0.55, Source.MODEL),
    ]
    merged = merge_spans(model, rule, text)
    print(f"  入力: model={[(s.label.value, s.text, s.score) for s in model]}")
    print(f"        rule ={[(s.label.value, s.text) for s in rule]}")
    print(f"  統合: {[(s.label.value, s.text, s.meta.get('from')) for s in merged]}")
    assert any(s.label is PIIType.PHONE and s.meta.get("from") == "rule" for s in merged), "規則が優先されていない"
    assert not any(s.label is PIIType.MEMBER_ID for s in merged), "規則と重なるモデル出力が残っている"
    assert any(s.text == "田中太郎" for s in merged), "高スコアのモデル出力が残っていない"
    assert not any(s.text == "田中" for s in merged), "低スコアの重なりが残っている"
    for i in range(1, len(merged)):
        assert merged[i - 1].end <= merged[i].start, "統合結果が重なっている"
    print("  規則優先 ✓ / 重なり破棄 ✓ / モデル同士は高score優先 ✓ / 非重複 ✓")

    print("\n" + "=" * 74)
    ok = (not miss) and fp_total == 0
    print("すべての自己テストに合格" if ok
          else f"注意: 取りこぼし {len(miss)} 件 / 否定例での誤検出 {fp_total} 件")
    print("=" * 74)


if __name__ == "__main__":
    _selftest()
