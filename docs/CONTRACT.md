> **Note for English readers.** This is the internal implementation contract that
> every module in `sumi/` is written against — interface signatures, offset
> conventions, and the hard rules the test suite enforces. It is kept in Japanese
> because that is the project's working language and the language of the domain.
> All *published* documentation (README, model card, dataset card) and all
> user-facing runtime strings are in English. See `CONTRIBUTING.md`.

# Sumi 実装契約 (すべてのモジュールがこの契約に従う)

リポジトリ root: `/Users/nagaoyuta/Desktop/Claude code/40-Sumi`
Python: `/opt/miniconda3/bin/python3` (3.13)。`cd` してから `python3 -c ...` で実行する。

## 0. 絶対規則

1. **実在の個人情報を一切書かない**。氏名・住所・番号はすべてプログラムで合成する。
   実在の人物名・実在の住所番地・実在の口座番号をソースにハードコードしない。
   都道府県名・市区町村名・一般的な姓/名の語such は公開された地理的/統計的事実として
   利用してよいが、**組合せは乱数で生成**し、番地は必ず乱数化する。
2. **チェックディジットのある識別子は「形式は正しく、値は無効」に生成する**
   (クレジットカード様式は Luhn を **わざと外す**、マイナンバー様式は検査数字を **わざと外す**)。
   規則層は **形式で検出し、checksum の成否は `meta["checksum_valid"]` に記録するだけ** で、
   検出可否の条件にはしない (redaction では「それらしい物」を落としてはならない)。
3. **すべての public 関数・メソッドの docstring に `Claim:` 行を入れる**。
   `Claim:` の後に、その関数が実証する主張を
   「検出率 / 低誤検出 / CPU速度 / 可逆性 / 較正」のいずれか (複数可) で明記する。
   これは `tests/test_docstrings.py` が機械的に検査する。
4. 法令遵守を保証する表現を書かない。「リスクを下げる道具」と書く。
5. 日本語校正・誤字脱字検出には踏み込まない。
6. 外部ネットワークに触るのは `sumi/corpus.py` と benchmarks のモデル取得のみ。
   取得物は `data/raw/` にキャッシュし、2回目以降はオフラインで動く。

## 1. 共有型 (`sumi/types.py` — 実装済み・変更禁止)

```python
class PIIType(str, Enum):   # .ja で日本語名
    NAME ADDRESS PHONE EMAIL DOB BANK_ACCOUNT CREDIT_CARD MYNUMBER MEMBER_ID POSTAL_CODE
class Source(str, Enum):    # MODEL RULE MERGED GOLD BASELINE
RULE_DETERMINISTIC: frozenset[PIIType]   # EMAIL PHONE POSTAL_CODE BANK_ACCOUNT CREDIT_CARD MYNUMBER MEMBER_ID
MODEL_DRIVEN: frozenset[PIIType]         # NAME ADDRESS DOB
ALL_TYPES: tuple[PIIType, ...]

@dataclass(frozen=True, slots=True)
class Span:
    start:int; end:int; label:PIIType; text:str=""; score:float=1.0
    source:Source=Source.MODEL; meta:dict=...
    .length .key() .overlaps(o) .iou(o) .with_(**kw) .slice_of(text) .to_dict() Span.from_dict(d)

@dataclass
class Document:
    text:str; spans:list[Span]; doc_id:str; subset:str; genre:str
    source_license:str; source_ref:str; negative_kinds:list[str]; meta:dict
    .sorted_spans() .validate() .to_dict() Document.from_dict(d)

def normalize(text)->str                 # NFKC + 改行統一 + ダッシュ統一 (長音符ーは保持)
def bio_labels(types=ALL_TYPES)->list[str]   # ["O","B-NAME","I-NAME",...] 計21
def spans_to_bio(spans, offsets, label2id)->list[int]   # 特殊トークンは -100
```

**オフセットは常に NFKC 正規化後の Python 文字インデックス、半開区間 `[start,end)`。**
`Document.text` は必ず `normalize()` を通した後の文字列。正規化は土台テキストに **1度だけ**
適用し、PII 挿入はその後に行う (挿入後に再正規化してはならない = 座標が壊れる)。

## 2. モジュール別インターフェース (この名前・引数で実装すること)

### `sumi/corpus.py` — 土台テキスト取得 + ライセンス台帳
```python
@dataclass
class CorpusItem:
    text:str; license:str; source:str; genre:str; attribution:str; item_id:str

LICENSES: dict[str, dict]   # {"wikipedia_ja": {"license":"CC BY-SA 4.0","url":...,"note":...}, ...}

def fetch_wikipedia(n:int, *, seed:int=0, cache_dir:str="data/raw", min_chars:int=120)->list[CorpusItem]
def fetch_egov_laws(n:int, *, seed:int=0, cache_dir:str="data/raw")->list[CorpusItem]
def fetch_aozora(n:int, *, seed:int=0, cache_dir:str="data/raw")->list[CorpusItem]
def load_base_corpus(n:int, *, seed:int=0, cache_dir:str="data/raw",
                     sources=("wikipedia","egov","aozora"))->list[CorpusItem]
def license_table()->list[dict]   # データセットカードに貼る表
```
- Wikipedia: `https://ja.wikipedia.org/w/api.php` の `generator=random&prop=extracts&explaintext=1`
  (User-Agent 必須)。license `"CC BY-SA 4.0"`。
- e-Gov: `https://laws.e-gov.go.jp/api/2/law_data/{law_id}?response_format=json` (JSON)。
  法令は著作権法13条により著作権の対象外 → license `"Public Domain (著作権法第13条)"`。
  法令 ID は `https://laws.e-gov.go.jp/api/2/laws` 等で取得するか、既知の ID 定数表を持つ。
- 青空文庫: HF `globis-university/aozorabunko-clean` を streaming で先頭 N 件。
  license `"Public Domain (青空文庫・保護期間満了)"`。取得失敗時は空リストで縮退。
- **どの取得関数もネットワーク失敗時は例外を投げずに、キャッシュがあればキャッシュを、
  無ければ空リストを返す**。`load_base_corpus` は取得できたものだけで動く。
- 段落単位に切って 120〜600 文字程度のチャンクにする。

### `sumi/synth.py` — 合成 PII + 業務文書テンプレート + 位置記録挿入
```python
@dataclass
class PIIValue: text:str; label:PIIType; meta:dict

class PIIFactory:
    def __init__(self, seed:int=0)
    def name(self, *, full:bool|None=None)->PIIValue        # meta: sei, mei, reading, honorific_ok
    def address(self, *, with_postal:bool=False)->PIIValue  # meta: pref, city, has_banchi
    def phone(self, kind:str|None=None)->PIIValue           # kind: mobile|landline|tollfree|ip
    def email(self, name_meta:dict|None=None)->PIIValue
    def dob(self, era:str|None=None)->PIIValue              # era: seireki|wareki
    def bank_account(self)->PIIValue
    def credit_card(self)->PIIValue                          # meta: checksum_valid=False (Luhn故意不一致)
    def mynumber(self)->PIIValue                             # meta: checksum_valid=False
    def member_id(self)->PIIValue
    def postal_code(self)->PIIValue
    def make(self, t:PIIType)->PIIValue

GENRES = ("email","minutes","application","inquiry")
def render_document(factory:PIIFactory, *, genre:str, base_text:str="",
                    base_license:str="synthetic (CC0-1.0)", base_ref:str="",
                    doc_id:str="", n_pii:int|None=None, subset:str="train")->Document
def build_documents(n:int, *, seed:int=0, base_items=None, genres=GENRES,
                    subset:str="train")->list[Document]
```
- 挿入は **プレースホルダ展開時に開始位置を記録** する方式で行い、
  「後から `text.index()` で探す」ことは禁止 (同じ文字列が複数出ると壊れる)。
- 生成した全 `Document` は `.validate()` を通ること。
- 姓名は公開の頻度分布に近い重み付きサンプリング (よくある姓ほど高確率)。
- 住所は 都道府県+市区町村+町名 (公開の地理名) + **乱数の丁目-番-号**。

### `sumi/negatives.py` — HardNegativeGenerator (差別化の中心)
```python
NEGATIVE_KINDS: tuple[str,...] = (
  "common_noun_surname",    # 森/林/泉/大和/青木 を普通名詞として使う
  "place_as_person",        # 地名と同形の人名/人名と同形の地名
  "company_as_person",      # 企業名と同形の人名 (大和商事 / 本田技研 など)
  "honorific_boundary",     # 様/さん/氏/殿 の有無で境界が揺れる文脈
  "phone_like_id",          # 電話番号に見える型番・注文番号・日時
  "address_like_facility",  # 住所に見える施設名
  "number_like_id",         # 会員番号/口座に見えるが違う数字列
  "date_like_nondob",       # 生年月日に見えるが違う日付
)

@dataclass
class NegativeItem: text:str; kind:str; note:str=""

class HardNegativeGenerator:
    def __init__(self, seed:int=0, weights:dict[str,float]|None=None)
    def sample(self, kind:str|None=None)->NegativeItem
    def inject(self, doc:Document, k:int=2)->Document
        # 紛らわしい表現を本文へ挿入する。**gold span は増やさない**。
        # 既存 gold span の座標を必ずずらして更新すること (挿入位置以降を +len)。
        # doc.negative_kinds に混入 kind を追記。
    def build_negative_documents(self, n:int, *, base_items=None, subset:str="negatives")->list[Document]
        # gold span が 0 件、あるいは真の PII と紛らわしい否定例だけの文書群
    def reweight_from_errors(self, fp_counts:dict[str,int], *, strength:float=1.0)->dict[str,float]
        # 閉ループ: 誤検出の多い kind の生成確率を上げ、self.weights を更新して返す

def classify_false_positive(span:Span, doc:Document)->str
    # 誤検出スパンがどの negative kind に当たるかを推定して返す (閉ループの入力)
```

### `sumi/rules.py` — RuleLayer + 明示的優先順位の統合
```python
def luhn_ok(digits:str)->bool
def mynumber_check_ok(digits:str)->bool     # 総務省の検査用数字アルゴリズム
def is_valid_jp_phone(s:str)->bool

@dataclass
class RuleSpec: rule_id:str; label:PIIType; pattern:str; confidence:float; require_context:bool=False

class RuleLayer:
    def __init__(self, *, types=None, context_window:int=12)
    def detect(self, text:str)->list[Span]
        # Source.RULE、meta に rule_id / checksum_valid / matched_context を入れる
    def explain(self, text:str)->list[dict]

def merge_spans(model_spans, rule_spans, text:str, *,
                rule_types=RULE_DETERMINISTIC, min_model_score:float=0.0)->list[Span]
    """規則が確実な箇所は規則を優先、それ以外はモデル。優先順位は明示的:
       1. rule_types に属する規則スパンは常に採用 (Source.RULE のまま)
       2. 規則スパンと重なるモデルスパンは、同種別なら破棄、異種別でも規則を優先して破棄
       3. 残りのモデルスパンを採用
       4. モデル同士の重なりは score の高い方を残す
       5. 結果は start 昇順・非重複。採用時に source=Source.MERGED を付ける (rule 由来は meta['from']='rule')
    """
```
- 電話: 日本の市外局番桁数 (総桁数 10 桁、携帯 070/080/090 で 11 桁、0120/0800 フリーダイヤル)
  を満たすものだけ。`03-1234-5678` は可、`03-1234` は不可、`2024-01-15` は不可 (日付形状を除外)。
- 型番/注文番号との区別のため、`TEL/電話/連絡先/携帯/Tel` などの文脈語ボーナスを実装。
- 郵便番号 `\d{3}-\d{4}`、口座 `(銀行コード4)-?(支店3)-?(口座7)` および `普通 1234567`、
  マイナンバー様式 12 桁 (区切りあり/なし)、会員番号は接頭辞つき英数。

### `sumi/calibrate.py` — 較正と主要指標
```python
class SpanCalibrator:
    def __init__(self, method:str="temperature")   # "temperature" | "isotonic"
    def fit(self, scores:Sequence[float], labels:Sequence[int])->"SpanCalibrator"
    def transform(self, scores)->list[float]
    def save(self, path); @classmethod load(cls, path)

def expected_calibration_error(scores, labels, bins:int=15)->float
def reliability_diagram(scores, labels, *, bins:int=15, title:str="", out_path:str|None=None)
    # matplotlib figure を返す。日本語フォントが無い環境でも落ちないこと

def match_spans(gold:list[Span], pred:list[Span], *, mode:str="exact")->tuple[list,list,list]
    # mode: "exact" | "partial" (1文字でも重なれば可) -> (tp_pairs, fp, fn)

def detection_rates(gold_docs, pred_per_doc, *, mode="partial")->dict
    # 種別ごとの precision/recall/f1 と全体

def false_positive_rate(neg_docs, pred_per_doc)->float
    # 否定例文書あたりの誤検出スパン数 (= 誤検出率の定義。0.0 が理想)

def recall_at_fixed_fpr(gold_docs, pred_per_doc_scored, neg_docs, neg_pred_per_doc_scored,
                        *, target_fpr:float=0.05, mode:str="partial", by_type:bool=True)->dict
    """**主要指標**。否定例側の誤検出率が target_fpr 以下になる最小の閾値を score 上で
       二分探索し、その閾値での陽性側の検出率 (recall) を種別ごとに返す。
       戻り値: {"threshold":float, "fpr":float, "overall_recall":float, "by_type":{...}}
       Claim: 低誤検出 — 「誤検出を実務で許せる水準に固定したときに、どれだけ拾えるか」
       という運用上の問いに直接答える指標。"""
```

### `sumi/model.py` — TokenClassifier
```python
DEFAULT_BACKBONE = "sbintuitions/modernbert-ja-130m"   # 132M, fast tokenizer + offsets, MIT

@dataclass
class TrainConfig:
    backbone:str=DEFAULT_BACKBONE; epochs:float=3.0; lr:float=3e-5; batch_size:int=16
    max_length:int=256; warmup_ratio:float=0.1; weight_decay:float=0.01
    seed:int=0; device:str|None=None   # None -> mps > cpu 自動
    output_dir:str="artifacts/sumi-model"

class TokenClassifier:
    def __init__(self, model, tokenizer, label_list)
    @classmethod
    def from_backbone(cls, backbone:str=DEFAULT_BACKBONE)->"TokenClassifier"
    @classmethod
    def load(cls, path:str, *, device:str|None=None)->"TokenClassifier"
    def save(self, path:str)->None
    def train(self, train_docs:list[Document], val_docs:list[Document], cfg:TrainConfig)->dict
    def predict(self, texts:list[str], *, batch_size:int=16, max_length:int=256,
                threshold:float=0.5, refine:bool=True)->list[list[Span]]
    def predict_with_probs(self, texts, **kw)->tuple[list[list[Span]], list]

def decode_bio(probs, offsets, text:str, label_list:list[str], *, threshold:float=0.5)->list[Span]
    # span score = 構成トークン確率の最小値 (最も弱い根拠) を採用する
def refine_boundaries(span:Span, text:str)->Span
    # 敬称 (様/さん/氏/殿/君) と助詞 (は/が/の/を/に/へ/と/より/から) を末尾から剥がし、
    # 先頭の空白・記号も剥がす。NAME/ADDRESS に適用。
```
- 学習は自前ループ (transformers の Trainer に依存しない) で書いてよい。torch 直書き推奨。
- `device` は `mps` が使えれば `mps`、無ければ `cpu`。**推論のベンチは必ず CPU で測れる**ようにする。
- 長文は max_length を超えたら **重なりありのスライディングウィンドウ** で処理し、
  文字オフセットを元テキスト基準に戻すこと。

### `sumi/mask.py` + `sumi/egress.py` — 可逆マスキングと外部送信境界
```python
# egress.py
class Transport(Protocol):
    def send(self, payload:str)->str: ...
class RecordingTransport:            # テスト用。送信されたペイロードを全部保持
    sent:list[str]
    def __init__(self, responder=None)
    def send(self, payload)->str
class EchoTransport: ...             # payload をそのまま返す
class EgressGuard:
    """送信直前に、対応表の元値が payload に含まれていないかを検査して例外を投げる番人。
       Claim: 可逆性 — 対応表が外部へ出ないことを実行時にも強制する。"""
    def __init__(self, forbidden:Iterable[str])
    def check(self, payload:str)->None      # 含まれていたら EgressViolation
class EgressViolation(RuntimeError): ...
def guarded(transport:Transport, guard:EgressGuard)->Transport

# mask.py
@dataclass
class MaskEntry: placeholder:str; original:str; label:PIIType; start:int; end:int
@dataclass
class MaskMap:
    entries:list[MaskEntry]; doc_id:str=""; version:str="1"
    def to_dict(); from_dict(); def originals()->list[str]
    def redact_summary()->list[dict]   # 元値を含まない要約 (UI 表示用)

class ReversibleMasker:
    def __init__(self, *, style:str="angle")    # <NAME_1>
    def mask(self, text:str, spans:list[Span], *, doc_id:str="")->tuple[str, MaskMap]
        # 同一の元値は同一の置換子に安定して割り当てる (安定性)
    def unmask(self, text:str, mmap:MaskMap)->str
    def save_map(self, mmap:MaskMap, path:str)->None    # ファイルモード 0600
    @staticmethod load_map(path:str)->MaskMap

class LLMRoundTrip:
    def __init__(self, transport:Transport, masker:ReversibleMasker|None=None)
    def run(self, text:str, spans:list[Span], *, instruction:str="")->dict
        # {"masked":..., "response_masked":..., "response":..., "map":MaskMap}
        # 送信前に EgressGuard を必ず通す
```

### `sumi/presidio_plugin/__init__.py`
```python
class SumiRecognizer(EntityRecognizer):
    def __init__(self, model_path:str|None=None, *, supported_language:str="ja",
                 threshold:float=0.5, use_rules:bool=True, detector=None)
    def load(self)->None
    def analyze(self, text:str, entities, nlp_artifacts=None)->list[RecognizerResult]
SUMI_TO_PRESIDIO: dict[PIIType,str]   # NAME->PERSON, ADDRESS->LOCATION, PHONE->PHONE_NUMBER,
                                      # EMAIL->EMAIL_ADDRESS, DOB->DATE_TIME, CREDIT_CARD->CREDIT_CARD,
                                      # BANK_ACCOUNT->JP_BANK_ACCOUNT, MYNUMBER->JP_MY_NUMBER,
                                      # MEMBER_ID->JP_MEMBER_ID, POSTAL_CODE->JP_POSTAL_CODE
def register(registry, **kw)->None
def build_analyzer(model_path=None, **kw)->AnalyzerEngine
```
presidio は import できるが spacy の英語モデルが無い可能性があるため、
**NlpEngine を使わずに済む経路** (`AnalyzerEngine(nlp_engine=..., registry=...)` の
最小構成、または `SumiDetector` を直接呼ぶ薄い経路) を用意し、import 失敗時も
`sumi` 本体は動くようにする (`try/except ImportError`)。

### `sumi/detector.py` — 3層を束ねる公開ファサード (新規)
```python
@dataclass
class DetectResult: text:str; spans:list[Span]; timings:dict
class SumiDetector:
    def __init__(self, model_path:str|None=None, *, use_rules:bool=True, use_model:bool=True,
                 threshold:float=0.5, calibrator=None, device:str="cpu", onnx:bool=False)
    def detect(self, text:str)->list[Span]
    def detect_batch(self, texts:list[str])->list[list[Span]]
    def redact(self, text:str)->tuple[str, "MaskMap"]
```

## 3. コーディング規約
- 標準ライブラリ + numpy/torch/transformers/matplotlib のみに依存。scipy/sklearn は可。
- 型ヒントを付ける。`from __future__ import annotations` を先頭に。
- 乱数は必ず `random.Random(seed)` / `np.random.default_rng(seed)` をインスタンス化して使う
  (グローバル `random` を汚さない)。
- 日本語コメント可。docstring の `Claim:` 行は必須。
- **各モジュールは `if __name__ == "__main__":` に自己テストを持ち、
  `python3 -m sumi.<mod>` で動作確認できること。**
