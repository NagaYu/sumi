"""合成日本語 PII と業務文書の生成器 (Sumi のデータ基盤)。

このモジュールは **実在の個人情報を一切含まない**。氏名・住所・番号はすべて
公開の統計的/地理的事実 (よくある姓、都道府県名、市区町村名、市外局番の桁数など) を
素材にして、**組合せと番地・番号を乱数で生成** する。チェックディジットを持つ識別子
(クレジットカード様式・マイナンバー様式) は「形式は正しく値は無効」に作る。

文書生成の要は **挿入位置の構成的記録** である。テンプレートを走査しながら
出力文字列を継ぎ足し、PII を書き出した瞬間に ``(start, end)`` を記録する。
生成後に ``text.index()`` で探し直すことは一切しない (同一値が複数回現れると壊れるため)。

Claim: 検出率 / 低誤検出 — 正解スパンが構成的に (探索ではなく記録で) 得られるため、
検出率・誤検出率の分母と分子が定義上正しい。同一値の反復・敬称の隣接・
番号様式の紛らわしさを意図的に含めることで、低誤検出の主張を検証可能にする。
"""

from __future__ import annotations

import datetime as _dt
import random
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from sumi.types import Document, PIIType, Source, Span, normalize

__all__ = [
    "PIIValue",
    "PIIFactory",
    "GENRES",
    "TEMPLATES",
    "render_document",
    "build_documents",
]

# ---------------------------------------------------------------------------
# 1. 素材テーブル (公開の統計的・地理的事実。個人を特定する情報は含まない)
# ---------------------------------------------------------------------------

#: 頻度順に並べた姓 (漢字, カタカナ読み)。上位は公開されている姓名統計の順序に近い。
_SURNAME_RANKED: tuple[tuple[str, str], ...] = (
    ("佐藤", "サトウ"), ("鈴木", "スズキ"), ("高橋", "タカハシ"), ("田中", "タナカ"),
    ("伊藤", "イトウ"), ("渡辺", "ワタナベ"), ("山本", "ヤマモト"), ("中村", "ナカムラ"),
    ("小林", "コバヤシ"), ("加藤", "カトウ"), ("吉田", "ヨシダ"), ("山田", "ヤマダ"),
    ("佐々木", "ササキ"), ("山口", "ヤマグチ"), ("松本", "マツモト"), ("井上", "イノウエ"),
    ("木村", "キムラ"), ("林", "ハヤシ"), ("斎藤", "サイトウ"), ("清水", "シミズ"),
    ("山崎", "ヤマザキ"), ("阿部", "アベ"), ("森", "モリ"), ("池田", "イケダ"),
    ("橋本", "ハシモト"), ("山下", "ヤマシタ"), ("石川", "イシカワ"), ("中島", "ナカジマ"),
    ("前田", "マエダ"), ("藤田", "フジタ"), ("後藤", "ゴトウ"), ("小川", "オガワ"),
    ("岡田", "オカダ"), ("村上", "ムラカミ"), ("長谷川", "ハセガワ"), ("近藤", "コンドウ"),
    ("石井", "イシイ"), ("斉藤", "サイトウ"), ("坂本", "サカモト"), ("遠藤", "エンドウ"),
    ("藤井", "フジイ"), ("青木", "アオキ"), ("福田", "フクダ"), ("三浦", "ミウラ"),
    ("西村", "ニシムラ"), ("藤原", "フジワラ"), ("太田", "オオタ"), ("松田", "マツダ"),
    ("原田", "ハラダ"), ("岡本", "オカモト"), ("中野", "ナカノ"), ("中川", "ナカガワ"),
    ("小野", "オノ"), ("田村", "タムラ"), ("竹内", "タケウチ"), ("金子", "カネコ"),
    ("和田", "ワダ"), ("中山", "ナカヤマ"), ("石田", "イシダ"), ("上田", "ウエダ"),
    ("森田", "モリタ"), ("原", "ハラ"), ("柴田", "シバタ"), ("酒井", "サカイ"),
    ("工藤", "クドウ"), ("横山", "ヨコヤマ"), ("宮崎", "ミヤザキ"), ("宮本", "ミヤモト"),
    ("内田", "ウチダ"), ("高木", "タカギ"), ("谷口", "タニグチ"), ("安藤", "アンドウ"),
    ("丸山", "マルヤマ"), ("今井", "イマイ"), ("高田", "タカダ"), ("藤本", "フジモト"),
    ("河野", "コウノ"), ("大野", "オオノ"), ("上野", "ウエノ"), ("武田", "タケダ"),
    ("菅原", "スガワラ"), ("千葉", "チバ"), ("久保", "クボ"), ("松井", "マツイ"),
    ("小島", "コジマ"), ("岩崎", "イワサキ"), ("桜井", "サクライ"), ("木下", "キノシタ"),
    ("野口", "ノグチ"), ("松尾", "マツオ"), ("菊地", "キクチ"), ("野村", "ノムラ"),
    ("新井", "アライ"), ("渡部", "ワタベ"), ("佐野", "サノ"), ("市川", "イチカワ"),
    ("水野", "ミズノ"), ("大塚", "オオツカ"), ("小松", "コマツ"), ("島田", "シマダ"),
    ("古川", "フルカワ"), ("杉山", "スギヤマ"), ("増田", "マスダ"), ("小山", "コヤマ"),
    ("大西", "オオニシ"), ("平野", "ヒラノ"), ("秋山", "アキヤマ"), ("石原", "イシハラ"),
    ("松浦", "マツウラ"), ("大橋", "オオハシ"), ("吉川", "ヨシカワ"), ("荒木", "アラキ"),
    ("星野", "ホシノ"), ("岡崎", "オカザキ"), ("岩田", "イワタ"), ("松岡", "マツオカ"),
    ("内藤", "ナイトウ"), ("川口", "カワグチ"), ("平田", "ヒラタ"), ("大久保", "オオクボ"),
    ("樋口", "ヒグチ"), ("川崎", "カワサキ"), ("飯田", "イイダ"), ("大石", "オオイシ"),
)

#: 普通名詞・地名・企業名と同形の姓。``negatives.py`` の曖昧性はここに依存するので
#: **意図的に** 一定の重みで出現させる (実データでも決して珍しくない姓ばかり)。
_SURNAME_AMBIGUOUS: tuple[tuple[str, str], ...] = (
    ("泉", "イズミ"), ("大和", "ヤマト"), ("石", "イシ"), ("谷", "タニ"),
    ("島", "シマ"), ("川", "カワ"), ("富士", "フジ"), ("東", "アズマ"),
    ("西", "ニシ"), ("南", "ミナミ"), ("北", "キタ"), ("上", "ウエ"),
    ("中", "ナカ"), ("下", "シモ"), ("本田", "ホンダ"), ("日高", "ヒダカ"),
    ("関", "セキ"), ("辻", "ツジ"), ("堀", "ホリ"), ("畑", "ハタ"),
    ("庄司", "ショウジ"), ("郷", "ゴウ"), ("城", "ジョウ"), ("福島", "フクシマ"),
)

#: 上位20姓のおおよその出現率 (%)。21位以降はなだらかに減衰させる。
_SURNAME_HEAD_W: tuple[float, ...] = (
    1.53, 1.45, 1.14, 1.07, 0.85, 0.84, 0.82, 0.79, 0.78, 0.72,
    0.63, 0.62, 0.55, 0.50, 0.48, 0.47, 0.45, 0.44, 0.43, 0.42,
)


def _surname_table() -> tuple[list[tuple[str, str]], list[float]]:
    """姓テーブルと重みを構築する (降順の重み付き分布)。"""
    rows: list[tuple[str, str]] = list(_SURNAME_RANKED)
    weights: list[float] = []
    for i in range(len(rows)):
        if i < len(_SURNAME_HEAD_W):
            weights.append(_SURNAME_HEAD_W[i])
        else:
            # 21位以降: w = 0.42 * (20 / rank) ** 0.7 (単調減少)
            weights.append(0.42 * (20.0 / (i + 1)) ** 0.7)
    for kanji, kana in _SURNAME_AMBIGUOUS:
        if any(kanji == k for k, _ in rows):
            continue
        rows.append((kanji, kana))
        weights.append(0.22)  # 珍しすぎず、曖昧性の学習に足りる程度
    return rows, weights


_SURNAMES, _SURNAME_WEIGHTS = _surname_table()

#: 名 (漢字, カタカナ, 性別 m/f/x, 世代 showa/heisei/reiwa/any)
_GIVEN: tuple[tuple[str, str, str, str], ...] = (
    # --- 昭和期に多い男性名 ---
    ("博", "ヒロシ", "m", "showa"), ("茂", "シゲル", "m", "showa"), ("隆", "タカシ", "m", "showa"),
    ("清", "キヨシ", "m", "showa"), ("実", "ミノル", "m", "showa"), ("誠", "マコト", "m", "showa"),
    ("修", "オサム", "m", "showa"), ("明", "アキラ", "m", "showa"), ("勇", "イサム", "m", "showa"),
    ("進", "ススム", "m", "showa"), ("豊", "ユタカ", "m", "showa"), ("武", "タケシ", "m", "showa"),
    ("正雄", "マサオ", "m", "showa"), ("一郎", "イチロウ", "m", "showa"), ("健一", "ケンイチ", "m", "showa"),
    ("和夫", "カズオ", "m", "showa"), ("幸雄", "ユキオ", "m", "showa"), ("昭夫", "アキオ", "m", "showa"),
    ("義明", "ヨシアキ", "m", "showa"), ("敏夫", "トシオ", "m", "showa"), ("光男", "ミツオ", "m", "showa"),
    ("秀夫", "ヒデオ", "m", "showa"), ("隆司", "タカシ", "m", "showa"), ("正治", "ショウジ", "m", "showa"),
    ("二郎", "ジロウ", "m", "showa"), ("孝治", "コウジ", "m", "showa"), ("信夫", "ノブオ", "m", "showa"),
    ("太郎", "タロウ", "m", "showa"),
    # --- 昭和期に多い女性名 ---
    ("幸子", "サチコ", "f", "showa"), ("和子", "カズコ", "f", "showa"), ("洋子", "ヨウコ", "f", "showa"),
    ("京子", "キョウコ", "f", "showa"), ("恵子", "ケイコ", "f", "showa"), ("節子", "セツコ", "f", "showa"),
    ("久美子", "クミコ", "f", "showa"), ("真由美", "マユミ", "f", "showa"), ("由美子", "ユミコ", "f", "showa"),
    ("裕子", "ユウコ", "f", "showa"), ("美智子", "ミチコ", "f", "showa"), ("順子", "ジュンコ", "f", "showa"),
    ("智子", "トモコ", "f", "showa"), ("陽子", "ヨウコ", "f", "showa"), ("直子", "ナオコ", "f", "showa"),
    ("千代子", "チヨコ", "f", "showa"), ("文子", "フミコ", "f", "showa"), ("悦子", "エツコ", "f", "showa"),
    ("民子", "タミコ", "f", "showa"), ("春美", "ハルミ", "f", "showa"), ("明美", "アケミ", "f", "showa"),
    ("典子", "ノリコ", "f", "showa"), ("良子", "リョウコ", "f", "showa"), ("弘子", "ヒロコ", "f", "showa"),
    ("房子", "フサコ", "f", "showa"),
    # --- 平成期に多い男性名 ---
    ("大輔", "ダイスケ", "m", "heisei"), ("翔太", "ショウタ", "m", "heisei"), ("拓也", "タクヤ", "m", "heisei"),
    ("健太", "ケンタ", "m", "heisei"), ("亮", "リョウ", "m", "heisei"), ("涼太", "リョウタ", "m", "heisei"),
    ("雄太", "ユウタ", "m", "heisei"), ("智也", "トモヤ", "m", "heisei"), ("直樹", "ナオキ", "m", "heisei"),
    ("和也", "カズヤ", "m", "heisei"), ("達也", "タツヤ", "m", "heisei"), ("翔", "ショウ", "m", "heisei"),
    ("大樹", "ダイキ", "m", "heisei"), ("駿", "シュン", "m", "heisei"), ("悠人", "ユウト", "m", "heisei"),
    ("陸", "リク", "m", "heisei"), ("颯太", "ソウタ", "m", "heisei"), ("悠斗", "ユウト", "m", "heisei"),
    ("海斗", "カイト", "m", "heisei"), ("太一", "タイチ", "m", "heisei"), ("亮太", "リョウタ", "m", "heisei"),
    ("圭吾", "ケイゴ", "m", "heisei"), ("隼人", "ハヤト", "m", "heisei"), ("慎也", "シンヤ", "m", "heisei"),
    ("裕貴", "ユウキ", "m", "heisei"),
    # --- 平成期に多い女性名 ---
    ("愛", "アイ", "f", "heisei"), ("美咲", "ミサキ", "f", "heisei"), ("彩", "アヤ", "f", "heisei"),
    ("舞", "マイ", "f", "heisei"), ("遥", "ハルカ", "f", "heisei"), ("さくら", "サクラ", "f", "heisei"),
    ("結衣", "ユイ", "f", "heisei"), ("陽菜", "ヒナ", "f", "heisei"), ("葵", "アオイ", "f", "heisei"),
    ("七海", "ナナミ", "f", "heisei"), ("莉子", "リコ", "f", "heisei"), ("結菜", "ユイナ", "f", "heisei"),
    ("咲良", "サクラ", "f", "heisei"), ("凛", "リン", "f", "heisei"), ("芽衣", "メイ", "f", "heisei"),
    ("杏", "アン", "f", "heisei"), ("楓", "カエデ", "f", "heisei"), ("美優", "ミユ", "f", "heisei"),
    ("優花", "ユウカ", "f", "heisei"), ("桃子", "モモコ", "f", "heisei"), ("千尋", "チヒロ", "f", "heisei"),
    ("麻衣", "マイ", "f", "heisei"), ("沙織", "サオリ", "f", "heisei"), ("奈々", "ナナ", "f", "heisei"),
    ("瑞希", "ミズキ", "f", "heisei"),
    # --- 令和期に多い名 ---
    ("陽翔", "ハルト", "m", "reiwa"), ("湊", "ミナト", "m", "reiwa"), ("悠真", "ユウマ", "m", "reiwa"),
    ("大翔", "ヒロト", "m", "reiwa"), ("律", "リツ", "m", "reiwa"), ("樹", "イツキ", "m", "reiwa"),
    ("朝陽", "アサヒ", "m", "reiwa"), ("碧", "アオ", "m", "reiwa"), ("蒼", "ソウ", "m", "reiwa"),
    ("暖", "ダン", "m", "reiwa"), ("新", "アラタ", "m", "reiwa"), ("匠", "タクミ", "m", "reiwa"),
    ("蓮", "レン", "m", "reiwa"), ("陽向", "ヒナタ", "m", "reiwa"), ("湊斗", "ミナト", "m", "reiwa"),
    ("陽葵", "ヒマリ", "f", "reiwa"), ("詩", "ウタ", "f", "reiwa"), ("芽依", "メイ", "f", "reiwa"),
    ("紬", "ツムギ", "f", "reiwa"), ("澪", "ミオ", "f", "reiwa"), ("心春", "コハル", "f", "reiwa"),
    ("凪", "ナギ", "f", "reiwa"), ("環奈", "カンナ", "f", "reiwa"), ("結愛", "ユア", "f", "reiwa"),
    ("咲希", "サキ", "f", "reiwa"), ("美桜", "ミオ", "f", "reiwa"), ("心結", "ミユ", "f", "reiwa"),
    ("灯", "アカリ", "f", "reiwa"), ("花", "ハナ", "f", "reiwa"), ("琴音", "コトネ", "f", "reiwa"),
    # --- 中性的な名 (世代を問わない) ---
    ("薫", "カオル", "x", "any"), ("真", "マコト", "x", "any"), ("望", "ノゾミ", "x", "any"),
    ("翼", "ツバサ", "x", "any"), ("郁", "イク", "x", "any"), ("千秋", "チアキ", "x", "any"),
    ("蛍", "ホタル", "x", "any"), ("泉", "イズミ", "x", "any"), ("圭", "ケイ", "x", "any"),
    ("和", "ナゴミ", "x", "any"),
)

#: 世代の出現重み (母集団の年齢構成をおおまかに模す)。
_ERA_STYLE_W: dict[str, float] = {"showa": 0.34, "heisei": 0.38, "reiwa": 0.18, "any": 0.10}

#: (都道府県, 市区町村, 町名候補)。47 都道府県すべてを含む。番地は必ず乱数化する。
_PLACES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("北海道", "札幌市中央区", ("大通西", "北一条西", "南三条西", "宮の森")),
    ("北海道", "函館市", ("本町", "五稜郭町", "湯川町")),
    ("北海道", "旭川市", ("宮下通", "常盤通", "神楽")),
    ("青森県", "青森市", ("新町", "古川", "長島")),
    ("岩手県", "盛岡市", ("中央通", "大通", "内丸")),
    ("宮城県", "仙台市青葉区", ("中央", "一番町", "国分町", "本町")),
    ("秋田県", "秋田市", ("中通", "山王", "大町")),
    ("山形県", "山形市", ("香澄町", "城南町", "七日町")),
    ("福島県", "福島市", ("栄町", "太田町", "置賜町")),
    ("茨城県", "水戸市", ("三の丸", "泉町", "桜川")),
    ("栃木県", "宇都宮市", ("馬場通り", "本町", "東宿郷")),
    ("群馬県", "前橋市", ("大手町", "本町", "千代田町")),
    ("埼玉県", "さいたま市大宮区", ("桜木町", "大門町", "宮町")),
    ("埼玉県", "川口市", ("本町", "栄町", "幸町")),
    ("埼玉県", "川越市", ("元町", "脇田町", "新富町")),
    ("千葉県", "千葉市中央区", ("中央", "富士見", "新町")),
    ("千葉県", "船橋市", ("本町", "湊町", "海神")),
    ("千葉県", "柏市", ("柏", "中央町", "旭町")),
    ("東京都", "新宿区", ("西新宿", "歌舞伎町", "高田馬場", "四谷")),
    ("東京都", "千代田区", ("丸の内", "大手町", "神田佐久間町", "有楽町")),
    ("東京都", "港区", ("六本木", "赤坂", "芝公園", "白金台")),
    ("東京都", "世田谷区", ("三軒茶屋", "経堂", "用賀", "成城")),
    ("東京都", "八王子市", ("旭町", "子安町", "南大沢")),
    ("東京都", "立川市", ("曙町", "錦町", "柴崎町")),
    ("神奈川県", "横浜市西区", ("みなとみらい", "北幸", "南幸", "高島")),
    ("神奈川県", "川崎市川崎区", ("砂子", "駅前本町", "日進町")),
    ("神奈川県", "相模原市中央区", ("中央", "富士見", "矢部")),
    ("神奈川県", "藤沢市", ("藤沢", "鵠沼石上", "辻堂神台")),
    ("新潟県", "新潟市中央区", ("万代", "東大通", "弁天")),
    ("富山県", "富山市", ("桜町", "総曲輪", "新富町")),
    ("石川県", "金沢市", ("広坂", "香林坊", "木ノ新保町")),
    ("福井県", "福井市", ("中央", "大手", "順化")),
    ("山梨県", "甲府市", ("丸の内", "中央", "朝気")),
    ("長野県", "長野市", ("南長野", "鶴賀", "中御所")),
    ("岐阜県", "岐阜市", ("司町", "神田町", "橋本町")),
    ("静岡県", "静岡市葵区", ("追手町", "呉服町", "紺屋町")),
    ("静岡県", "浜松市中央区", ("砂山町", "板屋町", "中央")),
    ("愛知県", "名古屋市中区", ("栄", "錦", "丸の内", "大須")),
    ("愛知県", "豊田市", ("西町", "若宮町", "竹生町")),
    ("愛知県", "岡崎市", ("明大寺町", "康生通", "羽根町")),
    ("三重県", "津市", ("広明町", "羽所町", "栄町")),
    ("滋賀県", "大津市", ("京町", "打出浜", "におの浜")),
    ("京都府", "京都市下京区", ("東塩小路町", "四条通", "烏丸通")),
    ("京都府", "宇治市", ("宇治", "大久保町", "広野町")),
    ("大阪府", "大阪市北区", ("梅田", "中之島", "曽根崎", "天神橋")),
    ("大阪府", "大阪市中央区", ("難波", "本町", "心斎橋筋", "大手前")),
    ("大阪府", "堺市堺区", ("中瓦町", "南瓦町", "戎島町")),
    ("兵庫県", "神戸市中央区", ("三宮町", "加納町", "磯上通", "港島中町")),
    ("兵庫県", "姫路市", ("本町", "駅前町", "豆腐町")),
    ("兵庫県", "西宮市", ("六湛寺町", "甲風園", "池田町")),
    ("奈良県", "奈良市", ("登大路町", "三条本町", "大宮町")),
    ("和歌山県", "和歌山市", ("七番丁", "美園町", "屏風丁")),
    ("鳥取県", "鳥取市", ("東町", "富安", "今町")),
    ("島根県", "松江市", ("殿町", "朝日町", "白潟本町")),
    ("岡山県", "岡山市北区", ("内山下", "表町", "駅元町")),
    ("広島県", "広島市中区", ("基町", "紙屋町", "大手町", "八丁堀")),
    ("山口県", "山口市", ("滝町", "中園町", "亀山町")),
    ("徳島県", "徳島市", ("万代町", "寺島本町西", "幸町")),
    ("香川県", "高松市", ("番町", "サンポート", "丸亀町")),
    ("愛媛県", "松山市", ("一番町", "二番町", "千舟町")),
    ("高知県", "高知市", ("丸ノ内", "本町", "帯屋町")),
    ("福岡県", "福岡市博多区", ("博多駅前", "中洲", "店屋町", "住吉")),
    ("福岡県", "北九州市小倉北区", ("城内", "浅野", "魚町")),
    ("佐賀県", "佐賀市", ("城内", "駅前中央", "白山")),
    ("長崎県", "長崎市", ("尾上町", "出島町", "万才町")),
    ("熊本県", "熊本市中央区", ("手取本町", "花畑町", "水前寺")),
    ("大分県", "大分市", ("荷揚町", "府内町", "金池町")),
    ("宮崎県", "宮崎市", ("橘通東", "橘通西", "高千穂通")),
    ("鹿児島県", "鹿児島市", ("山下町", "中央町", "東千石町")),
    ("沖縄県", "那覇市", ("泉崎", "久茂地", "おもろまち", "牧志")),
)

#: 都道府県ごとの郵便番号上位3桁 (公開の番号帯)。下4桁は乱数。
_POSTAL_PREFIX: dict[str, tuple[str, ...]] = {
    "北海道": ("060", "001", "040", "070", "080"),
    "青森県": ("030", "031", "036"), "岩手県": ("020", "024", "026"),
    "宮城県": ("980", "981", "983"), "秋田県": ("010", "011", "015"),
    "山形県": ("990", "992", "997"), "福島県": ("960", "963", "965"),
    "茨城県": ("310", "300", "305"), "栃木県": ("320", "321", "326"),
    "群馬県": ("371", "370", "376"), "埼玉県": ("330", "332", "350", "336"),
    "千葉県": ("260", "261", "272", "273"),
    "東京都": ("100", "101", "105", "150", "151", "160", "163", "170", "190"),
    "神奈川県": ("220", "221", "231", "210", "252"),
    "新潟県": ("950", "951", "940"), "富山県": ("930", "939", "933"),
    "石川県": ("920", "921", "923"), "福井県": ("910", "914", "916"),
    "山梨県": ("400", "403", "405"), "長野県": ("380", "381", "390"),
    "岐阜県": ("500", "501", "503"), "静岡県": ("420", "430", "410"),
    "愛知県": ("460", "461", "450", "471"), "三重県": ("514", "510", "515"),
    "滋賀県": ("520", "522", "525"), "京都府": ("600", "604", "611"),
    "大阪府": ("530", "531", "540", "590"), "兵庫県": ("650", "651", "670", "662"),
    "奈良県": ("630", "631", "634"), "和歌山県": ("640", "641", "644"),
    "鳥取県": ("680", "683", "689"), "島根県": ("690", "692", "699"),
    "岡山県": ("700", "701", "710"), "広島県": ("730", "732", "733"),
    "山口県": ("753", "750", "755"), "徳島県": ("770", "771", "779"),
    "香川県": ("760", "761", "769"), "愛媛県": ("790", "791", "799"),
    "高知県": ("780", "781", "787"), "福岡県": ("810", "812", "802", "814"),
    "佐賀県": ("840", "841", "849"), "長崎県": ("850", "852", "857"),
    "熊本県": ("860", "861", "862"), "大分県": ("870", "874", "879"),
    "宮崎県": ("880", "882", "889"), "鹿児島県": ("890", "892", "899"),
    "沖縄県": ("900", "901", "904"),
}

#: 建物名の部品 (実在建物ではなく、町名 + 一般語の合成)。
_BLDG_SUFFIX: tuple[str, ...] = (
    "ビル", "第一ビル", "センタービル", "ハイツ", "コーポ", "メゾン",
    "マンション", "レジデンス", "パークハイム", "スカイタワー", "プラザ",
)
_BLDG_PREFIX: tuple[str, ...] = (
    "みどり", "さくら", "あおば", "ひまわり", "けやき", "しらかば", "つばき", "こもれび",
)

#: 市外局番。桁数の内訳は「市外局番 + 市内局番 + 加入者番号 = 10 桁」を満たす。
_AREA2: tuple[str, ...] = ("03", "06")                      # 2桁 + 4 + 4
_AREA3: tuple[str, ...] = (                                  # 3桁 + 3 + 4
    "011", "015", "017", "018", "019", "022", "023", "024", "025", "026",
    "027", "028", "029", "042", "043", "044", "045", "046", "047", "048",
    "049", "052", "053", "054", "055", "058", "059", "072", "073", "075",
    "076", "077", "078", "079", "082", "083", "084", "086", "087", "088",
    "089", "092", "093", "095", "096", "097", "098", "099",
)
_AREA4: tuple[str, ...] = (                                  # 4桁 + 2 + 4
    "0134", "0138", "0143", "0155", "0166", "0176", "0178", "0182", "0187",
    "0191", "0223", "0234", "0246", "0250", "0257", "0263", "0265", "0267",
    "0270", "0276", "0280", "0287", "0294", "0299", "0428", "0438", "0463",
    "0465", "0467", "0470", "0475", "0479", "0480", "0493", "0532", "0533",
    "0537", "0538", "0545", "0555", "0561", "0566", "0568", "0584", "0742",
    "0743", "0744", "0745", "0748", "0749", "0790", "0794", "0797", "0798",
)

#: 金融機関名 (実在名。ただし付随する番号はすべて乱数で、実口座を指さない)。
_BANKS: tuple[str, ...] = (
    "みずほ銀行", "三菱UFJ銀行", "三井住友銀行", "りそな銀行", "ゆうちょ銀行",
    "横浜銀行", "千葉銀行", "静岡銀行", "常陽銀行", "福岡銀行", "京都銀行",
    "広島銀行", "北陸銀行", "七十七銀行", "群馬銀行", "八十二銀行", "中国銀行",
    "伊予銀行", "十六銀行", "南都銀行", "山口銀行", "大垣共立銀行", "北海道銀行",
    "琉球銀行", "楽天銀行", "住信SBIネット銀行", "PayPay銀行", "ソニー銀行",
    "イオン銀行", "西日本シティ銀行", "北國銀行", "第四北越銀行",
)
_BRANCHES: tuple[str, ...] = (
    "本店営業部", "新宿支店", "渋谷支店", "梅田支店", "名古屋駅前支店", "博多支店",
    "札幌支店", "仙台支店", "横浜支店", "京都支店", "神戸支店", "広島支店",
    "金沢支店", "高松支店", "那覇支店", "大手町支店", "駅前支店", "中央支店",
    "北支店", "南支店", "港支店", "緑支店",
)
_ACCOUNT_KIND: tuple[str, ...] = ("普通", "当座", "貯蓄")

#: クレジットカードの IIN 形状 (発行会社の桁構成に似せた接頭辞。値は無効)。
_IIN: tuple[str, ...] = (
    "4", "41", "42", "45", "4539", "4556", "4916", "4024",
    "51", "52", "53", "54", "55", "5105", "5425", "5334",
    "35", "3528", "3540", "3562", "3566",
)

_MEMBER_PREFIX: tuple[tuple[str, str], ...] = (
    ("M", "member"), ("MB", "member"), ("CU", "customer"), ("CS", "customer"),
    ("KT", "contract"), ("CT", "contract"), ("EMP", "employee"), ("SB", "subscriber"),
    ("PT", "patient"), ("AC", "account"),
)

# ---------------------------------------------------------------------------
# 2. かな・ローマ字・和暦などの小道具
# ---------------------------------------------------------------------------

_ROMAJI2: dict[str, str] = {
    "キャ": "kya", "キュ": "kyu", "キョ": "kyo", "ギャ": "gya", "ギュ": "gyu", "ギョ": "gyo",
    "シャ": "sha", "シュ": "shu", "ショ": "sho", "ジャ": "ja", "ジュ": "ju", "ジョ": "jo",
    "チャ": "cha", "チュ": "chu", "チョ": "cho", "ニャ": "nya", "ニュ": "nyu", "ニョ": "nyo",
    "ヒャ": "hya", "ヒュ": "hyu", "ヒョ": "hyo", "ビャ": "bya", "ビュ": "byu", "ビョ": "byo",
    "ピャ": "pya", "ピュ": "pyu", "ピョ": "pyo", "ミャ": "mya", "ミュ": "myu", "ミョ": "myo",
    "リャ": "rya", "リュ": "ryu", "リョ": "ryo", "シェ": "she", "ジェ": "je", "チェ": "che",
    "ティ": "ti", "ディ": "di", "ファ": "fa", "フィ": "fi", "フェ": "fe", "フォ": "fo",
}
_ROMAJI1: dict[str, str] = {
    "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o",
    "カ": "ka", "キ": "ki", "ク": "ku", "ケ": "ke", "コ": "ko",
    "サ": "sa", "シ": "shi", "ス": "su", "セ": "se", "ソ": "so",
    "タ": "ta", "チ": "chi", "ツ": "tsu", "テ": "te", "ト": "to",
    "ナ": "na", "ニ": "ni", "ヌ": "nu", "ネ": "ne", "ノ": "no",
    "ハ": "ha", "ヒ": "hi", "フ": "fu", "ヘ": "he", "ホ": "ho",
    "マ": "ma", "ミ": "mi", "ム": "mu", "メ": "me", "モ": "mo",
    "ヤ": "ya", "ユ": "yu", "ヨ": "yo",
    "ラ": "ra", "リ": "ri", "ル": "ru", "レ": "re", "ロ": "ro",
    "ワ": "wa", "ヲ": "o", "ン": "n",
    "ガ": "ga", "ギ": "gi", "グ": "gu", "ゲ": "ge", "ゴ": "go",
    "ザ": "za", "ジ": "ji", "ズ": "zu", "ゼ": "ze", "ゾ": "zo",
    "ダ": "da", "ヂ": "ji", "ヅ": "zu", "デ": "de", "ド": "do",
    "バ": "ba", "ビ": "bi", "ブ": "bu", "ベ": "be", "ボ": "bo",
    "パ": "pa", "ピ": "pi", "プ": "pu", "ペ": "pe", "ポ": "po",
    "ヴ": "vu", "ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o",
    "ャ": "ya", "ュ": "yu", "ョ": "yo",
}


def _to_romaji(kana: str, *, long_vowel: bool = True) -> str:
    """カタカナ列をヘボン式ローマ字に変換する (メールアドレス生成用の内部関数)。"""
    out: list[str] = []
    i = 0
    n = len(kana)
    while i < n:
        two = kana[i : i + 2]
        if two in _ROMAJI2:
            out.append(_ROMAJI2[two])
            i += 2
            continue
        ch = kana[i]
        if ch == "ッ":
            nxt = kana[i + 1 : i + 3]
            r = _ROMAJI2.get(nxt) or _ROMAJI1.get(kana[i + 1 : i + 2], "")
            if r:
                out.append(r[0])
            i += 1
            continue
        if ch == "ー":
            if long_vowel and out and out[-1] and out[-1][-1] in "aiueo":
                out.append(out[-1][-1])
            i += 1
            continue
        out.append(_ROMAJI1.get(ch, ""))
        i += 1
    s = "".join(out)
    if not long_vowel:
        s = s.replace("ou", "o").replace("uu", "u").replace("oo", "o")
    return s or "user"


def _to_hiragana(kana: str) -> str:
    """カタカナをひらがなに変換する (読み仮名表記の揺れを作るための内部関数)。"""
    out = []
    for ch in kana:
        o = ord(ch)
        out.append(chr(o - 0x60) if 0x30A1 <= o <= 0x30F6 else ch)
    return "".join(out)


_KANJI_DIGITS = "〇一二三四五六七八九"


def _kanji_number(n: int) -> str:
    """1..99 の整数を漢数字にする (「二丁目」等の表記のための内部関数)。"""
    if n < 10:
        return _KANJI_DIGITS[n]
    if n < 20:
        return "十" + (_KANJI_DIGITS[n % 10] if n % 10 else "")
    return _KANJI_DIGITS[n // 10] + "十" + (_KANJI_DIGITS[n % 10] if n % 10 else "")


#: 元号 (名称, 英字略号, 開始日, 終了日)。改元日をまたぐ計算を正しく行う。
_ERAS: tuple[tuple[str, str, _dt.date, _dt.date], ...] = (
    ("明治", "M", _dt.date(1868, 10, 23), _dt.date(1912, 7, 29)),
    ("大正", "T", _dt.date(1912, 7, 30), _dt.date(1926, 12, 24)),
    ("昭和", "S", _dt.date(1926, 12, 25), _dt.date(1989, 1, 7)),
    ("平成", "H", _dt.date(1989, 1, 8), _dt.date(2019, 4, 30)),
    ("令和", "R", _dt.date(2019, 5, 1), _dt.date(2100, 12, 31)),
)


def _to_wareki(d: _dt.date) -> tuple[str, str, int]:
    """西暦日付を (元号名, 英字略号, 元号年) に変換する内部関数。"""
    for name, alpha, start, end in _ERAS:
        if start <= d <= end:
            return name, alpha, d.year - start.year + 1
    raise ValueError(f"date out of supported era range: {d}")


def _fullwidth(s: str) -> str:
    """ASCII 英数字と記号を全角にする (正規化前の揺れを作る内部関数)。"""
    out = []
    for ch in s:
        o = ord(ch)
        if 0x21 <= o <= 0x7E:
            out.append(chr(o + 0xFEE0))
        elif ch == " ":
            out.append("　")
        else:
            out.append(ch)
    return "".join(out)


def _luhn_check_digit(body: str) -> int:
    """Luhn チェックディジットを計算する内部関数 (合成では **わざと外す**)。"""
    total = 0
    for i, ch in enumerate(reversed(body)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10


def _mynumber_check_digit(body11: str) -> int:
    """個人番号 (12桁) の検査用数字を総務省告示の算式で計算する内部関数。

    n = 1..11 を右から数えた位置とし、Pn = n+1 (n<=6) / n-5 (n>=7)。
    検査用数字 = 11 - (Σ Pn*Cn mod 11)、ただし 10 以上なら 0。
    """
    total = 0
    for n in range(1, 12):
        c = int(body11[11 - n])
        p = n + 1 if n <= 6 else n - 5
        total += p * c
    r = 11 - (total % 11)
    return 0 if r >= 10 else r


# ---------------------------------------------------------------------------
# 3. PIIValue / PIIFactory
# ---------------------------------------------------------------------------


@dataclass
class PIIValue:
    """合成された1件の PII 値。文字列と種別と生成時の内訳を持つ。

    Claim: 検出率 / 可逆性 — 生成時の内訳 (姓・名・読み・チェックディジットの成否など) を
    ``meta`` に残すことで、検出結果の内訳評価と、マスク後の復元検証ができる。

    ``meta["parts"]`` に ``[(start, end, "LABEL"), ...]`` (値文字列内の相対位置) を
    置くと、1つの値の中の部分だけを正解スパンにできる (例: ``〒160-0023`` の
    ``〒`` を除いた数字部分、住所の先頭に付いた郵便番号を別種別にする、など)。
    """

    text: str
    label: PIIType
    meta: dict[str, Any] = field(default_factory=dict)

    def parts(self) -> list[tuple[int, int, PIIType]]:
        """この値の中で正解スパンにすべき区間を返す (既定は値全体)。

        Claim: 検出率 — 「郵便番号記号 〒」のような非個人情報を正解から外し、
        種別ごとの検出率が記号の扱いで変わらないようにする。
        """
        raw = self.meta.get("parts")
        if not raw:
            return [(0, len(self.text), self.label)]
        out: list[tuple[int, int, PIIType]] = []
        for a, b, lab in raw:
            out.append((int(a), int(b), PIIType(lab)))
        return out


class PIIFactory:
    """種別ごとの合成 PII 生成器 (乱数はインスタンス内で完結する)。

    Claim: 検出率 / 低誤検出 — 実在の個人情報を使わずに、実データと同じ *形式の分布*
    (姓の頻度分布、市外局番の桁数規則、和暦と西暦、区切り記号の揺れ) を再現する。
    形式は正しく値は無効な識別子を作るため、検出器を「形式で検出し値で判断しない」
    設計に強制でき、誤検出を減らす主張の土台になる。
    """

    def __init__(self, seed: int = 0) -> None:
        """乱数シードを固定して生成器を作る。

        Claim: 検出率 — データセットが完全に再現可能であることが、
        検出率の数値を後から検証可能にする前提条件になる。
        """
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

    # -- 氏名 -------------------------------------------------------------

    def name(self, *, full: bool | None = None) -> PIIValue:
        """氏名を生成する (姓のみ / 姓名 / 読み仮名表記を確率的に切り替える)。

        Claim: 検出率 / 低誤検出 — 姓は公開頻度分布に近い重み付きで引き、
        普通名詞や地名と同形の姓 (森・泉・大和・本田 など) も一定割合で出す。
        表記も漢字・カタカナ・ひらがな、区切りの有無を混ぜるため、
        「表層形が珍しいから拾えない」型の取りこぼしを検出率に反映できる。
        """
        rng = self.rng
        sei, sei_kana = self._pick_surname()
        mei, mei_kana, gender, era_style = self._pick_given()

        if full is None:
            r = rng.random()
            if r < 0.58:
                form = "full"
            elif r < 0.80:
                form = "sei"
            elif r < 0.88:
                form = "kana_full"
            elif r < 0.94:
                form = "kana_sei"
            else:
                form = "mei"
        else:
            form = "full" if full else "sei"

        sep = ""
        if form in ("full", "kana_full"):
            r = rng.random()
            sep = "" if r < 0.55 else (" " if r < 0.80 else "　")

        if form == "full":
            text = f"{sei}{sep}{mei}"
            reading = f"{sei_kana}{' ' if sep else ''}{mei_kana}"
            honorific_ok = True
        elif form == "sei":
            text, reading, mei, mei_kana = sei, sei_kana, "", ""
            honorific_ok = True
        elif form == "mei":
            text, reading, sei, sei_kana = mei, mei_kana, "", ""
            honorific_ok = False
        elif form == "kana_full":
            kana = f"{sei_kana}{sep}{mei_kana}"
            text = kana if rng.random() < 0.6 else _to_hiragana(kana)
            reading = f"{sei_kana} {mei_kana}"
            honorific_ok = True
        else:  # kana_sei
            text = sei_kana if rng.random() < 0.6 else _to_hiragana(sei_kana)
            reading = sei_kana
            mei, mei_kana = "", ""
            honorific_ok = True

        return PIIValue(
            text=text,
            label=PIIType.NAME,
            meta={
                "sei": sei,
                "mei": mei,
                "reading": reading,
                "sei_kana": sei_kana,
                "mei_kana": mei_kana,
                "honorific_ok": honorific_ok,
                "form": form,
                "gender": gender,
                "era_style": era_style,
                "kana_form": form.startswith("kana"),
            },
        )

    def reading_of(self, name_meta: dict[str, Any], *, style: str | None = None) -> PIIValue:
        """氏名メタからフリガナ表記の氏名 (カタカナ/ひらがな) を作る。

        Claim: 検出率 — 申込書の「フリガナ」欄のように、同一人物が同一文書内で
        別表記で二度現れる状況を作り、表記違いの取りこぼしを測れるようにする。
        """
        rng = self.rng
        sei_kana = str(name_meta.get("sei_kana") or "")
        mei_kana = str(name_meta.get("mei_kana") or "")
        if not sei_kana and not mei_kana:
            sei_kana = str(name_meta.get("reading") or "ヤマダ")
        kana = sei_kana + (" " if sei_kana and mei_kana else "") + mei_kana
        if style is None:
            style = "katakana" if rng.random() < 0.75 else "hiragana"
        text = kana if style == "katakana" else _to_hiragana(kana)
        return PIIValue(
            text=text,
            label=PIIType.NAME,
            meta={
                "sei": name_meta.get("sei", ""),
                "mei": name_meta.get("mei", ""),
                "reading": kana,
                "sei_kana": sei_kana,
                "mei_kana": mei_kana,
                "honorific_ok": True,
                "form": "furigana",
                "kana_form": True,
            },
        )

    def _pick_surname(self) -> tuple[str, str]:
        idx = self.rng.choices(range(len(_SURNAMES)), weights=_SURNAME_WEIGHTS, k=1)[0]
        return _SURNAMES[idx]

    def _pick_given(self) -> tuple[str, str, str, str]:
        styles = list(_ERA_STYLE_W)
        style = self.rng.choices(styles, weights=[_ERA_STYLE_W[s] for s in styles], k=1)[0]
        pool = [g for g in _GIVEN if g[3] == style] or list(_GIVEN)
        return self.rng.choice(pool)

    # -- 住所 -------------------------------------------------------------

    def address(self, *, with_postal: bool = False) -> PIIValue:
        """住所を生成する (都道府県+市区町村+町名 + 乱数の丁目-番-号)。

        Claim: 検出率 — 地名部分は公開の地理名だが、**番地は必ず乱数**であり
        実在の居所を指さない。ハイフン式と漢数字丁目式、建物名や部屋番号の有無を
        混ぜることで、住所検出の境界のゆれ (どこまでが住所か) を評価できる。
        """
        rng = self.rng
        pref, city, towns = rng.choice(_PLACES)
        town = rng.choice(towns)
        chome = rng.randint(1, 9)
        ban = rng.randint(1, 40)
        go = rng.randint(1, 30)

        r = rng.random()
        has_banchi = True
        if r < 0.42:
            banchi = f"{chome}-{ban}-{go}"
        elif r < 0.66:
            banchi = f"{chome}丁目{ban}番{go}号"
        elif r < 0.78:
            banchi = f"{_kanji_number(chome)}丁目{ban}番{go}号"
        elif r < 0.88:
            banchi = f"{chome}-{ban}"
        elif r < 0.94:
            banchi = f"{ban}番{go}号"
        else:
            banchi = ""
            has_banchi = False

        head = f"{pref}{city}" if rng.random() < 0.82 else city
        text = f"{head}{town}{banchi}"

        building = ""
        if rng.random() < 0.30:
            bname = rng.choice(_BLDG_PREFIX) + rng.choice(_BLDG_SUFFIX)
            room = rng.choice(
                [f"{rng.randint(1, 12)}0{rng.randint(1, 9)}", f"{rng.randint(1, 9)}{rng.randint(1,9)}{rng.randint(0,9)}"]
            )
            r2 = rng.random()
            if r2 < 0.45:
                building = f" {bname}{room}"
            elif r2 < 0.8:
                building = f" {bname} {room}号室"
            else:
                building = f"-{bname}{room}"
            text += building

        meta: dict[str, Any] = {
            "pref": pref,
            "city": city,
            "town": town,
            "chome": chome,
            "ban": ban,
            "go": go,
            "has_banchi": has_banchi,
            "building": building.strip(),
            "has_pref": head.startswith(pref),
        }

        if with_postal:
            pv = self._postal_code(pref=pref, mark=True)
            sep = " " if rng.random() < 0.75 else ""
            addr_start = len(pv.text) + len(sep)
            full = f"{pv.text}{sep}{text}"
            meta["postal"] = pv.text
            meta["parts"] = [
                (1, len(pv.text), PIIType.POSTAL_CODE.value),   # 〒 を除いた数字部分
                (addr_start, len(full), PIIType.ADDRESS.value),
            ]
            return PIIValue(text=full, label=PIIType.ADDRESS, meta=meta)

        return PIIValue(text=text, label=PIIType.ADDRESS, meta=meta)

    # -- 電話 -------------------------------------------------------------

    def phone(self, kind: str | None = None) -> PIIValue:
        """日本の番号計画に沿った電話番号を生成する。

        Claim: 検出率 / 低誤検出 — 携帯 11 桁 (070/080/090)、固定 10 桁
        (市外局番 2/3/4 桁に応じて市内局番の桁が変わる)、フリーダイヤル
        0120/0800、IP 電話 050 を桁数まで正しく作る。区切りの揺れ
        (ハイフン / なし / 括弧 / 全角) を混ぜるので、「桁数を数えない」実装や
        「日付を電話と誤る」実装をこのデータで検出できる。
        """
        rng = self.rng
        if kind is None:
            kind = rng.choices(
                ["mobile", "landline", "tollfree", "ip"], weights=[0.5, 0.34, 0.1, 0.06], k=1
            )[0]
        if kind == "mobile":
            head = rng.choice(("070", "080", "090"))
            parts = [head, f"{rng.randrange(1000, 10000)}", f"{rng.randrange(0, 10000):04d}"]
        elif kind == "tollfree":
            if rng.random() < 0.6:
                parts = ["0120", f"{rng.randrange(100, 1000)}", f"{rng.randrange(100, 1000)}"]
            else:
                parts = ["0800", f"{rng.randrange(100, 1000)}", f"{rng.randrange(0, 10000):04d}"]
        elif kind == "ip":
            parts = ["050", f"{rng.randrange(1000, 10000)}", f"{rng.randrange(0, 10000):04d}"]
        else:
            kind = "landline"
            r = rng.random()
            if r < 0.30:
                area = rng.choice(_AREA2)
                parts = [area, f"{rng.randrange(1000, 10000)}", f"{rng.randrange(0, 10000):04d}"]
            elif r < 0.80:
                area = rng.choice(_AREA3)
                parts = [area, f"{rng.randrange(100, 1000)}", f"{rng.randrange(0, 10000):04d}"]
            else:
                area = rng.choice(_AREA4)
                parts = [area, f"{rng.randrange(10, 100)}", f"{rng.randrange(0, 10000):04d}"]

        r = rng.random()
        if r < 0.58:
            text, style = "-".join(parts), "hyphen"
        elif r < 0.74:
            text, style = "".join(parts), "plain"
        elif r < 0.84:
            text, style = f"({parts[0]}){parts[1]}-{parts[2]}", "paren"
        elif r < 0.90:
            text, style = f"{parts[0]}({parts[1]}){parts[2]}", "paren_mid"
        elif r < 0.96:
            text, style = " ".join(parts), "space"
        else:
            text, style = _fullwidth("-".join(parts)), "fullwidth"

        digits = "".join(parts)
        return PIIValue(
            text=text,
            label=PIIType.PHONE,
            meta={
                "kind": kind,
                "digits": digits,
                "n_digits": len(digits),
                "style": style,
                "area": parts[0],
            },
        )

    # -- メール -----------------------------------------------------------

    #: 予約済みドメインのみ (RFC 2606 / JPRS の例示用ドメイン)。実在アドレスは決して作らない。
    DOMAINS: tuple[str, ...] = (
        "example.com", "example.co.jp", "example.net", "example.org", "example.jp",
        "example.ne.jp", "example.or.jp", "mail.example.com", "corp.example.co.jp",
        "example.test", "example.invalid", "sub.example.test",
    )

    def email(self, name_meta: dict[str, Any] | None = None) -> PIIValue:
        """メールアドレスを生成する (氏名メタがあればローマ字から導出)。

        Claim: 検出率 / 低誤検出 — ドメインは **予約済みの例示用ドメインだけ** を使うので、
        実在のアドレスが生成される余地がない。氏名由来のローカル部にすることで、
        「氏名を伏せてもメールから同定できてしまう」実務上の穴を評価対象にできる。
        """
        rng = self.rng
        derived = "random"
        if name_meta is None:
            name_meta = self.name(full=True).meta
        sei_kana = str(name_meta.get("sei_kana") or name_meta.get("reading") or "ヤマダ")
        mei_kana = str(name_meta.get("mei_kana") or "")
        long_vowel = rng.random() < 0.5
        sei_r = _to_romaji(sei_kana, long_vowel=long_vowel)
        mei_r = _to_romaji(mei_kana, long_vowel=long_vowel) if mei_kana else ""
        derived = "name"

        if not mei_r:
            choices = [
                f"{sei_r}{rng.randrange(1, 99)}",
                f"{sei_r}_{rng.randrange(100, 999)}",
                f"{sei_r}.{rng.choice('abcdefghijklmnopqrstuvwxyz')}",
                sei_r,
            ]
        else:
            choices = [
                f"{sei_r}.{mei_r}",
                f"{mei_r}.{sei_r}",
                f"{sei_r}_{mei_r}",
                f"{mei_r[0]}.{sei_r}",
                f"{sei_r}-{mei_r[0]}",
                f"{sei_r}{mei_r}",
                f"{sei_r}{mei_r[0]}{rng.randrange(1, 99)}",
            ]
        local = rng.choice(choices)
        if rng.random() < 0.06:
            local = rng.choice(("info", "support", "contact", "sales", "office")) + str(rng.randrange(1, 9))
            derived = "role"
        domain = rng.choice(self.DOMAINS)
        text = f"{local}@{domain}"
        if rng.random() < 0.05:
            text = _fullwidth(text)
        return PIIValue(
            text=text,
            label=PIIType.EMAIL,
            meta={"local": local, "domain": domain, "derived_from": derived,
                  "sei": name_meta.get("sei", ""), "mei": name_meta.get("mei", "")},
        )

    # -- 生年月日 ---------------------------------------------------------

    def dob(self, era: str | None = None) -> PIIValue:
        """生年月日を生成する (西暦・和暦、区切りの揺れを含む)。

        Claim: 検出率 — 和暦と西暦の対応 (改元日をまたぐ元号年計算) を正しく行い、
        暦として妥当な日付だけを作る。「日付らしい文字列」と「生年月日」を
        文脈で見分けられるかを評価するための正解データになる。
        """
        rng = self.rng
        start = _dt.date(1935, 1, 1).toordinal()
        end = _dt.date(2008, 12, 31).toordinal()
        d = _dt.date.fromordinal(rng.randint(start, end))
        if era is None:
            era = "seireki" if rng.random() < 0.55 else "wareki"

        if era == "seireki":
            r = rng.random()
            if r < 0.50:
                text, fmt = f"{d.year}年{d.month}月{d.day}日", "kanji"
            elif r < 0.70:
                text, fmt = f"{d.year}/{d.month:02d}/{d.day:02d}", "slash"
            elif r < 0.85:
                text, fmt = f"{d.year}.{d.month}.{d.day}", "dot"
            else:
                text, fmt = f"{d.year}-{d.month:02d}-{d.day:02d}", "hyphen"
            era_name, era_alpha, era_year = _to_wareki(d)
        else:
            era_name, era_alpha, era_year = _to_wareki(d)
            y = "元" if era_year == 1 else str(era_year)
            r = rng.random()
            if r < 0.55:
                text, fmt = f"{era_name}{y}年{d.month}月{d.day}日", "wareki_kanji"
            elif r < 0.75:
                text, fmt = f"{era_alpha}{era_year}.{d.month}.{d.day}", "wareki_alpha"
            elif r < 0.88:
                text, fmt = f"{era_name}{y}年{d.month:02d}月{d.day:02d}日", "wareki_kanji2"
            else:
                text, fmt = f"{era_alpha}{era_year}/{d.month:02d}/{d.day:02d}", "wareki_slash"

        return PIIValue(
            text=text,
            label=PIIType.DOB,
            meta={
                "iso": d.isoformat(),
                "year": d.year,
                "month": d.month,
                "day": d.day,
                "era": era,
                "era_name": era_name,
                "era_year": era_year,
                "format": fmt,
                "age_ref": 2025 - d.year,
            },
        )

    # -- 金融 -------------------------------------------------------------

    def bank_account(self) -> PIIValue:
        """金融機関口座を生成する (コード形式と、支店名を伴う文章形式)。

        Claim: 検出率 — 「銀行コード4-支店3-口座7」の数字列と、
        「みずほ銀行 新宿支店 普通 1234567」の散文形式の両方を作る。
        銀行名は実在名だが **番号はすべて乱数** で、実在の口座を指さない。
        """
        rng = self.rng
        bank_code = f"{rng.randrange(1, 10000):04d}"
        branch_code = f"{rng.randrange(1, 1000):03d}"
        account = f"{rng.randrange(0, 10_000_000):07d}"
        bank = rng.choice(_BANKS)
        branch = rng.choice(_BRANCHES)
        kind = rng.choices(_ACCOUNT_KIND, weights=[0.78, 0.16, 0.06], k=1)[0]

        r = rng.random()
        if r < 0.28:
            text, fmt = f"{bank_code}-{branch_code}-{account}", "code_hyphen"
        elif r < 0.38:
            text, fmt = f"{bank_code}{branch_code}{account}", "code_plain"
        elif r < 0.70:
            text, fmt = f"{bank} {branch} {kind} {account}", "prose"
        elif r < 0.84:
            text, fmt = f"{bank}{branch} {kind} {account}", "prose_tight"
        elif r < 0.93:
            text, fmt = f"{bank} {branch}（{branch_code}） {kind} {account}", "prose_code"
        else:
            text, fmt = f"{kind} {account}", "kind_only"

        return PIIValue(
            text=text,
            label=PIIType.BANK_ACCOUNT,
            meta={
                "bank": bank, "branch": branch, "bank_code": bank_code,
                "branch_code": branch_code, "account": account,
                "account_kind": kind, "format": fmt,
            },
        )

    def credit_card(self) -> PIIValue:
        """クレジットカード様式の16桁を生成する (**Luhn を故意に外す**)。

        Claim: 低誤検出 / 検出率 — 実在しうる番号を絶対に作らないため、
        検査数字をわざと誤らせる。規則層は「形式で検出し、checksum の成否は
        ``meta['checksum_valid']`` に記録するだけ」でなければならず、
        このデータはその設計を強制する (checksum を検出条件にすると全滅する)。
        """
        rng = self.rng
        prefix = rng.choice(_IIN)
        body = prefix + "".join(str(rng.randrange(10)) for _ in range(15 - len(prefix)))
        correct = _luhn_check_digit(body)
        wrong = (correct + rng.randint(1, 9)) % 10
        digits = body + str(wrong)
        assert len(digits) == 16

        r = rng.random()
        if r < 0.45:
            text, fmt = "-".join(digits[i : i + 4] for i in range(0, 16, 4)), "hyphen"
        elif r < 0.75:
            text, fmt = " ".join(digits[i : i + 4] for i in range(0, 16, 4)), "space"
        else:
            text, fmt = digits, "plain"

        return PIIValue(
            text=text,
            label=PIIType.CREDIT_CARD,
            meta={
                "digits": digits, "iin": prefix, "format": fmt,
                "checksum_valid": False, "correct_check_digit": correct,
                "used_check_digit": wrong, "scheme": _iin_scheme(digits),
            },
        )

    def mynumber(self) -> PIIValue:
        """個人番号 (マイナンバー) 様式の12桁を生成する (**検査数字を故意に外す**)。

        Claim: 低誤検出 / 検出率 — 桁数と区切りは実物どおり、検査数字だけを外すことで
        「実在しうる番号を作らない」制約と「形式で検出できる」要件を両立させる。
        """
        rng = self.rng
        body = "".join(str(rng.randrange(10)) for _ in range(11))
        correct = _mynumber_check_digit(body)
        wrong = (correct + rng.randint(1, 10)) % 11
        if wrong > 9:
            wrong = (correct + 1) % 10
        if wrong == correct:
            wrong = (correct + 1) % 10
        digits = body + str(wrong)
        assert len(digits) == 12

        r = rng.random()
        if r < 0.5:
            text, fmt = digits, "plain"
        elif r < 0.8:
            text, fmt = f"{digits[0:4]} {digits[4:8]} {digits[8:12]}", "space"
        else:
            text, fmt = f"{digits[0:4]}-{digits[4:8]}-{digits[8:12]}", "hyphen"

        return PIIValue(
            text=text,
            label=PIIType.MYNUMBER,
            meta={
                "digits": digits, "format": fmt, "checksum_valid": False,
                "correct_check_digit": correct, "used_check_digit": int(digits[-1]),
            },
        )

    # -- 会員番号・郵便番号 -----------------------------------------------

    def member_id(self) -> PIIValue:
        """会員番号・顧客ID・契約番号などの接頭辞付き英数IDを生成する。

        Claim: 低誤検出 — 型番や注文番号と形が近いため、文脈語 (会員番号/顧客ID/契約番号)
        がないときに拾いすぎないかを試す素材になる。
        """
        rng = self.rng
        prefix, kind = rng.choice(_MEMBER_PREFIX)
        r = rng.random()
        if r < 0.30:
            body = f"{rng.randrange(2015, 2026)}-{rng.randrange(0, 1_000_000):06d}"
            text = f"{prefix}-{body}"
            fmt = "prefix-year-serial"
        elif r < 0.55:
            text = f"{prefix}{rng.randrange(0, 100_000_000):08d}"
            fmt = "prefix+8"
        elif r < 0.72:
            text = f"{prefix}-{rng.randrange(0, 100_000_000):08d}"
            fmt = "prefix-8"
        elif r < 0.86:
            text = f"{prefix}{rng.randrange(0, 100_000):05d}"
            fmt = "prefix+5"
        else:
            grp = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(2))
            text = f"{prefix}{grp}-{rng.randrange(0, 10000):04d}-{rng.randrange(0, 10000):04d}"
            fmt = "prefix+alpha-groups"
        return PIIValue(
            text=text,
            label=PIIType.MEMBER_ID,
            meta={"prefix": prefix, "kind": kind, "format": fmt},
        )

    def postal_code(self) -> PIIValue:
        """郵便番号を生成する (都道府県に整合する上位3桁 + 乱数の下4桁)。

        Claim: 検出率 — 〒記号の有無・ハイフンの有無という表記の揺れを含み、
        記号を含めるか否かで境界がぶれる問題を評価できる。
        """
        return self._postal_code()

    def _postal_code(self, *, pref: str | None = None, mark: bool | None = None) -> PIIValue:
        rng = self.rng
        if pref is None:
            pref = rng.choice(tuple(_POSTAL_PREFIX))
        head = rng.choice(_POSTAL_PREFIX.get(pref, ("100",)))
        tail = f"{rng.randrange(0, 10000):04d}"
        if mark is None:
            mark = rng.random() < 0.45
        if rng.random() < 0.9:
            code = f"{head}-{tail}"
            fmt = "hyphen"
        else:
            code = f"{head}{tail}"
            fmt = "plain"
        meta: dict[str, Any] = {"pref": pref, "head": head, "tail": tail,
                                "format": fmt, "has_mark": bool(mark)}
        if mark:
            text = f"〒{code}"
            meta["parts"] = [(1, len(text), PIIType.POSTAL_CODE.value)]
        else:
            text = code
        return PIIValue(text=text, label=PIIType.POSTAL_CODE, meta=meta)

    # -- ディスパッチ -----------------------------------------------------

    def make(self, t: PIIType) -> PIIValue:
        """種別を指定して1件生成する (テンプレート展開の共通入口)。

        Claim: 検出率 — 全10種別を同じ入口から生成できることで、
        種別ごとの件数バランスを制御した評価データを組める。
        """
        fn = {
            PIIType.NAME: self.name,
            PIIType.ADDRESS: self.address,
            PIIType.PHONE: self.phone,
            PIIType.EMAIL: self.email,
            PIIType.DOB: self.dob,
            PIIType.BANK_ACCOUNT: self.bank_account,
            PIIType.CREDIT_CARD: self.credit_card,
            PIIType.MYNUMBER: self.mynumber,
            PIIType.MEMBER_ID: self.member_id,
            PIIType.POSTAL_CODE: self.postal_code,
        }[PIIType(t)]
        return fn()


def _iin_scheme(digits: str) -> str:
    """IIN から発行ブランドの *形状* 名を返す内部関数 (値は無効なので実カードではない)。"""
    if digits.startswith("4"):
        return "visa-shaped"
    if digits[:2] in {"51", "52", "53", "54", "55"}:
        return "mastercard-shaped"
    if digits[:2] == "35":
        return "jcb-shaped"
    return "unknown-shaped"


# ---------------------------------------------------------------------------
# 4. 業務文書テンプレート
# ---------------------------------------------------------------------------

GENRES: tuple[str, ...] = ("email", "minutes", "application", "inquiry")

#: PII を差し込むスロットの基底名 -> 生成方法。末尾の数字は「同一文書内の同一人物/同一値」を表す。
_PII_BASES: dict[str, str] = {
    "name": "NAME",
    "kana": "READING",
    "addr": "ADDRESS",
    "tel": "PHONE",
    "mobile": "PHONE_MOBILE",
    "fax": "PHONE_LANDLINE",
    "email": "EMAIL",
    "dob": "DOB",
    "bank": "BANK_ACCOUNT",
    "card": "CREDIT_CARD",
    "mynum": "MYNUMBER",
    "member": "MEMBER_ID",
    "zip": "POSTAL_CODE",
}

#: n_pii で削られたスロットに入る、PII でない代替表記 (実務でもよくある伏せ字)。
_OMIT_TEXT: dict[str, str] = {
    "name": "ご本人", "kana": "(省略)", "addr": "(登録住所に同じ)",
    "tel": "(記載なし)", "mobile": "(記載なし)", "fax": "(記載なし)",
    "email": "(未登録)", "dob": "(非公開)", "bank": "(別途ご案内)",
    "card": "(別途ご案内)", "mynum": "(未提出)", "member": "(新規)",
    "zip": "(未記入)",
}

_TOKEN_RE = re.compile(r"\{([a-z_]+[0-9]*)\}")

TEMPLATES: dict[str, tuple[tuple[str, str], ...]] = {
    "email": (
        (
            "email_business_reply",
            "件名: {subject}の件（ご連絡）\n\n"
            "{company1}\n{dept1} {name1} 様\n\n"
            "拝啓 時下ますますご清栄のこととお慶び申し上げます。\n"
            "平素は格別のご高配を賜り、厚く御礼申し上げます。\n\n"
            "{base}さて、先般ご依頼いただきました{subject}につきまして、"
            "下記のとおりご連絡いたします。\n\n"
            "  ご登録氏名   {name2}\n"
            "  ご登録住所   〒{zip1} {addr1}\n"
            "  ご連絡先     {tel1}\n"
            "  メール       {email2}\n\n"
            "ご不明な点がございましたら、下記までお問い合わせくださいますようお願い申し上げます。\n"
            "敬具\n\n"
            "------------------------------\n"
            "{company2} {dept2}\n"
            "{name3}\n"
            "TEL {tel3} / FAX {fax3}\n"
            "{email3}\n"
            "------------------------------\n",
        ),
        (
            "email_internal_share",
            "{dept1}各位\n\n"
            "お疲れさまです。{dept2}の{name1}です。\n"
            "{base}標記の件について、下記のとおり共有します。\n\n"
            "・対象      {subject}\n"
            "・担当      {name2}（内線 {ext}）\n"
            "・連絡先    {tel2}\n"
            "・提出先    {email2}\n"
            "・期限      {date1}\n\n"
            "資料は共有フォルダ（管理番号 {order}）に格納しています。\n"
            "ご対応のほど、よろしくお願いいたします。\n\n"
            "{name1}\n{dept2} / 内線 {ext}\n",
        ),
        (
            "email_customer_notice",
            "{name1} 様\n\n"
            "平素より{company1}をご利用いただき、誠にありがとうございます。\n"
            "{dept1}でございます。\n\n"
            "{base}このたび、ご登録情報の確認をお願いしたくご連絡いたしました。\n"
            "お手数ですが、下記の内容に相違がないかご確認ください。\n\n"
            "  会員番号     {member1}\n"
            "  お名前       {name1}\n"
            "  生年月日     {dob1}\n"
            "  ご住所       〒{zip1} {addr1}\n"
            "  お電話番号   {tel1}\n"
            "  メール       {email1}\n\n"
            "相違がある場合は、{date1}までに本メールへご返信ください。\n"
            "今後とも変わらぬご愛顧を賜りますようお願い申し上げます。\n\n"
            "{company1} {dept1}\n{name2}\n{tel2}\n",
        ),
        (
            "email_invoice",
            "{company1} {dept1} {name1} 様\n\n"
            "拝啓 貴社ますますご清栄のこととお慶び申し上げます。\n"
            "平素は格別のお引き立てを賜り、誠にありがとうございます。\n\n"
            "{base}さて、{date1}付にてご請求申し上げました{subject}につきまして、"
            "下記口座へお振込みくださいますようお願い申し上げます。\n\n"
            "  お振込先   {bank1}\n"
            "  ご請求額   {amount}円（税込）\n"
            "  お支払期限 {date2}\n"
            "  請求書番号 {order}\n\n"
            "なお、行き違いにお振込みいただいておりました節は、何卒ご容赦ください。\n"
            "敬具\n\n"
            "{company2} 経理部 {name2}\n"
            "TEL {tel2} / FAX {fax2}\n{email2}\n",
        ),
        (
            "email_support_reply",
            "{name1} 様\n\n"
            "お問い合わせいただき、誠にありがとうございます。\n"
            "{company1} カスタマーサポート担当の{name2}でございます。\n\n"
            "  お問い合わせ番号 {ticket}\n"
            "  受付日時         {date1} {time1}\n\n"
            "{base}ご照会の件につきまして確認いたしましたところ、"
            "ご登録のご連絡先は {tel1}、ご住所は {addr1} となっておりました。\n"
            "ご変更をご希望の場合は、会員番号 {member1} をお手元にご用意のうえ、"
            "本メールへご返信ください。\n\n"
            "引き続きよろしくお願いいたします。\n\n"
            "{company1} カスタマーサポート\n{email2} / {tel2}\n",
        ),
        (
            "email_appointment",
            "{name1} 様\n\n"
            "いつもお世話になっております。{company1}の{name2}でございます。\n\n"
            "{base}先日ご相談いただいた{subject}につきまして、"
            "下記日程でお打ち合わせをお願いできればと存じます。\n\n"
            "  日時   {date1} {time1}から\n"
            "  場所   {place}\n"
            "  当日連絡先 {mobile2}\n\n"
            "ご都合が合わない場合は、{date2}までにお知らせください。\n"
            "また、当日は受付にて会員番号 {member1} をご提示ください。\n"
            "何卒よろしくお願い申し上げます。\n\n"
            "{company1} {dept1} {name2}\n{email2}\n",
        ),
    ),
    "minutes": (
        (
            "minutes_project",
            "{subject}に関する打合せ 議事録\n\n"
            "日時   {date1} {time1} - {time2}\n"
            "場所   {place}\n"
            "出席者 {name1}（{dept1}）、{name2}（{dept2}）、{name3}（{company2}）\n"
            "記録者 {name4}\n\n"
            "1. 議題\n"
            "   (1) {topic1}\n"
            "   (2) {topic2}\n\n"
            "2. 報告事項\n"
            "{base}   {name1}より{topic1}について報告があった。"
            "進捗は概ね予定どおりであり、次回までに詳細資料を配布する。\n\n"
            "3. 決定事項\n"
            "   ・{topic2}の担当を{name2}とする。\n"
            "   ・問い合わせ窓口は {tel1}（{dept1}直通）とする。\n"
            "   ・資料送付先は {email3} に統一する。\n\n"
            "4. 次回予定\n"
            "   {date2} {time1}より {place} にて実施。\n\n"
            "以上\n",
        ),
        (
            "minutes_hr_interview",
            "採用面接記録\n\n"
            "実施日 {date1}   実施時間 {time1} - {time2}\n"
            "場所   {place}\n"
            "面接官 {name2}（{dept1}）、{name3}（{dept2}）\n\n"
            "【応募者情報】\n"
            "  氏名       {name1}\n"
            "  フリガナ   {kana1}\n"
            "  生年月日   {dob1}\n"
            "  現住所     {addr1}\n"
            "  連絡先     {mobile1}\n"
            "  メール     {email1}\n"
            "  応募番号   {member1}\n\n"
            "【所見】\n"
            "{base}  {name1}は前職での{topic1}の経験について具体的に説明した。\n"
            "  {name2}より{topic2}に関する確認を行い、いずれも十分な回答が得られた。\n\n"
            "【結論】\n"
            "  二次面接に進める。日程は{date2}を第一候補とする。\n\n"
            "以上\n",
        ),
        (
            "minutes_management",
            "管理組合 定例理事会 議事録\n\n"
            "日時   {date1} {time1}より\n"
            "場所   {place}\n"
            "出席者 理事長 {name1}、副理事長 {name2}、会計 {name3}、監事 {name4}\n"
            "欠席者 なし\n\n"
            "議題1 {topic1}について\n"
            "{base}  {name1}より{topic1}の現況説明があり、審議の結果、原案どおり承認された。\n\n"
            "議題2 {topic2}について\n"
            "  {name3}より収支報告があった。修繕積立金の振込先を\n"
            "  {bank1} に変更する件、全員一致で承認。\n\n"
            "議題3 連絡体制について\n"
            "  緊急連絡先を {tel1}（{name2}）とし、掲示板に掲出する。\n"
            "  書面送付先は {addr1} で変更なし。\n\n"
            "次回開催 {date2} {time1}より 同会場\n\n"
            "以上\n"
            "記録 {name4}\n",
        ),
        (
            "minutes_incident",
            "障害対応 報告会 議事録\n\n"
            "件名   {subject}\n"
            "日時   {date1} {time1} - {time2}\n"
            "場所   {place}（一部オンライン参加）\n"
            "出席者 {name1}（{dept1}）、{name2}（{dept2}）、{name3}（{company2}）\n"
            "記録   {name2}\n\n"
            "1. 事象概要\n"
            "{base}   管理番号 {order} の設備において{topic1}が発生した。\n"
            "   一次受付は {tel1}（{dept1}）で行い、{time1}に{name1}へエスカレーションした。\n\n"
            "2. 対応経過\n"
            "   {time2}までに暫定対応を完了。顧客窓口の{name3}へ経過を報告した。\n"
            "   顧客連絡先: {tel2} / {email3}\n\n"
            "3. 再発防止策\n"
            "   ・{topic2}の手順書を{date2}までに改訂する（担当 {name2}）。\n"
            "   ・受付番号 {ticket} を親番号として関連記録を紐付ける。\n\n"
            "以上\n",
        ),
        (
            "minutes_sales_review",
            "{dept1} 週次営業会議 議事録\n\n"
            "日時 {date1} {time1}から\n"
            "場所 {place}\n"
            "参加 {name1}、{name2}、{name3}\n\n"
            "1. 前週実績\n"
            "{base}   {name1}より、{company2}向け{subject}の進捗報告。"
            "先方窓口は {name4} 様（{tel1}）。\n"
            "   見積番号 {order}、提示金額 {amount}円。\n\n"
            "2. 課題\n"
            "   ・{topic1}について、先方{dept2}の承認待ち。\n"
            "   ・請求先住所が {addr1} に変更となる見込み。\n\n"
            "3. 決定事項\n"
            "   ・{name2}が{date2}までに再見積を提出する。\n"
            "   ・連絡は {email1} に一本化する。\n\n"
            "以上\n",
        ),
    ),
    "application": (
        (
            "application_service",
            "{service}申込書\n\n"
            "受付番号 {ticket}     受付日 {date1}\n\n"
            "【申込者】\n"
            "  フリガナ     {kana1}\n"
            "  氏名         {name1}\n"
            "  生年月日     {dob1}\n"
            "  郵便番号     〒{zip1}\n"
            "  住所         {addr1}\n"
            "  電話番号     {tel1}\n"
            "  携帯電話     {mobile1}\n"
            "  メール       {email1}\n\n"
            "【お支払い】\n"
            "  支払方法     クレジットカード\n"
            "  カード番号   {card1}\n"
            "  有効期限     {exp}\n"
            "  引落口座     {bank1}\n\n"
            "【本人確認】\n"
            "  個人番号     {mynum1}\n"
            "  会員番号     {member1}\n\n"
            "{base}上記のとおり相違ありません。{service}の利用規約に同意のうえ申し込みます。\n\n"
            "  申込日       {date2}\n"
            "  申込者署名   {name1}\n",
        ),
        (
            "application_membership",
            "会員登録申込書（{service}）\n\n"
            "  整理番号 {ticket}\n\n"
            "1. 申込者情報\n"
            "  フリガナ   {kana1}\n"
            "  お名前     {name1}\n"
            "  生年月日   {dob1}\n"
            "  ご住所     〒{zip1} {addr1}\n"
            "  日中連絡先 {tel1}\n"
            "  メール     {email1}\n\n"
            "2. ご家族（同居）\n"
            "  お名前     {name2}\n"
            "  続柄       配偶者\n"
            "  連絡先     {mobile2}\n\n"
            "3. 引落口座\n"
            "  {bank1}\n"
            "  口座名義   {kana1}\n\n"
            "4. 確認事項\n"
            "{base}  ・記載内容に虚偽がないこと\n"
            "  ・規約第{clause}条に同意すること\n\n"
            "  申込日 {date1}     受付担当 {name3}\n",
        ),
        (
            "application_change",
            "登録内容変更届\n\n"
            "受付番号 {ticket}   受付日 {date1}   受付担当 {name3}（{dept1}）\n\n"
            "会員番号 {member1}\n"
            "氏名     {name1}（{kana1}）\n"
            "生年月日 {dob1}\n\n"
            "【変更前】\n"
            "  住所     〒{zip2} {addr2}\n"
            "  電話     {tel2}\n\n"
            "【変更後】\n"
            "  住所     〒{zip1} {addr1}\n"
            "  電話     {tel1}\n"
            "  メール   {email1}\n\n"
            "【変更理由】\n"
            "{base}  転居のため（旧住所からの転送手続き済み）。\n\n"
            "上記のとおり変更を届け出ます。\n"
            "  届出日 {date2}   署名 {name1}\n",
        ),
        (
            "application_insurance",
            "{service}加入申込書\n\n"
            "証券番号（仮） {order}\n"
            "受付番号       {ticket}\n\n"
            "■契約者\n"
            "  フリガナ {kana1}\n"
            "  氏名     {name1}\n"
            "  生年月日 {dob1}\n"
            "  住所     〒{zip1} {addr1}\n"
            "  電話     {tel1}\n"
            "  メール   {email1}\n\n"
            "■被保険者（契約者と異なる場合のみ記入）\n"
            "  氏名     {name2}\n"
            "  生年月日 {dob2}\n"
            "  続柄     子\n\n"
            "■保険料\n"
            "  月額     {amount}円\n"
            "  払込方法 口座振替\n"
            "  振替口座 {bank1}\n\n"
            "■本人確認書類\n"
            "  個人番号カード（個人番号 {mynum1}）\n\n"
            "{base}以上の内容で申し込みます。\n"
            "  申込日 {date1}   募集人 {name3}（登録番号 {order}）\n",
        ),
        (
            "application_event",
            "{service} 参加申込書\n\n"
            "申込ID {ticket}\n\n"
            "参加者\n"
            "  氏名     {name1}\n"
            "  フリガナ {kana1}\n"
            "  所属     {company1} {dept1}\n"
            "  連絡先   {mobile1}\n"
            "  メール   {email1}\n"
            "  緊急連絡先 {tel2}（{name2}）\n\n"
            "会場までの送迎を希望する場合は下記へ配車します。\n"
            "  住所 {addr1}\n\n"
            "参加費 {amount}円（当日会場でお支払い、またはカード決済）\n"
            "  カード番号 {card1}\n\n"
            "{base}開催日 {date1} {time1}より、{place}にて。\n"
            "  申込日 {date2}\n",
        ),
    ),
    "inquiry": (
        (
            "inquiry_history",
            "【問い合わせ履歴】\n\n"
            "受付番号 {ticket}\n"
            "受付日時 {date1} {time1}\n"
            "チャネル 電話（{tel3} 着信）\n"
            "対応者   {name2}（{dept1}）\n\n"
            "■お客様情報\n"
            "  お名前   {name1} 様\n"
            "  フリガナ {kana1}\n"
            "  会員番号 {member1}\n"
            "  ご連絡先 {tel1}\n"
            "  メール   {email1}\n"
            "  ご住所   {addr1}\n\n"
            "■お問い合わせ内容\n"
            "{base}  {name1}様より、{topic1}についてお問い合わせをいただいた。\n"
            "  ご登録の生年月日（{dob1}）にて本人確認を実施済み。\n\n"
            "■対応内容\n"
            "  {name2}が{topic2}についてご案内。\n"
            "  折り返しのご連絡先として {mobile1} を伺った。\n\n"
            "■ステータス\n"
            "  対応中（次回連絡予定日 {date2}）\n",
        ),
        (
            "inquiry_complaint",
            "お客様対応記録（重要度: 高）\n\n"
            "受付番号 {ticket}   受付 {date1} {time1}   一次対応 {name2}\n"
            "エスカレーション先 {name3}（{dept1}、内線 {ext}）\n\n"
            "1. お客様\n"
            "   {name1} 様（{kana1}）／会員番号 {member1}\n"
            "   ご住所 〒{zip1} {addr1}\n"
            "   ご連絡先 {tel1}（日中）／{mobile1}（携帯）\n\n"
            "2. 申し出内容\n"
            "{base}   {date2}に納品された{subject}（管理番号 {order}）について、"
            "{topic1}との申し出。\n\n"
            "3. 対応経過\n"
            "   {time1} 一次受付（{name2}）。\n"
            "   {time2} {name3}より折り返しご連絡。{topic2}を提案し、ご了承いただいた。\n\n"
            "4. 今後の対応\n"
            "   返金分は下記口座へ振込予定。\n"
            "   {bank1}\n"
            "   完了報告は {email1} 宛に送付する。\n",
        ),
        (
            "inquiry_web_form",
            "Webフォーム受信内容\n\n"
            "受付番号 {ticket}\n"
            "送信日時 {date1} {time1}\n"
            "フォーム {service}に関するお問い合わせ\n\n"
            "お名前     {name1}\n"
            "フリガナ   {kana1}\n"
            "メール     {email1}\n"
            "電話番号   {tel1}\n"
            "郵便番号   {zip1}\n"
            "ご住所     {addr1}\n"
            "生年月日   {dob1}\n"
            "会員番号   {member1}\n\n"
            "お問い合わせ本文:\n"
            "{base}{topic1}について確認したく連絡しました。"
            "先日{date2}に申し込んだ{service}の件です。よろしくお願いします。\n\n"
            "---\n"
            "自動返信済み（{email1} 宛）／担当割当 {name2}（{dept1}）\n",
        ),
        (
            "inquiry_callback",
            "架電記録\n\n"
            "受付番号 {ticket}   架電日 {date1}   担当 {name2}（{dept1}）\n\n"
            "架電先   {tel1}（{name1} 様）\n"
            "結果     応答あり（{time1}、通話 {minutes}分）\n\n"
            "内容\n"
            "{base}  {name1}様へ{topic1}のご案内。ご住所（{addr1}）に案内状を"
            "再送する旨をお伝えした。\n"
            "  ご本人確認は生年月日 {dob1} と会員番号 {member1} で実施。\n"
            "  カード情報の変更希望あり。新しいカード番号は {card1}、\n"
            "  引落口座の変更は {bank1} へ。\n\n"
            "次回架電予定 {date2} {time2}頃、携帯 {mobile1} 宛。\n",
        ),
        (
            "inquiry_ticket_summary",
            "サポートチケット要約\n\n"
            "チケット {ticket} / 優先度 中 / 状態 保留\n"
            "起票 {date1} {time1}   起票者 {name2}（{dept1}）\n\n"
            "顧客: {name1}（{company1} {dept2}）\n"
            "  連絡先 {tel1} ／ {email1}\n"
            "  請求先 〒{zip1} {addr1}\n"
            "  契約番号 {member1}\n\n"
            "事象:\n"
            "{base}  {topic1}が再現するとの申告。対象は管理番号 {order}、\n"
            "  型番 {model} の製品。\n\n"
            "調査結果:\n"
            "  {topic2}が原因と判断。代替品を{date2}に発送予定。\n"
            "  送付先は上記住所、受取人は {name1} 様。\n\n"
            "備考:\n"
            "  費用が発生する場合は {bank1} へご請求。\n",
        ),
    ),
}

# --- 非 PII の穴埋め素材 (自然な「紛らわしい数字」を含む) -------------------

_COMPANY_HEAD: tuple[str, ...] = (
    "青葉", "光洋", "明和", "大成", "三葉", "東洋", "新和", "千歳", "白鳥", "双葉",
    "みどり", "山手", "けやき", "あかつき", "大和", "本田", "富士見", "北斗", "南星",
)
_COMPANY_TAIL: tuple[str, ...] = (
    "商事株式会社", "工業株式会社", "システムズ株式会社", "運輸株式会社",
    "電機株式会社", "物産株式会社", "サービス株式会社", "建設株式会社",
)
_DEPTS: tuple[str, ...] = (
    "営業部", "営業第一課", "総務部", "人事部", "経理部", "情報システム部",
    "カスタマーサポート部", "品質保証部", "開発部", "購買部", "法務部", "広報部",
    "お客様相談室", "施設管理課",
)
_SUBJECTS: tuple[str, ...] = (
    "定期メンテナンス", "料金プラン変更", "納品スケジュール", "新サービス導入",
    "セキュリティ研修", "見積書", "契約更新", "請求書発行", "システム移行",
    "在庫調整", "配送遅延", "会員規約改定", "年次点検", "設備更新工事",
)
_TOPICS: tuple[str, ...] = (
    "在庫管理システムの刷新", "問い合わせ対応の一次受付フロー", "配送遅延の原因究明",
    "個人情報の取扱い手順", "見積条件の見直し", "会員規約の改定案",
    "定期点検の実施時期", "サポート窓口の受付時間", "返品処理の運用",
    "データ移行の検証結果", "料金体系の説明資料", "接続障害の再発防止",
)
_PLACES_MEET: tuple[str, ...] = (
    "本社 第1会議室", "本社 第2会議室", "オンライン（Web会議）", "支店 3階会議室",
    "研修センター A室", "共用ラウンジ", "現地事務所", "本社 応接室",
)
_SERVICES: tuple[str, ...] = (
    "インターネット回線", "電気料金プラン", "会員カード", "総合保険", "定期購読サービス",
    "動画配信サービス", "スポーツクラブ", "宅配サービス", "オンライン講座",
)


def _pick_base_sentences(base_text: str, rng: random.Random, *, max_chars: int = 170) -> str:
    """土台テキストから、氏名らしさの薄い1〜2文を抜き出す内部関数。

    土台コーパス (Wikipedia 等) には実在の人名が含まれうる。合成 PII 以外を
    正解にしないという設計上、敬称・鉤括弧を含む文は避けて混入量を抑える。
    """
    if not base_text:
        return ""
    flat = " ".join(base_text.split())
    raw = [s for s in re.split(r"(?<=。)", flat) if s.strip()]
    cands = [
        s for s in raw
        if 20 <= len(s) <= max_chars
        and not re.search(r"[「」『』]|氏|さん|様|君|@|\d{3}-\d{4}", s)
    ]
    if not cands:
        cands = [s for s in raw if 20 <= len(s) <= max_chars]
    if not cands:
        return ""
    k = 1 if (len(cands) == 1 or rng.random() < 0.7) else 2
    start = rng.randrange(0, max(1, len(cands) - k + 1))
    out = "".join(cands[start : start + k])
    return out[:max_chars] + ("" if out.endswith("。") else "。")


class _SlotResolver:
    """テンプレートのスロットを解決し、同一文書内で値を安定させる内部ヘルパ。"""

    def __init__(self, factory: PIIFactory, base_text: str) -> None:
        self.f = factory
        self.rng = factory.rng
        self.base_text = base_text
        self.pii: dict[str, PIIValue] = {}
        self.filler: dict[str, str] = {}
        self.base_used = False

    # -- PII スロット ----------------------------------------------------
    def name(self, idx: str) -> PIIValue:
        key = "name" + idx
        if key not in self.pii:
            self.pii[key] = self.f.name()
        return self.pii[key]

    def addr(self, idx: str) -> PIIValue:
        key = "addr" + idx
        if key not in self.pii:
            self.pii[key] = self.f.address()
        return self.pii[key]

    def pii_value(self, base: str, idx: str) -> PIIValue:
        key = base + idx
        if key in self.pii:
            return self.pii[key]
        if base == "name":
            v = self.f.name()
        elif base == "kana":
            v = self.f.reading_of(self.name(idx).meta)
        elif base == "addr":
            v = self.f.address()
        elif base == "tel":
            v = self.f.phone()
        elif base == "mobile":
            v = self.f.phone("mobile")
        elif base == "fax":
            v = self.f.phone("landline")
        elif base == "email":
            v = self.f.email(self.name(idx).meta)
        elif base == "dob":
            v = self.f.dob()
        elif base == "bank":
            v = self.f.bank_account()
        elif base == "card":
            v = self.f.credit_card()
        elif base == "mynum":
            v = self.f.mynumber()
        elif base == "member":
            v = self.f.member_id()
        elif base == "zip":
            v = self.f._postal_code(pref=self.addr(idx).meta.get("pref"), mark=False)
        else:  # pragma: no cover - 未知スロットは呼ばれない
            raise KeyError(base)
        self.pii[key] = v
        return v

    # -- 非 PII スロット --------------------------------------------------
    def fill(self, slot: str, base: str) -> str:
        if slot in self.filler:
            return self.filler[slot]
        rng = self.rng
        if base == "company":
            s = rng.choice(_COMPANY_HEAD) + rng.choice(_COMPANY_TAIL)
        elif base == "dept":
            s = rng.choice(_DEPTS)
        elif base == "subject":
            s = rng.choice(_SUBJECTS)
        elif base == "topic":
            s = rng.choice(_TOPICS)
        elif base == "place":
            s = rng.choice(_PLACES_MEET)
        elif base == "service":
            s = rng.choice(_SERVICES)
        elif base == "date":
            s = self._date_nondob()
        elif base == "time":
            s = self._time()
        elif base == "ticket":
            s = f"{rng.randrange(2023, 2026)}{rng.randrange(1, 13):02d}{rng.randrange(1, 29):02d}-{rng.randrange(0, 10000):04d}"
        elif base == "ext":
            s = str(rng.randrange(1000, 9999))
        elif base == "amount":
            s = f"{rng.randrange(3, 900) * 1000:,}"
        elif base == "exp":
            s = f"{rng.randrange(1, 13):02d}/{rng.randrange(26, 33)}"
        elif base == "order":
            s = rng.choice([
                f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randrange(100000, 999999)}",
                f"NO.{rng.randrange(1000, 99999)}",
                f"{rng.randrange(2023, 2026)}-{rng.randrange(100, 999)}",
            ])
        elif base == "model":
            s = f"{rng.choice(('XR', 'GT', 'PL', 'SV', 'NX'))}-{rng.randrange(100, 999)}{rng.choice('ABCDEFGH')}"
        elif base == "clause":
            s = str(rng.randrange(3, 40))
        elif base == "minutes":
            s = str(rng.randrange(3, 45))
        else:
            s = ""
        self.filler[slot] = s
        return s

    def _date_nondob(self) -> str:
        """会議日・受付日など、**生年月日ではない** 日付 (自然な否定例)。"""
        rng = self.rng
        d = _dt.date(2024, 1, 1) + _dt.timedelta(days=rng.randrange(0, 700))
        r = rng.random()
        if r < 0.45:
            return f"{d.year}年{d.month}月{d.day}日"
        if r < 0.65:
            return f"{d.year}/{d.month:02d}/{d.day:02d}"
        if r < 0.80:
            era_name, _alpha, ey = _to_wareki(d)
            return f"{era_name}{ey}年{d.month}月{d.day}日"
        if r < 0.92:
            return f"{d.month}月{d.day}日"
        return f"{d.year}-{d.month:02d}-{d.day:02d}"

    def _time(self) -> str:
        rng = self.rng
        h = rng.randrange(9, 19)
        m = rng.choice((0, 15, 30, 45))
        return f"{h}:{m:02d}" if rng.random() < 0.6 else f"{h}時{m:02d}分"

    def base_sentence(self) -> str:
        s = _pick_base_sentences(self.base_text, self.rng)
        if s:
            self.base_used = True
        return s


def _split_slot(slot: str) -> tuple[str, str]:
    """``"name12"`` -> ``("name", "12")`` の分解 (内部関数)。"""
    m = re.match(r"^([a-z_]+?)([0-9]*)$", slot)
    if not m:
        return slot, ""
    return m.group(1), m.group(2)


def render_document(
    factory: PIIFactory,
    *,
    genre: str,
    base_text: str = "",
    base_license: str = "synthetic (CC0-1.0)",
    base_ref: str = "",
    doc_id: str = "",
    n_pii: int | None = None,
    subset: str = "train",
) -> Document:
    """テンプレートを走査しながら文書を組み立て、PII の位置を **記録** する。

    出力文字列は左から継ぎ足して作り、PII 値を書き出した瞬間の長さから
    ``(start, end)`` を確定させる。生成後に ``text.index()`` で探し直すことは
    しない (同じ値が複数回現れても座標が壊れない)。

    Claim: 検出率 / 低誤検出 — 正解スパンが構成的に正しいので、検出率の分母が
    信頼できる。同一人物の氏名が本文と署名に二度出る、受付番号が会員番号と
    紛らわしい、といった実務的な難所をテンプレート側に埋め込んである。

    Args:
        factory: 値の生成器 (乱数状態を共有する)。
        genre: ``GENRES`` のいずれか。
        base_text: 土台コーパスの本文 (``{base}`` に1〜2文だけ織り込む)。
        base_license: 土台テキストのライセンス表記。実際に使われたときだけ記録する。
        base_ref: 土台テキストの出典参照。
        doc_id: 文書 ID (空なら乱数由来の ID を振る)。
        n_pii: 挿入する PII スロット数の目安 (None ならテンプレートのまま)。
        subset: ``train`` / ``validation`` / ``test`` / ``negatives``。

    Returns:
        ``validate()`` を通る ``Document``。
    """
    if genre not in TEMPLATES:
        raise ValueError(f"unknown genre: {genre!r} (expected one of {GENRES})")
    rng = factory.rng
    tpl_id, tpl = rng.choice(TEMPLATES[genre])
    res = _SlotResolver(factory, base_text)

    # テンプレート内の PII スロットを出現順に列挙 (重複は先頭のみ)。
    order: list[str] = []
    for m in _TOKEN_RE.finditer(tpl):
        slot = m.group(1)
        base, _idx = _split_slot(slot)
        if base in _PII_BASES and slot not in order:
            order.append(slot)

    omit: set[str] = set()
    extra_types: list[PIIType] = []
    if n_pii is not None:
        n_pii = max(0, int(n_pii))
        if n_pii < len(order):
            omit = set(order[n_pii:])
        elif n_pii > len(order):
            pool = list(PIIType)
            extra_types = [rng.choice(pool) for _ in range(n_pii - len(order))]

    out: list[str] = []
    spans: list[Span] = []
    pos = 0

    def emit_text(s: str) -> None:
        nonlocal pos
        if not s:
            return
        s = normalize(s)
        out.append(s)
        pos += len(s)

    def emit_value(v: PIIValue, slot: str) -> None:
        nonlocal pos
        t = normalize(v.text)
        parts = v.parts()
        if len(t) != len(v.text):  # 正規化で長さが変わったら値全体を1スパンに縮退
            parts = [(0, len(t), v.label)]
        out.append(t)
        for a, b, lab in parts:
            if b <= a:
                continue
            meta = {k: val for k, val in v.meta.items() if k != "parts"}
            meta["slot"] = slot
            meta["synthetic"] = True
            spans.append(
                Span(
                    start=pos + a,
                    end=pos + b,
                    label=lab,
                    text=t[a:b],
                    score=1.0,
                    source=Source.GOLD,
                    meta=meta,
                )
            )
        pos += len(t)

    last = 0
    for m in _TOKEN_RE.finditer(tpl):
        emit_text(tpl[last : m.start()])
        last = m.end()
        slot = m.group(1)
        base, idx = _split_slot(slot)
        if base == "base":
            emit_text(res.base_sentence())
        elif base in _PII_BASES:
            if slot in omit:
                emit_text(_OMIT_TEXT.get(base, "(省略)"))
            else:
                emit_value(res.pii_value(base, idx), slot)
        else:
            emit_text(res.fill(slot, base))
    emit_text(tpl[last:])

    # n_pii がテンプレートのスロット数を超える場合は「備考」として追記する。
    if extra_types:
        emit_text("\n備考\n")
        for i, t in enumerate(extra_types, 1):
            v = factory.make(t)
            emit_text(f"  ({i}) {t.ja}: ")
            emit_value(v, f"extra{i}")
            emit_text("\n")

    text = "".join(out)
    if not doc_id:
        doc_id = f"synth-{genre}-{rng.randrange(16 ** 8):08x}"

    used_base = res.base_used
    doc = Document(
        text=text,
        spans=spans,
        doc_id=doc_id,
        subset=subset,
        genre=genre,
        source_license=base_license if used_base else "synthetic (CC0-1.0)",
        source_ref=base_ref if used_base else "",
        negative_kinds=[],
        meta={
            "template_id": tpl_id,
            "n_spans": len(spans),
            "n_pii_requested": n_pii,
            "base_used": used_base,
            "factory_seed": factory.seed,
            "synthetic": True,
        },
    )
    doc.validate()
    return doc


def build_documents(
    n: int,
    *,
    seed: int = 0,
    base_items: Sequence[Any] | None = None,
    genres: Iterable[str] = GENRES,
    subset: str = "train",
) -> list[Document]:
    """合成文書を n 件、ジャンルをならして生成する。

    ``base_items`` は ``sumi.corpus.CorpusItem`` 互換 (``text`` / ``license`` /
    ``source`` / ``attribution`` / ``item_id`` 属性、または素の文字列) を受ける。
    土台テキストが実際に織り込まれたときだけ、ライセンスと出典を文書に引き継ぐ。

    Claim: 検出率 / 低誤検出 — 種別とジャンルの分布を制御した学習・評価データを
    一括で作る入口。シードを固定すれば完全に同じデータが再現でき、
    検出率・誤検出率の比較が同一データ上で行える。
    """
    genre_list = [g for g in genres]
    if not genre_list:
        raise ValueError("genres must not be empty")
    for g in genre_list:
        if g not in TEMPLATES:
            raise ValueError(f"unknown genre: {g!r}")

    rng = random.Random(seed * 1_000_003 + 17)
    factory = PIIFactory(seed=seed)

    plan = [genre_list[i % len(genre_list)] for i in range(n)]
    rng.shuffle(plan)

    items = list(base_items) if base_items else []
    docs: list[Document] = []
    for i, genre in enumerate(plan):
        base_text, lic, ref = "", "synthetic (CC0-1.0)", ""
        if items and rng.random() < 0.75:
            it = items[rng.randrange(len(items))]
            if isinstance(it, str):
                base_text = it
            else:
                base_text = str(getattr(it, "text", "") or "")
                lic = str(getattr(it, "license", "") or lic)
                ref = str(
                    getattr(it, "attribution", "")
                    or getattr(it, "source", "")
                    or getattr(it, "item_id", "")
                )
        n_pii = None
        if rng.random() < 0.35:
            n_pii = rng.randint(2, 10)
        docs.append(
            render_document(
                factory,
                genre=genre,
                base_text=base_text,
                base_license=lic,
                base_ref=ref,
                doc_id=f"synth-{subset}-{i:05d}",
                n_pii=n_pii,
                subset=subset,
            )
        )
    return docs


# ---------------------------------------------------------------------------
# 5. 自己テスト
# ---------------------------------------------------------------------------

def _mark_spans(doc: Document, *, width: int = 999) -> str:
    """正解スパンを 【】 で囲んで表示用の文字列を作る (自己テスト用)。"""
    t = doc.text
    out = []
    prev = 0
    for s in doc.sorted_spans():
        out.append(t[prev : s.start])
        out.append(f"【{s.slice_of(t)}|{s.label.value}】")
        prev = s.end
    out.append(t[prev:])
    return "".join(out)


def _selftest() -> None:  # pragma: no cover - 実行時の自己点検
    import collections

    print("=" * 72)
    print("sumi.synth 自己テスト")
    print("=" * 72)

    # --- 1. 個別生成器の健全性 ---------------------------------------
    f = PIIFactory(seed=20240901)

    n_check = 400
    cards = [f.credit_card() for _ in range(n_check)]
    for v in cards:
        d = re.sub(r"\D", "", v.text)
        assert len(d) == 16, f"card digits != 16: {v.text}"
        assert re.fullmatch(r"(\d{4}[- ]){3}\d{4}|\d{16}", v.text), v.text
        assert v.meta["checksum_valid"] is False
        body, last = d[:15], int(d[15])
        assert _luhn_check_digit(body) != last, f"Luhn accidentally valid: {v.text}"
    print(f"[card ] {n_check}件: 16桁・書式OK・Luhn は全件わざと不一致 ✓  例 {cards[0].text}")

    mynums = [f.mynumber() for _ in range(n_check)]
    for v in mynums:
        d = re.sub(r"\D", "", v.text)
        assert len(d) == 12, f"mynumber digits != 12: {v.text}"
        assert v.meta["checksum_valid"] is False
        assert _mynumber_check_digit(d[:11]) != int(d[11]), f"check digit valid: {v.text}"
    print(f"[mynum] {n_check}件: 12桁・書式OK・検査数字は全件わざと不一致 ✓  例 {mynums[0].text}")

    kinds = collections.Counter()
    for _ in range(600):
        p = f.phone()
        kinds[p.meta["kind"]] += 1
        nd = p.meta["n_digits"]
        if p.meta["kind"] == "mobile":
            assert nd == 11, p.text
        elif p.meta["kind"] == "ip":
            assert nd == 11, p.text
        elif p.meta["kind"] == "tollfree":
            assert nd in (10, 11), p.text
        else:
            assert nd == 10, (p.text, p.meta)
    print(f"[phone] 桁数規則OK {dict(kinds)}")

    doms = collections.Counter()
    for _ in range(300):
        e = f.email()
        dom = e.meta["domain"]
        doms[dom] += 1
        assert dom in PIIFactory.DOMAINS
        assert dom.startswith("example.") or ".example." in dom or dom.startswith(("mail.example", "corp.example", "sub.example"))
    print(f"[email] 予約ドメインのみ {len(doms)}種  例 {f.email().text}")

    dobs = [f.dob() for _ in range(300)]
    wareki = [d for d in dobs if d.meta["era"] == "wareki"]
    for d in wareki:
        y, m_, dd = d.meta["year"], d.meta["month"], d.meta["day"]
        name, alpha, ey = _to_wareki(_dt.date(y, m_, dd))
        assert d.meta["era_name"] == name and d.meta["era_year"] == ey
    print(f"[dob  ] 和暦換算OK ({len(wareki)}/{len(dobs)} 和暦)  例 "
          f"{dobs[0].text} / {wareki[0].text if wareki else '-'}")

    prefs = {f.address().meta["pref"] for _ in range(3000)}
    print(f"[addr ] 都道府県カバレッジ {len(prefs)}/47  例 {f.address(with_postal=True).text}")
    assert len(prefs) == 47, sorted(set(_POSTAL_PREFIX) - prefs)

    names = [f.name() for _ in range(4000)]
    top = collections.Counter(n.meta["sei"] for n in names).most_common(5)
    forms = collections.Counter(n.meta["form"] for n in names)
    print(f"[name ] 姓の上位5 {top}")
    print(f"        表記形の内訳 {dict(forms)}")

    # --- 2. 文書生成 -------------------------------------------------
    base_items = [
        "日本の郵便制度は明治期に整備され、全国一律の料金体系が導入された。"
        "現在では郵便番号によって配達区域が細かく区分されている。",
        "気象庁は各地の観測点で得られたデータをもとに、天気予報を発表している。"
        "観測点は全国におよそ千三百か所設置されている。",
        "図書館は資料の収集と保存を担う施設であり、地域の情報拠点としての役割も持つ。",
    ]
    docs = build_documents(300, seed=7, base_items=base_items, subset="train")
    assert len(docs) == 300
    for d in docs:
        d.validate()
        assert normalize(d.text) == d.text, f"{d.doc_id}: text is not normalized"
        for s in d.spans:
            assert d.text[s.start : s.end] == s.text

    per_type = collections.Counter(s.label.value for d in docs for s in d.spans)
    per_genre = collections.Counter(d.genre for d in docs)
    per_tpl = collections.Counter(d.meta["template_id"] for d in docs)
    n_spans = sum(len(d.spans) for d in docs)
    chars = sum(len(d.text) for d in docs)

    print()
    print(f"文書 {len(docs)} 件 / 正解スパン {n_spans} 件 / 平均 {n_spans/len(docs):.1f} 件・文書")
    print(f"平均文字数 {chars/len(docs):.0f}  ジャンル内訳 {dict(per_genre)}  テンプレ {len(per_tpl)}種")
    print()
    print("種別ごとの正解スパン数")
    print("-" * 46)
    for t in PIIType:
        c = per_type.get(t.value, 0)
        bar = "#" * min(40, c // 8)
        print(f"  {t.value:<13}{t.ja:<8}{c:>5}  {bar}")
    print("-" * 46)
    assert all(per_type.get(t.value, 0) > 0 for t in PIIType), per_type

    # --- 3. サンプル表示 ---------------------------------------------
    shown = 0
    seen_genres: set[str] = set()
    for d in docs:
        if d.genre in seen_genres or shown >= 3:
            continue
        seen_genres.add(d.genre)
        shown += 1
        print()
        print(f"--- 例{shown}: {d.doc_id} [{d.genre}/{d.meta['template_id']}] "
              f"spans={len(d.spans)} license={d.source_license} ---")
        print(_mark_spans(d))

    # --- 4. 再現性と否定例的要素 -------------------------------------
    a = build_documents(20, seed=99, base_items=base_items)
    b = build_documents(20, seed=99, base_items=base_items)
    assert [x.to_dict() for x in a] == [y.to_dict() for y in b], "not reproducible"
    c = build_documents(20, seed=100, base_items=base_items)
    assert [x.text for x in a] != [y.text for y in c], "seed had no effect"

    small = [render_document(PIIFactory(seed=i), genre=g, n_pii=3, doc_id=f"t{i}")
             for i, g in enumerate(GENRES)]
    for d in small:
        d.validate()
        # n_pii=3 でも住所スロットが 郵便番号+住所 の2スパンに割れる等で数件増える
        assert 1 <= len(d.spans) <= 8, (d.doc_id, len(d.spans))
    big = render_document(PIIFactory(seed=5), genre="email", n_pii=30, doc_id="big")
    big.validate()
    assert len(big.spans) >= 25, len(big.spans)
    print()
    print(f"[repro] 同一シードで完全一致 ✓ / n_pii 制御 ✓ (少 {[len(d.spans) for d in small]}, 多 {len(big.spans)})")

    amb = sum(1 for n in names if n.meta["sei"] in {k for k, _ in _SURNAME_AMBIGUOUS})
    print(f"[amb  ] 普通名詞・地名と同形の姓の出現率 {amb/len(names):.1%} (negatives.py の土台)")
    print()
    print(f"素材: 姓 {len(_SURNAMES)} / 名 {len(_GIVEN)} / 市区町村 {len(_PLACES)} "
          f"(都道府県 {len({p for p,_,_ in _PLACES})}) / テンプレ "
          f"{sum(len(v) for v in TEMPLATES.values())} ({', '.join(f'{k}:{len(v)}' for k,v in TEMPLATES.items())})")
    print("すべての自己テストに合格しました。")


if __name__ == "__main__":  # pragma: no cover
    _selftest()
