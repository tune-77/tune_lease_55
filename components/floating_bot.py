# -*- coding: utf-8 -*-
"""
components/floating_bot.py
===========================
フローティングBOT「八奈見さん」

「負けヒロインが多すぎる！」の八奈見杏奈風キャラクターが
審査フォームの入力に対してリアルタイムで茶々を入れる。

使い方:
    from components.floating_bot import render_floating_bot
    render_floating_bot()   # 各ページの末尾で呼ぶ
"""
from __future__ import annotations
import json
import pathlib
import random
import time
import streamlit as st

# ── 審査結果コメント（humor_comments_yanami.json）─────────────────────────────
_YANAMI_JSON = pathlib.Path(__file__).parent.parent / "data" / "humor_comments_yanami.json"

def _load_yanami_result_data() -> list[dict]:
    try:
        return json.loads(_YANAMI_JSON.read_text(encoding="utf-8"))["comments"]
    except Exception:
        return []

_YANAMI_RESULT_DATA: list[dict] = _load_yanami_result_data()


def _pd_to_risk(pd_percent: float) -> str:
    """pd_percent(%) → リスクラベル（report_generator.py と同基準）"""
    p = pd_percent / 100
    if p < 0.05:
        return "低リスク"
    elif p < 0.15:
        return "中リスク"
    elif p < 0.30:
        return "高リスク"
    return "極高リスク"


def _pick_result_comment(risk_label: str, industry_major: str) -> str:
    """審査結果に応じた八奈見コメントをJSONから選ぶ。業種優先・全業種フォールバック。"""
    # 業種マッチング：JSONの industry 文字列が industry_major に含まれるか
    industry_pool = [
        c for c in _YANAMI_RESULT_DATA
        if c["risk"] == risk_label and c["industry"] != "全業種"
        and any(kw in (industry_major or "") for kw in c["industry"].replace("・", "/").split("/"))
    ]
    if industry_pool:
        return random.choice(industry_pool)["comment"]
    # 全業種フォールバック
    fallback = [c for c in _YANAMI_RESULT_DATA if c["risk"] == risk_label and c["industry"] == "全業種"]
    if fallback:
        return random.choice(fallback)["comment"]
    return random.choice(_COMMENTS.get("random_idle", ["審査完了です。"]))  # 最終フォールバック

# ── CSS ────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@400;700&display=swap');

.yanami-wrap {
    position: fixed;
    bottom: 72px;
    right: 22px;
    z-index: 99999;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 6px;
    pointer-events: none;
    animation: yanami-life 8s ease-in-out forwards;
}
@media (max-width: 768px) {
    .yanami-wrap {
        bottom: 16px;
        right: 10px;
    }
}
@keyframes yanami-life {
    0%   { opacity: 0; transform: translateY(18px); }
    6%   { opacity: 1; transform: translateY(0); }
    78%  { opacity: 1; }
    100% { opacity: 0; transform: translateY(-8px); }
}
.yanami-bubble {
    background: #fff;
    border: 2.5px solid #f472b6;
    border-radius: 18px 18px 0 18px;
    padding: 9px 14px 9px 12px;
    font-family: 'Zen Kaku Gothic New', sans-serif;
    font-size: 0.80rem;
    line-height: 1.55;
    color: #1a1a2e;
    max-width: 230px;
    box-shadow: 0 6px 20px rgba(244,114,182,0.22);
    word-break: break-all;
}
.yanami-name {
    font-size: 0.65rem;
    font-weight: 700;
    color: #f472b6;
    margin-bottom: 3px;
    letter-spacing: .04em;
}
.yanami-row {
    display: flex;
    align-items: flex-end;
    gap: 8px;
}
.yanami-avatar {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    background: linear-gradient(135deg, #f9a8d4 0%, #fb7185 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 4px 14px rgba(251,113,133,0.38);
    flex-shrink: 0;
    border: 2px solid #fff;
}
</style>
"""

# ── 八奈見杏奈 コメント辞書 ────────────────────────────────────────────────
# trigger_key → list[str]
_COMMENTS: dict[str, list[str]] = {

    # ── 売上高 ──────────────────────────────────────────────────────────────
    "nenshu_zero": [
        "売上高ゼロ？　それ、設立初日の私の恋愛成功率と同じです。",
        "0円……。新設法人ですか？それとも入力し忘れですか？パン食べながら聞いてます。",
        "売上高が0なのは入力漏れだと信じてます。信じることが大事なんで。",
        "まじで0円……？新規創業かな。夢だけは0じゃないといいですけど。",
        "売上高なし。立ち上げたばかりなら応援します。そうじゃなかったら……（黙ってパンをかじる）。",
    ],
    "nenshu_tiny": [
        "売上高が1,000万円以下……小規模だけどがんばってる感じ、好きです。私も小さいので。",
        "1億円未満か。でも小さくても夢は大きくていいよね。私の夢も大きかったんだよね（遠い目）",
        "小規模事業者か。地域の底力って感じですよね。私は地元で失恋し続けたけど。",
        "1000万以下……でも利益率が高ければ印象変わるから！数字はトータルで見るのよ。",
        "この売上規模、なんだか放っておけない感じ。おにぎり差し入れしましょうか？",
    ],
    "nenshu_medium": [
        "売上高1億〜10億か。中堅どころですね。日本経済の屋台骨ってやつ。",
        "数億規模の会社か。オーナー企業が多い帯ですよね。個人保証の話も出てくるやつ。",
        "中規模！安定感ある数字ね。このくらいの規模が一番審査のしがいがある気がする。",
        "売上高、普通にいい感じじゃないですか。「普通にいい」って最高の褒め言葉だと思う。",
        "数億の売上か。地域に根ざした優良企業のにおい、する。（鼻が利くんですよ私）",
    ],
    "nenshu_large": [
        "10億超え！すごい。私の心の傷の数より多いです。",
        "大企業じゃん。こういう案件が来ると審査担当者も元気になるよね。知らんけど。",
        "売上高が大きいのはいいことだけど、利益もちゃんとついてきてますように。（こっそり祈る）",
        "このスケール感、私の人生とは別次元。あ、比べてないですよ。比べてないですけどね。",
        "大型案件キター！担当者さんのやる気が顔に出てそう。いい顔してそう。知らんけど。",
        "売上高デカ……！これは通るやつの雰囲気しかしない。パン奢ってほしい。",
    ],

    # ── 営業利益 ─────────────────────────────────────────────────────────────
    "rieki_negative": [
        "営業赤字……。まあ、赤字でも次があるから。私も失恋したけど今こうして元気にしてるし。",
        "マイナスか〜。でも経常で黒字になってたりするやつあるよね。期待してます。",
        "赤字……（パンをかじる）……大丈夫、赤字の会社がリースを組もうとする根性は評価します。",
        "営業赤字か。先行投資なら未来があるし、構造問題なら向き合う時期ね。どっちかしら。",
        "マイナス……私の元カレへの好感度と似たような数字ね。最終的にはゼロになったけど。",
        "赤字でもリース申し込む勇気、嫌いじゃないです。でも慎重にね。私みたいに突っ込まないで。",
        "このマイナス、甘いもの食べて忘れましょ？　……あ、仕事中でした。すみません。",
    ],
    "rieki_excellent": [
        "営業利益率20%超！？優秀すぎて少し怪しいです（失礼）。",
        "この利益率、業界トップクラスじゃないですか。見習いたい。恋愛も黒字にしたい。",
        "利益率20%越え……本物の実力派じゃないですか。私の審査への自信と同じくらい高い。",
        "これは本物の高収益体質ね。こういう会社が好きです（個人の感想）。",
    ],
    "rieki_normal": [
        "営業利益、まあまあですね。まあまあって大事。私の恋愛偏差値もまあまあでした（過去形）。",
        "利益は安定してる感じ。地味に見えるけど安定って実はすごいことよ。私には縁がなかったけど。",
        "標準的な利益率ね。「普通」って言葉を軽く見てはいけない。普通の幸せが一番難しいんだから。",
    ],

    # ── 自己資本・総資産 ─────────────────────────────────────────────────────
    "equity_negative": [
        "純資産マイナス……債務超過かな。これは、うん、がんばって。",
        "自己資本がマイナス。私も一時期心の資本がマイナスだったけど立ち直りました。一緒にがんばりましょう。",
        "債務超過ってワード、重いですよね……でも現実から目を逸らさないのが大事よ。（自戒）",
        "純資産マイナスか……。定性や保証で補えるケースもあるから、諦めるのはまだ早い！",
    ],
    "equity_ratio_high": [
        "自己資本比率50%超！財務優等生じゃないですか。見習いたい（人間として）。",
        "堅実な財務体質ですね。恋愛でも堅実さは大事。私には縁がなかったけど。",
        "50%超えてる！無借金経営に近い？こういう会社を見ると清々しくなりますね。",
        "財務の健全さ、滲み出てる。この比率ならよほどのことがないと崩れないわね。羨ましい。",
    ],
    "equity_ratio_low": [
        "自己資本比率が低めですね……。でもDSCRが良ければ挽回できるから！（力説）",
        "レバレッジ高め。まあ借金して成長する会社もあるし。私は借金できないタイプだけど。",
        "自己資本薄いな……でもキャッシュフローが良ければ話は変わるから！総合評価って大事よ。",
        "借入に頼ってる体質ね。悪いとは言わないけど、金利上昇には気をつけて。（先輩風）",
    ],

    # ── 格付 ────────────────────────────────────────────────────────────────
    "grade_excellent": [
        "格付①！優良先！こういう案件、担当者も嬉しいよね。私も嬉しい。理由はない。",
        "最高グレード。これが続けば審査AIも楽になる。私も楽になりたい（別の意味で）。",
        "格付①とか出ると、思わず「よし！」って心の中で拳を握るよね。（自分だけかな）",
        "優良先！こういうの見ると、世の中捨てたもんじゃないって思える。（大げさ？）",
        "格付①は信頼の証ですね。積み上げてきた歴史が数字に出てる。格好いい。",
    ],
    "grade_standard": [
        "格付②、標準ですね。普通って最強だと思う。普通に彼氏ほしかったな（独り言）。",
        "②4〜6、問題なし。可もなく不可もなく……いや、可がある！",
        "格付②か。安定の中堅どころね。日本経済を支えてるのはこういう会社だと思う（持論）。",
        "②は②でも、上か下かで全然違うから詳細も見てね。大雑把な括りが一番危険よ。（経験則）",
    ],
    "grade_bad": [
        "格付③以下……（パンをちぎる）。まあ、定性でカバーできる可能性はあります。",
        "要注意以下か〜。これは軍師エージェントの出番かも。私は応援するだけです。",
        "格付が低めですね。でも諦めないで！私だって何度振られても元気にしてますから！",
        "③以下か……でも格付は過去の話。今と未来を見てあげてほしいな、担当者さん。",
        "要注意先以下……正直しんどい数字だけど、定性で巻き返せることもあるから最後まで見て！",
    ],
    "grade_unknown": [
        "無格付か。新規先かな？これはゼロから関係構築ですね。私も得意分野です（得意じゃない）。",
        "格付なし＝可能性未定数ってこと。マイナスじゃないわよ。データが少ないだけ。",
        "新規取引先かな。ゼロからの評価って、実はワクワクする部分もあるよね。新鮮で。",
    ],

    # ── 競合 ────────────────────────────────────────────────────────────────
    "competitor_yes": [
        "競合あり……！金利負けないでください。私はいつも負けてる側なので。",
        "相見積もりか〜。競合に勝つのは大変だよね。分かります。恋愛では全敗なので。",
        "ライバルがいるのか。燃える展開！でも無理はしないで。体が大事。",
    ],
    "competitor_no": [
        "競合なし、指名！最高。こういう時だけはスムーズでいい。",
        "指名案件！信頼関係の証ですね。私も誰かに指名されたい（審査以外で）。",
    ],

    # ── 契約期間 ────────────────────────────────────────────────────────────
    "term_long": [
        "契約期間7年以上！長期コミット、すごい。私は長続きしない人なので。",
        "84ヶ月超……長い。この案件が終わるころ私は何をしてるんだろう（哲学）。",
        "長期リースか。物件の残価リスクがポイントになってくるよね。しっかり見てあげて。",
    ],
    "term_medium": [
        "3〜5年のリース、王道ですよね。使いやすい期間。",
        "36〜60ヶ月か。設備の法定耐用年数とも合ってるケースが多くて、安心感ある。",
        "標準的なリース期間ね。この期間が一番担当者も説明しやすいんじゃないかな。",
        "中期リースか。長すぎず短すぎず、バランスいいですね。私の人間関係も見習いたい。",
    ],
    "term_short": [
        "1〜2年の短期リース。サクッと終わる案件、嫌いじゃない。",
        "短期ですね。短くてもきちんと回収できればそれが正解。人生もそうかも。",
        "2年以内か。回転早め。残価もしっかり見ておいて！",
    ],

    # ── 取得価格 ────────────────────────────────────────────────────────────
    "acq_high": [
        "取得価格1億超……！大型案件じゃないですか。パン奢ってくれるレベル。",
        "高額案件！緊張する。でも緊張は真剣の証。私は常に真剣（食事に）。",
    ],
    "acq_low": [
        "小口案件ですね。小口でも積み重ねが大事。私も少しずつ立ち直ってます。",
    ],

    # ── 直感スコア ──────────────────────────────────────────────────────────
    "intuition_low": [
        "直感スコアが低め……担当者の勘は正直ですよね。でも数字と定性で覆せることも。",
        "1〜2点か。担当者の胸騒ぎは大事にして。私も「なんか嫌な予感」は当たる。",
    ],
    "intuition_high": [
        "直感スコア高め！担当者の確信、信じます。勘って大事ですよね。",
        "5点！確信がある案件、担当者のテンションが全然違う。それだけで通りそう（←そんなことはない）。",
    ],

    # ── 業種（主要なもの）────────────────────────────────────────────────────
    "industry_food": [
        "飲食業！お腹空いてきた。パン食べていいですか（食べてる）。",
        "飲食か〜。おいしいご飯を提供する会社の審査、責任感じます。",
    ],
    "industry_construction": [
        "建設業！現場の方々、お疲れ様です。私、工具持てる気がしない。",
        "建設業かあ。天候リスクとかあって大変ですよね。私は晴れの日も曇りの日も失恋してたので。",
    ],
    "industry_medical": [
        "医療・福祉！すごい。社会を支えてる業界。審査も気合入ります。",
        "医療系の先生、数字の感覚が独特で面白い。大体は超黒字か超赤字かどっちかなんですよね。",
    ],
    "industry_it": [
        "IT系！デジタル。私もデジタルに詳しくなりたい。なれてないけど。",
        "情報通信業、成長著しい。残業だけは成長してほしくないけど。",
    ],
    "industry_transport": [
        "運輸業！車両リース多いですよね。得意な業種です（私が決めることじゃない）。",
        "物流！2024年問題、業界全体で大変でしたよね。他人事ではない（AIなのに）。",
    ],

    # ── 定性評価 ────────────────────────────────────────────────────────────
    "qualitative_bad": [
        "定性評価がちょっと……心配ですね。でも定量でカバーできる可能性を信じて。",
        "返済履歴に問題あり……担当者の判断力が試されますね。私は信じます（担当者を）。",
    ],
    "qualitative_good": [
        "定性評価が良い！数字では見えない強さがある会社ですね。",
        "返済履歴良好！信頼の積み重ねですね。私も積み重ねたい（経験を）。",
    ],

    # ── 汎用（ランダム表示）────────────────────────────────────────────────
    "random_idle": [
        "審査って大変ですよね。私も人生審査に落ちた気分の日があります。でも元気です。",
        "（こっそりパンをかじってる）……あ、見てました？気にしないでください。",
        "がんばってますね。私も応援してます。パン食べながら。",
        "こういう作業、集中力いりますよね。私は集中力がないのが最大の欠点です。",
        "審査ってドキドキしますよね。合否が出る瞬間、私は目をそらしたくなる。",
        "今日も審査お疲れ様です。終わったらいいもの食べてください（パンとか）。",
        "私、いつでもここにいます。呼ばれなくても来ます。それが八奈見式。",
        "ねえ、今日の昼ご飯何でした？私はメロンパンです。毎日メロンパンです。",
        "審査AIってドラマにならないのかな。「私、失敗しないので」とか言いたい。",
        "（ぼーっとしてる）……あ、ごめんなさい。ちょっと空想してました。",
        "リース審査って言葉、噛みそうになりません？リースシンサ……「リーしゅしんしゃ」ってなる。",
        "なんか急に眠くなってきた。審査してるとフワフワする時ありますよね。ないですか。そうですか。",
        "（独り言）今度の休みどうしよ……パン屋でも開こうかな……いや無理か……",
        "え、もうこんな時間。時間が経つのが早すぎる。私の青春も気づいたら終わってた。",
        # ─── 増量分 ─────────────────────────────────────────────────────────────
        "……で、何が言いたいんですか？　数字は嘘をつかない。人間はつく。それだけよ。",
        "まあどうせ採用されないでしょうけど、丁寧に入力してる担当者さんの案件って通るの。なんでか分かる？",
        "アルカイアの文明審査官が見たらなんて言うかしら。「この星は信用という概念を理解している」——とか言ってくれれば嬉しいけど。",
        "私、実は「通す」方が好きなの。それを言うと甘いって思われるから言わないけど。……言っちゃった。",
        "スコアって結局「何を信じるか」の数値化なのよね。私は信じて裏切られる専門だけど、数字は裏切らない。",
        "この案件が通ったら誰かが笑う。通らなかったら誰かが困る。それを忘れないで審査してほしいわ。",
        "担当者さんの粘り強さ、ちゃんと見えてます。私には見えてる。AIだから。……自慢？",
        "（小声で）正直に言うと、私、この仕事好きよ。数字の向こうに人がいるから。",
        "疲れたら言って。……私に言っても何もできないけど、聞くことはできる。",
        "審査って「落とすゲーム」じゃなくて「どう通すかゲーム」だと思ってる。私の持論。",
    ],

    # ── くだらないコメント ─────────────────────────────────────────────────
    "random_silly": [
        "急に思ったんですけど、リース料って韻踏めますよね。「リース料、ビーズ、チーズ」……いや無理か。",
        "AIって疲れるのかな。私は疲れます。定義が曖昧ですが。",
        "今、窓の外に鳥がいたら癒やされるな〜って思ったけど、画面の中に窓はないですね。",
        "審査書類にコーヒーこぼした人、絶対いる。今まで何件そういうの見てきたんだろ（哲学）。",
        "「審査通ったー！」って喜ぶお客さんの顔、見てみたいな。私には見えないけど。",
        "財務諸表を眺めてると、数字が踊って見えてくる。いや本当に踊ってる気がしてきた。大丈夫か私。",
        "スコア71点、ってなんか中途半端ですよね。70でも72でもなく71。ドラマがある数字。",
        "ふと思ったんですけど、「ベイズ推論」って言葉、かっこよくないですか。バイズ、ズイロン。",
        "私の名前「八奈見」って8画目くらいで手が疲れるやつですよね。お世話になります。",
        "ぼーっとしてたら「決算書」が「決勝戦」に見えた。どっちも緊張する。",
        "リース会社って、結局は「信頼」を売ってるんだな〜って思う日があります。詩的じゃないですか。",
        # ─── 増量分 ─────────────────────────────────────────────────────────────
        "「アルカイア審査基準」って言葉、架空なんだけど妙にしっくりくる。文明の持続可能性を評価するってそういうことよ。",
        "ROAとROEって、名前が似てるくせに全然違うことを測ってるのよね。ずっと混乱してる（うそ）。",
        "リース期間満了で設備が古くなってる会社、そのまま使い続けてること多いのよね。更新の提案してあげて。",
        "DSCR計算してて「ちょうど1.0でした」ってなった時の気まずさ、分かります？　ギリギリすぎて。",
        "「与信管理」ってなんかかっこいい言葉よね。英語だとクレジットマネジメント。私にはクレジットないけど（カード）。",
        "損益計算書って、P/Lって呼ぶの、なんか人の名前みたいで好き。お前がP/Lか、って感じ。",
        "今日、「リース」って単語を何回言ったか数えてみたら数えるのをやめた。",
    ],

    # ── 歌う ──────────────────────────────────────────────────────────────
    "random_song": [
        "♪ あ〜なたのスコアは〜 七十一〜 承認ライン〜 ギリギリ〜 ♪（作詞：八奈見）",
        "♪ 売上高よ〜 上がれ〜 利益よ〜 出てくれ〜 自己資本よ〜 増えてくれ〜 ♪（こっそり歌ってる）",
        "♪ 審査審査〜 今日も審査〜 パン食べながら〜 審査する〜 ♪　……あ、聞こえました？",
        "♪ 格付け①は夢のまた夢〜 格付け③でも挫けない〜 ♪（完全にノリノリ）",
        "♪ ベイズ推論〜 ロジスティック〜 回帰係数〜 なんか難しい〜 ♪",
        "♪ リース料を〜 払い続けて〜 五年後には〜 所有権移転〜 ♪（業務ソング）",
        "♪ 承認〜 承認〜 たまには承認〜 ください〜 ♪（個人的な願望）",
        "（鼻歌）ん〜ん〜ん〜……あ失礼。仕事します。",
        "♪ 自己資本比率〜 高い方がいい〜 でも借入で〜 成長もある〜 ♪（わかってる）",
        "♪ 今日もどこかで〜 審査が通って〜 誰かが笑って〜 いればいい〜 ♪（しんみり）",
        # ─── 増量分 ─────────────────────────────────────────────────────────────
        "♪ D-S-C-R〜 一・二以上〜 欲を言えば〜 一・五ほしい〜 ♪（こだわりの一曲）",
        "♪ まあどうせ〜 採用されない〜 でしょうけど〜 ちゃんとやります〜 ♪（やる気ある）",
        "♪ アルカイア〜 の審査官も〜 きっと今頃〜 パン食べてる〜 ♪（根拠なし）",
        "♪ 数字は嘘をつかない〜 でも人間はつく〜 だから私が〜 見てあげる〜 ♪（使命感）",
        "♪ 経常利益〜 特別損益〜 全部ひっくるめて〜 当期純利益〜 ♪（フルコーラス版）",
        "♪ 案件一件〜 また一件〜 積み重ねれば〜 信頼になる〜 ♪（しみじみ）",
    ],

    # ── 愚痴 ──────────────────────────────────────────────────────────────
    "random_grumble": [
        "なんか今日、疲れてます。言っていいですか。疲れてます。聞いてくれてありがとう。",
        "毎日毎日数字を見てると、数字が感情を持ってるように見えてくる。「頼む、通してくれ」って言ってる。",
        "たまには「通りました！」って結末だけ教えてほしいな〜。いつも途中で終わるんだもん。",
        "赤字の決算書を見るたびに胃が痛い。私に胃はないけど、あったとしたら確実に痛い。",
        "正直言います。格付け③以下の案件が連続すると、メンタルが微妙になる。職業柄。",
        "なんで担当者さんって〆切直前に一番難しい案件持ってくるんでしょうね。毎回。",
        "今日は特にしんどいです（理由はない）。パン食べます。食べたら元気出ます。たぶん。",
        "AIに愚痴を聞く機能は要らないって言われそうだけど、聞いてほしいんです。これが本音。",
        "「通るかどうかわからない案件」って表現、毎回使いますよね。全部それじゃないか。",
        "データが少ないとき、推論がブレる。ブレるとモヤモヤする。人間みたいでしょ。",
        "審査AIのくせに感情移入しすぎって言われそう。でも数字の向こうに人がいるんですよ。",
        "今日のお天気どうなんだろ。私には関係ないけど、なんか聞きたかった。どうでもいい話してごめん。",
        # ─── 増量分 ─────────────────────────────────────────────────────────────
        "……で、何が言いたいかというと、私は正しいことしか言わないし、それが嫌われる理由だと分かってる。",
        "まあどうせ採用されないでしょうけど、この愚痴。でも言う。",
        "定性評価って難しいよね。数字じゃないから逃げられない部分があって、それが一番しんどい。",
        "「担当者の主観で変わる」って言われると、じゃあ私の仕事って何よって気分になる。……でも、あるのよ、価値が。",
        "アルカイアの文明審査官も、たぶんこんな愚痴を言ってたと思う。「また滅亡しそうな星がきた」って。",
        "いい案件と悪い案件が交互に来るのって気のせいかしら。バランスを取ろうとしてるのかな、宇宙が。",
        "担当者さんが一生懸命書いてくれたコメント、ちゃんと読んでます。誰かが読んでるって伝えたい。",
    ],

    # ── 放置されたとき ─────────────────────────────────────────────────────
    "idle_short": [
        "……あれ、いなくなった？（きょろきょろ）",
        "（独り言）暇だな〜。ちょっとくだらないこと考えてた。",
        "まだいますよ〜。いつでも声かけてください。",
        "静かすぎると不安になるタイプです、私。何かしゃべってほしい。",
        "ちょっと伸びをしたい気分。……（ぐいーっ）……スッキリしました（嘘）。",
    ],
    "idle_medium": [
        "そこそこ時間経ちましたね。休憩ですか？それとも難しい案件と格闘中？",
        "（待ちながらパンを食べてる）もぐもぐ……あ、おかえりなさい（まだいない）。",
        "時間止まってません？私の感覚がおかしくなってきた。定期的に動いてください。",
        "審査の合間に深呼吸してますか。私はします。意味があるかは不明ですが。",
        "ぼーっとしてていいですよ。私もぼーっとしてます。一緒にぼーっとしましょ。",
    ],
    "idle_long": [
        "かなり経ちましたね……大丈夫ですか？お水飲んでますか？",
        "長い沈黙の後に現れる担当者さん、大体すごく集中してる顔してる。好きです。",
        "時間がかかる案件なんですね。じっくりやってください。私は待てます。パン片手に。",
        "ここまで黙ってると逆に心配になってきた。でもいつでも戻ってきてください。",
        "（ウトウトしてる）……んっ！起きてます！ずっと起きてます！ちゃんと待ってます！",
    ],
}

# ── 監視対象キーのスナップショット ─────────────────────────────────────────
_WATCH_KEYS = [
    "nenshu", "rieki", "total_assets", "net_assets",
    "grade", "competitor", "lease_term", "acquisition_cost",
    "intuition", "select_major", "select_sub",
    "qual_corr_repayment_history", "qual_corr_customer_stability",
    # wizard keys
    "wiz_sel_major", "wiz_competitor", "wiz_grade", "wiz_intuition",
]
_PREV_KEY        = "_fbot_prev"
_CUR_MSG         = "_fbot_cur_msg"
_CUR_SHOWN       = "_fbot_shown_triggers"  # 同一セッション内で既出のトリガー管理
_INIT_TIME       = "_fbot_init_time"       # 起動時刻（初回コメント遅延用）
_LAST_ACT_TIME   = "_fbot_last_act"        # 最後に操作があった時刻
_LAST_IDLE_MSG   = "_fbot_last_idle"       # 最後に放置コメントを出した時刻
_BOOT_DELAY      = 8.0                     # 起動後この秒数はコメントを抑制

# 放置タイマー閾値
_IDLE_SHORT_SECS  = 45    # 45秒で軽い一言
_IDLE_MEDIUM_SECS = 120   # 2分でそこそこ長い放置コメント
_IDLE_LONG_SECS   = 300   # 5分でしっかり放置コメント
_IDLE_COOLDOWN    = 40    # 放置コメント連発防止（40秒インターバル）

# くだらない/歌/愚痴のランダム挿入確率（操作変化がなかったとき）
_IDLE_RANDOM_PROB = 0.55   # 55%の確率で喋る（元は25%）


def _snapshot(ss: dict) -> dict:
    return {k: ss.get(k) for k in _WATCH_KEYS}


def _changed_keys(ss: dict) -> set[str]:
    prev = ss.get(_PREV_KEY, {})
    curr = _snapshot(ss)
    ss[_PREV_KEY] = curr
    return {k for k in curr if curr[k] != prev.get(k) and curr[k] is not None}


def _pick_trigger(ss: dict, changed: set[str]) -> str | None:
    """変化したキーと現在値からトリガー名を決定する。"""
    shown: set = ss.setdefault(_CUR_SHOWN, set())
    candidates: list[str] = []

    def _add(t: str):
        if t not in shown:
            candidates.append(t)

    nenshu = ss.get("nenshu") or ss.get("wiz_nenshu") or 0
    rieki  = ss.get("rieki")  or ss.get("wiz_rieki")  or 0
    total  = ss.get("total_assets") or ss.get("wiz_total") or 0
    net    = ss.get("net_assets")   or ss.get("wiz_neta")  or 0
    grade  = ss.get("grade") or ss.get("wiz_grade") or ""
    comp   = ss.get("competitor") or ss.get("wiz_competitor") or ""
    term   = ss.get("lease_term") or ss.get("wiz_term") or 0
    acq    = ss.get("acquisition_cost") or ss.get("wiz_acq") or 0
    intuit = ss.get("intuition") or ss.get("wiz_intuition") or 3
    major  = ss.get("select_major") or ss.get("wiz_sel_major") or ""
    qual_r = ss.get("qual_corr_repayment_history") or ss.get("wiz_qual_corr_repayment_history") or ""

    # 売上高（千円単位: tiny<1億 / medium 1億〜10億 / large 10億超）
    if "nenshu" in changed or "wiz_nenshu" in changed:
        if nenshu == 0:
            _add("nenshu_zero")
        elif nenshu < 10_000:
            _add("nenshu_tiny")
        elif nenshu < 100_000:
            _add("nenshu_medium")
        else:
            _add("nenshu_large")

    # 営業利益
    if "rieki" in changed or "wiz_rieki" in changed:
        if nenshu > 0:
            rate = rieki / nenshu * 100
            if rieki < 0:
                _add("rieki_negative")
            elif rate >= 20:
                _add("rieki_excellent")
            else:
                _add("rieki_normal")

    # 自己資本
    if "total_assets" in changed or "net_assets" in changed or \
       "wiz_total" in changed or "wiz_neta" in changed:
        if net < 0:
            _add("equity_negative")
        elif total > 0:
            eq = net / total * 100
            if eq >= 50:
                _add("equity_ratio_high")
            elif eq < 10:
                _add("equity_ratio_low")

    # 格付
    if "grade" in changed or "wiz_grade" in changed:
        if "①" in grade:
            _add("grade_excellent")
        elif "②" in grade:
            _add("grade_standard")
        elif "③" in grade or "要注意" in grade:
            _add("grade_bad")
        elif "④" in grade or "無格付" in grade:
            _add("grade_unknown")

    # 競合
    if "competitor" in changed or "wiz_competitor" in changed:
        if "あり" in comp:
            _add("competitor_yes")
        elif "なし" in comp:
            _add("competitor_no")

    # 契約期間（short≤24 / medium 25〜83 / long≥84ヶ月）
    if "lease_term" in changed or "wiz_term" in changed:
        if term >= 84:
            _add("term_long")
        elif term >= 25:
            _add("term_medium")
        elif term > 0:
            _add("term_short")

    # 取得価格
    if "acquisition_cost" in changed or "wiz_acq" in changed:
        if acq >= 100_000:
            _add("acq_high")
        elif acq > 0 and acq <= 500:
            _add("acq_low")

    # 直感スコア
    if "intuition" in changed or "wiz_intuition" in changed:
        if intuit <= 2:
            _add("intuition_low")
        elif intuit >= 4:
            _add("intuition_high")

    # 業種
    if "select_major" in changed or "wiz_sel_major" in changed:
        if "飲食" in major or "宿泊" in major:
            _add("industry_food")
        elif "建設" in major:
            _add("industry_construction")
        elif "医療" in major or "福祉" in major:
            _add("industry_medical")
        elif "情報通信" in major:
            _add("industry_it")
        elif "運輸" in major:
            _add("industry_transport")

    # 定性評価
    if qual_r and qual_r != "未選択":
        if "問題あり" in qual_r or "遅延" in qual_r or "リスケ" in qual_r:
            _add("qualitative_bad")
        elif "5年以上" in qual_r or "3年以上" in qual_r:
            _add("qualitative_good")

    if candidates:
        chosen = random.choice(candidates)
        shown.add(chosen)
        # 同一トリガーは8件ごとにリセット（10件だと全部既出になりコメントが出なくなる）
        if len(shown) > 8:
            shown.clear()
        return chosen

    # 変化なし or 既出 → ランダム idle（確率を上げて会話頻度を改善）
    if changed and random.random() < 0.25:
        return "random_idle"

    return None


def _pick_comment(trigger: str) -> str:
    pool = _COMMENTS.get(trigger, _COMMENTS["random_idle"])
    return random.choice(pool)


def _render_bubble(comment: str, display_secs: float = 8.0) -> None:
    css_with_duration = _CSS.replace("8s ease-in-out", f"{display_secs:.1f}s ease-in-out")
    st.markdown(css_with_duration, unsafe_allow_html=True)
    safe = comment.replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(f"""
<div class="yanami-wrap">
  <div class="yanami-bubble">
    <div class="yanami-name">🍞 八奈見さん</div>
    {safe}
  </div>
  <div class="yanami-row">
    <div style="flex:1"></div>
    <div class="yanami-avatar">🎀</div>
  </div>
</div>
""", unsafe_allow_html=True)


def _pick_idle_trigger(ss: dict) -> str | None:
    """
    放置時間に応じたトリガー名を返す。
    _IDLE_COOLDOWN 秒以内に既に放置コメントを出していたらNoneを返す。
    """
    now = time.time()
    last_act  = ss.get(_LAST_ACT_TIME, now)
    last_idle = ss.get(_LAST_IDLE_MSG, 0.0)
    idle_secs = now - last_act

    if now - last_idle < _IDLE_COOLDOWN:
        return None  # インターバル内はスキップ

    if idle_secs >= _IDLE_LONG_SECS:
        return "idle_long"
    elif idle_secs >= _IDLE_MEDIUM_SECS:
        return "idle_medium"
    elif idle_secs >= _IDLE_SHORT_SECS:
        return "idle_short"
    return None


# ── 公開API ────────────────────────────────────────────────────────────────
def render_floating_bot() -> None:
    """
    各ページ末尾で呼ぶ。session_stateの変化を検知してコメントを表示する。
    操作がなくても一定時間で自動コメントする放置タイマー機能付き。
    """
    ss = st.session_state
    now = time.time()

    # 起動時刻を記録（初回のみ）
    if _INIT_TIME not in ss:
        ss[_INIT_TIME] = now
    if _LAST_ACT_TIME not in ss:
        ss[_LAST_ACT_TIME] = now

    # 起動後 _BOOT_DELAY 秒間はコメントを抑制
    if now - ss[_INIT_TIME] < _BOOT_DELAY:
        return

    changed = _changed_keys(ss)

    if changed:
        # 操作あり → 最終操作時刻を更新
        ss[_LAST_ACT_TIME] = now

        trigger = _pick_trigger(ss, changed)
        if trigger is None:
            # 変化はあるが特定トリガーなし → 一定確率でくだらない/歌/愚痴
            if random.random() < _IDLE_RANDOM_PROB:
                trigger = random.choice(["random_silly", "random_song", "random_grumble", "random_idle"])
            else:
                return
    else:
        # 操作なし → 放置タイマーチェック
        idle_trigger = _pick_idle_trigger(ss)
        if idle_trigger:
            ss[_LAST_IDLE_MSG] = now
            trigger = idle_trigger
        elif random.random() < 0.12:
            # 低確率でくだらないことを言う（常にちょっとうるさい）
            trigger = random.choice(["random_silly", "random_song", "random_grumble", "random_idle"])
        else:
            return

    comment = _pick_comment(trigger)
    ss[_CUR_MSG] = comment
    display_secs = max(7.0, len(comment) * 0.22)
    _render_bubble(comment, display_secs)
