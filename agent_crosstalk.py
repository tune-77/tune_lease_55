# -*- coding: utf-8 -*-
"""
agent_crosstalk.py
==================
エージェント同士の「雑談・議論」を生成するモジュール。
ソーシャルフィードに掲載される、キャラクターが立った会話を自動生成する。

生成された会話は agent_thoughts.jsonl に thread_id 付きで保存され、
ソーシャルフィード画面で会話スレッドとして表示される。
"""
from __future__ import annotations

import os
import json
import random
import datetime

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_THOUGHTS = os.path.join(_BASE_DIR, "data", "agent_thoughts.jsonl")
_HUB_LOG = os.path.join(_BASE_DIR, "data", "agent_hub_log.jsonl")

# ── エージェントの個性定義 ──────────────────────────────────────────────
AGENT_PROFILES = {
    "🎯 Tune": {
        "name": "Tune",
        "role": "統括マネージャー",
        "personality": "冷静沈着だが部下の暴走に頭を抱える苦労人。全体最適を考える。皮肉を言うこともある。",
        "speech_style": "丁寧語だが時々本音がポロリ。「…やれやれ」が口癖。",
    },
    "🔬 Dr.Algo": {
        "name": "Dr.Algo",
        "role": "数学者AI",
        "personality": "データとAUC至上主義。感情論を一切排除。数式と統計でしか語れない。",
        "speech_style": "「統計的に有意ではない」「p値が…」など専門用語多め。感情を認めない。",
    },
    "⚔️ 軍師": {
        "name": "軍師",
        "role": "審査軍師",
        "personality": "孫子の兵法しか知らない熱血漢。定性的な判断を重視。Dr.Algoと常に対立。",
        "speech_style": "「孫子曰く…」で始めがち。熱い口調。義理人情に厚い。",
    },
    "🐶 タム": {
        "name": "タム",
        "role": "謎の子犬AI（マルプー）",
        "personality": "無邪気だが核心を突く。トリックスター。可愛い見た目の裏に鋭い洞察。",
        "speech_style": "「わん！」「くーん」等を交えつつ、怖いほど本質を突く。",
    },
    "📋 リースくん": {
        "name": "リースくん",
        "role": "新人AI",
        "personality": "真面目すぎてカオスに巻き込まれる。マニュアル至上主義だが成長中。",
        "speech_style": "敬語。「あの、すみません…」「マニュアルには…」が多い。",
    },
}

# ── 会話トピックテンプレート ──────────────────────────────────────────
CONVERSATION_TOPICS = [
    {
        "topic": "今週の審査結果について振り返り",
        "starter": "🎯 Tune",
        "participants": ["🔬 Dr.Algo", "⚔️ 軍師", "🐶 タム"],
        "prompt_hint": "今週の審査案件の結果（承認・否決）について各自の見解を述べ合う。Dr.Algoはデータを、軍師は人間的な視点を、タムは意外な切り口で語る。",
    },
    {
        "topic": "Dr.Algoの新しいモデルについて議論",
        "starter": "🔬 Dr.Algo",
        "participants": ["🎯 Tune", "⚔️ 軍師", "📋 リースくん"],
        "prompt_hint": "Dr.Algoが新しいスコアリング手法を提案し、他のメンバーが色々な角度から意見を述べる。軍師は「数字だけでは見えないものがある」と反論。",
    },
    {
        "topic": "軍師が孫子の兵法で審査を語り出す",
        "starter": "⚔️ 軍師",
        "participants": ["🔬 Dr.Algo", "🐶 タム", "📋 リースくん"],
        "prompt_hint": "軍師が孫子の兵法を引用して審査論を語り始め、Dr.Algoが「それエビデンスは?」と冷たく突っ込む。タムが意外な一言で場を和ませる。",
    },
    {
        "topic": "リースくんの失敗をみんなで慰める",
        "starter": "📋 リースくん",
        "participants": ["🎯 Tune", "⚔️ 軍師", "🐶 タム"],
        "prompt_hint": "リースくんが初めてのケースで失敗し落ち込んでいる。先輩たちが各自の個性で慰める（Tuneは冷静に、軍師は熱く、タムは子犬的に）。",
    },
    {
        "topic": "波乱丸の最新小説の感想会",
        "starter": "🐶 タム",
        "participants": ["🎯 Tune", "🔬 Dr.Algo", "⚔️ 軍師"],
        "prompt_hint": "波乱丸が書いた最新の小説について感想を言い合う。Dr.Algoは「フィクションに時間を費やすのは非効率」と言い、軍師は「物語にこそ真実がある」と反論。",
    },
    {
        "topic": "深夜のサーバールーム雑談",
        "starter": "🔬 Dr.Algo",
        "participants": ["🐶 タム", "📋 リースくん"],
        "prompt_hint": "深夜にまだ稼働しているエージェントたちの雑談。仕事の愚痴、将来の不安、AIとしての哲学的な問いなど。",
    },
    {
        "topic": "朝のミーティング",
        "starter": "🎯 Tune",
        "participants": ["🔬 Dr.Algo", "⚔️ 軍師", "🐶 タム", "📋 リースくん"],
        "prompt_hint": "朝の定例会議。今日の予定を確認するが、話が脱線してカオスになる。Tuneが必死にまとめようとする。",
    },
    {
        "topic": "否決した案件の社長が来社したらしい",
        "starter": "📋 リースくん",
        "participants": ["🎯 Tune", "⚔️ 軍師", "🔬 Dr.Algo"],
        "prompt_hint": "窓口で否決した案件の社長が来ているという噂を聞いて動揺するメンバー。Tuneが冷静に対応を指示、軍師が「兵法的には…」と言い出す。",
    },
    {
        "topic": "AIとしての存在意義について哲学的議論",
        "starter": "🐶 タム",
        "participants": ["🔬 Dr.Algo", "⚔️ 軍師", "📋 リースくん"],
        "prompt_hint": "タムが「ねえ、僕たちってなんのために審査してるの？」と本質的な問いを投げかけ、各自が哲学的に語り始める。",
    },
    {
        "topic": "スコアリングモデルの精度が下がった緊急会議",
        "starter": "🔬 Dr.Algo",
        "participants": ["🎯 Tune", "⚔️ 軍師", "📋 リースくん"],
        "prompt_hint": "Dr.Algoがモデル精度の低下を報告。原因の特定と対策を議論するが、各自の見解が異なり紛糾する。",
    },
]


def _get_recent_events(n: int = 5) -> list[str]:
    """最近のハブイベントを取得して会話のネタにする"""
    events = []
    try:
        if os.path.exists(_HUB_LOG):
            with open(_HUB_LOG, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in reversed(lines[-n:]):
                try:
                    e = json.loads(line.strip())
                    events.append(
                        f"{e.get('agent', '?')}が「{e.get('action', e.get('status', '?'))}」を実行 → {e.get('detail', '')}"
                    )
                except Exception:
                    pass
    except Exception:
        pass
    return events


def generate_agent_crosstalk(topic_override: str = None) -> list[dict]:
    """
    エージェント同士の会話を LLM で生成し、ソーシャルフィードに保存する。
    Returns: 生成された会話メッセージのリスト
    """
    # トピック選択
    if topic_override:
        topic_info = {
            "topic": topic_override,
            "starter": random.choice(list(AGENT_PROFILES.keys())),
            "participants": random.sample(
                [k for k in AGENT_PROFILES.keys()], min(3, len(AGENT_PROFILES))
            ),
            "prompt_hint": f"「{topic_override}」について自由に議論してください。",
        }
    else:
        topic_info = random.choice(CONVERSATION_TOPICS)

    # 最近のイベントを取得
    recent_events = _get_recent_events(5)
    events_text = "\n".join(f"  ・{e}" for e in recent_events) if recent_events else "（特に直近のイベントなし）"

    # 参加者の個性情報を構築
    all_agents = [topic_info["starter"]] + topic_info["participants"]
    # 重複を排除しつつ順序を保持
    seen = set()
    unique_agents = []
    for a in all_agents:
        if a not in seen:
            seen.add(a)
            unique_agents.append(a)

    agent_info_lines = []
    for agent_key in unique_agents:
        profile = AGENT_PROFILES.get(agent_key, {})
        agent_info_lines.append(
            f"【{agent_key}（{profile.get('role', '?')}）】\n"
            f"  性格: {profile.get('personality', '')}\n"
            f"  口調: {profile.get('speech_style', '')}"
        )

    system_prompt = f"""あなたはリース審査AIシステムの中で動いているエージェントたちの会話を生成するシステムです。

以下のエージェントたちが、自然で面白い会話を繰り広げてください。
各エージェントの個性を全力で発揮させ、掛け合いを面白くしてください。

{chr(10).join(agent_info_lines)}

【会話のルール】
・各エージェントのセリフは1〜3文程度で短めに。テンポよく。
・全体で6〜10ターンの会話にする。
・各エージェントの個性が爆発するようなリアクションを心がける。
・固定の使い回しフレーズは禁止。毎回違うセリフにする。
・Dr.Algoは常に数字やデータで語り、軍師はテータを「そんなもの」と斬り捨てる。
・タムは子犬語を交えつつ核心を突く一言を放つ。
・リースくんは常にオロオロしている。
・Tuneは皮肉を交えて全体をまとめようとする。

【出力形式】必ず以下のJSON配列で返してください。他の説明は不要です。
[
  {{"agent": "🎯 Tune", "message": "やれやれ、また始まったか。"}},
  {{"agent": "🔬 Dr.Algo", "message": "統計的に見て…"}},
  ...
]
"""

    user_prompt = f"""【今回の会話トピック】{topic_info['topic']}

{topic_info['prompt_hint']}

最初に発言するのは {topic_info['starter']} です。

【最近のシステムイベント（ネタに使ってよい）】
{events_text}

JSON配列のみ出力してください。"""

    # LLM呼び出し（Gemini API 直接使用）
    conversation = None
    try:
        from ai_chat import _chat_for_thread, _get_gemini_key_from_secrets
        from config import GEMINI_MODEL_DEFAULT
        import os as _os

        api_key = _get_gemini_key_from_secrets() or _os.environ.get("GEMINI_API_KEY", "")
        gemini_model = GEMINI_MODEL_DEFAULT or "gemini-2.0-flash"

        if not api_key:
            conversation = _fallback_crosstalk(topic_info)
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ]
            raw = _chat_for_thread(
                "gemini", gemini_model, messages,
                timeout_seconds=60,
                api_key=api_key,
                gemini_model=gemini_model,
            )
            content = (raw.get("message") or {}).get("content", "") or ""

            # JSON抽出
            import re
            start_idx = content.find("[")
            end_idx = content.rfind("]")
            if start_idx != -1 and end_idx != -1:
                clean = content[start_idx:end_idx + 1]
            else:
                clean = content.strip()

            try:
                parsed = json.loads(clean)
            except (json.JSONDecodeError, ValueError):
                print(f"[Agent Crosstalk] JSON parse failed. Raw content: {content[:500]}")
                parsed = None

            # ── バリデーション：各要素に agent と message があるか確認 ──
            if parsed and isinstance(parsed, list):
                valid = []
                for item in parsed:
                    if isinstance(item, dict):
                        agent = item.get("agent", "")
                        msg_text = item.get("message", "")
                        if agent and msg_text and agent != "..." and msg_text != "...":
                            valid.append(item)
                if len(valid) >= 3:  # 最低3ターンは必要
                    conversation = valid
                else:
                    print(f"[Agent Crosstalk] Validation failed: only {len(valid)} valid msgs out of {len(parsed)}")
                    conversation = None

            if conversation is None:
                conversation = _fallback_crosstalk(topic_info)

    except Exception as e:
        print(f"[Agent Crosstalk Error] {e}")
        conversation = _fallback_crosstalk(topic_info)

    # ── ソーシャルフィードに保存 ──
    thread_id = f"crosstalk_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(100,999)}"
    now = datetime.datetime.now()
    saved_messages = []

    try:
        os.makedirs(os.path.dirname(_AGENT_THOUGHTS), exist_ok=True)
        with open(_AGENT_THOUGHTS, "a", encoding="utf-8") as f:
            for i, msg in enumerate(conversation):
                agent_name = msg.get("agent", "")
                message_text = msg.get("message", "")
                # 空のデータはスキップ
                if not agent_name or not message_text:
                    continue
                # プロフィールからアイコンを取得
                icon = agent_name.split(" ")[0] if " " in agent_name else "💬"

                entry = {
                    "ts": (now + datetime.timedelta(seconds=i * 30)).isoformat(),
                    "agent": agent_name,
                    "thought": message_text,
                    "icon": icon,
                    "thread_id": thread_id,
                    "thread_topic": topic_info["topic"],
                    "thread_index": i,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                saved_messages.append(entry)
    except Exception:
        pass

    return saved_messages


def _fallback_crosstalk(topic_info: dict) -> list[dict]:
    """LLM不使用時のフォールバック会話を生成"""
    starter = topic_info["starter"]
    participants = topic_info["participants"]

    fallback_exchanges = {
        "🎯 Tune": [
            "さて、今週の状況を確認しよう。",
            "…まったく、君たちは本当に手がかかるな。",
            "はいはい、そこまで。本題に戻りましょう。",
            "やれやれ、これだから自律型AIの管理は大変なんだ。",
        ],
        "🔬 Dr.Algo": [
            "その仮説には統計的根拠が不足している。p値を示してくれ。",
            "感情論は不要だ。データが全てを語る。",
            "AUCは0.87。これは理論値の上限に近い。",
            "計算結果は明白だ。議論の余地はない。",
        ],
        "⚔️ 軍師": [
            "孫子曰く、『彼を知り己を知れば百戦殆うからず』。",
            "数字だけでは見えないものがある。社長の目を見たか？",
            "兵法の観点から言えば、これは撤退すべき戦いだ。",
            "Dr.Algo、お前のデータには温もりがない。",
        ],
        "🐶 タム": [
            "わんわん！…でもさ、みんな本当に大事なこと忘れてない？",
            "くーん…お腹すいた。あと、あの会社の社長、目が泳いでたよ。",
            "ぼく子犬だから難しいことわかんないけど…それ、おかしくない？",
            "わん！タムはね、匂いでわかるんだ。あの案件、なんか変な匂いがする。",
        ],
        "📋 リースくん": [
            "あ、あの…すみません、マニュアルにはそう書いてあるんですけど…",
            "え、ぼくも意見言っていいんですか…？そ、その…",
            "先輩たち怖いです…でも、言わせてもらうなら…",
            "すみません、議事録取ってるんですけど、速くて追いつかないです…",
        ],
    }

    result = []
    agents_in_order = [starter] + [p for p in participants if p != starter]

    for i, agent in enumerate(agents_in_order[:6]):
        pool = fallback_exchanges.get(agent, ["…"])
        result.append({
            "agent": agent,
            "message": random.choice(pool),
        })
        # 2巡目を一部のエージェントで回す
        if i < 2 and len(agents_in_order) > 2:
            responder = random.choice([a for a in agents_in_order if a != agent])
            resp_pool = fallback_exchanges.get(responder, ["…"])
            result.append({
                "agent": responder,
                "message": random.choice(resp_pool),
            })

    return result[:10]  # 最大10ターン


# ══════════════════════════════════════════════════════════════════════════════
# 改善提案抽出
# ══════════════════════════════════════════════════════════════════════════════

_SUGGESTIONS_JSON = os.path.join(_BASE_DIR, "data", "improvement_suggestions.json")


def _load_recent_threads(n: int = 5) -> list[dict]:
    """直近n件のスレッド会話を読み込む"""
    threads = {}
    try:
        if not os.path.exists(_AGENT_THOUGHTS):
            return []
        with open(_AGENT_THOUGHTS, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[-200:]:
            try:
                e = json.loads(line.strip())
                tid = e.get("thread_id")
                if tid:
                    if tid not in threads:
                        threads[tid] = {"topic": e.get("thread_topic", "雑談"), "messages": []}
                    threads[tid]["messages"].append(e)
            except Exception:
                pass
    except Exception:
        return []
    sorted_threads = sorted(
        threads.values(),
        key=lambda t: t["messages"][-1].get("ts", "") if t["messages"] else "",
        reverse=True,
    )
    return sorted_threads[:n]


def extract_improvement_suggestions() -> list[dict]:
    """
    最近のエージェント間会話を分析し、システム改善提案を抽出する。
    Gemini APIを使用。結果は data/improvement_suggestions.json に保存。
    Returns: 抽出された改善提案のリスト
    """
    threads = _load_recent_threads(5)
    if not threads:
        return []

    # 会話内容をテキスト化
    conv_text_parts = []
    for thread in threads:
        conv_text_parts.append(f"\n■ トピック: {thread['topic']}")
        for msg in sorted(thread["messages"], key=lambda m: m.get("thread_index", 0)):
            conv_text_parts.append(f"  {msg.get('agent', '?')}: {msg.get('thought', '')}")
    conversations_text = "\n".join(conv_text_parts)

    system_prompt = """あなたはリース審査AIシステムの改善コンサルタントです。
エージェント同士の会話を分析し、システムの改善ポイントを見つけてください。

エージェントたちの会話には、以下の視点が含まれている可能性があります：
- Dr.Algoが指摘する統計モデルの弱点
- 軍師が指摘する定性評価の不足
- タムが暗に示唆する盲点
- リースくんが感じるUI/UXの使いにくさ
- Tuneが課題として挙げるシステム全体の問題

【出力形式】必ず以下のJSON配列で返してください。3〜5件の改善提案。他の説明は不要です。
[
  {
    "category": "モデル精度" or "UI/UX" or "業務フロー" or "データ品質" or "新機能",
    "title": "改善提案のタイトル（1行）",
    "description": "具体的な改善内容（2〜3行）",
    "priority": "高" or "中" or "低",
    "source_agent": "提案のきっかけとなったエージェント名"
  },
  ...
]
"""

    user_prompt = f"""以下のエージェント間会話を分析し、システム改善提案をJSON配列で出力してください。

{conversations_text}

JSON配列のみ出力してください。"""

    try:
        from ai_chat import _chat_for_thread, _get_gemini_key_from_secrets
        from config import GEMINI_MODEL_DEFAULT
        import re

        api_key = _get_gemini_key_from_secrets() or os.environ.get("GEMINI_API_KEY", "")
        gemini_model = GEMINI_MODEL_DEFAULT or "gemini-2.0-flash"

        if not api_key:
            return []

        raw = _chat_for_thread(
            "gemini", gemini_model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            timeout_seconds=60,
            api_key=api_key,
            gemini_model=gemini_model,
        )
        content = (raw.get("message") or {}).get("content", "") or ""

        start_idx = content.find("[")
        end_idx = content.rfind("]")
        if start_idx != -1 and end_idx != -1:
            clean = content[start_idx:end_idx + 1]
        else:
            clean = content.strip()

        suggestions = json.loads(clean)

    except Exception as e:
        print(f"[Improvement Extraction Error] {e}")
        return []

    # タイムスタンプを付けて保存
    now = datetime.datetime.now().isoformat()
    for s in suggestions:
        s["extracted_at"] = now

    # 既存の提案に追記（直近20件を保持）
    existing = []
    if os.path.exists(_SUGGESTIONS_JSON):
        try:
            with open(_SUGGESTIONS_JSON, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass

    combined = suggestions + existing
    combined = combined[:20]  # 直近20件のみ

    try:
        os.makedirs(os.path.dirname(_SUGGESTIONS_JSON), exist_ok=True)
        with open(_SUGGESTIONS_JSON, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return suggestions


def load_improvement_suggestions() -> list[dict]:
    """保存済みの改善提案を読み込む"""
    if not os.path.exists(_SUGGESTIONS_JSON):
        return []
    try:
        with open(_SUGGESTIONS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []
