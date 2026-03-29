import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import GEMINI_MODEL_DEFAULT

def generate_a4_summary_dashboard(result_data: dict, inputs: dict, ai_history: list, api_key: str, model_name: str = GEMINI_MODEL_DEFAULT) -> str:
    """
    1件の審査結果データ（スコア、財務データ、AIのチャット履歴や軍師コメント等）を受け取り、
    A4用紙1枚（または2枚）に収まるエグゼクティブ・サマリー（稟議書ベースのMarkdown）を生成するエージェント。
    """
    if not api_key:
        return "エラー: Gemini APIキーが設定されていません。"

    # 渡されたデータをテキスト化してプロンプトに埋め込む
    try:
        # 結果データの整形
        score = result_data.get("score", 0)
        hantei = result_data.get("hantei", "不明")
        
        # 不要に長くなりすぎるのを防ぐため、一部のデータだけを抽出
        system_prompt = (
            "あなたは非常に優秀なダッシュボードプランナー（エグゼクティブ・アシスタント）です。\n"
            "以下の【リース審査結果データ】を元に、経営層や上司へそのまま提出できる「A4サイズ1〜2枚に収まる、洗練されたエグゼクティブ・サマリー（稟議書形式のダッシュボード）」をMarkdownで作成してください。\n\n"
            "【出力形式のルール】\n"
            "1. 全体を美しく見やすい構成にするため、Markdownのヘッダー（##, ###）、太字、箇条書き、表（テーブル）を効果的に使ってください。\n"
            "2. 以下の構成要素を必ず入れてください：\n"
            "   - **【結論（サマリー）】**: 承認か否決か（スコアも併記）、およびその最大の決め手は何か（3行以内で端的に）\n"
            "   - **【ハイライト】**: 今回の案件の強み（ポジティブ要因）と、弱み・リスク（ネガティブ要因）の箇条書き\n"
            "   - **【定量データ（主要KPI）】**: 売上高、自己資本比率など、決裁に影響する極めて重要な数字だけを表形式等で抜粋して提示\n"
            "   - **【軍師（AI）の見解】**: なぜそのスコアや結果になったかの分析要約\n"
            "   - **【ネクストアクション】**: 営業担当者や審査担当者が「次に具体的に何をすべきか」の提案\n"
            "3. 長すぎる文章や不要なログは削り、エグゼクティブが「パッと見て理解できる」密度に圧縮してください。"
        )

        user_content = (
            f"【審査判定】: {hantei} (スコア: {score}点)\n\n"
            f"【入力データ抜粋】:\n{json.dumps(inputs, ensure_ascii=False, indent=2)}\n\n"
            f"【判定詳細・AIのコメント記録】:\n{json.dumps(ai_history, ensure_ascii=False, indent=2)}\n"
        )

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3, # 報告書なので低めに設定
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]

        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        return f"サマリーの生成中にエラーが発生しました: {e}"
