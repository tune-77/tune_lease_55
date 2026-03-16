"""
AIエージェントの処理モジュール（LangChain）

ルール・過去案件などの検索ツール（Tools）と、
それらを自律的に使いこなして回答を組み立てるAgent Executorを定義します。
"""
import os
import requests
import streamlit as st
import datetime
import re
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

from rule_manager import load_business_rules
from data_cases import find_similar_past_cases
from config import GEMINI_MODEL_DEFAULT

@tool
def search_business_rules(query: str) -> str:
    """
    システムの審査ロジック、閾値（しきい値）、業種ごとの独自ルールを検索・参照するためのツールです。
    ユーザーから「ルールはどうなっている？」「赤字の場合は？」等の質問があった場合に仕様を確認するために使います。
    引数 query: 検索したい内容（例: "建設業", "赤字", "しきい値" など）
    """
    rules = load_business_rules()
    # 簡易的にルールの全体をテキスト化して返す
    # ※本来はqueryに応じて絞り込むが、JSON全体がそこまで大きくないため文字列化して渡す
    import json
    try:
        rules_text = json.dumps(rules, ensure_ascii=False, indent=2)
        return f"【現在のシステムルール】\n{rules_text}"
    except Exception as e:
        return f"ルールの取得に失敗しました: {e}"

@tool
def search_past_cases(industry: str, equity_ratio: float) -> str:
    """
    過去の審査案件データベースから、指定した業種や自己資本比率に近い案件を検索するツールです。
    ユーザーから「過去に似た案件は通った？」等の質問があった場合に使います。
    引数 industry: 対象の業種名（例: "建設業"）
    引数 equity_ratio: 検索の基準となる自己資本比率（%）
    """
    try:
        # 最大5件引いてサマリーを返す
        cases = find_similar_past_cases(industry, equity_ratio, max_count=5)
        if not cases:
            return f"{industry}で自己資本比率が{equity_ratio}%に近い過去事例は見つかりませんでした。"
        
        summary = []
        for i, c in enumerate(cases):
            status = c.get("final_status", "不明")
            score = c.get("result", {}).get("score", 0)
            summary.append(f"事例{i+1}: 判定={status}, スコア={score:.1f}")
            
        return f"【似た過去事例の検索結果】\n" + "\n".join(summary)
    except Exception as e:
        return f"過去事例の検索に失敗しました: {e}"

@tool
def search_custom_manuals(query: str) -> str:
    """
    社内の独自の審査マニュアル、規定、FAQ、ナレッジベースなどの社内文書を横断検索するツールです。
    ユーザーから「〇〇の判定基準は？」「マニュアルにはどう書いてある？」等の質問があった場合に使います。
    引数 query: 検索したい内容（例: "建設業の特例", "赤字の場合の対応" など）
    """
    try:
        url = "http://localhost:3001/api/v1/workspace/71b18af2-ab59-4bcb-874c-4cb329bd1b41/chat"
        headers = {
            "Authorization": "Bearer RGK6FNE-HHK4VTT-NJVQS1S-CK2847K",
            "Content-Type": "application/json"
        }
        data = {
            "message": query,
            "mode": "chat"
        }
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("textResponse", "")
            return f"【社内マニュアルからの検索結果】\n{answer}"
        else:
            return f"マニュアル検索でエラーが発生しました。ステータスコード: {response.status_code}"
    except Exception as e:
        return f"マニュアル検索に失敗しました: {e}"

@tool
def search_web(query: str) -> str:
    """インターネット上から最新のニュースや企業情報などを検索するツールです。
    ユーザーから「最新の動向は？」「〇〇社のニュースはある？」と聞かれたときに使います。
    引数 query: 検索キーワード（例: "運送業 倒産 ニュース"）"""
    try:
        search = DuckDuckGoSearchRun()
        return search.invoke(query)
    except Exception as e:
        return f"Web検索に失敗しました: {e}"

@tool
def calculate_expression(expression: str) -> str:
    """数値の四則演算や複雑な計算を正確に行うツールです。
    ユーザーから「〇〇の〇〇%は？」「合計でいくら？」と聞かれたときに使います。
    引数 expression: 評価可能な算術式（例: "15000000 * 0.05", "(100 + 20) / 3"）"""
    try:
        # LLMからの数式を計算して返す
        # セキュリティ上、標準ビルトイン関数などは除外
        allowed_names = {"__builtins__": None}
        result = eval(expression, allowed_names, {})
        return f"計算結果: {result}"
    except Exception as e:
        return f"計算エラー: {expression} を計算できませんでした ({e})"

@tool
def read_pdf_document(file_path: str) -> str:
    """ローカルに保存されているPDFファイルからテキストを読み込むツールです。
    ユーザーから「〇〇の資料を要約して」とパスを指定されたときに使います。
    引数 file_path: PDFファイルの絶対パスまたは相対パス"""
    try:
        import PyPDF2
        if not os.path.exists(file_path):
            return f"エラー: 指定されたパスにファイルが見つかりません ({file_path})"
        
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            # 最初の5ページのみ読み込む
            for i in range(min(5, len(reader.pages))):
                page = reader.pages[i]
                text += page.extract_text() + "\n"
        
        if len(reader.pages) > 5:
            text += "\n... (以降のページは省略されました)"
            
        return f"【PDF {os.path.basename(file_path)} の読み込み結果】\n{text[:4000]}"
    except Exception as e:
        return f"PDFの読み込みに失敗しました: {e}"

@tool
def generate_report(title: str, content: str) -> str:
    """調査結果や審査の要約をレポートとしてMarkdownファイルに自動作成・保存するツールです。
    ユーザーから「レポートにまとめて」「ファイルに出力して」と依頼されたときに使います。
    引数 title: レポートのタイトル（ファイル名にも使用されます）
    引数 content: レポートの本文（Markdown形式の長文テキスト）"""
    try:
        out_dir = "outputs/reports"
        os.makedirs(out_dir, exist_ok=True)
        safe_title = "".join(c for c in title if c.isalnum() or c in (" ", "_", "-")).rstrip()
        filename = f"{safe_title}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.md"
        filepath = os.path.join(out_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n{content}")
            
        return f"レポートが正常に作成されました。保存先: {filepath}"
    except Exception as e:
        return f"レポート作成に失敗しました: {e}"

def mask_personal_info(text: str) -> str:
    """
    ユーザーの入力内容から企業名や電話番号、メールアドレスなどの個人/機密情報を
    簡易的に黒塗り（マスキング）するユーティリティ関数。
    """
    if not text:
        return text
    
    # 1. 電話番号のマスキング (例: 03-1234-5678, 090-1234-5678)
    text = re.sub(r'0\d{1,4}-\d{1,4}-\d{4}', '[電話番号]', text)
    # 2. メールアドレスのマスキング
    text = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[Email]', text)
    # 3. 企業名のマスキング (株式会社〇〇、〇〇(株) など)
    text = re.sub(r'(株式会社|有限会社|合同会社|合名会社|合資会社|一般社団法人|財団法人|医療法人)\s*[^\s　、。！？,.*\(\)（）]+', r'\1[マスキング済み]', text)
    text = re.sub(r'[^\s　、。！？,.*\(\)（）]+\s*(株式会社|有限会社|合同会社|合名会社|合資会社|一般社団法人|財団法人|医療法人)', r'[マスキング済み]\1', text)
    # (株) や (有) などの略称
    text = re.sub(r'(\(株\)|\(有\)|\(同\)|\(名\)|\(資\))\s*[^\s　、。！？,.*\(\)（）]+', r'\1[マスキング済み]', text)
    text = re.sub(r'[^\s　、。！？,.*\(\)（）]+\s*(\(株\)|\(有\)|\(同\)|\(名\)|\(資\))', r'[マスキング済み]\1', text)
    
    return text

def run_agent_query(user_input: str, system_context: str, api_key: str, model_name: str = GEMINI_MODEL_DEFAULT, image_base64: str = None, messages: list = None):
    """
    LangChainのTool-calling Agentを実行し、ユーザーの質問に対する回答を文字列で返す。
    """
    # LLMへ渡す前に入力コンテキストをマスキング
    user_input = mask_personal_info(user_input)
    system_context = mask_personal_info(system_context)
    
    if not api_key:
        yield {"type": "error", "content": "Gemini APIキーが設定されていません。"}
        return

    # 1. ツール定義
    tools = [
        search_business_rules, 
        search_past_cases, 
        search_custom_manuals,
        search_web,
        calculate_expression,
        read_pdf_document,
        generate_report
    ]
    
    # 2. LLMの初期化
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.7,
    )
    
    # 3. LangGraph の create_react_agent を使用
    system_message = (
        "あなたは法人リース審査のプロフェッショナルAIエージェントです。\n"
        "ユーザーの質問に対し、必要に応じて各ツール（ルール検索や過去事例検索など）を使って正確に答えてください。\n"
        "分からないことは適当に答えず、ツールで調べてください。\n"
        "※重要※ あなたは強力な『視覚（Vision）能力』をネイティブに持っています。ユーザーから画像が添付された場合は、ツールを使わずにあなた自身の目で直接画像を読み取り、分析して回答してください。\n\n"
        f"【現在の背景情報】\n{system_context}"
    )

    agent_executor = create_react_agent(
        llm,
        tools,
        prompt=system_message
    )
    
    # 5. 実行
    try:
        from langchain_core.messages import HumanMessage, AIMessage
        
        langchain_messages = []
        
        # もし過去のmessagesが渡されていれば構築
        if messages:
            for m in messages[:-1]: # 最後のメッセージは別で処理
                if m["role"] == "user":
                    langchain_messages.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=m["content"]))
            
            # 最後のメッセージに画像が保存されていればそれを優先する
            if messages[-1].get("image_base64") and not image_base64:
                image_base64 = messages[-1]["image_base64"]
                    
        # 最新のメッセージの構築
        if image_base64:
            message_content = [
                {"type": "text", "text": user_input},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64[:50]}..."}} # Debug log doesn't need full base64
            ]
            
            # 念のため、ちゃんと画像部分ができているかファイルに書き出して確認
            with open("/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/debug_agent_payload.txt", "w") as f:
                f.write(f"IMAGE WAS PASSED TO RUN_AGENT_QUERY!\n{str(message_content)}")
                
            # 実データは省略なし
            real_message_content = [
                {"type": "text", "text": user_input},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
            langchain_messages.append(HumanMessage(content=real_message_content))
        else:
            with open("/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/debug_agent_payload.txt", "w") as f:
                f.write(f"IMAGE WAS NOT PASSED TO RUN_AGENT_QUERY. BASE64 WAS NONE.\nUser Input: {user_input}")
                
            langchain_messages.append(HumanMessage(content=user_input))

        inputs = {"messages": langchain_messages}

        # streamを使用し、実行過程（イベント）を返す
        events = agent_executor.stream(
            inputs,
            stream_mode="values"
        )
        
        final_answer = ""
        # yieldを使ってUI側に途中経過を伝えるジェネレータに変更する
        # （ここでは、ツール呼び出し等のイベントをyieldし、最後に最終回答をyieldする仕様にする）
        for event in events:
            if "messages" in event:
                last_msg = event["messages"][-1]
                # ツール呼び出し要求の検知
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        tool_name = tc.get('name', '不明なツール')
                        args = tc.get('args', {})
                        yield {"type": "tool_start", "tool": tool_name, "input": str(args)}
                # ツール結果の検知
                elif getattr(last_msg, "type", "") == "tool":
                    yield {"type": "tool_end", "tool": getattr(last_msg, "name", ""), "output": last_msg.content}
                # AIの通常の返答（最終回答含む）
                elif getattr(last_msg, "type", "") == "ai" and not getattr(last_msg, "tool_calls", []):
                    final_answer = last_msg.content
                    
        if final_answer:
            yield {"type": "final_answer", "content": final_answer}
        else:
            yield {"type": "final_answer", "content": "（AIからの応答を解析できませんでした）"}
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            yield {"type": "error", "content": "Gemini APIの利用枠制限に達しました。"}
        else:
            yield {"type": "error", "content": f"AIエージェントの実行でエラーが発生しました: {e}"}
