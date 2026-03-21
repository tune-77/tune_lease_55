#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
send_slack_report.py
====================
4人のエージェントチームが議論した改善レポートをSlackに送信するスクリプト。

使い方:
    # 環境変数でWebhook URLを指定
    export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/T.../B.../xxx"
    python send_slack_report.py

    # またはコマンドライン引数で
    python send_slack_report.py --webhook "https://hooks.slack.com/services/T.../B.../xxx"
"""

import json
import sys
import argparse
import datetime
import requests

# ══════════════════════════════════════════════════════════════════════════════
# 4人のエージェントチーム — リースシステム改善ディスカッション結果
# ══════════════════════════════════════════════════════════════════════════════

REPORT = {
    "title": "リースシステム改善レポート — エージェントチーム議論結果",
    "date": datetime.datetime.now().strftime("%Y年%m月%d日"),
    "agents": [
        {
            "name": "プランナー",
            "avatar": "🔭",
            "improvements": [
                {
                    "priority": "🔴 高",
                    "title": "入力バリデーション不備",
                    "detail": (
                        "系のエントロピー増大を防ぐ第一歩として、フォーム入力値の検証が甘い。"
                        "年収・純資産にマイナス値や0が入力可能で、スコアリングが破綻するリスクがある。"
                        "プロスペクト理論で言えば「損失を未然に防ぐ」仕組みが欠如。"
                        "全入力フィールドにmin/max制約と型チェックを導入すべき。"
                    ),
                    "effort": "1日",
                },
                {
                    "priority": "🔴 高",
                    "title": "エラーハンドリングの空白地帯",
                    "detail": (
                        "analysis_regression.pyの行列演算、data_cases.pyのDB読み込みで"
                        "例外が握りつぶされ、ユーザーには白い画面が表示される。"
                        "アンカリング効果で「壊れた」という第一印象が固定される。"
                        "全てのexceptブロックでst.errorによるユーザー向けメッセージを出すべき。"
                    ),
                    "effort": "2日",
                },
                {
                    "priority": "🟡 中",
                    "title": "係数バージョン管理なし",
                    "detail": (
                        "coeff_auto.jsonが上書き保存されるだけで、ロールバック不可。"
                        "エントロピー的には不可逆過程。タイムスタンプ付きバックアップと差分表示が必要。"
                    ),
                    "effort": "3日",
                },
            ],
        },
        {
            "name": "ダッシュ",
            "avatar": "📊",
            "improvements": [
                {
                    "priority": "🔴 高",
                    "title": "モバイル表示の崩壊",
                    "detail": (
                        "PowerBIなら当然レスポンシブなのに、このシステムは1200px未満で"
                        "右カラム（AI相談）が切れる。CSS hardcoded min-widthが原因。"
                        "認知負荷が爆発する。st.columnsの比率を動的にし、"
                        "モバイルでは縦積みレイアウトに切り替えるべき。"
                    ),
                    "effort": "2日",
                },
                {
                    "priority": "🔴 高",
                    "title": "スコア可視化が数字の羅列",
                    "detail": (
                        "審査スコアの内訳がst.metricの数値だけ。"
                        "視線動線を考慮したゲージチャート+レーダーチャートで"
                        "直感的にスコアの強弱が分かるダッシュボードに作り変えるべき。"
                        "Plotly gaugeとradarで実装可能。"
                    ),
                    "effort": "3日",
                },
                {
                    "priority": "🟡 中",
                    "title": "ローディング体験の貧弱さ",
                    "detail": (
                        "st.spinnerだけでは何がどこまで進んだか不明。"
                        "プログレスバーと推定残り時間を表示すべき。情報密度を上げよ。美しくあれ。"
                    ),
                    "effort": "1日",
                },
            ],
        },
        {
            "name": "田中さん",
            "avatar": "💼",
            "improvements": [
                {
                    "priority": "🔴 高",
                    "title": "フォームの入力項目が多すぎる",
                    "detail": (
                        "お客さん目線で言うとね、47個もセッションキーがある入力フォームは使えない。"
                        "営業は移動中にスマホで入力するんだから、必須項目を10個以下に絞って、"
                        "「かんたんモード」と「詳細モード」の2段階にしてほしい。"
                    ),
                    "effort": "3日",
                },
                {
                    "priority": "🔴 高",
                    "title": "審査確率が表示されない",
                    "detail": (
                        "スコアは出るけど「結局、通るの？通らないの？」が分からない。"
                        "過去データから承認確率を％で出してほしい。"
                        "「このスコアなら承認率78%」って出れば、お客さんにも説明しやすい。"
                    ),
                    "effort": "2日",
                },
                {
                    "priority": "🟡 中",
                    "title": "検索機能が弱い",
                    "detail": (
                        "過去案件を業種やスコア範囲で検索したいのに、フルスキャンしかない。"
                        "「前に同じ業種で通った案件」を素早く見つけたい。"
                        "とにかく使いやすくしてほしいです！"
                    ),
                    "effort": "2日",
                },
            ],
        },
        {
            "name": "鈴木さん",
            "avatar": "💻",
            "improvements": [
                {
                    "priority": "🔴 高",
                    "title": "テストがゼロ",
                    "detail": (
                        "技術的には可能ですが…単体テストが1つもない。"
                        "スコアリングロジック（scoring_core.py）の回帰テスト、"
                        "DB操作のINTEGRATIONテスト、フォームバリデーションテスト、最低限この3つ。"
                        "pytest + fixtures で3日〜5日。"
                    ),
                    "effort": "3〜5日",
                },
                {
                    "priority": "🔴 高",
                    "title": "SQLiteの同時アクセス問題",
                    "detail": (
                        "lease_data.dbとscreening_db.sqliteの2つが並列で、"
                        "トランザクション管理もコネクションプーリングもない。"
                        "同時アクセスでロック待ちが発生する。"
                        "SQLAlchemy導入 or 最低限contextmanagerでwith文管理にすべき。"
                    ),
                    "effort": "4日",
                },
                {
                    "priority": "🟡 中",
                    "title": "キャッシュ未活用でページ表示が遅い",
                    "detail": (
                        "load_all_cases()が毎回フルスキャン、Plotlyチャートも毎回再生成。"
                        "@st.cache_dataを入れれば50%は高速化できる。"
                        "ちょっとトレードオフがあって…TTL設定を間違えると古いデータが出る。"
                        "なんとかやります…（泣）"
                    ),
                    "effort": "1〜2日",
                },
            ],
        },
    ],
    "summary": {
        "total_items": 12,
        "high_priority": 8,
        "medium_priority": 4,
        "estimated_total_effort": "22〜28日",
        "recommended_order": [
            "1. 入力バリデーション（プランナー提案 — 即効性あり）",
            "2. キャッシュ導入（鈴木さん — 体感速度改善）",
            "3. モバイル対応（ダッシュ — 営業現場で必須）",
            "4. かんたんモード（田中さん — ユーザー体験大幅改善）",
            "5. テスト整備（鈴木さん — 安全に開発するための基盤）",
            "6. エラーハンドリング（プランナー — 信頼性向上）",
            "7. スコア可視化（ダッシュ — 見た目の説得力）",
            "8. 承認確率表示（田中さん — 営業支援の要）",
        ],
    },
}


def _build_slack_blocks(report: dict) -> list[dict]:
    """レポートをSlack Block Kit形式に変換。"""
    blocks = []

    # ── ヘッダー ──
    blocks.append({
        "type": "header",
        "text": {"type": "plain_text", "text": f"🤝 {report['title']}", "emoji": True},
    })
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": f"📅 {report['date']} | 4人のエージェントによるコードベース分析に基づく改善提案"}],
    })
    blocks.append({"type": "divider"})

    # ── 各エージェントの改善提案 ──
    for agent in report["agents"]:
        # エージェントヘッダー
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*{agent['avatar']} {agent['name']}の提案:*"},
        })

        for imp in agent["improvements"]:
            text = f"{imp['priority']} *{imp['title']}*（工数: {imp['effort']}）\n{imp['detail']}"
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": text},
            })

        blocks.append({"type": "divider"})

    # ── サマリー ──
    summary = report["summary"]
    summary_text = (
        f"*📋 改善サマリー*\n"
        f"• 総項目数: {summary['total_items']}件\n"
        f"• 🔴 高優先: {summary['high_priority']}件 / 🟡 中優先: {summary['medium_priority']}件\n"
        f"• 推定総工数: {summary['estimated_total_effort']}\n\n"
        f"*🏆 推奨実施順:*\n"
    )
    summary_text += "\n".join(summary["recommended_order"])

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": summary_text},
    })

    # ── フッター ──
    blocks.append({"type": "divider"})
    blocks.append({
        "type": "context",
        "elements": [
            {"type": "mrkdwn", "text": "🤖 _リース審査システム エージェントチーム — Powered by Claude Opus 4.6_"},
        ],
    })

    return blocks


def _split_blocks_into_messages(blocks: list[dict], max_blocks: int = 50) -> list[list[dict]]:
    """Slack のブロック数制限（50）に収まるよう分割。"""
    messages = []
    current = []
    for block in blocks:
        current.append(block)
        if len(current) >= max_blocks:
            messages.append(current)
            current = []
    if current:
        messages.append(current)
    return messages


def send_report_to_slack(webhook_url: str) -> bool:
    """改善レポートをSlackに送信する。"""
    blocks = _build_slack_blocks(REPORT)
    message_chunks = _split_blocks_into_messages(blocks)

    fallback_text = (
        f"【リースシステム改善レポート】{REPORT['date']}\n"
        f"4人のエージェントが{REPORT['summary']['total_items']}件の改善点を提案しました。"
    )

    success = True
    for i, chunk in enumerate(message_chunks):
        payload = {
            "text": fallback_text if i == 0 else f"（続き {i+1}/{len(message_chunks)}）",
            "blocks": chunk,
        }
        try:
            resp = requests.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=15,
            )
            if resp.status_code != 200 or resp.text != "ok":
                print(f"❌ Slack送信エラー (チャンク{i+1}): HTTP {resp.status_code} — {resp.text}")
                success = False
            else:
                print(f"✅ チャンク {i+1}/{len(message_chunks)} 送信完了")
        except Exception as e:
            print(f"❌ Slack送信エラー (チャンク{i+1}): {e}")
            success = False

    return success


def main():
    parser = argparse.ArgumentParser(description="改善レポートをSlackに送信")
    parser.add_argument("--webhook", type=str, help="Slack Incoming Webhook URL")
    parser.add_argument("--dry-run", action="store_true", help="送信せずにJSON出力のみ")
    args = parser.parse_args()

    webhook_url = args.webhook or __import__("os").environ.get("SLACK_WEBHOOK_URL", "")

    if args.dry_run:
        blocks = _build_slack_blocks(REPORT)
        print(json.dumps({"blocks": blocks}, ensure_ascii=False, indent=2))
        print(f"\n📊 ブロック数: {len(blocks)}")
        return

    if not webhook_url:
        print("❌ Webhook URLが指定されていません。")
        print("   使い方:")
        print("   export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/T.../B.../xxx'")
        print("   python send_slack_report.py")
        print("")
        print("   または:")
        print("   python send_slack_report.py --webhook 'https://hooks.slack.com/services/...'")
        print("")
        print("   Webhook URLの取得方法:")
        print("   1. https://api.slack.com/apps にアクセス")
        print("   2. Create New App → From scratch")
        print("   3. App名: 'リース審査レポート' / Workspace: leasecorp")
        print("   4. Incoming Webhooks → Activate → Add New Webhook to Workspace")
        print("   5. 投稿先チャンネルを選択 → Webhook URLをコピー")
        sys.exit(1)

    if not webhook_url.startswith("https://hooks.slack.com/"):
        print(f"⚠️ URLが正しくありません: {webhook_url[:50]}...")
        print("   https://hooks.slack.com/services/... の形式が必要です")
        sys.exit(1)

    print(f"📡 Slackにレポートを送信します...")
    print(f"   ワークスペース: leasecorp")
    print(f"   改善項目数: {REPORT['summary']['total_items']}件")
    print(f"   高優先: {REPORT['summary']['high_priority']}件")
    print()

    if send_report_to_slack(webhook_url):
        print("\n🎉 レポート送信完了！Slackを確認してください。")
    else:
        print("\n⚠️ 一部の送信に失敗しました。")


if __name__ == "__main__":
    main()
