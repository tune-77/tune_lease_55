# 自動生成ファイル（improvement_report_data.py）
# generated: 2026-03-20 00:00

REPORT_DATA = {
    "generated_at": "2026-03-20 00:00",
    "stats": {
        "total_cases": 26,
        "this_month": 8,
        "this_week": 3,
        "avg_score": 70.0,
        "approval_rate": 57.7,
        "rejection_rate": 0.0,
    },
    "industry_ranking": [
        {"industry": "44 道路貨物運送業",       "count": 6, "avg_score": 82.5, "approval_rate": 83.3},
        {"industry": "09 食料品製造業",          "count": 5, "avg_score": 86.0, "approval_rate": 80.0},
        {"industry": "06 総合工事業",            "count": 4, "avg_score": 69.2, "approval_rate": 50.0},
        {"industry": "21 金属製品製造業",        "count": 4, "avg_score": 45.9, "approval_rate": 25.0},
        {"industry": "24 生産用機械器具製造業",  "count": 2, "avg_score": 67.3, "approval_rate": 50.0},
    ],
    "score_distribution": {"low": 10, "mid": 1, "high": 15},
    "recent_cases": [
        {"industry": "09 食料品製造業",         "score": 46.0, "hantei": "要審議",   "date": "2026-03-19"},
        {"industry": "83 医療業(病院・診療所)",  "score": 90.8, "hantei": "承認圏内", "date": "2026-03-16"},
        {"industry": "07 職別工事業(大工・とび等)", "score": 36.9, "hantei": "要審議", "date": "2026-03-16"},
        {"industry": "06 総合工事業",           "score": 47.3, "hantei": "要審議",   "date": "2026-03-14"},
        {"industry": "06 総合工事業",           "score": 96.7, "hantei": "承認圏内", "date": "2026-03-05"},
        {"industry": "83 医療業(病院・診療所)",  "score": 96.0, "hantei": "承認圏内", "date": "2026-03-03"},
        {"industry": "21 金属製品製造業",       "score": 37.0, "hantei": "要審議",   "date": "2026-03-03"},
        {"industry": "09 食料品製造業",         "score": 97.3, "hantei": "承認圏内", "date": "2026-03-02"},
        {"industry": "56-61 各種小売業",        "score": 69.9, "hantei": "要審議",   "date": "2026-02-27"},
        {"industry": "21 金属製品製造業",       "score": 40.6, "hantei": "要審議",   "date": "2026-02-27"},
    ],
    "improvement_proposals": [
        {
            "title": "金属製品製造業の審査基準見直し",
            "detail": (
                "金属製品製造業（21）は4件中3件が要審議、平均スコア45.9と最も低い。"
                "業種特有のリスク要因（素材価格変動・受注集中）を審査係数に反映し、"
                "スコア閾値の見直しまたは業種別条件付承認フローの整備を検討する。"
            ),
            "priority": "high",
            "category": "審査精度",
        },
        {
            "title": "「要審議」判定の後処理フロー整備",
            "detail": (
                "全26件中11件（42%）が「要審議」で終結し、承認・否決いずれにも分類されていない。"
                "要審議案件の最終ステータス（成約・失注・保留）を必ず記録する運用ルールを設け、"
                "実質的な否決率・成約率を正確に把握できる体制を構築する。"
            ),
            "priority": "high",
            "category": "運用",
        },
        {
            "title": "スコア低分散問題への対応（mid帯が極端に薄い）",
            "detail": (
                "スコア分布が low(0-49)=10件 / mid(50-69)=1件 / high(70-100)=15件 と二極化しており、"
                "中間帯がほぼ存在しない。スコアリングモデルの係数を調整し、"
                "グレーゾーン案件を適切に50〜69点帯に分布させることで判定精度を向上させる。"
            ),
            "priority": "high",
            "category": "審査精度",
        },
        {
            "title": "データ蓄積の加速（全体件数26件はまだ少ない）",
            "detail": (
                "総審査件数26件では統計的有意性が低く、業種別傾向の信頼性も限定的。"
                "Slack Bot経由の審査入力を促進し、過去案件のレトロフィット（遡及登録）も"
                "検討して最低100件以上のデータ基盤を整える。"
            ),
            "priority": "high",
            "category": "データ",
        },
        {
            "title": "医療業・食料品製造業を優良業種として積極対応",
            "detail": (
                "医療業（93.4点・承認率100%）と食料品製造業（86.0点・80%）は高スコア・高承認率。"
                "これら優良業種向けに提案資料テンプレートや審査簡略フローを用意し、"
                "フロントエンドの初期入力時に優良業種バッジを表示してリード獲得を促進する。"
            ),
            "priority": "medium",
            "category": "運用",
        },
        {
            "title": "総合工事業のスコア変動対策",
            "detail": (
                "総合工事業（06）は4件で47〜97点と変動幅が最大。"
                "受注残高・季節性など工事業特有の財務変動要因を入力項目に追加し、"
                "スコアのばらつきを抑える業種別補正係数の導入を検討する。"
            ),
            "priority": "medium",
            "category": "審査精度",
        },
        {
            "title": "週次ダッシュボードの自動更新・Slack配信",
            "detail": (
                "現在、週次サマリーはAgent 8（APScheduler）経由で配信可能だが、"
                "本ファイル（improvement_report_data.py）の自動再生成を週1回Cronに組み込み、"
                "最新統計を常にUIとSlackレポートに反映できる仕組みを整える。"
            ),
            "priority": "medium",
            "category": "UI/UX",
        },
        {
            "title": "今月・今週の件数トレンド監視アラート",
            "detail": (
                "今週3件・今月8件と案件流入は安定しているが、"
                "週あたり件数が閾値（例：2件未満）を下回った場合にSlack通知を出す"
                "活動量モニタリング機能を追加し、営業活動の停滞を早期検知する。"
            ),
            "priority": "low",
            "category": "運用",
        },
    ],
}
