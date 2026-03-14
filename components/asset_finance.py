"""
アセット・ファイナンス型 物件審査エンジン。
（lease_logic_sumaho12.py から切り出し）
"""


class AssetFinanceEngine:
    """
    ベイジアンネットワーク思想に基づく「アセット・ファイナンス型」リース審査エンジン。
    物件の担保価値（動的LGD/BEP）と財務スコアを統合し、
    銀行審査を超えた判定・逆転承認・ライフサイクル提案を行う。
    """

    ASSET_PARAMS = {
        '建機':    {'r': 0.15, 'priority': '中',   'priority_score': 2, 'info': '海外需要による底値の硬さ'},
        '工作機械': {'r': 0.20, 'priority': '高',   'priority_score': 3, 'info': '精度維持による長寿命価値'},
        'PC/IT':   {'r': 0.40, 'priority': '低',   'priority_score': 1, 'info': '高速な陳腐化・資産価値なし'},
        '医療機器': {'r': 0.10, 'priority': '極高', 'priority_score': 5, 'info': '診療報酬直結（最優先支払）'},
        'ドローン': {'r': 0.50, 'priority': '中',   'priority_score': 2, 'info': '物理破損と超高速陳腐化'},
        '車両':    {'r': 0.25, 'priority': '中高', 'priority_score': 3, 'info': '高換金性と確立された中古市場'},
    }

    def get_effective_depreciation_rate(self, asset_type, annual_km=0, has_maintenance_lease=False):
        """実効減価率（走行距離補正・メンテリース補正を加えた実効値）"""
        r = self.ASSET_PARAMS[asset_type]['r']
        if asset_type == '車両':
            if annual_km >= 20000:
                r += 0.10  # 過走行補正（年2万km以上）
            if has_maintenance_lease:
                r -= 0.05  # メンテリースによる価値維持補正
        return max(r, 0.0)

    def get_maintenance_lgd_bonus(self, asset_type, has_maintenance_lease):
        """メンテナンスリース受託時のLGD（中古売却価値）加算率"""
        if asset_type == '車両' and has_maintenance_lease:
            return 0.075  # 5〜10%の中間値：管理品質による与信枠拡大
        return 0.0

    def calculate_bep(self, asset_type, term_months, down_payment_rate,
                      annual_km=0, has_maintenance_lease=False):
        """
        損益分岐点（BEP）算出。
        V(t) = (1 + maint_bonus) × (1 - r)^(t/12)
        L(t) = (1 - down_payment) × (1 - t/term)
        V(t) > L(t) となる最初の月をBEPとする。
        """
        r = self.get_effective_depreciation_rate(asset_type, annual_km, has_maintenance_lease)
        maint_bonus = self.get_maintenance_lgd_bonus(asset_type, has_maintenance_lease)

        bep_month = term_months
        v_curve = []
        l_curve = []

        for m in range(0, term_months + 1):
            v = (1.0 + maint_bonus) * ((1 - r) ** (m / 12))
            l = (1 - down_payment_rate) * (1 - m / term_months) if m < term_months else 0.0
            v_curve.append(round(v, 4))
            l_curve.append(round(l, 4))
            if m > 0 and v > l and bep_month == term_months:
                bep_month = m

        return bep_month, v_curve, l_curve

    def calculate_score(self, data):
        """
        総合承認スコアの算出。
        定量（財務）最大40点 ＋ 物件LGD/BEP緩和最大50点 ＋ 逆転定性因子最大105点
        """
        asset_type = data['asset_type']
        term = data['term']
        down_payment = data['down_payment']
        financial_score = data['financial_score']
        annual_km = data.get('annual_km', 0)
        has_maintenance_lease = data.get('has_maintenance_lease', False)

        bep_month, v_curve, l_curve = self.calculate_bep(
            asset_type, term, down_payment, annual_km, has_maintenance_lease
        )
        bep_ratio = bep_month / term if term > 0 else 1.0
        priority_score = self.ASSET_PARAMS[asset_type]['priority_score']

        score = 0
        reasons = []
        deductions = []

        # --- 1. 財務スコア（銀行視点） ---
        fin_map = {'High': 40, 'Medium': 20, 'Low': -20}
        fin_pts = fin_map.get(financial_score, 0)
        score += fin_pts
        fin_labels = {'High': '優良', 'Medium': '標準', 'Low': '低評価（赤字・債務超過）'}
        fin_label = fin_labels.get(financial_score, financial_score)
        if fin_pts >= 0:
            reasons.append(f"財務{fin_label}（+{fin_pts}点）")
        else:
            deductions.append(f"財務{fin_label}（{fin_pts}点）")

        # --- 2. 物件・LGD/BEP 緩和（リース独自視点） ---
        if bep_ratio < 0.3:
            pts = 35
            reasons.append(f"BEP早期達成（{bep_month}ヶ月目 / {term}ヶ月）→ 物件保全性が極めて高い（+{pts}点）")
        elif bep_ratio < 0.5:
            pts = 20
            reasons.append(f"BEP前半達成（{bep_month}ヶ月目）→ 物件保全性が高い（+{pts}点）")
        elif bep_ratio < 0.7:
            pts = 10
            reasons.append(f"BEP中盤（{bep_month}ヶ月目）→ 物件保全性は標準（+{pts}点）")
        else:
            pts = 0
            deductions.append(f"BEP後半（{bep_month}ヶ月目 / {term}ヶ月）→ 物件価値がリース期間全体でリスク（0点）")
        score += pts

        # 支払優先度ボーナス
        if priority_score >= 5:
            score += 15
            reasons.append("支払優先度「極高」（診療報酬直結）→ +15点")
        elif priority_score >= 3:
            score += 8
            reasons.append(f"支払優先度「{self.ASSET_PARAMS[asset_type]['priority']}」→ +8点")

        # メンテナンスリース補正（車両）
        if asset_type == '車両' and has_maintenance_lease:
            score += 10
            reasons.append("メンテナンスリース受託 → 管理品質による中古価値向上（+10点）")

        # --- 3. 逆転因子（定性） ---
        if data.get('main_bank_support'):
            score += 50
            reasons.append("メイン銀行の支援先（最強の緩和因子）→ +50点")
        if data.get('bank_coordination'):
            score += 20
            reasons.append("銀行協調案件 → +20点")
        if data.get('core_business'):
            score += 20
            reasons.append("本業利用物件（支払優先度UP）→ +20点")
        if data.get('related_assets'):
            score += 15
            reasons.append("関係者資産による保全 → +15点")

        # --- 4. 車両独自リスク（過走行） ---
        if asset_type == '車両' and annual_km >= 20000:
            score -= 10
            deductions.append(f"過走行リスク（年{annual_km:,}km）→ 実効減価率+10%補正、−10点")

        return score, bep_month, bep_ratio, v_curve, l_curve, reasons, deductions

    def get_decision(self, score):
        """スコアから判定区分を決定"""
        if score >= 80:
            return "承認", "✅"
        elif score >= 50:
            return "条件付き承認", "⚠️"
        elif score >= 30:
            return "要審議（上位承認）", "🔶"
        else:
            return "否決", "❌"

    def get_marketing_advice(self, asset_type, term_months):
        """ライフサイクル・マーケティング提案"""
        if asset_type == '車両':
            check_month = max(term_months - 6, 1)
            return (
                f"**車検タイミング戦略**: {check_month}ヶ月目（車検前6ヶ月）に"
                f"リプレイス提案を予約。3〜5年サイクルでの入れ替えニーズを先行受注。"
            )
        elif asset_type == '医療機器':
            return (
                "**長期伴走戦略**: 満了後の再リース移行率が高い物件です。"
                "7年以上を見据えた再リース・保守契約のセットプランを今から提示してください。"
            )
        elif asset_type == '建機':
            return (
                "**海外輸出戦略**: 4〜5年後の海外輸出相場（東南アジア・中東）に"
                "合わせた下取り提案が有効。中古ブローカーとの連携も検討。"
            )
        elif asset_type == '工作機械':
            return (
                "**精度保証戦略**: 満了前にオーバーホール費用の試算を提示し、"
                "リニューアルリースへの誘導を図ってください。"
            )
        elif asset_type == 'PC/IT':
            return (
                "**早期入れ替え戦略**: 陳腐化が速いため、2〜3年での早期入れ替えを前提に"
                "短期リース設計を推奨。バルク更新需要を狙う。"
            )
        elif asset_type == 'ドローン':
            return (
                "**物理リスク対策**: 損害保険とのセット提案が必須。"
                "法規制変化に合わせた入れ替えサイクル（2〜3年）を明示。"
            )
        return "満了の6ヶ月前に入れ替え需要を調査し、次案件の先行受注を狙ってください。"

    def get_bank_comparison(self, asset_type, financial_score):
        """銀行システムとの差異解説"""
        asset_info = self.ASSET_PARAMS[asset_type]['info']
        base = (
            f"銀行は「財務諸表の健全性」のみで判断しますが、"
            f"当エンジンは **{asset_type}**（{asset_info}）の物件価値・換金性を定量評価します。"
        )
        if financial_score == 'Low':
            base += (
                "\n\n財務が低評価でも、物件の保全性（BEP・残価）と定性緩和因子により、"
                "銀行が『否決』とする案件を承認圏に引き上げることができます。"
            )
        return base

    def get_action_plan(self, score, data, bep_month, bep_ratio):
        """営業アクションプラン"""
        plans = []
        asset_type = data['asset_type']
        down = data['down_payment']
        decision, _ = self.get_decision(score)

        if decision in ["否決", "要審議（上位承認）"]:
            needed_down = min(down + 0.10, 0.50)
            plans.append(
                f"自己資金を **{needed_down*100:.0f}%以上**（現在{down*100:.0f}%）に引き上げると"
                f"承認確率が大幅に上昇します。"
            )
            if not data.get('main_bank_support'):
                plans.append("**メイン銀行の推薦・協調案件**として申請することで、最大+50点の緩和が可能です。")
            if not data.get('core_business'):
                plans.append("**本業利用**であることを明示する書類（事業計画書等）を追加してください。（+20点）")
        if asset_type == '車両':
            plans.append(
                f"**{bep_month}ヶ月目**（BEP到達）以降は物件の換金価値がリース残債を上回ります。"
                "この点を保全力として稟議書に記載してください。"
            )
            if not data.get('has_maintenance_lease'):
                plans.append("**メンテナンスリース**を付帯させると中古売却価値が5〜10%向上します。（+10点）")
        if asset_type == '医療機器':
            plans.append("診療報酬との連動を稟議書に明記し、支払優先度の高さを強調してください。")
            plans.append(
                f"**{data['term'] + 12}ヶ月後**（満了後1年）の再リースプランを今から提示し、関係継続を確約。"
            )
        if bep_ratio > 0.7 and decision != "承認":
            plans.append("物件の頭金比率を上げるか、リース期間を短縮してBEPを前倒しにすることを検討してください。")
        if not plans:
            plans.append("現状の条件で承認可能です。満了に向けて次回リプレイス提案の準備を進めてください。")
        return plans

    def run_inference(self, data):
        """総合審査を実行し、結果辞書を返す"""
        score, bep_month, bep_ratio, v_curve, l_curve, reasons, deductions = self.calculate_score(data)
        decision, icon = self.get_decision(score)
        return {
            'score': score,
            'decision': decision,
            'icon': icon,
            'bep_month': bep_month,
            'bep_ratio': bep_ratio,
            'v_curve': v_curve,
            'l_curve': l_curve,
            'reasons': reasons,
            'deductions': deductions,
            'marketing_advice': self.get_marketing_advice(data['asset_type'], data['term']),
            'bank_comparison': self.get_bank_comparison(data['asset_type'], data['financial_score']),
            'action_plan': self.get_action_plan(score, data, bep_month, bep_ratio),
        }
