"""
AI Budget Advisor — 100% free, no API keys required.
Uses rule-based expert system + statistical analysis to generate
personalized financial recommendations.
"""
import pandas as pd
import numpy as np
from datetime import datetime


# ── Budget benchmarks (50/30/20 rule variants) ──────────────────────────────
IDEAL_RATIOS = {
    "Housing":       (0.25, 0.30),
    "Food":          (0.10, 0.15),
    "Transport":     (0.10, 0.15),
    "Healthcare":    (0.05, 0.10),
    "Shopping":      (0.05, 0.10),
    "Entertainment": (0.03, 0.07),
    "Education":     (0.03, 0.07),
    "Savings":       (0.15, 0.25),   # target range
    "Travel":        (0.02, 0.06),
    "Miscellaneous": (0.02, 0.05),
}

SAVINGS_BENCHMARKS = {
    "Emergency Fund Months": 6,   # ideal months of expenses in emergency fund
    "Min Savings Rate": 0.20,
    "Good Savings Rate": 0.30,
}


def _savings_health(savings_rate: float) -> tuple:
    if savings_rate >= 30:
        return "Excellent", "🟢", f"Your {savings_rate:.1f}% savings rate is outstanding — you're building wealth fast."
    elif savings_rate >= 20:
        return "Good", "🔵", f"Your {savings_rate:.1f}% savings rate is healthy and above the recommended 20%."
    elif savings_rate >= 10:
        return "Fair", "🟡", f"Your {savings_rate:.1f}% savings rate is below ideal. Aim for at least 20%."
    elif savings_rate >= 0:
        return "Poor", "🔴", f"Your {savings_rate:.1f}% savings rate is critically low. Focus on cutting expenses immediately."
    else:
        return "Critical", "🚨", f"You are spending MORE than you earn ({savings_rate:.1f}%). Immediate action required."


def _category_alerts(cat_summary: pd.DataFrame, avg_monthly_income: float) -> list:
    alerts = []
    for _, row in cat_summary.iterrows():
        cat = row["category"]
        if cat not in IDEAL_RATIOS:
            continue
        pct_actual = row["Pct"] / 100
        lo, hi = IDEAL_RATIOS[cat]
        monthly_cat = row["Total"] / max(1, row.get("n_months", 1))

        if pct_actual > hi:
            over_pct = (pct_actual - hi) * 100
            save_potential = (pct_actual - hi) * avg_monthly_income * 12
            alerts.append({
                "level": "warning",
                "category": cat,
                "message": (
                    f"**{cat}** is {pct_actual*100:.1f}% of spending — "
                    f"{over_pct:.1f}% above the {hi*100:.0f}% ideal cap. "
                    f"Reducing to target could save **${save_potential:,.0f}/year**."
                ),
                "action": _get_reduction_tip(cat),
                "save_potential": save_potential,
            })
        elif pct_actual < lo and cat == "Savings":
            alerts.append({
                "level": "critical",
                "category": cat,
                "message": (
                    f"**Savings** allocation is only {pct_actual*100:.1f}% — "
                    f"well below the recommended {lo*100:.0f}% minimum."
                ),
                "action": "Set up an automatic transfer to savings on payday before spending.",
                "save_potential": 0,
            })
    return sorted(alerts, key=lambda x: -x.get("save_potential", 0))


def _get_reduction_tip(category: str) -> str:
    tips = {
        "Housing": "Consider roommates, negotiate rent renewal, or refinance mortgage at lower rates.",
        "Food": "Meal prep weekly, use grocery lists, limit restaurant visits to 2x/week.",
        "Transport": "Combine trips, use public transit 2 days/week, or carpool to save on fuel.",
        "Shopping": "Apply a 48-hour waiting rule before non-essential purchases. Unsubscribe from retail emails.",
        "Entertainment": "Audit streaming subscriptions — cancel unused ones. Use free library resources.",
        "Healthcare": "Use generic medications, compare pharmacy prices, take advantage of free preventive care.",
        "Travel": "Book 6+ weeks in advance, use points, travel during off-peak seasons.",
        "Education": "Explore free alternatives: Coursera audits, YouTube tutorials, local library e-learning.",
        "Miscellaneous": "Track every 'other' expense for 30 days — visibility alone reduces this category by 20%.",
    }
    return tips.get(category, "Review and categorize these expenses carefully to find reduction opportunities.")


def _savings_strategies(eda: dict) -> list:
    strategies = []
    avg_income = eda["avg_monthly_income"]
    avg_spend = eda["avg_monthly_expenses"]
    savings_rate = eda["overall_savings_rate"]
    recurring = eda["recurring_total"]

    # Strategy 1: Pay yourself first
    target_save = avg_income * 0.20
    if savings_rate < 20:
        gap = target_save - eda["avg_monthly_savings"]
        strategies.append({
            "title": "💰 Pay Yourself First",
            "desc": f"Automate ${target_save:,.0f}/month to savings on payday. You currently save ${eda['avg_monthly_savings']:,.0f}/month — close the ${gap:,.0f} gap.",
            "impact": "high",
            "annual_gain": round(gap * 12, 0),
        })

    # Strategy 2: 1% incremental increase
    strategies.append({
        "title": "📈 1% Monthly Increase Rule",
        "desc": f"Increase savings by just 1% each month. In 12 months, you'll save an extra ${avg_income*0.12:,.0f}/year automatically.",
        "impact": "medium",
        "annual_gain": round(avg_income * 0.12, 0),
    })

    # Strategy 3: Recurring audit
    strategies.append({
        "title": "🔄 Subscription Audit",
        "desc": f"You spend ${recurring:,.0f} total on recurring payments. Review every subscription — canceling just 2-3 unused services can free $50-150/month.",
        "impact": "medium",
        "annual_gain": 900,
    })

    # Strategy 4: 30-day rule
    strategies.append({
        "title": "⏳ The 30-Day Purchase Rule",
        "desc": "For non-essential purchases over $50, wait 30 days. Studies show 70% of impulse desires vanish — saving an estimated $200-400/month.",
        "impact": "medium",
        "annual_gain": 3000,
    })

    # Strategy 5: Emergency fund
    emergency_target = avg_spend * 6
    strategies.append({
        "title": "🛡️ Build Emergency Fund",
        "desc": f"Target: ${emergency_target:,.0f} (6 months of expenses at ${avg_spend:,.0f}/month). Start with ${avg_spend*0.05:,.0f}/month dedicated contributions.",
        "impact": "high",
        "annual_gain": 0,
    })

    return strategies


def _budget_plan(eda: dict) -> pd.DataFrame:
    """Generate a recommended monthly budget using 50/30/20 principle."""
    income = eda["avg_monthly_income"]
    rows = []
    for cat, (lo, hi) in IDEAL_RATIOS.items():
        ideal_pct = (lo + hi) / 2
        rows.append({
            "Category": cat,
            "Ideal %": f"{ideal_pct*100:.0f}%",
            "Recommended Budget": round(income * ideal_pct, 2),
            "Range": f"${income*lo:,.0f} – ${income*hi:,.0f}",
        })
    return pd.DataFrame(rows)


def _spending_alerts(df_monthly: pd.DataFrame) -> list:
    alerts = []
    if len(df_monthly) < 2:
        return alerts

    last = df_monthly.iloc[-1]
    prev = df_monthly.iloc[-2]

    spend_change = (last["expenses"] - prev["expenses"]) / prev["expenses"] * 100
    if spend_change > 15:
        alerts.append({
            "type": "spike",
            "icon": "📈",
            "msg": f"Spending jumped **{spend_change:.1f}%** vs last month (${last['expenses']:,.0f} vs ${prev['expenses']:,.0f}). Investigate what drove this increase.",
        })
    elif spend_change < -15:
        alerts.append({
            "type": "drop",
            "icon": "📉",
            "msg": f"Spending dropped **{abs(spend_change):.1f}%** vs last month — great discipline! Redirect the savings to your emergency fund.",
        })

    if last["savings_rate"] < 0:
        alerts.append({
            "type": "critical",
            "icon": "🚨",
            "msg": f"You spent **${abs(last['savings']):,.0f} MORE than you earned** last month. Immediate budget review required.",
        })

    if last["savings_rate"] > 0 and last["savings_rate"] < 5:
        alerts.append({
            "type": "warning",
            "icon": "⚠️",
            "msg": f"Savings rate last month was only **{last['savings_rate']:.1f}%**. One unexpected expense could put you in deficit.",
        })

    return alerts


def generate_budget_advice(eda: dict, df_monthly: pd.DataFrame,
                            prediction: dict = None) -> dict:
    """
    Main advisor function — returns complete structured advice.
    100% free: rule-based expert system, no API calls.
    """
    savings_rate = eda["overall_savings_rate"]
    health_label, health_icon, health_msg = _savings_health(savings_rate)

    # Category alerts — add n_months to cat_summary
    cat_df = eda["cat_summary"].copy()
    cat_df["n_months"] = eda["n_months"]
    alerts = _category_alerts(cat_df, eda["avg_monthly_income"])

    strategies = _savings_strategies(eda)
    budget_plan = _budget_plan(eda)
    spending_alerts = _spending_alerts(df_monthly)

    # Prediction-based advice
    pred_advice = []
    if prediction:
        pred = prediction["predicted_spend"]
        income = eda["avg_monthly_income"]
        pred_savings = income - pred
        pred_save_rate = pred_savings / income * 100 if income > 0 else 0

        if prediction["pct_change"] > 10:
            pred_advice.append(f"⚠️ Next month's spending is predicted to **rise {prediction['pct_change']:.1f}%** to **${pred:,.0f}**. Plan for this now by cutting discretionary spend this week.")
        elif prediction["pct_change"] < -10:
            pred_advice.append(f"✅ Next month's spending is predicted to **drop {abs(prediction['pct_change']):.1f}%** to **${pred:,.0f}** — ahead of plan!")
        else:
            pred_advice.append(f"📊 Predicted next-month spend: **${pred:,.0f}** (±{prediction['pct_change']:+.1f}% vs last month). On track.")

        if pred_save_rate < 10:
            pred_advice.append(f"🔴 At this spending rate, predicted savings next month are only **${pred_savings:,.0f}** ({pred_save_rate:.1f}%). Consider reducing by ${pred - income*0.80:,.0f}.")
        elif pred_save_rate >= 20:
            pred_advice.append(f"🟢 At predicted spending, you'll save **${pred_savings:,.0f}** ({pred_save_rate:.1f}%) next month — excellent!")

    # Financial health score (0–100)
    score = 0
    score += min(40, savings_rate * 1.5)           # savings rate: max 40pts
    if len(alerts) == 0: score += 20               # no category overruns
    elif len(alerts) <= 2: score += 10
    score += min(20, (savings_rate / 30) * 20)     # relative to 30% goal
    if df_monthly["savings_rate"].iloc[-3:].mean() > 15: score += 10  # recent trend
    if len(spending_alerts) == 0: score += 10
    score = min(100, max(0, round(score)))

    score_label = (
        "A+ Financial Health 🏆" if score >= 85 else
        "B+ Good Standing 🟢" if score >= 70 else
        "C Average 🟡" if score >= 50 else
        "D Needs Improvement 🔴" if score >= 30 else
        "F Critical Action Needed 🚨"
    )

    return {
        "health_label": health_label,
        "health_icon": health_icon,
        "health_msg": health_msg,
        "financial_score": score,
        "score_label": score_label,
        "category_alerts": alerts,
        "strategies": strategies,
        "budget_plan": budget_plan,
        "spending_alerts": spending_alerts,
        "prediction_advice": pred_advice,
        "summary_bullets": [
            f"📅 Analyzed **{eda['n_months']} months** of financial data",
            f"💵 Avg monthly income: **${eda['avg_monthly_income']:,.0f}**",
            f"💳 Avg monthly expenses: **${eda['avg_monthly_expenses']:,.0f}**",
            f"💰 Avg monthly savings: **${eda['avg_monthly_savings']:,.0f}** ({eda['overall_savings_rate']:.1f}%)",
            f"🔄 Total recurring payments: **${eda['recurring_total']:,.0f}**",
            f"🏆 Top expense: **{eda['cat_summary'].iloc[0]['category']}** (${eda['cat_summary'].iloc[0]['Total']:,.0f} total)",
        ],
    }
