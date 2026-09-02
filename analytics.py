import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def compute_expense_eda(df_tx: pd.DataFrame, df_monthly: pd.DataFrame) -> dict:
    """Full EDA summary of personal finance data."""
    expenses = df_tx[df_tx["type"] == "expense"].copy()
    income = df_tx[df_tx["type"] == "income"].copy()

    total_income = income["amount"].sum()
    total_expenses = expenses["amount"].sum()
    total_savings = total_income - total_expenses
    avg_tx = expenses["amount"].mean()
    n_months = df_monthly.shape[0]

    # Category breakdown
    cat_summary = (
        expenses.groupby("category")["amount"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "Total", "count": "Transactions", "mean": "Avg"})
        .sort_values("Total", ascending=False)
        .reset_index()
    )
    cat_summary["Pct"] = (cat_summary["Total"] / total_expenses * 100).round(1)

    # Monthly trend
    monthly_trend = df_monthly.copy()
    monthly_trend["month_str"] = monthly_trend["month"].dt.strftime("%b %Y")

    # Recurring vs one-time
    rec = expenses.groupby("is_recurring")["amount"].sum()
    recurring_total = rec.get(True, 0)
    onetime_total = rec.get(False, 0)

    # Weekday spending
    expenses["weekday"] = pd.to_datetime(expenses["date"]).dt.day_name()
    weekday_spend = expenses.groupby("weekday")["amount"].sum().reindex(
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    ).fillna(0).reset_index()
    weekday_spend.columns = ["Weekday", "Total"]

    # Subcategory deep dive
    subcat_summary = (
        expenses.groupby(["category","subcategory"])["amount"]
        .sum().reset_index()
        .sort_values("amount", ascending=False)
    )

    # Top 5 spending months
    top_months = df_monthly.nlargest(5, "expenses")[["month","expenses","income","savings"]]

    return {
        "total_income": round(total_income, 2),
        "total_expenses": round(total_expenses, 2),
        "total_savings": round(total_savings, 2),
        "overall_savings_rate": round(total_savings / total_income * 100, 1) if total_income > 0 else 0,
        "avg_monthly_income": round(df_monthly["income"].mean(), 2),
        "avg_monthly_expenses": round(df_monthly["expenses"].mean(), 2),
        "avg_monthly_savings": round(df_monthly["savings"].mean(), 2),
        "avg_transaction": round(avg_tx, 2),
        "n_months": n_months,
        "cat_summary": cat_summary,
        "monthly_trend": monthly_trend,
        "recurring_total": round(recurring_total, 2),
        "onetime_total": round(onetime_total, 2),
        "weekday_spend": weekday_spend,
        "subcat_summary": subcat_summary,
        "top_months": top_months,
    }


def kmeans_financial_segmentation(df_monthly: pd.DataFrame) -> pd.DataFrame:
    """K-Means clustering on monthly financial behavior."""
    features = df_monthly[["income", "expenses", "savings", "savings_rate", "n_transactions"]].copy()
    features = features.fillna(0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    best_k, best_sil = 3, -1
    for k in range(2, min(6, len(df_monthly))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        if len(set(labels)) > 1:
            sc = silhouette_score(scaled, labels)
            if sc > best_sil:
                best_sil, best_k = sc, k

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_monthly = df_monthly.copy()
    df_monthly["cluster"] = km_final.fit_predict(scaled)

    # Label clusters by avg savings rate
    means = df_monthly.groupby("cluster")["savings_rate"].mean().sort_values(ascending=False)
    labels_map = {cid: lbl for cid, lbl in zip(
        means.index,
        ["Excellent Saver 🟢", "Good Saver 🔵", "Average Spender 🟡",
         "Overspender 🔴", "Critical 🚨"][:best_k]
    )}
    df_monthly["segment"] = df_monthly["cluster"].map(labels_map)
    return df_monthly, best_k, round(best_sil, 3)


def spending_pattern_analysis(df_tx: pd.DataFrame) -> dict:
    """Analyze spending patterns: trends, velocity, anomalies."""
    expenses = df_tx[df_tx["type"] == "expense"].copy()
    expenses["date"] = pd.to_datetime(expenses["date"])
    expenses["month"] = expenses["date"].dt.to_period("M").astype(str)
    expenses["week"] = expenses["date"].dt.isocalendar().week.astype(int)

    # Month-over-month category change
    cat_monthly = expenses.groupby(["month","category"])["amount"].sum().unstack(fill_value=0)

    # Largest single transactions
    top_tx = expenses.nlargest(10, "amount")[["date","category","subcategory","amount","description"]]

    # Recurring payment total per month
    recurring = expenses[expenses["is_recurring"]].groupby("month")["amount"].sum().reset_index()
    recurring.columns = ["month", "recurring_amount"]

    return {
        "cat_monthly": cat_monthly,
        "top_transactions": top_tx,
        "recurring_monthly": recurring,
    }
