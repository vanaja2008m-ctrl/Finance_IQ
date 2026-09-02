import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


CATEGORIES = {
    "Housing":      {"weight": 0.28, "subcats": ["Rent", "Mortgage", "Utilities", "Maintenance"]},
    "Food":         {"weight": 0.14, "subcats": ["Groceries", "Restaurants", "Coffee", "Delivery"]},
    "Transport":    {"weight": 0.10, "subcats": ["Fuel", "Public Transit", "Uber/Lyft", "Parking", "Car Insurance"]},
    "Healthcare":   {"weight": 0.07, "subcats": ["Pharmacy", "Doctor", "Gym", "Insurance"]},
    "Shopping":     {"weight": 0.09, "subcats": ["Clothing", "Electronics", "Home Goods", "Personal Care"]},
    "Entertainment":{"weight": 0.06, "subcats": ["Streaming", "Movies", "Games", "Events"]},
    "Education":    {"weight": 0.04, "subcats": ["Courses", "Books", "Tuition", "Subscriptions"]},
    "Savings":      {"weight": 0.10, "subcats": ["Emergency Fund", "Investments", "Retirement"]},
    "Travel":       {"weight": 0.05, "subcats": ["Flights", "Hotels", "Activities"]},
    "Miscellaneous":{"weight": 0.07, "subcats": ["Gifts", "Charity", "Pet", "Other"]},
}

INCOME_PROFILES = {
    "Low":    {"income": (2000, 3500),  "save_rate": (0.02, 0.08)},
    "Mid":    {"income": (3500, 6500),  "save_rate": (0.08, 0.18)},
    "High":   {"income": (6500, 12000), "save_rate": (0.18, 0.35)},
    "Wealthy":{"income": (12000, 25000),"save_rate": (0.30, 0.55)},
}


def generate_finance_data(n_months: int = 18, seed: int = 42) -> tuple:
    np.random.seed(seed)
    random.seed(seed)

    profile_name = np.random.choice(list(INCOME_PROFILES.keys()), p=[0.25, 0.40, 0.25, 0.10])
    profile = INCOME_PROFILES[profile_name]
    monthly_income = np.random.uniform(*profile["income"])
    save_rate = np.random.uniform(*profile["save_rate"])
    monthly_budget = monthly_income * (1 - save_rate)

    transactions = []
    monthly_summary = []
    end_date = datetime(2024, 12, 31)
    start_date = end_date - timedelta(days=30 * n_months)

    cat_names = list(CATEGORIES.keys())
    cat_weights = [CATEGORIES[c]["weight"] for c in cat_names]

    for m in range(n_months):
        month_start = start_date + timedelta(days=30 * m)
        month_income = monthly_income * np.random.uniform(0.92, 1.08)  # slight variation
        month_spend = 0
        n_tx = np.random.randint(20, 55)

        for _ in range(n_tx):
            cat = np.random.choice(cat_names, p=cat_weights)
            subcat = random.choice(CATEGORIES[cat]["subcats"])
            # Seasonal & category-based amount
            base_amt = (monthly_budget * CATEGORIES[cat]["weight"]) / np.random.randint(2, 6)
            amount = round(abs(np.random.normal(base_amt, base_amt * 0.3)), 2)
            amount = max(1.0, amount)
            tx_date = month_start + timedelta(days=np.random.randint(0, 28))
            is_recurring = subcat in ["Rent", "Mortgage", "Streaming", "Gym", "Car Insurance",
                                       "Tuition", "Emergency Fund", "Investments"]
            transactions.append({
                "date": tx_date.strftime("%Y-%m-%d"),
                "category": cat,
                "subcategory": subcat,
                "amount": amount,
                "type": "expense",
                "is_recurring": is_recurring,
                "description": f"{subcat} - {tx_date.strftime('%b %Y')}",
            })
            month_spend += amount

        # Income entry
        transactions.append({
            "date": (month_start + timedelta(days=1)).strftime("%Y-%m-%d"),
            "category": "Income",
            "subcategory": "Salary",
            "amount": round(month_income, 2),
            "type": "income",
            "is_recurring": True,
            "description": f"Monthly Salary - {month_start.strftime('%b %Y')}",
        })

        month_key = month_start.strftime("%Y-%m")
        monthly_summary.append({
            "month": month_key,
            "income": round(month_income, 2),
            "expenses": round(month_spend, 2),
            "savings": round(month_income - month_spend, 2),
            "savings_rate": round((month_income - month_spend) / month_income * 100, 1),
            "n_transactions": n_tx,
        })

    df_tx = pd.DataFrame(transactions)
    df_tx["date"] = pd.to_datetime(df_tx["date"])
    df_tx = df_tx.sort_values("date").reset_index(drop=True)

    df_monthly = pd.DataFrame(monthly_summary)
    df_monthly["month"] = pd.to_datetime(df_monthly["month"])

    meta = {
        "profile": profile_name,
        "avg_monthly_income": round(monthly_income, 2),
        "target_save_rate": round(save_rate * 100, 1),
    }
    return df_tx, df_monthly, meta
