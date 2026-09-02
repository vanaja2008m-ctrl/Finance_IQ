# 💰 FinanceIQ — Personal Finance Assistant

> **100% Free · No API Keys · No Paid Services · Open Source**

An end-to-end AI + ML Personal Finance platform built with Python and Streamlit.
Analyzes your spending, predicts future expenses, and delivers personalized budget advice — all for free.

---

## 🚀 Live Demo

Deploy in 3 minutes on **Streamlit Community Cloud** (free forever) — instructions below.

---

## ✨ Feature Overview

| Module | What It Does |
|---|---|
| 📊 **Expense Analytics (EDA)** | Category breakdown, monthly trends, recurring vs one-time, weekday heatmap |
| 👥 **Financial Segmentation** | K-Means clustering of monthly behavior into Excellent/Good/Average/Overspender |
| 📈 **Spending Trends** | Subcategory sunburst, category heatmap, top transactions, recurring payments |
| 🤖 **ML Predictions** | Random Forest / Gradient Boosting predicts next month's total spending |
| 🧠 **AI Budget Advisor** | Rule-based expert AI: budget alerts, savings strategies, financial health score |
| 💡 **Insights & Alerts** | Real-time alerts, savings progress bars, best/worst months, overspend potential |

---

## 🛠️ Tech Stack (100% Free & Open Source)

| Tool | Purpose |
|---|---|
| **Python 3.10+** | Core language |
| **Streamlit** | Interactive web dashboard |
| **Pandas / NumPy** | Data processing |
| **Scikit-Learn** | ML models + K-Means clustering |
| **Plotly** | Interactive charts |
| **SciPy** | Statistical computations |
| **Rule-Based AI** | Budget advisor (no API needed) |

**Zero paid dependencies. Works on free-tier Streamlit Cloud.**

---

## 📦 Local Setup

```bash
# 1. Clone / unzip the project
cd finance_assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py

# 4. Open http://localhost:8501
```

---

## ☁️ Deploy to Streamlit Community Cloud (Free Public URL)

**Step 1** — Push to GitHub:
```bash
git init
git add .
git commit -m "FinanceIQ initial commit"
git remote add origin https://github.com/YOUR_USERNAME/financeiq.git
git push -u origin main
```

**Step 2** — Deploy:
1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repo → Branch: `main` → Main file: `app.py`
5. Click **"Deploy!"**

Your public URL:
```
https://YOUR_USERNAME-financeiq-app-XXXX.streamlit.app
```

No credit card. No API keys. Completely free.

---

## 📂 Project Structure

```
finance_assistant/
├── app.py                     # Main Streamlit application (7 pages)
├── requirements.txt           # Python dependencies
├── README.md
├── sample_finance_data.csv    # Ready-to-upload sample dataset (821 rows)
├── .streamlit/
│   └── config.toml            # Dark teal theme
└── utils/
    ├── __init__.py
    ├── data_generator.py      # Synthetic finance data generator
    ├── analytics.py           # EDA, RFM-style analysis, K-Means segmentation
    ├── ml_model.py            # Feature engineering + spending prediction model
    └── advisor.py             # Rule-based AI budget advisor (no API)
```

---

## 📋 CSV Upload Format

Upload your own data with these columns:

| Column | Type | Example |
|---|---|---|
| `date` | date | 2024-03-15 |
| `category` | string | Food |
| `subcategory` | string | Groceries |
| `amount` | float | 87.50 |
| `type` | string | expense or income |
| `is_recurring` | boolean | True / False |
| `description` | string | Weekly groceries |

---

## 🧠 How the AI Advisor Works (No API)

The Budget Advisor is a **rule-based expert system** — no ChatGPT, no Gemini, no paid APIs:

1. **Category ratio analysis** — compares your spending % to the 50/30/20 budgeting rule ideal ranges
2. **Spending velocity alerts** — detects month-over-month spikes/drops > 15%
3. **Savings health scoring** — 0–100 composite score from savings rate, trend, and alerts
4. **Personalized strategy generation** — 5 tailored strategies based on your actual numbers
5. **Prediction-aware advice** — integrates ML forecast to warn about upcoming overspend

---

## 📄 License

MIT — free to use, modify, and deploy commercially.
