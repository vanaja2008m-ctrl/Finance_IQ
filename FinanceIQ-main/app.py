import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

from utils.data_generator import generate_finance_data
from utils.analytics import compute_expense_eda, kmeans_financial_segmentation, spending_pattern_analysis
from utils.ml_model import engineer_features, train_spending_model, predict_next_month
from utils.advisor import generate_budget_advice

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinanceIQ — Personal Finance Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0A0F1E 0%, #0D1B2A 50%, #0A1628 100%);
    padding: 28px 32px; border-radius: 16px; margin-bottom: 24px;
    border: 1px solid #1E3A5F;
}
.main-title {
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(90deg, #00D4AA, #0099FF, #7B61FF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.2;
}
.main-sub { color: #94A3B8; font-size: 0.95rem; margin-top: 6px; }

/* Styled native st.metric cards - replaces custom kpi-card */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #111827 0%, #1a2035 100%);
    border: 1px solid #1E3A5F; border-radius: 14px;
    padding: 18px 20px !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    font-size: 0.72rem !important; color: #64748B !important;
    text-transform: uppercase; letter-spacing: 0.06em; font-weight: 500;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.8rem !important; font-weight: 700 !important; color: #00D4AA;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.82rem !important;
}

.section-title {
    font-size: 1.15rem; font-weight: 600; color: #E2E8F0;
    padding: 10px 0 10px 14px;
    border-left: 3px solid #00D4AA;
    margin: 24px 0 16px 0;
}

.advice-card {
    background: #0D1B2A; border: 1px solid #1E3A5F;
    border-radius: 12px; padding: 16px 18px; margin: 8px 0;
}
.advice-card.warning { border-left: 4px solid #F59E0B; }
.advice-card.critical { border-left: 4px solid #EF4444; }
.advice-card.success  { border-left: 4px solid #10B981; }
.advice-card.info     { border-left: 4px solid #3B82F6; }

.strategy-card {
    background: linear-gradient(135deg, #0D1B2A, #111827);
    border: 1px solid #1E3A5F; border-radius: 12px;
    padding: 16px 20px; margin: 8px 0;
}
.strategy-title { font-size: 1rem; font-weight: 600; color: #E2E8F0; }
.strategy-desc  { font-size: 0.85rem; color: #94A3B8; margin-top: 6px; line-height: 1.5; }
.strategy-gain  { font-size: 0.8rem; color: #00D4AA; margin-top: 8px; font-weight: 500; }

.health-score {
    background: linear-gradient(135deg, #0D1B2A, #0A1628);
    border: 2px solid #00D4AA; border-radius: 16px;
    padding: 24px; text-align: center;
}
.score-number { font-size: 4rem; font-weight: 800; color: #00D4AA; line-height: 1; }
.score-label  { font-size: 1rem; color: #E2E8F0; margin-top: 8px; font-weight: 500; }

.alert-box {
    padding: 12px 16px; border-radius: 10px; margin: 6px 0;
    font-size: 0.88rem; color: #E2E8F0;
}
.alert-spike    { background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.35); }
.alert-warning  { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.35); }
.alert-positive { background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.35); }
.alert-info     { background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.35); }

.pred-box {
    background: linear-gradient(135deg, #0D1B2A, #111827);
    border: 2px solid #7B61FF; border-radius: 16px; padding: 24px; text-align: center;
}
.pred-amount { font-size: 3rem; font-weight: 800; color: #7B61FF; }
.pred-label  { font-size: 0.9rem; color: #94A3B8; }

.sidebar-logo { font-size: 1.5rem; font-weight: 700; color: #00D4AA; }

.stButton>button {
    background: linear-gradient(90deg, #00D4AA, #0099FF);
    color: #0A0F1E; border: none; border-radius: 8px;
    padding: 10px 24px; font-weight: 700; font-size: 0.9rem;
    transition: all 0.2s;
}
.stButton>button:hover { opacity: 0.9; transform: translateY(-1px); box-shadow: 0 4px 15px rgba(0,212,170,0.35); }

.free-badge {
    display: inline-block; background: rgba(16,185,129,0.15);
    border: 1px solid #10B981; color: #10B981;
    padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def _init():
    defs = {
        "df_tx": None, "df_monthly": None, "meta": None,
        "eda": None, "patterns": None, "df_seg": None,
        "feats": None, "model": None, "scaler": None,
        "feat_cols": None, "ml_metrics": None, "prediction": None,
        "advice": None, "data_ready": False, "model_ready": False,
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ── Shared Plotly theme ───────────────────────────────────────────────────────
PT = dict(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#CBD5E1", family="Inter"), margin=dict(l=10,r=10,t=40,b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)
COLORS = ["#00D4AA","#0099FF","#7B61FF","#F59E0B","#EF4444",
          "#10B981","#EC4899","#F97316","#06B6D4","#84CC16"]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-logo">💰 FinanceIQ</div>', unsafe_allow_html=True)
    
    st.markdown("---")

    st.markdown("### 📂 Data Source")
    src = st.radio("", ["Sample Data", "Upload CSV"], label_visibility="collapsed")

    if src == "Upload CSV":
        f = st.file_uploader("Transactions CSV", type=["csv"])
        if f:
            try:
                df_up = pd.read_csv(f)
                req = ["date","category","amount","type"]
                miss = [c for c in req if c not in df_up.columns]
                if miss:
                    st.error(f"Missing columns: {miss}")
                else:
                    df_up["date"] = pd.to_datetime(df_up["date"])
                    if "is_recurring" not in df_up.columns:
                        df_up["is_recurring"] = False
                    if "subcategory" not in df_up.columns:
                        df_up["subcategory"] = df_up["category"]
                    if "description" not in df_up.columns:
                        df_up["description"] = ""
                    # Build monthly from uploaded
                    exp = df_up[df_up["type"]=="expense"]
                    inc = df_up[df_up["type"]=="income"]
                    df_up["month_key"] = df_up["date"].dt.to_period("M").astype(str)
                    monthly = df_up.groupby("month_key").apply(lambda g: pd.Series({
                        "month": pd.to_datetime(g["month_key"].iloc[0]),
                        "income": g[g["type"]=="income"]["amount"].sum(),
                        "expenses": g[g["type"]=="expense"]["amount"].sum(),
                        "n_transactions": (g["type"]=="expense").sum(),
                    })).reset_index(drop=True)
                    monthly["savings"] = monthly["income"] - monthly["expenses"]
                    monthly["savings_rate"] = (monthly["savings"] / monthly["income"].replace(0,1) * 100).round(1)
                    st.session_state.df_tx = df_up
                    st.session_state.df_monthly = monthly
                    st.session_state.meta = {"profile":"Custom","avg_monthly_income": monthly["income"].mean(),"target_save_rate":20}
                    st.session_state.data_ready = True
                    st.success(f"✅ {len(df_up)} rows loaded")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        n_mo = st.slider("Months of history", 6, 36, 18)
        if st.button("🔄 Generate Sample Data"):
            with st.spinner("Generating financial data..."):
                df_tx, df_monthly, meta = generate_finance_data(n_months=n_mo)
                st.session_state.df_tx = df_tx
                st.session_state.df_monthly = df_monthly
                st.session_state.meta = meta
                st.session_state.data_ready = True
            st.success(f"✅ {n_mo} months generated!")

    if st.session_state.data_ready:
        st.markdown("---")
        st.markdown("### ⚙️ Model Settings")
        algo = st.selectbox("Algorithm", ["Random Forest", "Gradient Boosting"])

        if st.button("🚀 Run Full Analysis"):
            with st.spinner("Running analytics + ML + AI Advisor..."):
                try:
                    df_tx = st.session_state.df_tx
                    df_monthly = st.session_state.df_monthly

                    eda = compute_expense_eda(df_tx, df_monthly)
                    patterns = spending_pattern_analysis(df_tx)
                    df_seg, _, _ = kmeans_financial_segmentation(df_monthly)

                    feats = engineer_features(df_tx, df_monthly)
                    model, scaler, feat_cols, ml_metrics = train_spending_model(feats, algo)
                    pred = predict_next_month(model, scaler, feat_cols, feats)
                    advice = generate_budget_advice(eda, df_monthly, pred)

                    st.session_state.update({
                        "eda": eda, "patterns": patterns, "df_seg": df_seg,
                        "feats": feats, "model": model, "scaler": scaler,
                        "feat_cols": feat_cols, "ml_metrics": ml_metrics,
                        "prediction": pred, "advice": advice, "model_ready": True,
                    })
                    st.success(f"✅ Done! Score: {advice['financial_score']}/100")
                except Exception as e:
                    st.error(f"Error: {e}")
                    raise

    st.markdown("---")
    st.markdown("### 🧭 Pages")
    page = st.selectbox("Navigate to", [
        "🏠 Dashboard",
        "🔍 Expense Analytics",
        "👥 Financial Segments",
        "📈 Spending Trends",
        "🤖 ML Predictions",
        "🧠 AI Budget Advisor",
        "💡 Insights & Alerts",
    ])

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem;color:#475569;text-align:center">
    🔒 All data stays local<br>
    🆓 No paid APIs used<br>
    ⚡ Powered by scikit-learn
    </div>
    """, unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
# kpi() removed — using native st.metric() for reliability across Streamlit versions


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def not_ready(msg="👈 Generate sample data or upload CSV, then click **Run Full Analysis**."):
    st.info(msg)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="main-header">
        <div class="main-title">💰 FinanceIQ — Personal Finance Assistant</div>
        <div class="main-sub">
            Expense Analytics · ML Spending Prediction · AI Budget Advisor &nbsp;|&nbsp;
            <span class="free-badge">100% Free & Open Source</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.model_ready:
        c1, c2, c3 = st.columns(3)
        for col, icon, title, desc in [
            (c1, "📊", "Expense Analytics", "EDA, category breakdown, spending patterns & K-Means clustering"),
            (c2, "🤖", "ML Predictions", "Random Forest / GBM predicts your next month's spending"),
            (c3, "🧠", "AI Budget Advisor", "Free rule-based AI gives personalized financial strategies"),
        ]:
            col.markdown(f"""<div class="advice-card info" style="text-align:center;padding:28px">
                <div style="font-size:2.5rem">{icon}</div>
                <div style="font-size:1.05rem;font-weight:600;color:#E2E8F0;margin-top:10px">{title}</div>
                <div style="font-size:0.82rem;color:#94A3B8;margin-top:8px">{desc}</div>
            </div>""", unsafe_allow_html=True)
        st.stop()

    eda = st.session_state.eda
    advice = st.session_state.advice
    pred = st.session_state.prediction

    # KPI row — native st.metric (renders reliably on all Streamlit versions)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("💵 Total Income",    f"${eda['total_income']:,.0f}")
    c2.metric("💳 Total Expenses",  f"${eda['total_expenses']:,.0f}")
    c3.metric("💰 Total Savings",   f"${eda['total_savings']:,.0f}")
    c4.metric("📊 Savings Rate",    f"{eda['overall_savings_rate']:.1f}%")
    c5.metric("🏆 Financial Score", f"{advice['financial_score']}/100")

    st.markdown("")
    col1, col2 = st.columns([2, 1])
    with col1:
        section("📈 Income vs Expenses Over Time")
        dm = st.session_state.df_seg
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dm["month"].astype(str), y=dm["income"],
                             name="Income", marker_color="#00D4AA", opacity=0.85))
        fig.add_trace(go.Bar(x=dm["month"].astype(str), y=dm["expenses"],
                             name="Expenses", marker_color="#EF4444", opacity=0.85))
        fig.add_trace(go.Scatter(x=dm["month"].astype(str), y=dm["savings"],
                                  name="Savings", line=dict(color="#7B61FF", width=2.5),
                                  mode="lines+markers"))
        fig.update_layout(**PT, barmode="group", title="Monthly Financial Overview", height=340)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("💳 Spending by Category")
        cat_df = eda["cat_summary"].head(8)
        fig2 = px.pie(cat_df, values="Total", names="category",
                      color_discrete_sequence=COLORS, hole=0.45)
        fig2.update_traces(textposition="inside", textinfo="percent+label",
                           textfont_size=11)
        fig2.update_layout(**PT, height=340, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Second row
    col3, col4 = st.columns([1, 1])
    with col3:
        section("🔮 Next Month Prediction")
        pct = pred["pct_change"]
        color = "#EF4444" if pct > 5 else "#10B981" if pct < -5 else "#F59E0B"
        st.markdown(f"""<div class="pred-box">
            <div class="pred-amount">${pred['predicted_spend']:,.0f}</div>
            <div class="pred-label">Predicted Next Month Spend</div>
            <div style="color:{color};font-size:1.1rem;margin-top:10px;font-weight:600">
                {'+' if pct>0 else ''}{pct:.1f}% vs last month
            </div>
            <div style="color:#475569;font-size:0.78rem;margin-top:6px">
                Range: ${pred['lower_bound']:,.0f} – ${pred['upper_bound']:,.0f}
            </div>
        </div>""", unsafe_allow_html=True)

    with col4:
        section("🏆 Financial Health Score")
        score = advice["financial_score"]
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 36, "color": "#00D4AA"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#475569"},
                "bar": {"color": "#00D4AA", "thickness": 0.25},
                "steps": [
                    {"range": [0, 30],  "color": "rgba(239,68,68,0.2)"},
                    {"range": [30, 60], "color": "rgba(245,158,11,0.2)"},
                    {"range": [60, 80], "color": "rgba(59,130,246,0.2)"},
                    {"range": [80,100], "color": "rgba(16,185,129,0.2)"},
                ],
                "threshold": {"line": {"color": "#00D4AA", "width": 3}, "value": score},
                "bgcolor": "rgba(0,0,0,0)",
            }
        ))
        fig3.update_layout(**PT, height=270)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown(f'<div style="text-align:center;font-size:0.9rem;color:#E2E8F0;font-weight:600">{advice["score_label"]}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPENSE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Expense Analytics":
    st.markdown("# 🔍 Expense Analytics")
    if not st.session_state.model_ready:
        not_ready()

    eda = st.session_state.eda

    # Summary KPIs
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("💵 Avg Monthly Income",   f"${eda['avg_monthly_income']:,.0f}")
    c2.metric("💳 Avg Monthly Expenses", f"${eda['avg_monthly_expenses']:,.0f}")
    c3.metric("💰 Avg Monthly Savings",  f"${eda['avg_monthly_savings']:,.0f}")
    c4.metric("🛒 Avg Transaction",      f"${eda['avg_transaction']:,.2f}")

    st.markdown("")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Category Analysis", "📅 Monthly Trends", "🔄 Recurring vs One-Time", "🗂️ Raw Data"])

    with tab1:
        section("Category Expense Breakdown")
        cat_df = eda["cat_summary"]
        col1, col2 = st.columns([3, 2])
        with col1:
            fig = px.bar(cat_df, x="Total", y="category", orientation="h",
                         color="Total", color_continuous_scale="teal",
                         text=cat_df["Pct"].apply(lambda x: f"{x}%"),
                         title="Total Spend by Category")
            fig.update_traces(textposition="outside")
            fig.update_layout(**PT, height=420, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.treemap(cat_df, path=["category"], values="Total",
                              color="Pct", color_continuous_scale="teal",
                              title="Spending Treemap")
            fig2.update_layout(**PT, height=420)
            st.plotly_chart(fig2, use_container_width=True)

        section("Category Details Table")
        st.dataframe(
            cat_df.rename(columns={"category":"Category","Total":"Total Spent","Transactions":"# Transactions",
                                    "Avg":"Avg Transaction","Pct":"% of Total"})
                  .style.format({"Total Spent":"${:,.2f}","Avg Transaction":"${:,.2f}","% of Total":"{:.1f}%"}),
            use_container_width=True, height=320
        )

    with tab2:
        section("Monthly Income vs Expenses")
        dm = eda["monthly_trend"]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=dm["month_str"], y=dm["income"],
                                   name="Income", line=dict(color="#00D4AA", width=2.5),
                                   fill="tozeroy", fillcolor="rgba(0,212,170,0.08)"))
        fig3.add_trace(go.Scatter(x=dm["month_str"], y=dm["expenses"],
                                   name="Expenses", line=dict(color="#EF4444", width=2.5),
                                   fill="tozeroy", fillcolor="rgba(239,68,68,0.08)"))
        fig3.add_trace(go.Scatter(x=dm["month_str"], y=dm["savings"],
                                   name="Savings", line=dict(color="#7B61FF", width=2, dash="dot")))
        fig3.update_layout(**PT, title="Monthly Financial Flow", height=380)
        st.plotly_chart(fig3, use_container_width=True)

        section("Savings Rate Trend")
        fig4 = px.area(dm, x="month_str", y="savings_rate",
                        color_discrete_sequence=["#10B981"], title="Monthly Savings Rate (%)")
        fig4.add_hline(y=20, line_dash="dash", line_color="#F59E0B",
                       annotation_text="20% Target", annotation_position="right")
        fig4.update_layout(**PT, height=280)
        st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        section("Recurring vs One-Time Spending")
        c1, c2 = st.columns(2)
        with c1:
            rec_data = pd.DataFrame({
                "Type": ["Recurring", "One-Time"],
                "Amount": [eda["recurring_total"], eda["onetime_total"]],
            })
            fig5 = px.pie(rec_data, values="Amount", names="Type",
                          color_discrete_sequence=["#00D4AA","#7B61FF"], hole=0.5,
                          title="Recurring vs One-Time Expenses")
            fig5.update_layout(**PT, height=320)
            st.plotly_chart(fig5, use_container_width=True)
        with c2:
            section("Spending by Day of Week")
            wd = eda["weekday_spend"]
            fig6 = px.bar(wd, x="Weekday", y="Total",
                          color="Total", color_continuous_scale="teal",
                          title="Total Spend by Weekday")
            fig6.update_layout(**PT, height=320, coloraxis_showscale=False)
            st.plotly_chart(fig6, use_container_width=True)

    with tab4:
        section("Transaction Records")
        df_show = st.session_state.df_tx.copy()
        cat_filter = st.multiselect("Filter Category", options=sorted(df_show["category"].unique()),
                                     default=sorted(df_show["category"].unique()))
        type_filter = st.selectbox("Type", ["All","expense","income"])
        if type_filter != "All":
            df_show = df_show[df_show["type"] == type_filter]
        df_show = df_show[df_show["category"].isin(cat_filter)]
        st.dataframe(df_show.sort_values("date", ascending=False).head(200),
                     use_container_width=True, height=420)
        csv = df_show.to_csv(index=False)
        st.download_button("📥 Download Filtered CSV", csv, "transactions.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FINANCIAL SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Financial Segments":
    st.markdown("# 👥 Financial Segmentation")
    if not st.session_state.model_ready:
        not_ready()

    df_seg = st.session_state.df_seg

    section("K-Means Monthly Financial Clustering")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df_seg, x="expenses", y="savings",
                         color="segment", size="income",
                         color_discrete_sequence=COLORS,
                         hover_data=["income","savings_rate","n_transactions"],
                         title="Expense vs Savings Clusters")
        fig.update_layout(**PT, height=380)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.scatter(df_seg, x="income", y="savings_rate",
                          color="segment", size="expenses",
                          color_discrete_sequence=COLORS,
                          title="Income vs Savings Rate")
        fig2.update_layout(**PT, height=380)
        st.plotly_chart(fig2, use_container_width=True)

    section("Segment Statistics")
    seg_stats = df_seg.groupby("segment").agg(
        Months=("income","count"),
        Avg_Income=("income","mean"),
        Avg_Expenses=("expenses","mean"),
        Avg_Savings=("savings","mean"),
        Avg_SaveRate=("savings_rate","mean"),
    ).round(1).reset_index()
    seg_stats.columns = ["Segment","Months","Avg Income","Avg Expenses","Avg Savings","Avg Save Rate %"]
    st.dataframe(seg_stats, use_container_width=True)

    section("Segment Timeline")
    fig3 = px.scatter(df_seg, x=df_seg["month"].astype(str), y="savings_rate",
                      color="segment", size="income",
                      color_discrete_sequence=COLORS,
                      title="Savings Rate Timeline by Segment")
    fig3.add_hline(y=20, line_dash="dash", line_color="#F59E0B",
                   annotation_text="20% Savings Goal")
    fig3.update_layout(**PT, height=350)
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SPENDING TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Spending Trends":
    st.markdown("# 📈 Spending Trends")
    if not st.session_state.model_ready:
        not_ready()

    patterns = st.session_state.patterns
    eda = st.session_state.eda

    section("Category Spending Heatmap (Month × Category)")
    cat_monthly = patterns["cat_monthly"]
    if not cat_monthly.empty:
        fig = px.imshow(cat_monthly.T, color_continuous_scale="teal",
                        aspect="auto", title="Spending Heatmap")
        fig.update_layout(**PT, height=400)
        st.plotly_chart(fig, use_container_width=True)

    section("Sub-Category Breakdown")
    subcat = eda["subcat_summary"]
    fig2 = px.sunburst(subcat, path=["category","subcategory"], values="amount",
                        color="amount", color_continuous_scale="teal",
                        title="Category → Subcategory Sunburst")
    fig2.update_layout(**PT, height=480)
    st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        section("💎 Top 10 Largest Transactions")
        top_tx = patterns["top_transactions"].copy()
        top_tx["date"] = top_tx["date"].astype(str)
        fig3 = px.bar(top_tx, x="amount", y="description", orientation="h",
                      color="category", color_discrete_sequence=COLORS,
                      title="Largest Single Transactions")
        fig3.update_layout(**PT, height=380)
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        section("🔄 Monthly Recurring Payments")
        rec = patterns["recurring_monthly"]
        if not rec.empty:
            fig4 = px.area(rec, x="month", y="recurring_amount",
                           color_discrete_sequence=["#F59E0B"],
                           title="Monthly Recurring Obligation")
            fig4.update_layout(**PT, height=380)
            st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ML PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Predictions":
    st.markdown("# 🤖 ML Spending Predictions")
    if not st.session_state.model_ready:
        not_ready()

    m = st.session_state.ml_metrics
    pred = st.session_state.prediction
    feats = st.session_state.feats

    section("Model Performance")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("📉 MAE",              f"${m['mae']:,.0f}")
    c2.metric("📐 RMSE",             f"${m['rmse']:,.0f}")
    c3.metric("🎯 R² Score",         f"{m['r2']:.3f}")
    c4.metric("🔁 CV MAE (5-fold)",  f"${m['cv_mae']:,.0f}")

    section("Next Month Spending Forecast")
    pct = pred["pct_change"]
    c = "#EF4444" if pct > 10 else "#10B981" if pct < -5 else "#F59E0B"
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""<div class="pred-box" style="margin-top:10px">
            <div style="color:#94A3B8;font-size:0.8rem;text-transform:uppercase;letter-spacing:0.08em">Predicted Spend</div>
            <div class="pred-amount">${pred['predicted_spend']:,.0f}</div>
            <div style="color:{c};font-size:1.2rem;font-weight:700;margin-top:8px">{'+' if pct>0 else ''}{pct:.1f}%</div>
            <div style="color:#475569;font-size:0.78rem;margin-top:6px">vs last month: ${pred['last_actual_spend']:,.0f}</div>
            <div style="color:#94A3B8;font-size:0.78rem;margin-top:8px">
                95% Range: ${pred['lower_bound']:,.0f} – ${pred['upper_bound']:,.0f}
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        section("Actual vs Predicted (Test Set)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=m["y_test"], name="Actual",
                                  line=dict(color="#00D4AA", width=2.5), mode="lines+markers"))
        fig.add_trace(go.Scatter(y=m["y_pred"], name="Predicted",
                                  line=dict(color="#7B61FF", width=2.5, dash="dash"), mode="lines+markers"))
        fig.update_layout(**PT, title="Predicted vs Actual Monthly Spend", height=280)
        st.plotly_chart(fig, use_container_width=True)

    section("Feature Importance")
    fi = m["feature_importance"]
    fi_df = pd.DataFrame(list(fi.items()), columns=["Feature","Importance"]).sort_values("Importance")
    fi_df["Feature"] = fi_df["Feature"].str.replace("cat_","").str.replace("_"," ").str.title()
    fig2 = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                  color="Importance", color_continuous_scale="teal",
                  title="Top Predictive Features")
    fig2.update_layout(**PT, height=420, coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    section("Historical Spending with Forecast")
    dm = st.session_state.df_seg
    hist_line = go.Figure()
    hist_line.add_trace(go.Scatter(
        x=dm["month"].astype(str), y=dm["expenses"],
        name="Actual Expenses", line=dict(color="#00D4AA", width=2.5)
    ))
    # Add forecast point
    last_month = dm["month"].max()
    next_month = (pd.Period(last_month, "M") + 1).to_timestamp()
    hist_line.add_trace(go.Scatter(
        x=[str(last_month), str(next_month)],
        y=[dm["expenses"].iloc[-1], pred["predicted_spend"]],
        name="Forecast", line=dict(color="#7B61FF", width=2.5, dash="dot"),
        mode="lines+markers", marker=dict(size=10, symbol="star")
    ))
    hist_line.add_hrect(y0=pred["lower_bound"], y1=pred["upper_bound"],
                         fillcolor="rgba(123,97,255,0.08)", line_width=0,
                         annotation_text="Forecast Range", annotation_position="top right")
    hist_line.update_layout(**PT, title="Spending History + ML Forecast", height=360)
    st.plotly_chart(hist_line, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI BUDGET ADVISOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 AI Budget Advisor":
    st.markdown("# 🧠 AI Budget Advisor")
    st.markdown('<span class="free-badge">⚡ Powered by Rule-Based Expert AI — No API Keys Required</span>', unsafe_allow_html=True)
    if not st.session_state.model_ready:
        not_ready()

    advice = st.session_state.advice

    # Financial health
    section("Financial Health Assessment")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"""<div class="health-score">
            <div style="font-size:3rem">{advice['health_icon']}</div>
            <div class="score-number">{st.session_state.advice['financial_score']}</div>
            <div class="score-label">{advice['score_label']}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="advice-card info">{advice["health_msg"]}</div>', unsafe_allow_html=True)
        for bullet in advice["summary_bullets"]:
            st.markdown(f'<div class="advice-card" style="border-left:3px solid #1E3A5F;padding:10px 14px;font-size:0.85rem;color:#CBD5E1">{bullet}</div>', unsafe_allow_html=True)

    # Category alerts
    if advice["category_alerts"]:
        section("⚠️ Budget Alerts — Categories Over Ideal Limits")
        for alert in advice["category_alerts"]:
            lvl_css = "warning" if alert["level"] == "warning" else "critical"
            save_str = f" · 💰 Potential saving: **${alert['save_potential']:,.0f}/yr**" if alert["save_potential"] > 0 else ""
            st.markdown(f"""<div class="advice-card {lvl_css}">
                <div style="color:#E2E8F0">{alert['message']}{save_str}</div>
                <div style="color:#94A3B8;font-size:0.82rem;margin-top:6px">💡 {alert['action']}</div>
            </div>""", unsafe_allow_html=True)

    # Prediction advice
    if advice["prediction_advice"]:
        section("🔮 ML-Driven Spending Outlook")
        for pa in advice["prediction_advice"]:
            st.markdown(f'<div class="advice-card info">{pa}</div>', unsafe_allow_html=True)

    # Strategies
    section("🎯 Personalized Savings Strategies")
    for s in advice["strategies"]:
        impact_color = {"high":"#10B981","medium":"#F59E0B","low":"#94A3B8"}.get(s["impact"],"#94A3B8")
        gain_str = f" · Est. gain: **${s['annual_gain']:,.0f}/yr**" if s["annual_gain"] > 0 else ""
        st.markdown(f"""<div class="strategy-card">
            <div class="strategy-title">{s['title']}
                <span style="float:right;color:{impact_color};font-size:0.75rem;font-weight:600">
                    {s['impact'].upper()} IMPACT
                </span>
            </div>
            <div class="strategy-desc">{s['desc']}</div>
            <div class="strategy-gain">{gain_str}</div>
        </div>""", unsafe_allow_html=True)

    # Budget Plan table
    section("📋 Recommended Monthly Budget (50/30/20 Framework)")
    bp = advice["budget_plan"]
    inc = st.session_state.eda["avg_monthly_income"]
    st.markdown(f'<div style="color:#94A3B8;font-size:0.85rem;margin-bottom:10px">Based on your average monthly income of <strong>${inc:,.0f}</strong></div>', unsafe_allow_html=True)
    st.dataframe(
        bp.style.format({"Recommended Budget": "${:,.2f}"}),
        use_container_width=True, height=380
    )

    # Visual budget vs actual
    section("Budget vs Actual Comparison")
    cat_actual = st.session_state.eda["cat_summary"][["category","Total"]].copy()
    cat_actual["Monthly Actual"] = cat_actual["Total"] / st.session_state.eda["n_months"]
    budget_compare = bp.merge(
        cat_actual.rename(columns={"category":"Category"}),
        on="Category", how="left"
    ).fillna(0)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Budget", x=budget_compare["Category"],
                         y=budget_compare["Recommended Budget"],
                         marker_color="#00D4AA", opacity=0.8))
    fig.add_trace(go.Bar(name="Actual (Avg/Month)", x=budget_compare["Category"],
                         y=budget_compare["Monthly Actual"],
                         marker_color="#EF4444", opacity=0.8))
    fig.update_layout(**PT, barmode="group", title="Recommended Budget vs Your Actual Spending",
                      height=380, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INSIGHTS & ALERTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Insights & Alerts":
    st.markdown("# 💡 Financial Insights & Alerts")
    if not st.session_state.model_ready:
        not_ready()

    advice = st.session_state.advice
    eda   = st.session_state.eda
    dm    = st.session_state.df_seg

    # Spending alerts
    section("🚨 Real-Time Spending Alerts")
    if advice["spending_alerts"]:
        for a in advice["spending_alerts"]:
            css = ("alert-spike" if a["type"] in ("spike","critical") else
                   "alert-warning" if a["type"] == "warning" else
                   "alert-positive")
            st.markdown(f'<div class="alert-box {css}">{a["icon"]} {a["msg"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-box alert-positive">✅ No critical spending alerts this period. Keep up the good work!</div>', unsafe_allow_html=True)

    section("📊 Key Financial Metrics")
    col1, col2 = st.columns(2)
    with col1:
        # Savings progress bar
        target_rate = 20
        actual_rate = eda["overall_savings_rate"]
        progress = min(1.0, actual_rate / target_rate)
        st.markdown(f"""
        <div class="advice-card info">
            <div style="font-weight:600;color:#E2E8F0;margin-bottom:10px">🎯 Savings Rate Progress</div>
            <div style="color:#94A3B8;font-size:0.82rem">Target: 20% | Actual: {actual_rate:.1f}%</div>
            <div style="background:#1E3A5F;border-radius:8px;height:14px;margin-top:10px;overflow:hidden">
                <div style="background:linear-gradient(90deg,#00D4AA,#0099FF);
                            width:{min(100,progress*100):.0f}%;height:100%;border-radius:8px;
                            transition:width 0.5s"></div>
            </div>
            <div style="color:#00D4AA;font-size:0.85rem;margin-top:6px;font-weight:600">
                {min(100,progress*100):.0f}% of target achieved
            </div>
        </div>""", unsafe_allow_html=True)

        # Emergency fund
        ef_target = eda["avg_monthly_expenses"] * 6
        ef_current = eda["total_savings"]
        ef_pct = min(100, ef_current / ef_target * 100) if ef_target > 0 else 0
        st.markdown(f"""
        <div class="advice-card warning" style="margin-top:10px">
            <div style="font-weight:600;color:#E2E8F0;margin-bottom:10px">🛡️ Emergency Fund</div>
            <div style="color:#94A3B8;font-size:0.82rem">Target: ${ef_target:,.0f} (6 months) | Current: ${ef_current:,.0f}</div>
            <div style="background:#1E3A5F;border-radius:8px;height:14px;margin-top:10px;overflow:hidden">
                <div style="background:linear-gradient(90deg,#F59E0B,#EF4444);
                            width:{ef_pct:.0f}%;height:100%;border-radius:8px"></div>
            </div>
            <div style="color:#F59E0B;font-size:0.85rem;margin-top:6px;font-weight:600">
                {ef_pct:.0f}% funded
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        # Monthly savings trend sparkline
        section("Monthly Savings Rate")
        fig = px.line(dm, x=dm["month"].astype(str), y="savings_rate",
                      color_discrete_sequence=["#10B981"], markers=True,
                      title="Savings Rate Over Time")
        fig.add_hline(y=20, line_dash="dash", line_color="#F59E0B",
                      annotation_text="20% Goal")
        fig.update_layout(**PT, height=270)
        st.plotly_chart(fig, use_container_width=True)

    section("📉 Top Overspending Categories vs Budget")
    alerts = advice["category_alerts"]
    if alerts:
        alert_df = pd.DataFrame([{
            "Category": a["category"],
            "Saving Potential ($/yr)": a["save_potential"],
            "Tip": a["action"][:60] + "...",
        } for a in alerts if a["save_potential"] > 0])
        if not alert_df.empty:
            fig2 = px.bar(alert_df, x="Category", y="Saving Potential ($/yr)",
                          color="Saving Potential ($/yr)", color_continuous_scale="reds",
                          title="Annual Saving Potential by Category")
            fig2.update_layout(**PT, height=300, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.markdown('<div class="alert-box alert-positive">✅ All categories are within recommended limits!</div>', unsafe_allow_html=True)

    section("📆 Best & Worst Financial Months")
    col1, col2 = st.columns(2)
    with col1:
        best = dm.nlargest(3, "savings_rate")[["month","income","expenses","savings","savings_rate"]]
        st.markdown("**🏆 Best Months (Highest Savings Rate)**")
        st.dataframe(best.assign(month=best["month"].astype(str)), use_container_width=True)
    with col2:
        worst = dm.nsmallest(3, "savings_rate")[["month","income","expenses","savings","savings_rate"]]
        st.markdown("**⚠️ Worst Months (Lowest Savings Rate)**")
        st.dataframe(worst.assign(month=worst["month"].astype(str)), use_container_width=True)
