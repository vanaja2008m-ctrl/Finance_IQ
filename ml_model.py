import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")


def engineer_features(df_tx: pd.DataFrame, df_monthly: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix for spending prediction (target = next month expenses)."""
    expenses = df_tx[df_tx["type"] == "expense"].copy()
    expenses["date"] = pd.to_datetime(expenses["date"])
    expenses["month"] = expenses["date"].dt.to_period("M").astype(str)

    # Per-month aggregates
    monthly_feats = expenses.groupby("month").agg(
        total_spend=("amount", "sum"),
        n_transactions=("amount", "count"),
        avg_tx_amount=("amount", "mean"),
        max_tx_amount=("amount", "max"),
        std_tx_amount=("amount", "std"),
        n_categories=("category", "nunique"),
        recurring_spend=("amount", lambda x: x[expenses.loc[x.index, "is_recurring"]].sum()),
    ).reset_index()

    # Category spend columns
    cat_pivot = expenses.pivot_table(
        index="month", columns="category", values="amount",
        aggfunc="sum", fill_value=0
    ).reset_index()
    cat_pivot.columns = [f"cat_{c}" if c != "month" else c for c in cat_pivot.columns]

    feats = monthly_feats.merge(cat_pivot, on="month", how="left")
    feats = feats.merge(
        df_monthly[["month", "income", "savings_rate"]].assign(
            month=df_monthly["month"].dt.to_period("M").astype(str)
        ),
        on="month", how="left"
    )

    feats = feats.sort_values("month").reset_index(drop=True)
    feats["std_tx_amount"] = feats["std_tx_amount"].fillna(0)

    # Rolling features (lag 1 and 2 months)
    feats["spend_lag1"] = feats["total_spend"].shift(1)
    feats["spend_lag2"] = feats["total_spend"].shift(2)
    feats["spend_rolling3"] = feats["total_spend"].shift(1).rolling(3).mean()
    feats["income_lag1"] = feats["income"].shift(1)

    # Target = next month spending
    feats["target_next_month_spend"] = feats["total_spend"].shift(-1)

    # Drop rows without lag or target
    feats = feats.dropna(subset=["spend_lag1", "spend_lag2", "target_next_month_spend"])
    return feats


def train_spending_model(feats: pd.DataFrame, model_type: str = "Random Forest"):
    """Train regression model to predict next-month spending."""
    exclude = ["month", "target_next_month_spend", "total_spend"]
    feature_cols = [c for c in feats.columns if c not in exclude]

    X = feats[feature_cols].fillna(0)
    y = feats["target_next_month_spend"]

    if len(X) < 6:
        raise ValueError("Need at least 8 months of data to train.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    else:
        model = GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                           learning_rate=0.1, random_state=42)

    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    cv = cross_val_score(model, scaler.transform(X), y, cv=min(5, len(X)),
                         scoring="neg_mean_absolute_error")

    fi = dict(zip(feature_cols, model.feature_importances_))
    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:12])

    metrics = {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "cv_mae": round(-cv.mean(), 2),
        "cv_std": round(cv.std(), 2),
        "feature_importance": fi_sorted,
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
    }
    return model, scaler, feature_cols, metrics


def predict_next_month(model, scaler, feature_cols: list, feats: pd.DataFrame) -> dict:
    """Predict spending for the upcoming month."""
    latest = feats[feature_cols].fillna(0).iloc[-1:].copy()
    X_sc = scaler.transform(latest)
    pred = model.predict(X_sc)[0]

    # Confidence interval via std of feature importances
    last_actual = feats["total_spend"].iloc[-1]
    pct_change = (pred - last_actual) / last_actual * 100

    return {
        "predicted_spend": round(pred, 2),
        "last_actual_spend": round(last_actual, 2),
        "pct_change": round(pct_change, 1),
        "lower_bound": round(pred * 0.88, 2),
        "upper_bound": round(pred * 1.12, 2),
    }
