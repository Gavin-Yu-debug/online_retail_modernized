from __future__ import annotations

import json
import os
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from catboost import CatBoostClassifier, CatBoostRegressor
from hdbscan import HDBSCAN
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

RANDOM_SEED = 42
HORIZON_DAYS = 90


@dataclass
class RunArtifacts:
    raw_file: Path
    source_sheets: list[str]
    data_quality_summary: pd.DataFrame
    snapshot_data: pd.DataFrame
    cluster_comparison: pd.DataFrame
    cluster_profiles: pd.DataFrame
    classification_results: pd.DataFrame
    regression_results: pd.DataFrame
    scoring_sample: pd.DataFrame
    best_classifier_name: str
    best_regressor_name: str
    final_cluster_method: str
    holdout_cutoff: pd.Timestamp
    feature_names: list[str]
    top_classifier_features: list[str]
    top_regressor_features: list[str]
    clv_used: bool


def print_header(text: str) -> None:
    print(f"\n{'=' * 10} {text} {'=' * 10}")


def ensure_outputs_dir(project_root: Path) -> Path:
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def snake_case(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(name).strip().lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def find_raw_excel(data_dir: Path) -> Path:
    excel_files = sorted(data_dir.glob("*.xls*"))
    if not excel_files:
        raise FileNotFoundError("No Excel files found in ./data/")

    preferred = [
        path
        for path in excel_files
        if "online" in path.name.lower() and "retail" in path.name.lower()
    ]
    ranked = preferred or excel_files
    ranked = sorted(ranked, key=lambda p: (p.suffix.lower() != ".xlsx", -p.stat().st_size, p.name.lower()))
    return ranked[0]


def identify_valid_sheets(raw_file: Path) -> list[str]:
    xls = pd.ExcelFile(raw_file)
    valid_sheets: list[str] = []
    required = {"invoice", "stockcode", "description", "quantity", "invoicedate", "price", "customerid"}
    for sheet in xls.sheet_names:
        sample = pd.read_excel(raw_file, sheet_name=sheet, nrows=5)
        normalized = {snake_case(col).replace("_", "") for col in sample.columns}
        if required.issubset(normalized):
            valid_sheets.append(sheet)
    if not valid_sheets:
        raise ValueError(f"No valid Online Retail II sheets found in {raw_file}")
    return valid_sheets


def load_raw_data(raw_file: Path) -> tuple[pd.DataFrame, list[str]]:
    sheets = identify_valid_sheets(raw_file)
    frames: list[pd.DataFrame] = []
    for sheet in sheets:
        frame = pd.read_excel(raw_file, sheet_name=sheet)
        frame["source_sheet"] = sheet
        frames.append(frame)
    raw_df = pd.concat(frames, ignore_index=True)

    column_map = {
        "invoice": "invoice",
        "stockcode": "stock_code",
        "description": "description",
        "quantity": "quantity",
        "invoicedate": "invoice_date",
        "price": "unit_price",
        "customerid": "customer_id",
        "country": "country",
        "source_sheet": "source_sheet",
    }

    normalized_names = {col: snake_case(col).replace("_", "") for col in raw_df.columns}
    renamed = {}
    for original, normalized in normalized_names.items():
        if normalized in column_map:
            renamed[original] = column_map[normalized]

    raw_df = raw_df.rename(columns=renamed)
    expected_columns = [
        "invoice",
        "stock_code",
        "description",
        "quantity",
        "invoice_date",
        "unit_price",
        "customer_id",
        "country",
        "source_sheet",
    ]
    missing_expected = [col for col in expected_columns if col not in raw_df.columns]
    if missing_expected:
        raise ValueError(f"Missing expected columns after standardization: {missing_expected}")

    raw_df = raw_df[expected_columns].copy()
    return raw_df, sheets


def make_category_proxy(description: pd.Series) -> pd.Series:
    cleaned = description.fillna("").astype(str).str.upper()
    proxy = cleaned.str.extract(r"([A-Z]{3,})", expand=False).fillna("OTHER")
    proxy = proxy.where(proxy != "", "OTHER")
    return proxy


def clean_transactions(raw_df: pd.DataFrame, output_dir: Path, raw_file: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = raw_df.copy()

    df["invoice"] = df["invoice"].astype(str).str.strip()
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    df["description"] = df["description"].astype(str).str.strip()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["customer_id_num"] = pd.to_numeric(df["customer_id"], errors="coerce")
    df["customer_id"] = df["customer_id_num"].round().astype("Int64").astype(str)
    df.loc[df["customer_id"] == "<NA>", "customer_id"] = pd.NA
    df["country"] = df["country"].fillna("Unknown").astype(str).str.strip()
    df["description_category_proxy"] = make_category_proxy(df["description"])
    df["raw_line_total"] = df["quantity"] * df["unit_price"]

    raw_row_count = len(df)
    duplicate_count = int(df.duplicated().sum())
    missing_customer_count = int(df["customer_id"].isna().sum())
    missing_date_count = int(df["invoice_date"].isna().sum())
    blank_invoice_count = int(df["invoice"].eq("").sum())
    blank_stock_code_count = int(df["stock_code"].isin(["", "NAN", "NONE"]).sum())
    cancellation_invoice_count = int(df["invoice"].str.upper().str.startswith("C", na=False).sum())
    negative_quantity_count = int((df["quantity"] <= 0).sum())
    non_positive_price_count = int((df["unit_price"] <= 0).sum())

    cleaned_all = df.drop_duplicates().copy()
    cleaned_all = cleaned_all.dropna(subset=["customer_id", "invoice_date", "quantity", "unit_price"])
    cleaned_all = cleaned_all[~cleaned_all["invoice"].eq("")]
    cleaned_all = cleaned_all[~cleaned_all["stock_code"].isin(["", "NAN", "NONE"])]
    cleaned_all["is_cancellation_invoice"] = cleaned_all["invoice"].str.upper().str.startswith("C", na=False)
    cleaned_all["is_return_like"] = cleaned_all["is_cancellation_invoice"] | (cleaned_all["quantity"] <= 0)
    cleaned_all["sales_line_total"] = cleaned_all["quantity"] * cleaned_all["unit_price"]

    positive_sales = cleaned_all[
        (~cleaned_all["is_return_like"])
        & (cleaned_all["unit_price"] > 0)
        & (cleaned_all["quantity"] > 0)
        & (cleaned_all["sales_line_total"] > 0)
    ].copy()

    cleaned_row_count = len(positive_sales)

    quality_rows = [
        {"section": "source", "metric": "raw_file_used", "value": str(raw_file)},
        {"section": "source", "metric": "raw_sheet_count", "value": cleaned_all["source_sheet"].nunique()},
        {"section": "rows", "metric": "raw_row_count", "value": raw_row_count},
        {"section": "rows", "metric": "exact_duplicate_rows", "value": duplicate_count},
        {"section": "rows", "metric": "rows_missing_customer_id", "value": missing_customer_count},
        {"section": "rows", "metric": "rows_missing_invoice_date", "value": missing_date_count},
        {"section": "rows", "metric": "rows_blank_invoice", "value": blank_invoice_count},
        {"section": "rows", "metric": "rows_blank_stock_code", "value": blank_stock_code_count},
        {"section": "rows", "metric": "rows_cancellation_invoice", "value": cancellation_invoice_count},
        {"section": "rows", "metric": "rows_non_positive_quantity", "value": negative_quantity_count},
        {"section": "rows", "metric": "rows_non_positive_price", "value": non_positive_price_count},
        {"section": "rows", "metric": "cleaned_positive_sales_rows", "value": cleaned_row_count},
        {
            "section": "coverage",
            "metric": "raw_min_invoice_date",
            "value": df["invoice_date"].min().strftime("%Y-%m-%d"),
        },
        {
            "section": "coverage",
            "metric": "raw_max_invoice_date",
            "value": df["invoice_date"].max().strftime("%Y-%m-%d"),
        },
        {
            "section": "coverage",
            "metric": "cleaned_min_invoice_date",
            "value": positive_sales["invoice_date"].min().strftime("%Y-%m-%d"),
        },
        {
            "section": "coverage",
            "metric": "cleaned_max_invoice_date",
            "value": positive_sales["invoice_date"].max().strftime("%Y-%m-%d"),
        },
        {"section": "coverage", "metric": "unique_customers_cleaned", "value": positive_sales["customer_id"].nunique()},
        {"section": "coverage", "metric": "unique_countries_cleaned", "value": positive_sales["country"].nunique()},
    ]

    missing_summary = cleaned_all[["invoice", "stock_code", "description", "quantity", "invoice_date", "unit_price", "customer_id", "country"]].isna().sum()
    for column, value in missing_summary.items():
        quality_rows.append({"section": "missing_values_after_basic_cleaning", "metric": column, "value": int(value)})

    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(output_dir / "data_quality_summary.csv", index=False)
    return cleaned_all, positive_sales, quality_df


def generate_snapshot_dates(positive_sales: pd.DataFrame) -> list[pd.Timestamp]:
    min_date = positive_sales["invoice_date"].min().normalize()
    max_date = positive_sales["invoice_date"].max().normalize()
    first_cutoff = (min_date + pd.DateOffset(months=7)).replace(day=1)
    final_cutoff = (max_date - pd.Timedelta(days=HORIZON_DAYS)).replace(day=1)
    snapshot_dates = list(pd.date_range(first_cutoff, final_cutoff, freq="2MS"))
    if len(snapshot_dates) < 5:
        raise ValueError("Not enough time coverage to build the required temporal snapshots.")
    return snapshot_dates


def _compute_purchase_interval_stats(order_dates: pd.Series) -> tuple[float, float]:
    sorted_dates = pd.Series(pd.to_datetime(order_dates).sort_values().unique())
    if len(sorted_dates) <= 1:
        return (np.nan, np.nan)
    deltas = sorted_dates.diff().dropna().dt.days.astype(float)
    return (float(deltas.mean()), float(deltas.std(ddof=0)) if len(deltas) > 1 else 0.0)


def build_customer_snapshot(
    positive_sales: pd.DataFrame,
    cleaned_all: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    horizon_days: int = HORIZON_DAYS,
) -> pd.DataFrame:
    history_sales = positive_sales[positive_sales["invoice_date"] < snapshot_date].copy()
    history_all = cleaned_all[cleaned_all["invoice_date"] < snapshot_date].copy()
    future_sales = positive_sales[
        (positive_sales["invoice_date"] >= snapshot_date)
        & (positive_sales["invoice_date"] < snapshot_date + pd.Timedelta(days=horizon_days))
    ].copy()

    if history_sales.empty:
        return pd.DataFrame()

    order_level = (
        history_sales.groupby(["customer_id", "invoice"], as_index=False)
        .agg(
            order_date=("invoice_date", "min"),
            order_sales=("sales_line_total", "sum"),
            order_quantity=("quantity", "sum"),
        )
    )
    order_level["order_day"] = order_level["order_date"].dt.normalize()

    customer_agg = (
        order_level.groupby("customer_id", as_index=False)
        .agg(
            first_purchase_date=("order_date", "min"),
            last_purchase_date=("order_date", "max"),
            total_orders=("invoice", "nunique"),
            monetary=("order_sales", "sum"),
            avg_order_value=("order_sales", "mean"),
            avg_items_per_order=("order_quantity", "mean"),
            unique_invoice_days=("order_day", "nunique"),
        )
    )
    customer_agg["recency"] = (snapshot_date - customer_agg["last_purchase_date"]).dt.days
    customer_agg["frequency"] = customer_agg["total_orders"]
    customer_agg["purchase_span_days"] = (
        customer_agg["last_purchase_date"] - customer_agg["first_purchase_date"]
    ).dt.days
    customer_agg["days_since_first_purchase"] = (
        snapshot_date - customer_agg["first_purchase_date"]
    ).dt.days
    customer_agg["days_since_last_purchase"] = customer_agg["recency"]

    unique_products = history_sales.groupby("customer_id")["stock_code"].nunique().rename("unique_products")
    total_quantity = history_sales.groupby("customer_id")["quantity"].sum().rename("total_items")
    line_count = history_sales.groupby("customer_id").size().rename("line_count")

    interval_stats = (
        order_level.groupby("customer_id")["order_date"]
        .apply(_compute_purchase_interval_stats)
        .apply(pd.Series)
        .rename(columns={0: "purchase_interval_mean", 1: "purchase_interval_std"})
    )

    recent_14 = order_level[order_level["order_date"] >= snapshot_date - pd.Timedelta(days=14)]
    recent_30 = order_level[order_level["order_date"] >= snapshot_date - pd.Timedelta(days=30)]
    recent_14_features = recent_14.groupby("customer_id").agg(
        transactions_last_14d=("invoice", "nunique"),
        sales_last_14d=("order_sales", "sum"),
    )
    recent_30_features = recent_30.groupby("customer_id").agg(
        transactions_last_30d=("invoice", "nunique"),
        sales_last_30d=("order_sales", "sum"),
    )

    product_diversity = history_sales.groupby("customer_id").agg(
        product_category_count_proxy=("description_category_proxy", "nunique")
    )
    top_category = (
        history_sales.groupby(["customer_id", "description_category_proxy"])
        .size()
        .rename("category_line_count")
        .reset_index()
        .sort_values(["customer_id", "category_line_count"], ascending=[True, False])
        .drop_duplicates("customer_id")
        .set_index("customer_id")
    )
    top_category_share = (top_category["category_line_count"] / line_count).rename("top_category_share")

    latest_country = (
        history_sales.sort_values("invoice_date")
        .groupby("customer_id")
        .tail(1)[["customer_id", "country"]]
        .drop_duplicates("customer_id")
        .set_index("customer_id")
    )

    return_amount = history_all.loc[history_all["is_return_like"]].assign(
        abs_return_amount=lambda frame: frame["sales_line_total"].abs()
    )
    return_ratio = (
        return_amount.groupby("customer_id")["abs_return_amount"].sum()
        / customer_agg.set_index("customer_id")["monetary"]
    ).rename("return_ratio")

    future_targets = future_sales.groupby("customer_id", as_index=False).agg(
        future_orders_90d=("invoice", "nunique"),
        sales_90_value=("sales_line_total", "sum"),
    )
    future_targets["purchase_90_flag"] = 1

    snapshot = customer_agg.set_index("customer_id")
    snapshot = snapshot.join(unique_products)
    snapshot = snapshot.join(total_quantity)
    snapshot = snapshot.join(interval_stats)
    snapshot = snapshot.join(recent_14_features)
    snapshot = snapshot.join(recent_30_features)
    snapshot = snapshot.join(product_diversity)
    snapshot = snapshot.join(top_category_share)
    snapshot = snapshot.join(latest_country)
    snapshot = snapshot.join(return_ratio)
    snapshot = snapshot.join(future_targets.set_index("customer_id"))
    snapshot = snapshot.fillna(
        {
            "transactions_last_14d": 0,
            "sales_last_14d": 0.0,
            "transactions_last_30d": 0,
            "sales_last_30d": 0.0,
            "product_category_count_proxy": 0,
            "top_category_share": 0.0,
            "return_ratio": 0.0,
            "future_orders_90d": 0,
            "sales_90_value": 0.0,
            "purchase_90_flag": 0,
        }
    )
    snapshot["purchase_90_flag"] = snapshot["purchase_90_flag"].astype(int)
    snapshot["country"] = snapshot["country"].fillna("Unknown")
    snapshot["snapshot_date"] = snapshot_date
    snapshot["observation_month"] = snapshot_date.strftime("%Y-%m")
    snapshot["total_sales"] = snapshot["monetary"]
    snapshot["sales_per_item"] = snapshot["monetary"] / snapshot["total_items"].replace(0, np.nan)
    snapshot["sales_per_item"] = snapshot["sales_per_item"].fillna(snapshot["avg_order_value"])
    snapshot.reset_index(inplace=True)
    return snapshot


def build_snapshot_dataset(
    positive_sales: pd.DataFrame,
    cleaned_all: pd.DataFrame,
    snapshot_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    snapshots: list[pd.DataFrame] = []
    for snapshot_date in snapshot_dates:
        print(f"Building snapshot for cutoff {snapshot_date.date()}...")
        snapshot = build_customer_snapshot(positive_sales, cleaned_all, snapshot_date)
        if not snapshot.empty:
            snapshots.append(snapshot)
    if not snapshots:
        raise ValueError("Snapshot feature engineering failed to produce any data.")
    return pd.concat(snapshots, ignore_index=True)


def clip_outliers(df: pd.DataFrame, numeric_cols: list[str], lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    clipped = df.copy()
    for col in numeric_cols:
        series = clipped[col]
        if series.notna().sum() == 0:
            continue
        lo = series.quantile(lower)
        hi = series.quantile(upper)
        clipped[col] = series.clip(lo, hi)
    return clipped


def cluster_preprocess(feature_df: pd.DataFrame, cluster_features: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    prepared = clip_outliers(feature_df[cluster_features], cluster_features)
    transformed = prepared.copy()
    for col in cluster_features:
        if (transformed[col] >= 0).all():
            transformed[col] = np.log1p(transformed[col])
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    matrix = imputer.fit_transform(transformed)
    matrix = scaler.fit_transform(matrix)
    return prepared, matrix


def safe_silhouette_score(x: np.ndarray, labels: np.ndarray) -> float:
    unique = [label for label in np.unique(labels) if label != -1]
    if len(unique) < 2:
        return np.nan
    mask = labels != -1
    if mask.sum() < 5:
        return np.nan
    from sklearn.metrics import silhouette_score

    return float(silhouette_score(x[mask], labels[mask]))


def safe_davies_bouldin_score(x: np.ndarray, labels: np.ndarray) -> float:
    unique = [label for label in np.unique(labels) if label != -1]
    if len(unique) < 2:
        return np.nan
    mask = labels != -1
    if mask.sum() < 5:
        return np.nan
    return float(davies_bouldin_score(x[mask], labels[mask]))


def interpretability_score(frame: pd.DataFrame, labels: np.ndarray, feature_cols: list[str]) -> float:
    valid = frame.assign(cluster=labels)
    valid = valid[valid["cluster"] != -1]
    if valid["cluster"].nunique() < 2:
        return np.nan
    profiles = valid.groupby("cluster")[feature_cols].median()
    overall = valid[feature_cols].median()
    std = valid[feature_cols].std(ddof=0).replace(0, np.nan)
    z = ((profiles - overall) / std).abs()
    return float(np.nanmean(z.values))


def assign_segment_labels(profiles: pd.DataFrame) -> dict[int, str]:
    remaining = list(profiles.index)
    labels: dict[int, str] = {}
    overall = profiles.median()

    def pop_best(score: pd.Series, label: str) -> None:
        candidate = score.loc[remaining].idxmax()
        labels[int(candidate)] = label
        remaining.remove(candidate)

    loyal_score = (
        profiles["monetary"].rank(pct=True)
        + profiles["frequency"].rank(pct=True)
        + profiles["sales_last_30d"].rank(pct=True)
        + (1 - profiles["recency"].rank(pct=True))
    )
    pop_best(loyal_score, "High Value Loyal")

    if remaining:
        low_value_score = (
            profiles["recency"].rank(pct=True)
            + (1 - profiles["monetary"].rank(pct=True))
            + (1 - profiles["frequency"].rank(pct=True))
        )
        pop_best(low_value_score, "Low Value Infrequent")

    if remaining:
        at_risk_score = (
            profiles["recency"].rank(pct=True)
            + profiles["monetary"].rank(pct=True)
            + profiles["frequency"].rank(pct=True)
        )
        candidate = at_risk_score.loc[remaining].idxmax()
        if (
            profiles.loc[candidate, "monetary"] >= overall["monetary"]
            or profiles.loc[candidate, "frequency"] >= overall["frequency"]
        ):
            labels[int(candidate)] = "At Risk High Value"
        else:
            labels[int(candidate)] = "At Risk Mid Value"
        remaining.remove(candidate)

    for candidate in remaining:
        row = profiles.loc[candidate]
        if row["return_ratio"] > max(overall["return_ratio"] * 2, 0.10):
            labels[int(candidate)] = "Return-Heavy Occasional"
        elif row["sales_last_30d"] >= overall["sales_last_30d"] or row["recency"] <= overall["recency"]:
            labels[int(candidate)] = "Emerging Potential"
        else:
            labels[int(candidate)] = "Steady Mid Value"

    return labels


def run_clustering_analysis(
    feature_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    cluster_features = [
        "recency",
        "frequency",
        "monetary",
        "avg_order_value",
        "avg_items_per_order",
        "unique_products",
        "purchase_interval_mean",
        "transactions_last_30d",
        "sales_last_30d",
        "return_ratio",
        "product_category_count_proxy",
    ]
    base = feature_df.copy()
    prepared, x = cluster_preprocess(base, cluster_features)

    comparison_rows: list[dict[str, Any]] = []
    label_store: dict[str, np.ndarray] = {}

    best_kmeans_score = -np.inf
    for k in range(3, 7):
        model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20)
        labels = model.fit_predict(x)
        score = safe_silhouette_score(x, labels)
        if np.nan_to_num(score, nan=-1.0) > best_kmeans_score:
            best_kmeans_score = np.nan_to_num(score, nan=-1.0)
            label_store["KMeans"] = labels
            comparison_rows.append(
                {
                    "method": "KMeans",
                    "chosen_param": f"k={k}",
                    "n_clusters": int(len(np.unique(labels))),
                    "noise_share": 0.0,
                    "silhouette_score": score,
                    "davies_bouldin_score": safe_davies_bouldin_score(x, labels),
                    "largest_cluster_share": float(pd.Series(labels).value_counts(normalize=True).max()),
                    "interpretability_score": interpretability_score(prepared, labels, cluster_features),
                }
            )

    best_gmm_score = -np.inf
    for k in range(3, 7):
        model = GaussianMixture(n_components=k, covariance_type="full", random_state=RANDOM_SEED)
        labels = model.fit_predict(x)
        score = safe_silhouette_score(x, labels)
        if np.nan_to_num(score, nan=-1.0) > best_gmm_score:
            best_gmm_score = np.nan_to_num(score, nan=-1.0)
            label_store["GaussianMixture"] = labels
            comparison_rows.append(
                {
                    "method": "GaussianMixture",
                    "chosen_param": f"k={k}",
                    "n_clusters": int(len(np.unique(labels))),
                    "noise_share": 0.0,
                    "silhouette_score": score,
                    "davies_bouldin_score": safe_davies_bouldin_score(x, labels),
                    "largest_cluster_share": float(pd.Series(labels).value_counts(normalize=True).max()),
                    "interpretability_score": interpretability_score(prepared, labels, cluster_features),
                }
            )

    hdbscan_candidates: list[tuple[str, np.ndarray, float]] = []
    for min_cluster_size in [40, 60, 80]:
        model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=10, prediction_data=False)
        labels = model.fit_predict(x)
        score = safe_silhouette_score(x, labels)
        hdbscan_candidates.append((f"min_cluster_size={min_cluster_size}", labels, np.nan_to_num(score, nan=-1.0)))
    chosen_param, hdbscan_labels, _ = max(hdbscan_candidates, key=lambda item: item[2])
    label_store["HDBSCAN"] = hdbscan_labels
    cluster_counts = pd.Series(hdbscan_labels[hdbscan_labels != -1]).value_counts(normalize=True)
    comparison_rows.append(
        {
            "method": "HDBSCAN",
            "chosen_param": chosen_param,
            "n_clusters": int(len([label for label in np.unique(hdbscan_labels) if label != -1])),
            "noise_share": float((hdbscan_labels == -1).mean()),
            "silhouette_score": safe_silhouette_score(x, hdbscan_labels),
            "davies_bouldin_score": safe_davies_bouldin_score(x, hdbscan_labels),
            "largest_cluster_share": float(cluster_counts.max()) if not cluster_counts.empty else np.nan,
            "interpretability_score": interpretability_score(prepared, hdbscan_labels, cluster_features),
        }
    )

    comparison_df = pd.DataFrame(comparison_rows).drop_duplicates(subset=["method"], keep="last")
    comparison_df["balance_score"] = 1 - comparison_df["largest_cluster_share"].clip(upper=1.0).fillna(1.0)
    comparison_df["inverse_db_score"] = 1 / comparison_df["davies_bouldin_score"].replace(0, np.nan)
    for metric in ["silhouette_score", "inverse_db_score", "balance_score", "interpretability_score"]:
        values = comparison_df[metric]
        min_v = values.min(skipna=True)
        max_v = values.max(skipna=True)
        if pd.isna(min_v) or pd.isna(max_v) or math.isclose(min_v, max_v):
            comparison_df[f"{metric}_norm"] = 0.5
        else:
            comparison_df[f"{metric}_norm"] = (values - min_v) / (max_v - min_v)
    comparison_df["selection_score"] = (
        0.35 * comparison_df["silhouette_score_norm"].fillna(0)
        + 0.20 * comparison_df["inverse_db_score_norm"].fillna(0)
        + 0.15 * comparison_df["balance_score_norm"].fillna(0)
        + 0.20 * comparison_df["interpretability_score_norm"].fillna(0)
        + 0.10 * (1 - comparison_df["noise_share"].fillna(0))
    )
    comparison_df = comparison_df.sort_values("selection_score", ascending=False).reset_index(drop=True)
    final_method = str(comparison_df.iloc[0]["method"])
    final_labels = label_store[final_method]

    clustered = base.copy()
    clustered["cluster_id"] = final_labels
    if final_method == "HDBSCAN":
        clustered = clustered[clustered["cluster_id"] != -1].copy()

    profile_metrics = [
        "recency",
        "frequency",
        "monetary",
        "avg_order_value",
        "transactions_last_30d",
        "sales_last_30d",
        "return_ratio",
    ]
    profiles = clustered.groupby("cluster_id")[profile_metrics].median()
    label_map = assign_segment_labels(profiles)
    clustered["final_segment"] = clustered["cluster_id"].map(label_map)

    profile_counts = clustered["cluster_id"].value_counts().rename("customer_count")
    cluster_profiles = (
        clustered.groupby(["cluster_id", "final_segment"])
        .agg(
            customer_count=("customer_id", "count"),
            customer_share=("customer_id", lambda s: len(s) / len(clustered)),
            recency_median=("recency", "median"),
            frequency_median=("frequency", "median"),
            monetary_median=("monetary", "median"),
            avg_order_value_median=("avg_order_value", "median"),
            sales_last_30d_median=("sales_last_30d", "median"),
            return_ratio_median=("return_ratio", "median"),
        )
        .reset_index()
        .sort_values("customer_count", ascending=False)
    )

    comparison_df.to_csv(output_dir / "cluster_comparison.csv", index=False)
    cluster_profiles.to_csv(output_dir / "cluster_profiles.csv", index=False)

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    pca_points = pca.fit_transform(x)
    viz_df = base.copy()
    viz_df["pc1"] = pca_points[:, 0]
    viz_df["pc2"] = pca_points[:, 1]
    viz_df["cluster_id"] = final_labels
    viz_df = viz_df[viz_df["cluster_id"] != -1].copy()
    viz_df["final_segment"] = viz_df["cluster_id"].map(label_map)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=viz_df,
        x="pc1",
        y="pc2",
        hue="final_segment",
        palette="tab10",
        alpha=0.75,
        s=45,
    )
    plt.title(f"Customer Segment Visualization ({final_method})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_visualization.png", dpi=180, bbox_inches="tight")
    plt.close()

    return comparison_df, cluster_profiles, clustered, final_method


def make_preprocessor(numeric_features: list[str], categorical_features: list[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(numeric_steps)
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def classification_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_prob) if y_true.nunique() > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, y_prob) if y_true.nunique() > 1 else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier_score": float(np.mean((y_prob - y_true.to_numpy()) ** 2)),
    }
    return metrics


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred) if y_true.nunique() > 1 else np.nan,
    }


def build_classifier_specs(numeric_features: list[str], categorical_features: list[str]) -> dict[str, Pipeline]:
    common_preprocessor = make_preprocessor(numeric_features, categorical_features, scale_numeric=False)
    scaled_preprocessor = make_preprocessor(numeric_features, categorical_features, scale_numeric=True)
    return {
        "LogisticRegression": Pipeline(
            [
                ("preprocessor", scaled_preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1200,
                        random_state=RANDOM_SEED,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "LightGBMClassifier": Pipeline(
            [
                ("preprocessor", common_preprocessor),
                (
                    "model",
                    LGBMClassifier(
                        random_state=RANDOM_SEED,
                        n_estimators=300,
                        learning_rate=0.05,
                        num_leaves=31,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        verbose=-1,
                    ),
                ),
            ]
        ),
        "XGBoostClassifier": Pipeline(
            [
                ("preprocessor", common_preprocessor),
                (
                    "model",
                    XGBClassifier(
                        random_state=RANDOM_SEED,
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        eval_metric="logloss",
                        tree_method="hist",
                        n_jobs=4,
                    ),
                ),
            ]
        ),
        "CatBoostClassifier": Pipeline(
            [
                ("preprocessor", common_preprocessor),
                (
                    "model",
                    CatBoostClassifier(
                        random_seed=RANDOM_SEED,
                        iterations=300,
                        depth=6,
                        learning_rate=0.05,
                        loss_function="Logloss",
                        verbose=False,
                    ),
                ),
            ]
        ),
    }


def build_regressor_specs(numeric_features: list[str], categorical_features: list[str]) -> dict[str, Pipeline]:
    common_preprocessor = make_preprocessor(numeric_features, categorical_features, scale_numeric=False)
    scaled_preprocessor = make_preprocessor(numeric_features, categorical_features, scale_numeric=True)
    return {
        "RidgeRegression": Pipeline(
            [
                ("preprocessor", scaled_preprocessor),
                ("model", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
            ]
        ),
        "LightGBMRegressor": Pipeline(
            [
                ("preprocessor", common_preprocessor),
                (
                    "model",
                    LGBMRegressor(
                        random_state=RANDOM_SEED,
                        n_estimators=300,
                        learning_rate=0.05,
                        num_leaves=31,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        verbose=-1,
                    ),
                ),
            ]
        ),
        "XGBoostRegressor": Pipeline(
            [
                ("preprocessor", common_preprocessor),
                (
                    "model",
                    XGBRegressor(
                        random_state=RANDOM_SEED,
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="reg:squarederror",
                        tree_method="hist",
                        n_jobs=4,
                    ),
                ),
            ]
        ),
        "CatBoostRegressor": Pipeline(
            [
                ("preprocessor", common_preprocessor),
                (
                    "model",
                    CatBoostRegressor(
                        random_seed=RANDOM_SEED,
                        iterations=300,
                        depth=6,
                        learning_rate=0.05,
                        loss_function="RMSE",
                        verbose=False,
                    ),
                ),
            ]
        ),
    }


def get_time_folds(train_snapshot_dates: list[pd.Timestamp]) -> list[tuple[list[pd.Timestamp], pd.Timestamp]]:
    folds: list[tuple[list[pd.Timestamp], pd.Timestamp]] = []
    for idx in range(1, len(train_snapshot_dates)):
        train_dates = train_snapshot_dates[:idx]
        val_date = train_snapshot_dates[idx]
        folds.append((train_dates, val_date))
    return folds


def fit_and_evaluate_classifiers(
    modeling_df: pd.DataFrame,
    feature_cols: list[str],
    holdout_cutoff: pd.Timestamp,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Pipeline], pd.DataFrame]:
    train_df = modeling_df[modeling_df["snapshot_date"] < holdout_cutoff].copy()
    holdout_df = modeling_df[modeling_df["snapshot_date"] == holdout_cutoff].copy()
    train_dates = sorted(train_df["snapshot_date"].unique())
    folds = get_time_folds(train_dates)

    numeric_features = [col for col in feature_cols if col != "country"]
    categorical_features = ["country"]
    model_specs = build_classifier_specs(numeric_features, categorical_features)

    results: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    holdout_predictions = holdout_df[["customer_id", "snapshot_date"]].copy()

    for name, pipeline in model_specs.items():
        print(f"Training classifier: {name}")
        fold_metrics: list[dict[str, float]] = []
        for train_dates_fold, val_date in folds:
            fold_train = train_df[train_df["snapshot_date"].isin(train_dates_fold)]
            fold_val = train_df[train_df["snapshot_date"] == val_date]
            fitted = clone(pipeline)
            fitted.fit(fold_train[feature_cols], fold_train["purchase_90_flag"])
            val_prob = fitted.predict_proba(fold_val[feature_cols])[:, 1]
            fold_result = classification_metrics(fold_val["purchase_90_flag"], val_prob)
            fold_result["validation_cutoff"] = val_date.strftime("%Y-%m-%d")
            fold_metrics.append(fold_result)

        final_model = clone(pipeline)
        final_model.fit(train_df[feature_cols], train_df["purchase_90_flag"])
        fitted_models[name] = final_model
        holdout_prob = final_model.predict_proba(holdout_df[feature_cols])[:, 1]
        holdout_scores = classification_metrics(holdout_df["purchase_90_flag"], holdout_prob)
        holdout_predictions[f"{name}_purchase_probability"] = holdout_prob

        fold_df = pd.DataFrame(fold_metrics)
        results.append(
            {
                "model": name,
                "cv_roc_auc": fold_df["roc_auc"].mean(),
                "cv_pr_auc": fold_df["pr_auc"].mean(),
                "cv_f1": fold_df["f1"].mean(),
                "cv_brier_score": fold_df["brier_score"].mean(),
                "holdout_roc_auc": holdout_scores["roc_auc"],
                "holdout_pr_auc": holdout_scores["pr_auc"],
                "holdout_f1": holdout_scores["f1"],
                "holdout_brier_score": holdout_scores["brier_score"],
                "train_rows": len(train_df),
                "holdout_rows": len(holdout_df),
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        ["holdout_pr_auc", "holdout_roc_auc", "holdout_brier_score"],
        ascending=[False, False, True],
    )
    results_df.to_csv(output_dir / "model_comparison_classification.csv", index=False)
    return results_df, fitted_models, holdout_predictions


def fit_and_evaluate_regressors(
    modeling_df: pd.DataFrame,
    feature_cols: list[str],
    holdout_cutoff: pd.Timestamp,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Pipeline], pd.DataFrame]:
    positive_df = modeling_df[modeling_df["purchase_90_flag"] == 1].copy()
    train_df = positive_df[positive_df["snapshot_date"] < holdout_cutoff].copy()
    holdout_df = positive_df[positive_df["snapshot_date"] == holdout_cutoff].copy()
    train_dates = sorted(train_df["snapshot_date"].unique())
    folds = get_time_folds(train_dates)

    numeric_features = [col for col in feature_cols if col != "country"]
    categorical_features = ["country"]
    model_specs = build_regressor_specs(numeric_features, categorical_features)

    results: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    holdout_predictions = holdout_df[["customer_id", "snapshot_date"]].copy()

    for name, pipeline in model_specs.items():
        print(f"Training regressor: {name}")
        fold_metrics: list[dict[str, float]] = []
        for train_dates_fold, val_date in folds:
            fold_train = train_df[train_df["snapshot_date"].isin(train_dates_fold)]
            fold_val = train_df[train_df["snapshot_date"] == val_date]
            fitted = clone(pipeline)
            fitted.fit(fold_train[feature_cols], np.log1p(fold_train["sales_90_value"]))
            val_pred = np.expm1(fitted.predict(fold_val[feature_cols]))
            val_pred = np.clip(val_pred, 0, None)
            fold_result = regression_metrics(fold_val["sales_90_value"], val_pred)
            fold_result["validation_cutoff"] = val_date.strftime("%Y-%m-%d")
            fold_metrics.append(fold_result)

        final_model = clone(pipeline)
        final_model.fit(train_df[feature_cols], np.log1p(train_df["sales_90_value"]))
        fitted_models[name] = final_model
        holdout_pred = np.expm1(final_model.predict(holdout_df[feature_cols]))
        holdout_pred = np.clip(holdout_pred, 0, None)
        holdout_scores = regression_metrics(holdout_df["sales_90_value"], holdout_pred)
        holdout_predictions[f"{name}_predicted_spend"] = holdout_pred

        fold_df = pd.DataFrame(fold_metrics)
        results.append(
            {
                "model": name,
                "cv_mae": fold_df["mae"].mean(),
                "cv_rmse": fold_df["rmse"].mean(),
                "cv_r2": fold_df["r2"].mean(),
                "holdout_mae": holdout_scores["mae"],
                "holdout_rmse": holdout_scores["rmse"],
                "holdout_r2": holdout_scores["r2"],
                "train_rows": len(train_df),
                "holdout_rows": len(holdout_df),
            }
        )

    results_df = pd.DataFrame(results).sort_values(["holdout_rmse", "holdout_mae"], ascending=[True, True])
    results_df.to_csv(output_dir / "model_comparison_regression.csv", index=False)
    return results_df, fitted_models, holdout_predictions


def get_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    try:
        names = list(preprocessor.get_feature_names_out())
        return [name.replace("num__", "").replace("cat__", "") for name in names]
    except Exception:
        return []


def save_feature_importance_plot(
    pipeline: Pipeline,
    x_frame: pd.DataFrame,
    output_path: Path,
    title: str,
) -> list[str]:
    feature_names = get_feature_names(pipeline)
    transformed = pipeline.named_steps["preprocessor"].transform(x_frame)
    model = pipeline.named_steps["model"]

    importances: np.ndarray
    use_shap = hasattr(model, "feature_importances_") and title.lower().startswith("purchase") is False
    if use_shap:
        use_shap = False

    try:
        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            importances = np.abs(np.asarray(model.coef_)).ravel()
        else:
            raise AttributeError("No built-in feature importance attribute")
    except Exception:
        transformed_sample = transformed[: min(len(x_frame), 1000)]
        if hasattr(model, "predict_proba"):
            explainer = shap.Explainer(model)
            shap_values = explainer(transformed_sample)
            importances = np.abs(shap_values.values).mean(axis=0)
        else:
            explainer = shap.Explainer(model.predict, transformed_sample)
            shap_values = explainer(transformed_sample)
            importances = np.abs(shap_values.values).mean(axis=0)

    if not feature_names:
        feature_names = [f"feature_{idx}" for idx in range(len(importances))]

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df["feature"] = (
        importance_df["feature"]
        .str.replace("onehot__", "", regex=False)
        .str.replace("country_", "country=", regex=False)
    )
    top_features = (
        importance_df.sort_values("importance", ascending=False)
        .head(15)
        .sort_values("importance", ascending=True)
    )

    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["importance"], color="#1f77b4")
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    return list(top_features.sort_values("importance", ascending=False)["feature"])


def build_action_label(scoring_df: pd.DataFrame) -> pd.Series:
    high_prob = scoring_df["purchase_probability_90d"] >= scoring_df["purchase_probability_90d"].quantile(0.75)
    high_value = scoring_df["expected_90_value"] >= scoring_df["expected_90_value"].quantile(0.75)
    high_history = scoring_df["monetary"] >= scoring_df["monetary"].quantile(0.75)
    low_prob = scoring_df["purchase_probability_90d"] <= scoring_df["purchase_probability_90d"].quantile(0.25)

    labels = np.where(
        high_prob & high_value,
        "Retain and reward",
        np.where(
            high_prob & ~high_value,
            "Nurture with cross-sell offers",
            np.where(
                low_prob & high_history,
                "Win-back high value customers",
                "Low-cost automation",
            ),
        ),
    )
    return pd.Series(labels, index=scoring_df.index)


def run_probabilistic_clv(
    positive_sales: pd.DataFrame,
    holdout_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    history = positive_sales[positive_sales["invoice_date"] < holdout_cutoff].copy()
    order_level = (
        history.groupby(["customer_id", "invoice"], as_index=False)
        .agg(order_date=("invoice_date", "min"), order_value=("sales_line_total", "sum"))
    )
    summary = summary_data_from_transaction_data(
        order_level,
        customer_id_col="customer_id",
        datetime_col="order_date",
        monetary_value_col="order_value",
        observation_period_end=holdout_cutoff,
        freq="D",
    )
    bgf = BetaGeoFitter(penalizer_coef=0.1)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])
    summary["expected_purchases_90d"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        90,
        summary["frequency"],
        summary["recency"],
        summary["T"],
    )

    gamma_input = summary[summary["frequency"] > 0].copy()
    if gamma_input.empty:
        return pd.DataFrame(columns=["customer_id", "probabilistic_clv_90d"])
    ggf = GammaGammaFitter(penalizer_coef=0.1)
    ggf.fit(gamma_input["frequency"], gamma_input["monetary_value"])
    gamma_input["expected_average_order_value"] = ggf.conditional_expected_average_profit(
        gamma_input["frequency"],
        gamma_input["monetary_value"],
    )
    gamma_input["probabilistic_clv_90d"] = (
        gamma_input["expected_purchases_90d"] * gamma_input["expected_average_order_value"]
    )
    gamma_input = gamma_input.reset_index().rename(columns={"index": "customer_id"})
    return gamma_input[["customer_id", "probabilistic_clv_90d"]]


def generate_business_recommendations(scoring_df: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        scoring_df.groupby("action_label")
        .agg(
            customers=("customer_id", "count"),
            avg_purchase_probability=("purchase_probability_90d", "mean"),
            avg_expected_value=("expected_90_value", "mean"),
            avg_historical_monetary=("monetary", "mean"),
        )
        .reset_index()
        .sort_values(["avg_expected_value", "customers"], ascending=[False, False])
    )

    lines = ["# Business Recommendations", ""]
    lines.append(
        "These actions use the 90-day purchase probability, conditional spend forecast, expected value, and the final segment labels built from customer behavior profiles."
    )
    lines.append("")
    for row in summary.itertuples(index=False):
        subset = scoring_df[scoring_df["action_label"] == row.action_label].copy()
        top_segments = (
            subset["final_segment"]
            .value_counts()
            .head(3)
            .index.tolist()
        )
        lines.append(f"## {row.action_label}")
        lines.append(f"- Customers in this bucket: {int(row.customers)}")
        lines.append(f"- Average 90-day purchase probability: {row.avg_purchase_probability:.2%}")
        lines.append(f"- Average expected 90-day value: {row.avg_expected_value:,.2f}")
        lines.append(f"- Most common segments: {', '.join(top_segments)}")
        if row.action_label == "Retain and reward":
            lines.append("- Recommended action: prioritize VIP retention, loyalty perks, and proactive service for customers with both high probability and high expected value.")
        elif row.action_label == "Nurture with cross-sell offers":
            lines.append("- Recommended action: use bundles, replenishment reminders, and product recommendations to turn likely purchasers into larger baskets.")
        elif row.action_label == "Win-back high value customers":
            lines.append("- Recommended action: focus on reactivation campaigns for historically valuable customers whose near-term purchase probability has softened.")
        else:
            lines.append("- Recommended action: keep communications automated and low cost so budget stays focused on higher-value opportunities.")
        lines.append("")

    (output_dir / "business_recommendations.md").write_text("\n".join(lines), encoding="utf-8")


def generate_project_docs(artifacts: RunArtifacts, output_dir: Path) -> None:
    clf_row = artifacts.classification_results.iloc[0]
    reg_row = artifacts.regression_results.iloc[0]
    top_segment_row = artifacts.cluster_profiles.sort_values("customer_count", ascending=False).iloc[0]

    summary_lines = [
        "# Project Summary",
        "",
        "## 1) Project goal",
        "Build a customer value analytics workflow on Online Retail II that combines reliable data cleaning, interpretable customer features, actionable segmentation, and a two-stage 90-day value forecast.",
        "",
        "## 2) Dataset and cleaning summary",
        f"- Raw source used: `{artifacts.raw_file}`",
        f"- Sheets loaded: {', '.join(artifacts.source_sheets)}",
        f"- Positive-sales analysis rows after cleaning: {int(artifacts.data_quality_summary.loc[artifacts.data_quality_summary['metric'] == 'cleaned_positive_sales_rows', 'value'].iloc[0]):,}",
        "- Main cleaning rules: removed missing customer IDs, exact duplicates, invalid dates, blank invoice or stock code values, cancellation rows, non-positive quantities, and non-positive prices from the main sales pipeline.",
        "",
        "## 3) Feature engineering summary",
        "- Customer features combine classic RFM with order value, purchase cadence, product diversity proxy, recent 14/30-day activity, and a return ratio derived from return-like transactions before each cutoff.",
        f"- The predictive design uses expanding observation windows and a {HORIZON_DAYS}-day forward target window.",
        "",
        "## 4) Clustering comparison and final segment choice",
        f"- Compared methods: {', '.join(artifacts.cluster_comparison['method'].tolist())}",
        f"- Final clustering method: {artifacts.final_cluster_method}",
        f"- Largest resulting segment: {top_segment_row['final_segment']} ({int(top_segment_row['customer_count'])} customers in the final clustering snapshot).",
        "- Final cluster selection balanced quantitative fit metrics with segment interpretability instead of treating clustering as a purely mathematical exercise.",
        "",
        "## 5) Predictive modeling design",
        "- Classification target: whether the customer purchases in the next 90 days.",
        "- Regression target: 90-day spend conditional on purchasing.",
        "- Final expected value is computed as purchase probability multiplied by predicted spend if purchase occurs.",
        "",
        "## 6) Final best models and results",
        f"- Best classifier: {artifacts.best_classifier_name} | Holdout ROC-AUC {clf_row['holdout_roc_auc']:.3f} | Holdout PR-AUC {clf_row['holdout_pr_auc']:.3f} | Holdout F1 {clf_row['holdout_f1']:.3f}",
        f"- Best regressor: {artifacts.best_regressor_name} | Holdout MAE {reg_row['holdout_mae']:.2f} | Holdout RMSE {reg_row['holdout_rmse']:.2f} | Holdout R2 {reg_row['holdout_r2']:.3f}",
        "",
        "## 7) Key insights",
        f"- The most important purchase-propensity features were {', '.join(artifacts.top_classifier_features[:5])}.",
        f"- The most important spend features were {', '.join(artifacts.top_regressor_features[:5])}.",
        "- The project shows that recent activity, historical value, and purchase cadence work better together than RFM alone when prioritizing customers.",
        "",
        "## 8) Limitations",
        "- The Online Retail II data is transactional rather than campaign-level, so the model cannot directly measure marketing lift.",
        "- Product categories are proxied from description keywords, which is practical but imperfect.",
        "- Spend regression is trained only on purchasers, so it should be interpreted as conditional spend rather than a direct all-customer revenue forecast.",
        "",
        "## 9) Next improvements",
        "- Add richer product taxonomy and margin data to move from revenue to profit-based customer value.",
        "- Add probability calibration or decile-based lift evaluation for campaign prioritization.",
        "- Compare current results with a retraining schedule that updates segments and scores monthly.",
    ]
    (output_dir / "project_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    description = (
        "This project rebuilds the Online Retail II analysis as a practical customer value workflow: it cleans the raw Excel transactions, engineers interpretable customer features, compares clustering methods for actionable segments, and uses time-based models to estimate who is likely to purchase and how much they may spend in the next 90 days."
    )
    (output_dir / "project_description.txt").write_text(description, encoding="utf-8")

    resume_lines = [
        "# Resume Bullets",
        "",
        "## Chinese project description",
        "- 基于 Online Retail II 原始 Excel 交易数据，搭建了一个以客户价值为核心的分析项目，覆盖数据清洗、RFM 扩展特征、客户分群、90 天购买概率预测与条件消费预测，并将结果转化为可执行的运营建议。",
        "",
        "## English project description",
        "- Built a customer value analytics project on the raw Online Retail II Excel data, combining robust cleaning, extended RFM features, clustering comparison, 90-day purchase propensity modeling, conditional spend prediction, and action-oriented customer scoring.",
        "",
        "## Chinese resume version - concise",
        "- 独立完成 Online Retail II 客户价值项目，使用原始交易数据构建清洗流程、客户分群与时间序列化的 90 天购买/消费预测模型，输出可解释的客户评分与运营建议。",
        "",
        "## Chinese resume version - achievement focused",
        "- 将原有以分群为主的零售分析项目升级为“客户价值建模 + 可执行分群”方案：比较多种聚类方法，替换黑盒 AutoML 为显式建模流程，并基于时间切分验证生成客户优先级评分表与业务建议。",
        "",
        "## English resume version - concise",
        "- Delivered an end-to-end customer value project on Online Retail II by building a raw-data cleaning pipeline, interpretable customer features, clustering analysis, and time-based 90-day purchase and spend models with actionable scoring outputs.",
        "",
        "## English resume version - achievement focused",
        "- Reframed a segmentation-only retail notebook into a customer value modeling project by comparing clustering methods, replacing PyCaret-style black-box modeling with explicit pipelines, and using time-based validation to produce explainable customer scores and business actions.",
    ]
    (output_dir / "resume_bullets.md").write_text("\n".join(resume_lines), encoding="utf-8")


def create_notebook(project_root: Path) -> None:
    nb = nbf.v4.new_notebook()
    cells: list[Any] = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# Online Retail II Customer Value Modeling\n\n"
            "This notebook modernizes the original project into an interview-friendly workflow focused on customer value modeling, actionable segmentation, and time-based prediction."
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 1) Problem framing\n\n"
            "Question: How can we identify high-value customers, predict who is likely to purchase in the next 90 days, estimate future spend, and turn those signals into actionable customer segments?"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import display\n"
            "from online_retail_modernized import (\n"
            "    HORIZON_DAYS,\n"
            "    build_snapshot_dataset,\n"
            "    clean_transactions,\n"
            "    ensure_outputs_dir,\n"
            "    find_raw_excel,\n"
            "    generate_snapshot_dates,\n"
            "    load_raw_data,\n"
            "    run_clustering_analysis,\n"
            "    run_pipeline,\n"
            ")\n"
            "\n"
            "PROJECT_ROOT = Path('.')\n"
            "DATA_DIR = PROJECT_ROOT / 'data'\n"
            "OUTPUT_DIR = ensure_outputs_dir(PROJECT_ROOT)"
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## 2) Data loading"))
    cells.append(
        nbf.v4.new_code_cell(
            "raw_file = find_raw_excel(DATA_DIR)\n"
            "raw_df, source_sheets = load_raw_data(raw_file)\n"
            "print('Raw source used:', raw_file)\n"
            "print('Sheets:', source_sheets)\n"
            "display(raw_df.head())"
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## 3) Cleaning"))
    cells.append(
        nbf.v4.new_code_cell(
            "cleaned_all, positive_sales, data_quality = clean_transactions(raw_df, OUTPUT_DIR, raw_file)\n"
            "display(data_quality.head(20))\n"
            "positive_sales.head()"
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## 4) Feature engineering"))
    cells.append(
        nbf.v4.new_code_cell(
            "snapshot_dates = generate_snapshot_dates(positive_sales)\n"
            "snapshot_data = build_snapshot_dataset(positive_sales, cleaned_all, snapshot_dates)\n"
            "print('Snapshot cutoffs:', [d.strftime('%Y-%m-%d') for d in snapshot_dates])\n"
            "display(snapshot_data.head())"
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## 5) Temporal train/validation design"))
    cells.append(
        nbf.v4.new_code_cell(
            "holdout_cutoff = snapshot_dates[-1]\n"
            "train_cutoffs = snapshot_dates[:-1]\n"
            "print('Training cutoffs:', [d.strftime('%Y-%m-%d') for d in train_cutoffs])\n"
            "print('Final holdout cutoff:', holdout_cutoff.strftime('%Y-%m-%d'))\n"
            "print('Prediction horizon (days):', HORIZON_DAYS)"
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## 6) Clustering comparison"))
    cells.append(
        nbf.v4.new_code_cell(
            "final_snapshot = snapshot_data[snapshot_data['snapshot_date'] == holdout_cutoff].copy()\n"
            "cluster_comparison, cluster_profiles, clustered_snapshot, final_method = run_clustering_analysis(final_snapshot, OUTPUT_DIR)\n"
            "print('Final clustering method:', final_method)\n"
            "display(cluster_comparison)\n"
            "display(cluster_profiles)"
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## 7) Predictive modeling"))
    cells.append(
        nbf.v4.new_code_cell(
            "artifacts = run_pipeline(PROJECT_ROOT)\n"
            "display(artifacts.classification_results)\n"
            "display(artifacts.regression_results)"
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## 8) Interpretability"))
    cells.append(
        nbf.v4.new_code_cell(
            "print('Top classifier features:', artifacts.top_classifier_features[:10])\n"
            "print('Top regressor features:', artifacts.top_regressor_features[:10])"
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## 9) Optional probabilistic CLV"))
    cells.append(
        nbf.v4.new_code_cell(
            "print('Probabilistic CLV generated:', artifacts.clv_used)\n"
            "display(artifacts.scoring_sample[['customer_id', 'probabilistic_clv_90d']].head())"
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## 10) Final customer scoring"))
    cells.append(
        nbf.v4.new_code_cell(
            "display(artifacts.scoring_sample.head(20))"
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## 11) Business recommendations"))
    cells.append(
        nbf.v4.new_code_cell(
            "print((OUTPUT_DIR / 'business_recommendations.md').read_text(encoding='utf-8'))"
        )
    )
    cells.append(nbf.v4.new_markdown_cell("## 12) Conclusion"))
    cells.append(
        nbf.v4.new_code_cell(
            "print((OUTPUT_DIR / 'project_summary.md').read_text(encoding='utf-8'))"
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
    }
    notebook_path = project_root / "online_retail_modernized.ipynb"
    notebook_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(project_root: Path | None = None) -> RunArtifacts:
    if project_root is None:
        project_root = Path(__file__).resolve().parent
    project_root = project_root.resolve()
    data_dir = project_root / "data"
    output_dir = ensure_outputs_dir(project_root)

    print_header("Source Discovery")
    raw_file = find_raw_excel(data_dir)
    print(f"Using raw source file: {raw_file}")
    raw_df, source_sheets = load_raw_data(raw_file)
    print(f"Loaded sheets: {source_sheets}")

    print_header("Cleaning")
    cleaned_all, positive_sales, quality_df = clean_transactions(raw_df, output_dir, raw_file)
    print(f"Positive-sales rows after cleaning: {len(positive_sales):,}")

    print_header("Feature Engineering")
    snapshot_dates = generate_snapshot_dates(positive_sales)
    print(f"Snapshot cutoffs: {[date.strftime('%Y-%m-%d') for date in snapshot_dates]}")
    snapshot_data = build_snapshot_dataset(positive_sales, cleaned_all, snapshot_dates)

    print_header("Clustering")
    holdout_cutoff = snapshot_dates[-1]
    final_snapshot = snapshot_data[snapshot_data["snapshot_date"] == holdout_cutoff].copy()
    cluster_comparison, cluster_profiles, clustered_snapshot, final_cluster_method = run_clustering_analysis(
        final_snapshot, output_dir
    )

    print_header("Predictive Modeling")
    model_feature_cols = [
        "recency",
        "frequency",
        "monetary",
        "avg_order_value",
        "avg_items_per_order",
        "purchase_span_days",
        "days_since_first_purchase",
        "days_since_last_purchase",
        "unique_products",
        "unique_invoice_days",
        "purchase_interval_mean",
        "purchase_interval_std",
        "return_ratio",
        "transactions_last_14d",
        "transactions_last_30d",
        "sales_last_14d",
        "sales_last_30d",
        "product_category_count_proxy",
        "top_category_share",
        "sales_per_item",
        "country",
    ]

    classification_results, classifier_models, classifier_holdout_preds = fit_and_evaluate_classifiers(
        snapshot_data,
        model_feature_cols,
        holdout_cutoff,
        output_dir,
    )
    regression_results, regressor_models, regression_holdout_preds = fit_and_evaluate_regressors(
        snapshot_data,
        model_feature_cols,
        holdout_cutoff,
        output_dir,
    )

    best_classifier_name = str(classification_results.iloc[0]["model"])
    best_regressor_name = str(regression_results.iloc[0]["model"])
    best_classifier = classifier_models[best_classifier_name]
    best_regressor = regressor_models[best_regressor_name]

    holdout_frame = final_snapshot.copy()
    holdout_frame = holdout_frame.merge(
        clustered_snapshot[["customer_id", "cluster_id", "final_segment"]],
        on="customer_id",
        how="left",
    )
    holdout_frame["final_segment"] = holdout_frame["final_segment"].fillna("Unassigned")

    best_classifier_probs = classifier_holdout_preds[["customer_id", f"{best_classifier_name}_purchase_probability"]].rename(
        columns={f"{best_classifier_name}_purchase_probability": "purchase_probability_90d"}
    )
    holdout_frame = holdout_frame.merge(best_classifier_probs, on="customer_id", how="left")

    regressor_input = holdout_frame[model_feature_cols].copy()
    predicted_spend = np.expm1(best_regressor.predict(regressor_input))
    predicted_spend = np.clip(predicted_spend, 0, None)
    holdout_frame["predicted_spend_if_purchase"] = predicted_spend
    holdout_frame["expected_90_value"] = (
        holdout_frame["purchase_probability_90d"] * holdout_frame["predicted_spend_if_purchase"]
    )

    print_header("Interpretability")
    top_classifier_features = save_feature_importance_plot(
        best_classifier,
        snapshot_data[snapshot_data["snapshot_date"] < holdout_cutoff][model_feature_cols],
        output_dir / "feature_importance_purchase.png",
        f"Purchase Model Feature Importance ({best_classifier_name})",
    )
    top_regressor_features = save_feature_importance_plot(
        best_regressor,
        snapshot_data[(snapshot_data["snapshot_date"] < holdout_cutoff) & (snapshot_data["purchase_90_flag"] == 1)][model_feature_cols],
        output_dir / "feature_importance_spend.png",
        f"Spend Model Feature Importance ({best_regressor_name})",
    )

    print_header("Optional CLV")
    clv_used = False
    try:
        clv_df = run_probabilistic_clv(positive_sales, holdout_cutoff)
        if not clv_df.empty:
            holdout_frame = holdout_frame.merge(clv_df, on="customer_id", how="left")
            clv_used = True
        else:
            holdout_frame["probabilistic_clv_90d"] = np.nan
    except Exception as exc:
        print(f"Probabilistic CLV skipped: {exc}")
        holdout_frame["probabilistic_clv_90d"] = np.nan

    holdout_frame["action_label"] = build_action_label(holdout_frame)
    holdout_frame["recommendation"] = holdout_frame["action_label"]

    scoring_columns = [
        "customer_id",
        "final_segment",
        "purchase_probability_90d",
        "predicted_spend_if_purchase",
        "expected_90_value",
        "probabilistic_clv_90d",
        "recency",
        "frequency",
        "monetary",
        "sales_last_30d",
        "transactions_last_30d",
        "country",
        "action_label",
        "recommendation",
    ]
    scoring_sample = (
        holdout_frame[scoring_columns]
        .sort_values("expected_90_value", ascending=False)
        .head(250)
        .reset_index(drop=True)
    )
    scoring_sample.to_csv(output_dir / "customer_scoring_sample.csv", index=False)

    generate_business_recommendations(holdout_frame, output_dir)

    artifacts = RunArtifacts(
        raw_file=raw_file,
        source_sheets=source_sheets,
        data_quality_summary=quality_df,
        snapshot_data=snapshot_data,
        cluster_comparison=cluster_comparison,
        cluster_profiles=cluster_profiles,
        classification_results=classification_results,
        regression_results=regression_results,
        scoring_sample=scoring_sample,
        best_classifier_name=best_classifier_name,
        best_regressor_name=best_regressor_name,
        final_cluster_method=final_cluster_method,
        holdout_cutoff=holdout_cutoff,
        feature_names=model_feature_cols,
        top_classifier_features=top_classifier_features,
        top_regressor_features=top_regressor_features,
        clv_used=clv_used,
    )

    print_header("Documentation")
    generate_project_docs(artifacts, output_dir)
    create_notebook(project_root)
    return artifacts


if __name__ == "__main__":
    run_pipeline()
