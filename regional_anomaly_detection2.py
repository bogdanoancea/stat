#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Regional anomaly detection for Eurostat NUTS2 data (year 2022).

Outputs:
  - regional_indicators.csv
  - regional_anomaly_results.csv
  - fig_pca_iforest.pdf

Requires:
  pip install eurostat pandas numpy scikit-learn matplotlib
"""

from __future__ import annotations

import logging
import re
from typing import List

import numpy as np
import pandas as pd
import eurostat

from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
# ==========================================================
# Logging
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ==========================================================
# Helpers: Eurostat reshape and filters
# ==========================================================

def eurostat_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Eurostat wide format (freq, unit, geo\\TIME_PERIOD, 2000, 2001, ..., 2023)
    to long format with columns: freq, unit, geo, time, values.

    If the DataFrame already has a 'time' column, it is returned unchanged.
    """
    if "time" in df.columns:
        return df

    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
    if not year_cols:
        raise ValueError(
            f"Could not find year columns in Eurostat data. "
            f"Columns: {df.columns.tolist()}"
        )

    id_cols = [c for c in df.columns if c not in year_cols]

    df_long = df.melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name="time",
        value_name="values",
    )

    if "geo\\TIME_PERIOD" in df_long.columns:
        df_long = df_long.rename(columns={"geo\\TIME_PERIOD": "geo"})

    df_long["time"] = df_long["time"].astype(str)
    return df_long


def filter_nuts2_and_year(df: pd.DataFrame, year: int, time_col: str = "time") -> pd.DataFrame:
    """
    Filter Eurostat long-format DataFrame to:
      - a single reference year
      - NUTS2 regions (geo codes length == 4, e.g. RO32, DE21)
    """
    df = df.copy()
    df[time_col] = df[time_col].astype(str)

    df = df[df[time_col] == str(year)]
    df = df[df["geo"].notna()]

    # Keep only NUTS2 (4-char codes)
    df["geo"] = df["geo"].astype(str)
    df = df[df["geo"].str.len() == 4]

    return df


# ==========================================================
# Download functions
# ==========================================================

def download_gdp_pc(year: int) -> pd.DataFrame:
    """
    GDP per capita in PPS at NUTS2 from nama_10r_2gdp.

    Try units:
      - PPS_HAB_EU27_2020
      - PPS_EU27_2020_HAB
      - PPS_HAB
    """
    logging.info("Downloading GDP per capita (PPS) from nama_10r_2gdp ...")
    df = eurostat.get_data_df("nama_10r_2gdp")
    df = eurostat_wide_to_long(df)

    units = sorted(df["unit"].unique().tolist())
    logging.info(f"Available units in nama_10r_2gdp: {units}")

    preferred_units = ["PPS_HAB_EU27_2020", "PPS_EU27_2020_HAB", "PPS_HAB"]
    chosen_unit = None
    for u in preferred_units:
        if u in units:
            chosen_unit = u
            break

    if chosen_unit is None:
        raise ValueError(f"No expected PPS per capita unit found. Units: {units}")

    logging.info(f"Using unit = {chosen_unit} for GDP per capita in PPS")
    df = df[df["unit"] == chosen_unit]

    df = filter_nuts2_and_year(df, year)
    df = df[["geo", "time", "values"]].rename(columns={"values": "gdp_pc_pps"})
    return df


def download_unemployment_rate(year: int) -> pd.DataFrame:
    """
    Unemployment rate 15–74 at NUTS2:
      dataset: lfst_r_lfu3rt
      filter: sex = T, age = Y15-74, unit = PC
    """
    logging.info("Downloading unemployment rate from lfst_r_lfu3rt ...")
    df = eurostat.get_data_df("lfst_r_lfu3rt")
    df = eurostat_wide_to_long(df)

    df = df[
        (df["sex"] == "T") &
        (df["age"] == "Y15-74") &
        (df["unit"] == "PC")
    ]

    df = filter_nuts2_and_year(df, year)
    df = df[["geo", "time", "values"]].rename(columns={"values": "unemployment_rate"})
    return df


def download_tertiary_share(year: int) -> pd.DataFrame:
    """
    Share of population with tertiary education 25–64 at NUTS2:
      dataset: edat_lfse_04
      filter: sex = T, age = Y25-64, isced11 = ED5-8, unit = PC
    """
    logging.info("Downloading tertiary education share from edat_lfse_04 ...")
    df = eurostat.get_data_df("edat_lfse_04")
    df = eurostat_wide_to_long(df)

    df = df[
        (df["sex"] == "T") &
        (df["age"] == "Y25-64") &
        (df["isced11"] == "ED5-8") &
        (df["unit"] == "PC")
    ]

    df = filter_nuts2_and_year(df, year)
    df = df[["geo", "time", "values"]].rename(columns={"values": "tertiary_share_25_64"})
    return df


def build_regional_dataset(year: int) -> pd.DataFrame:
    """
    Download all indicators for the specified year and merge by (geo, time).

    Returns columns:
      region_code, year, gdp_pc_pps, unemployment_rate, tertiary_share_25_64
    """
    logging.info(f"Building regional dataset for year {year} ...")

    df_gdp = download_gdp_pc(year)
    df_unemp = download_unemployment_rate(year)
    df_tert = download_tertiary_share(year)

    # Merge step-by-step
    df = df_gdp.copy()
    for other in [df_unemp, df_tert]:
        df = pd.merge(df, other, on=["geo", "time"], how="outer")

    # Rename and basic cleaning
    df = df.rename(columns={"geo": "region_code", "time": "year"})
    df["year"] = df["year"].astype(int)

    # Keep only valid NUTS2 region codes: non-null, length 4
    df = df[df["region_code"].notna()].copy()
    df["region_code"] = df["region_code"].astype(str)
    df = df[df["region_code"].str.len() == 4].copy()

    indicator_cols = ["gdp_pc_pps", "unemployment_rate", "tertiary_share_25_64"]

    # Collapse duplicates per region_code: first non-null per column
    def first_non_null(series: pd.Series):
        return series.dropna().iloc[0] if series.dropna().size > 0 else np.nan

    df = df.sort_values(["region_code", "year"])
    df = df.groupby("region_code", as_index=False).agg(
        {
            "year": "first",
            "gdp_pc_pps": first_non_null,
            "unemployment_rate": first_non_null,
            "tertiary_share_25_64": first_non_null,
        }
    )

    # Drop rows where all indicators are NaN
    all_nan_rows = df[indicator_cols].isna().all(axis=1)
    if all_nan_rows.any():
        logging.info(f"Dropping {all_nan_rows.sum()} rows with all indicators NaN")
        df = df[~all_nan_rows].copy()

    return df


# ==========================================================
# Preprocessing (no duplication of original columns)
# ==========================================================

def preprocess_indicators(df: pd.DataFrame, indicator_cols: List[str]):
    """
    Transform skewed variables, impute missing values, and standardise indicators.

    Returns:
      df_std: dataframe with std_ columns only (same index as df)
      scaler: fitted StandardScaler
    """
    # Work only on a numeric copy
    X = df[indicator_cols].copy()

    # Log-transform skewed GDP-per-capita indicator
    skewed_candidates = ["gdp_pc_pps"]
    for col in indicator_cols:
        if col in skewed_candidates and X[col].notna().any():
            min_val = X[col].min()
            shift = 1.0 - min_val if (pd.notna(min_val) and min_val <= 0) else 0.0
            X[col] = np.log(X[col] + shift)

    # Median imputation
    for col in indicator_cols:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)

    # Sanity check: no NaNs allowed now
    if X.isna().any().any():
        bad_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaNs remain in columns after imputation: {bad_cols}")

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X.values)
    df_std = pd.DataFrame(X_std, columns=[f"std_{c}" for c in indicator_cols], index=df.index)

    return df_std, scaler


# ==========================================================
# Classical outlier detection
# ==========================================================

def compute_univariate_z_outliers(df_std: pd.DataFrame, indicator_cols: List[str], threshold: float = 3.0) -> pd.Series:
    std_cols = [f"std_{c}" for c in indicator_cols]
    Z = df_std[std_cols].values
    mask = np.any(np.abs(Z) > threshold, axis=1)
    return pd.Series(mask, index=df_std.index, name="flag_zscore")


def compute_mahalanobis_outliers(df_std: pd.DataFrame, indicator_cols: List[str], quantile: float = 0.99) -> pd.Series:
    std_cols = [f"std_{c}" for c in indicator_cols]
    X = df_std[std_cols].values

    cov = EmpiricalCovariance().fit(X)
    mahal_dist = cov.mahalanobis(X)

    # We don't store mahal_dist inside df_std here; we'll attach it outside
    threshold = np.quantile(mahal_dist, quantile)
    flags = mahal_dist > threshold
    return pd.Series(flags, index=df_std.index, name="flag_mahal"), pd.Series(mahal_dist, index=df_std.index, name="mahal_dist")


# ==========================================================
# ML-based anomaly detection
# ==========================================================

def run_isolation_forest(df_std: pd.DataFrame, indicator_cols: List[str], contamination: float = 0.05) -> pd.Series:
    X = df_std[[f"std_{c}" for c in indicator_cols]].values
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X)
    labels = model.predict(X)  # -1 outlier, 1 inlier
    flags = (labels == -1)
    return pd.Series(flags, index=df_std.index, name="flag_iforest")


def run_lof(df_std: pd.DataFrame, indicator_cols: List[str], contamination: float = 0.05, n_neighbors: int = 20) -> pd.Series:
    X = df_std[[f"std_{c}" for c in indicator_cols]].values
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False,
    )
    labels = model.fit_predict(X)  # -1 outlier, 1 inlier
    flags = (labels == -1)
    return pd.Series(flags, index=df_std.index, name="flag_lof")


def run_one_class_svm(df_std: pd.DataFrame, indicator_cols: List[str], nu: float = 0.05, gamma: str = "scale") -> pd.Series:
    X = df_std[[f"std_{c}" for c in indicator_cols]].values
    model = OneClassSVM(
        nu=nu,
        kernel="rbf",
        gamma=gamma,
    )
    model.fit(X)
    labels = model.predict(X)  # -1 outlier, 1 inlier
    flags = (labels == -1)
    return pd.Series(flags, index=df_std.index, name="flag_ocsvm")


# ==========================================================
# PCA + plot
# ==========================================================

def compute_pca_projection(df_std: pd.DataFrame, indicator_cols: List[str], n_components: int = 2) -> pd.DataFrame:
    X = df_std[[f"std_{c}" for c in indicator_cols]].values
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X)
    cols = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(coords, columns=cols, index=df_std.index)
    return df_pca


def plot_pca_iforest(df_out: pd.DataFrame, output_path: str = "fig_pca_iforest.pdf"):
    if not {"PC1", "PC2", "flag_iforest"}.issubset(df_out.columns):
        logging.warning("PCA or flag_iforest not found; skipping PCA plot.")
        return

    flags = df_out["flag_iforest"].astype(bool)

    plt.figure(figsize=(8, 6))
    normal = df_out[~flags]
    anomalies = df_out[flags]

    plt.scatter(normal["PC1"], normal["PC2"], alpha=0.6, label="Normal regions")
    plt.scatter(anomalies["PC1"], anomalies["PC2"], marker="x", s=80, label="IForest anomalies")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of regional indicators (Isolation Forest anomalies)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"PCA plot saved to {output_path}")


# ==========================================================
# Main
# ==========================================================

def main():
    YEAR = 2022

    logging.info("Downloading Eurostat data and building regional_indicators.csv ...")
    df_regions = build_regional_dataset(YEAR)

    logging.info(f"Dataset shape after building & dedup: {df_regions.shape}")

    # Identify indicator columns
    indicator_cols = [c for c in df_regions.columns if c not in ["region_code", "year"]]

    # Ensure region_code clean: non-null, length 4
    df_regions = df_regions[df_regions["region_code"].notna()].copy()
    df_regions["region_code"] = df_regions["region_code"].astype(str)
    df_regions = df_regions[df_regions["region_code"].str.len() == 4].copy()

    logging.info(f"Dataset shape after cleaning region_code: {df_regions.shape}")

    df_regions = df_regions.sort_values(["region_code", "year"])
    df_regions.to_csv("regional_indicators.csv", index=False)
    logging.info("Saved regional indicators to regional_indicators.csv")

    logging.info("Preprocessing indicators (standardisation)...")
    df_std, scaler = preprocess_indicators(df_regions, indicator_cols)

    logging.info("Running classical and ML anomaly detection...")

    flag_z = compute_univariate_z_outliers(df_std, indicator_cols, threshold=3.0)
    flag_mahal, mahal_dist = compute_mahalanobis_outliers(df_std, indicator_cols, quantile=0.99)
    flag_if = run_isolation_forest(df_std, indicator_cols, contamination=0.05)
    flag_lof = run_lof(df_std, indicator_cols, contamination=0.05, n_neighbors=20)
    flag_oc = run_one_class_svm(df_std, indicator_cols, nu=0.05, gamma="scale")

    df_flags = pd.DataFrame(
        {
            "flag_zscore": flag_z,
            "flag_mahal": flag_mahal,
            "flag_iforest": flag_if,
            "flag_lof": flag_lof,
            "flag_ocsvm": flag_oc,
        },
        index=df_regions.index,
    )

    summary = df_flags.mean().sort_values(ascending=False) * 100.0
    logging.info("Percentage of regions flagged by each method:")
    for method, pct in summary.items():
        logging.info(f"  {method}: {pct:.1f}%")

    df_flags["flag_at_least_3"] = df_flags.sum(axis=1) >= 3
    logging.info(f"Regions flagged by at least 3 methods: {df_flags['flag_at_least_3'].sum()}")

    logging.info("Computing PCA projection...")
    df_pca = compute_pca_projection(df_std, indicator_cols, n_components=2)

    # Build final output: one row per region, no duplicate columns
    df_out = pd.concat(
        [
            df_regions.reset_index(drop=True),
            df_std.reset_index(drop=True),
            mahal_dist.reset_index(drop=True),
            df_flags.reset_index(drop=True),
            df_pca.reset_index(drop=True),
        ],
        axis=1,
    )

    # Final safeguard: region_code must be valid
    df_out = df_out[df_out["region_code"].notna()].copy()
    df_out["region_code"] = df_out["region_code"].astype(str)
    df_out = df_out[df_out["region_code"].str.len() == 4].copy()

    df_out.to_csv("regional_anomaly_results.csv", index=False)
    logging.info("Saved anomaly results to regional_anomaly_results.csv")

    plot_pca_iforest(df_out, output_path="fig_pca_iforest.pdf")


    # ----------------------------------------------------------
    # Load anomaly results
    # ----------------------------------------------------------
    df = pd.read_csv("regional_anomaly_results.csv")

    # Keep only valid NUTS2 regions (length 4)
    df = df[df["region_code"].astype(str).str.len() == 4].copy()

    # Convert flags to bool
    flag_cols = ['flag_zscore','flag_mahal','flag_iforest','flag_lof','flag_ocsvm']
    for col in flag_cols:
        df[col] = df[col].astype(bool)

    # Count number of flags
    df["flag_count"] = df[flag_cols].sum(axis=1)

    # Keep only regions with >=3 flags (the true anomalies)
    anom = df[df["flag_count"] >= 3].copy()

    # Deduplicate (first occurrence of each NUTS2 region)
    anom = anom.groupby("region_code").first().reset_index()

    # ----------------------------------------------------------
    # Select the indicator columns
    # ----------------------------------------------------------
    indicators = ["gdp_pc_pps", "unemployment_rate", "tertiary_share_25_64"]

    # Extract indicator matrix
    X = anom[indicators].copy()

    # Standardize indicators
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Build heatmap DataFrame with region_code as index
    heat_df = pd.DataFrame(
        X_std,
        columns=["GDP per capita (std)", "Unemployment rate (std)", "Tertiary education (std)"],
        index=anom["region_code"]
    )

    # ----------------------------------------------------------
    # Plot heatmap
    # ----------------------------------------------------------
    plt.figure(figsize=(10, 16))
    sns.heatmap(
        heat_df,
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        linecolor="grey",
        cbar_kws={"label": "Standardized value (z-score)"}
    )

    plt.title("Standardized Indicators for Anomalous NUTS2 Regions (2022)", fontsize=14)
    plt.xlabel("Indicators")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig("heatmap_anomalies.pdf")
    plt.close()

    print("Saved heatmap to heatmap_anomalies.pdf")


if __name__ == "__main__":
    main()
