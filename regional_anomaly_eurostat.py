#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end pipeline for:
  1. Downloading Eurostat regional (NUTS2) indicators for a given year.
  2. Building a single CSV of indicators.
  3. Running anomaly detection (classical + ML) on the regional data.
  4. Computing PCA for visualisation and exporting a PCA plot.

Outputs:
  - regional_indicators.csv
  - regional_anomaly_results.csv
  - fig_pca_iforest.pdf

Requirements (install via pip):

  pip install eurostat pandas numpy scikit-learn matplotlib

"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
import eurostat
import re

from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

# Standardize indicators
from sklearn.preprocessing import StandardScaler
# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def eurostat_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Eurostat wide format (freq, unit, geo\\TIME_PERIOD, 2000, 2001, ..., 2023)
    to long format with columns: freq, unit, geo, time, values.

    If the DataFrame already has a 'time' column, it is returned unchanged.
    """
    if "time" in df.columns:
        # Already long format
        return df

    # Identify year-like columns (e.g. "2000", "2001", ...)
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
    if not year_cols:
        raise ValueError(f"Could not find year columns in Eurostat data. Columns are: {df.columns.tolist()}")

    id_cols = [c for c in df.columns if c not in year_cols]

    df_long = df.melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name="time",
        value_name="values",
    )

    # Rename geo\\TIME_PERIOD -> geo if present
    if "geo\\TIME_PERIOD" in df_long.columns:
        df_long = df_long.rename(columns={"geo\\TIME_PERIOD": "geo"})

    # Make 'time' a string (or int later)
    df_long["time"] = df_long["time"].astype(str)

    return df_long


# ----------------------------------------------------------------------
# Eurostat download helpers
# ----------------------------------------------------------------------

def filter_nuts2_and_year(df: pd.DataFrame, year: int, time_col: str = "time") -> pd.DataFrame:
    """
    Filter Eurostat long-format DataFrame to:
      - a single reference year
      - NUTS2 regions (geo codes length == 4, e.g. RO32, DE21)

    Assumes 'geo' and 'time' columns exist.
    """
    df = df.copy()
    df[time_col] = df[time_col].astype(str)
    year_str = str(year)
    df = df[df[time_col] == year_str]

    # NUTS2: geo codes length == 4 (RO11, RO32, DE21, etc.)
    df = df[df["geo"].str.len() == 4]

    return df


def download_gdp_pc(year: int) -> pd.DataFrame:
    """
    GDP per capita in PPS at NUTS2 from nama_10r_2gdp.

    We try to keep a unit that corresponds to:
      - GDP per inhabitant in PPS, EU27=100, base year 2020.

    Eurostat unit codes can be, e.g.:
      - PPS_HAB_EU27_2020
      - PPS_EU27_2020_HAB
      - PPS_HAB

    We pick the first preferred code that is actually present.
    """
    logging.info("Downloading GDP per capita (PPS) from nama_10r_2gdp ...")
    df = eurostat.get_data_df("nama_10r_2gdp")

    # Convert wide to long (freq, unit, geo, time, values)
    df = eurostat_wide_to_long(df)

    # Inspect available units for debugging
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


def download_population_density(year: int) -> pd.DataFrame:
    """
    Approximate population density at NUTS2 by aggregating NUTS3 data.

    Steps:
      - Load demo_r_d3dens (population density, P_KM2).
      - Convert wide -> long.
      - Filter to unit = P_KM2.
      - Keep NUTS3 (length 5) and NUTS2 (length 4) codes.
      - Map NUTS3 to NUTS2 by truncating last character.
      - Aggregate to NUTS2 by simple mean (for illustration).
    """
    logging.info("Downloading population density from demo_r_d3dens ...")
    df = eurostat.get_data_df("demo_r_d3dens")
    df = eurostat_wide_to_long(df)

    # Keep only density unit
    df = df[df["unit"] == "P_KM2"]

    # Keep only valid regional codes (length 4 or 5)
    df = df[df["geo"].str.len().isin([4, 5])].copy()

    # Map NUTS3 -> NUTS2 by truncating last char if length==5
    df["nuts2"] = df["geo"].str.slice(0, 4)

    # Filter year
    df = df[df["time"] == str(year)]

    # Aggregate to NUTS2 (simple mean of densities across NUTS3 inside each NUTS2)
    df_agg = (
        df.groupby(["nuts2", "time"], as_index=False)["values"]
        .mean()
    )

    df_agg = df_agg.rename(columns={"nuts2": "geo"})
    df_agg = df_agg[["geo", "time", "values"]].rename(columns={"values": "population_density"})

    return df_agg


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
    Returns a wide DataFrame:
      region_code, year, gdp_pc_pps, unemployment_rate, population_density, tertiary_share_25_64
    """
    logging.info(f"Building regional dataset for year {year} ...")

    df_gdp = download_gdp_pc(year)
    df_unemp = download_unemployment_rate(year)
    df_dens = download_population_density(year)
    df_tert = download_tertiary_share(year)

    df = df_gdp.copy()
    for other in [df_unemp, df_dens, df_tert]:
        df = pd.merge(df, other, on=["geo", "time"], how="outer")

    df = df.rename(columns={"geo": "region_code", "time": "year"})
    df["year"] = df["year"].astype(int)

    return df


# ----------------------------------------------------------------------
# Preprocessing for anomaly detection
# ----------------------------------------------------------------------

def preprocess_indicators(df: pd.DataFrame, indicator_cols: List[str]) -> pd.DataFrame:
    """
    Handle missing values, optional transformations, and standardise indicators.

    Returns:
      df_proc: original cols + std_<indicator> columns
      scaler: fitted StandardScaler (if needed later)
    """
    df_proc = df.copy()

    # Example: log-transform skewed indicators if present
    skewed_candidates = ["gdp_pc_pps", "population_density"]
    for col in indicator_cols:
        if col in skewed_candidates:
            min_val = df_proc[col].min()
            shift = 1.0 - min_val if (pd.notna(min_val) and min_val <= 0) else 0.0
            df_proc[col] = np.log(df_proc[col] + shift)

    # Handle missing values: median imputation
    for col in indicator_cols:
        median_val = df_proc[col].median()
        df_proc[col] = df_proc[col].fillna(median_val)

    # Standardise
    scaler = StandardScaler()
    X = scaler.fit_transform(df_proc[indicator_cols].values)
    df_std = pd.DataFrame(X, columns=[f"std_{c}" for c in indicator_cols], index=df_proc.index)

    out = pd.concat([df_proc[["region_code", "year"]], df_proc[indicator_cols], df_std], axis=1)
    return out, scaler


# ----------------------------------------------------------------------
# Classical outlier detection
# ----------------------------------------------------------------------

def compute_univariate_z_outliers(df_std: pd.DataFrame, indicator_cols: List[str], threshold: float = 3.0) -> pd.Series:
    """
    Flag regions with |z| > threshold in at least one indicator.
    df_std should contain 'std_<indicator>' columns.
    Returns a boolean Series indexed by row.
    """
    std_cols = [f"std_{c}" for c in indicator_cols]
    Z = df_std[std_cols].values
    mask = np.any(np.abs(Z) > threshold, axis=1)
    return pd.Series(mask, index=df_std.index, name="flag_zscore")


def compute_mahalanobis_outliers(df_std: pd.DataFrame, indicator_cols: List[str], quantile: float = 0.99) -> pd.Series:
    """
    Compute squared Mahalanobis distance and flag top (1-quantile) fraction.
    """
    std_cols = [f"std_{c}" for c in indicator_cols]
    X = df_std[std_cols].values

    cov = EmpiricalCovariance().fit(X)
    mahal_dist = cov.mahalanobis(X)  # squared distances

    df_std["mahal_dist"] = mahal_dist
    threshold = np.quantile(mahal_dist, quantile)
    flags = mahal_dist > threshold
    return pd.Series(flags, index=df_std.index, name="flag_mahal")


# ----------------------------------------------------------------------
# ML-based anomaly detection
# ----------------------------------------------------------------------

def run_isolation_forest(df_std: pd.DataFrame, indicator_cols: List[str], contamination: float = 0.05) -> pd.Series:
    """
    Isolation Forest anomaly detection.
    Returns boolean Series: True if anomaly.
    """
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
    """
    Local Outlier Factor anomaly detection.
    Returns boolean Series: True if anomaly.
    """
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
    """
    One-Class SVM anomaly detection.
    Returns boolean Series: True if anomaly.
    """
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


# ----------------------------------------------------------------------
# PCA for visualisation
# ----------------------------------------------------------------------

def compute_pca_projection(df_std: pd.DataFrame, indicator_cols: List[str], n_components: int = 2) -> pd.DataFrame:
    """
    Compute PCA projection for visualisation.
    """
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

    # Ensure we have a boolean mask
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


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def main():
    YEAR = 2022  # choose a reference year with decent coverage

    # 1. Download Eurostat data and build regional_indicators.csv
    logging.info("Downloading Eurostat data and building regional_indicators.csv ...")
    df_regions = build_regional_dataset(YEAR)

    logging.info(f"Dataset shape before dropping all-NaN rows: {df_regions.shape}")
    indicator_cols = [c for c in df_regions.columns if c not in ["region_code", "year"]]

    # Drop rows where all indicators are NaN
    all_nan_mask = df_regions[indicator_cols].isna().all(axis=1)
    df_regions = df_regions[~all_nan_mask].copy()
    logging.info(f"Dataset shape after dropping all-NaN rows: {df_regions.shape}")

    # NEW: drop indicator columns that are entirely NaN
    non_empty_indicator_cols = [c for c in indicator_cols if df_regions[c].notna().any()]
    dropped = sorted(set(indicator_cols) - set(non_empty_indicator_cols))
    indicator_cols = non_empty_indicator_cols

    if dropped:
        logging.warning(f"Dropping indicator columns with no data: {dropped}")
    logging.info(f"Using indicators: {indicator_cols}")

    df_regions = df_regions.sort_values(["region_code"])
    df_regions.to_csv("regional_indicators.csv", index=False)
    logging.info("Saved regional indicators to regional_indicators.csv")

    # 2. Preprocess and standardise indicators
    logging.info("Preprocessing indicators...")
    df_std, scaler = preprocess_indicators(df_regions, indicator_cols)

    # 3. Run anomaly detection methods
    logging.info("Running classical and ML anomaly detection...")

    flag_z = compute_univariate_z_outliers(df_std, indicator_cols, threshold=3.0)
    flag_mahal = compute_mahalanobis_outliers(df_std, indicator_cols, quantile=0.99)
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
        index=df_std.index,
    )

    # Summary in logs
    summary = df_flags.mean().sort_values(ascending=False) * 100.0
    logging.info("Percentage of regions flagged by each method:")
    for method, pct in summary.items():
        logging.info(f"  {method}: {pct:.1f}%")

    df_flags["flag_at_least_3"] = df_flags.sum(axis=1) >= 3
    logging.info(f"Regions flagged by at least 3 methods: {df_flags['flag_at_least_3'].sum()}")

    # 4. PCA for visualisation
    logging.info("Computing PCA projection...")
    df_pca = compute_pca_projection(df_std, indicator_cols, n_components=2)

    # 5. Combine everything and export
    df_out = pd.concat([df_regions.reset_index(drop=True), df_std.drop(columns=["region_code", "year"]), df_flags, df_pca], axis=1)
    df_out.to_csv("regional_anomaly_results.csv", index=False)
    logging.info("Saved anomaly results to regional_anomaly_results.csv")

    # 6. PCA plot (Isolation Forest anomalies)
    plot_pca_iforest(df_out, output_path="fig_pca_iforest.pdf")



    X = final_table[['gdp_pc_pps','unemployment_rate','tertiary_share_25_64']].copy()
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    heat_df = pd.DataFrame(X_std, columns=['GDP pc (std)', 'Unemployment (std)', 'Tertiary (std)'])
    heat_df['region_code'] = final_table['region_code']

    plt.figure(figsize=(10, 16))
    sns.heatmap(heat_df.set_index('region_code'),
                cmap='coolwarm', center=0,
                linewidths=.5, linecolor='grey')
    plt.title("Standardized Indicators for Anomalous Regions (z-scores)")
    plt.tight_layout()
    plt.savefig("heatmap_anomalies.pdf")
    plt.close()

    
if __name__ == "__main__":
    main()
