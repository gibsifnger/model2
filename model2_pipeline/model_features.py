from __future__ import annotations

import pandas as pd

from .config import PipelineConfig


def build_model_feature_frame(decision_master_df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """HGB용 feature frame 생성.

    목적
    ----
    - raw/helper/wide 컬럼 중 scorer에 바로 넣을 숫자 feature만 추린다.
    - train / inference에서 같은 feature 집합을 쓰도록 고정한다.
    """
    df = decision_master_df.copy()

    df["usage_horizon_total_ton"] = sum(df[f"usage_m{i}_ton"] for i in range(1, cfg.horizon_months + 1))
    df["open_po_horizon_total_ton"] = sum(df[f"open_po_m{i}_ton"] for i in range(1, cfg.horizon_months + 1))
    df["open_po_coverage_ratio"] = df["open_po_horizon_total_ton"] / df["usage_horizon_total_ton"].clip(lower=1.0)
    df["cost_path_peak_minus_now"] = df[[f"expected_landed_cost_m{i}_per_ton" for i in range(1, cfg.horizon_months + 1)]].max(axis=1) - df[cfg.current_landed_cost_col]
    df["cost_path_mean_vs_now_pct"] = (
        df[[f"expected_landed_cost_m{i}_per_ton" for i in range(1, cfg.horizon_months + 1)]].mean(axis=1) / df[cfg.current_landed_cost_col].clip(lower=1.0)
    ) - 1.0

    feature_cols = [
        "current_inventory_ton",
        cfg.current_landed_cost_col,
        "global_raw_sugar_price",
        "usdkrw",
        "freight_index",
        "sugar_ret_1m",
        "usdkrw_ret_1m",
        "freight_ret_1m",
        "sugar_ret_3m",
        "usdkrw_ret_3m",
        "freight_ret_3m",
        "working_capital_pressure_score",
        "a_min_end_inv_ton",
        "a_min_cover_months",
        "a_emergency_buy_needed_flag",
        "a_first_shortage_month_idx",
        "b_peak_cost_vs_now_pct",
        "b_forced_buy_flag",
        "b_forced_buy_cost_vs_now_pct",
        "b_emergency_premium_score",
        "b_high_cost_month_count",
        "baseline_total_shortage_ton",
        "max_cum_gap_arrival_ton",
        "required_buy_qty_arrival_ton",
        "required_buy_qty_first_shortage_ton",
        "usage_horizon_total_ton",
        "open_po_horizon_total_ton",
        "open_po_coverage_ratio",
        "cost_path_peak_minus_now",
        "cost_path_mean_vs_now_pct",
    ]

    X = df[feature_cols].copy().fillna(0.0)
    return X
