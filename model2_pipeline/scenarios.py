from __future__ import annotations

from typing import Tuple

import pandas as pd

from .config import PipelineConfig


def shift_month_quantity(flow_df: pd.DataFrame, qty_col: str, delay_months: int, horizon_months: int) -> pd.Series:
    shifted = {month_idx: 0.0 for month_idx in range(1, horizon_months + 1)}
    for _, row in flow_df.iterrows():
        new_month = int(row["month_idx"]) + int(delay_months)
        if 1 <= new_month <= horizon_months:
            shifted[new_month] += float(row[qty_col])
    return pd.Series([shifted[int(m)] for m in flow_df["month_idx"]], index=flow_df.index)


def apply_scenario(flow_subset: pd.DataFrame, candidate_row: pd.Series, scenario_name: str, cfg: PipelineConfig) -> Tuple[pd.DataFrame, int]:
    scenario = cfg.scenario_library[scenario_name]
    flow = flow_subset.sort_values("month_idx").copy()

    flow["scenario_name"] = scenario_name
    flow["scenario_demand_ton"] = flow["demand_ton"] * scenario["demand_mult"]
    flow["scenario_unit_cost_per_ton"] = flow["expected_unit_cost_per_ton"] * scenario["cost_mult"]
    flow["scenario_open_po_ton"] = shift_month_quantity(
        flow_df=flow[["month_idx", "open_po_ton"]],
        qty_col="open_po_ton",
        delay_months=int(scenario["open_po_delay_months"]),
        horizon_months=cfg.horizon_months,
    ).values

    candidate_arrival_month_idx = int(candidate_row["candidate_arrival_month_idx"])
    if candidate_arrival_month_idx > 0:
        candidate_arrival_month_idx += int(scenario["candidate_delay_months"])

    return flow, candidate_arrival_month_idx
