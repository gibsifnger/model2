from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .config import PipelineConfig


def _get_value(row: pd.Series, key: str, default: float) -> float:
    value = row.get(key, default)
    if pd.isna(value):
        return float(default)
    return float(value)


def build_baseline_flow_df(decision_master_df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """decision row를 horizon long flow로 펼친다.

    중심축
    ------
    - row unit은 decision master에서는 `원재료-월-의사결정시점`
    - baseline flow에서는 `원재료-월-의사결정시점-미래월`
    - helper와 simulation이 같은 미래월 축을 보도록 만드는 연결층이다.
    """
    rows: List[Dict] = []
    for _, row in decision_master_df.iterrows():
        for month_idx in range(1, cfg.horizon_months + 1):
            rows.append({
                "decision_id": row["decision_id"],
                "decision_month": row["decision_month"],
                "material_code": row.get("material_code", cfg.material_code),
                "month_idx": month_idx,
                "demand_ton": _get_value(row, f"usage_m{month_idx}_ton", cfg.monthly_usage_base_ton),
                "open_po_ton": _get_value(row, f"open_po_m{month_idx}_ton", 0.0),
                "expected_unit_cost_per_ton": _get_value(
                    row,
                    f"expected_landed_cost_m{month_idx}_per_ton",
                    _get_value(row, cfg.current_landed_cost_col, 0.0),
                ),
            })

    return pd.DataFrame(rows).sort_values(["decision_id", "month_idx"]).reset_index(drop=True)
