from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .config import PipelineConfig


def simulate_inventory_only(
    flow_subset: pd.DataFrame,
    current_inventory_ton: float,
    candidate_qty_ton: float,
    arrival_month_idx: int,
) -> pd.DataFrame:
    """gate용 간이 재고 경로.

    주의
    ----
    - ending inventory 뿐 아니라 receipt 직후 inventory도 같이 본다.
    - warehouse gate는 ending만 보면 과대수량을 놓칠 수 있다.
    """
    detail_rows = []
    begin_inv = float(current_inventory_ton)

    for _, row in flow_subset.sort_values("month_idx").iterrows():
        receipt_candidate = candidate_qty_ton if int(row["month_idx"]) == int(arrival_month_idx) else 0.0
        receipt_open_po = float(row["open_po_ton"])
        inventory_after_receipts = begin_inv + receipt_open_po + receipt_candidate
        demand = float(row["demand_ton"])
        shortage = max(0.0, demand - inventory_after_receipts)
        ending_inv = max(0.0, inventory_after_receipts - demand)

        detail_rows.append({
            "month_idx": int(row["month_idx"]),
            "begin_inv_ton": begin_inv,
            "receipt_open_po_ton": receipt_open_po,
            "receipt_candidate_ton": receipt_candidate,
            "inventory_after_receipts_ton": inventory_after_receipts,
            "demand_ton": demand,
            "shortage_ton": shortage,
            "ending_inv_ton": ending_inv,
        })
        begin_inv = ending_inv

    return pd.DataFrame(detail_rows)


def apply_operating_gate(
    decision_master_df: pd.DataFrame,
    baseline_flow_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    decision_map = decision_master_df.set_index("decision_id")
    results: List[Dict] = []

    for _, cand in candidate_df.iterrows():
        decision_id = cand["decision_id"]
        row = decision_map.loc[decision_id]
        flow_subset = baseline_flow_df[baseline_flow_df["decision_id"] == decision_id].copy()

        candidate_qty_ton = float(cand["candidate_qty_ton"])
        arrival_month_idx = int(cand["candidate_arrival_month_idx"])

        moq_ton = float(row.get("moq_ton", cfg.moq_ton))
        lot_multiple_ton = float(row.get("lot_multiple_ton", cfg.lot_multiple_ton))
        warehouse_capacity_ton = float(row.get("warehouse_capacity_ton", cfg.warehouse_capacity_ton))

        inventory_path = simulate_inventory_only(
            flow_subset=flow_subset,
            current_inventory_ton=float(row.get("current_inventory_ton", 0.0)),
            candidate_qty_ton=candidate_qty_ton,
            arrival_month_idx=arrival_month_idx,
        )

        moq_pass = (candidate_qty_ton == 0.0) or (candidate_qty_ton >= moq_ton)
        lot_pass = (candidate_qty_ton == 0.0) or np.isclose(candidate_qty_ton % lot_multiple_ton, 0.0)

        # ending inventory가 아니라 receipt 직후 재고가 cap을 넘는지 본다.
        warehouse_pass = inventory_path["inventory_after_receipts_ton"].max() <= warehouse_capacity_ton

        wc_pressure = float(row.get("working_capital_pressure_score", 0.0))
        if candidate_qty_ton == 0.0:
            wc_gate = "pass"
        elif wc_pressure >= cfg.wc_pressure_block_threshold:
            wc_gate = "blocked"
        elif wc_pressure >= cfg.wc_pressure_conditional_threshold:
            wc_gate = "conditional"
        else:
            wc_gate = "pass"

        first_shortage_month_idx = row.get("a_first_shortage_month_idx", np.nan)
        if candidate_qty_ton == 0.0:
            arrival_gate = "pass"
        elif pd.isna(first_shortage_month_idx):
            arrival_gate = "pass"
        elif arrival_month_idx > int(first_shortage_month_idx):
            arrival_gate = "blocked"
        elif arrival_month_idx == int(first_shortage_month_idx):
            arrival_gate = "conditional"
        else:
            arrival_gate = "pass"

        hard_fail_reasons = []
        if not moq_pass:
            hard_fail_reasons.append("MOQ gate fail")
        if not lot_pass:
            hard_fail_reasons.append("lot multiple gate fail")
        if not warehouse_pass:
            hard_fail_reasons.append("warehouse gate fail")
        if arrival_gate == "blocked":
            hard_fail_reasons.append("arrival timing gate fail")
        if wc_gate == "blocked":
            hard_fail_reasons.append("working capital gate fail")

        soft_warnings = []
        if wc_gate == "conditional":
            soft_warnings.append("working capital pressure high")
        if arrival_gate == "conditional":
            soft_warnings.append("arrival timing tight")

        if hard_fail_reasons:
            candidate_status = "blocked"
        elif soft_warnings:
            candidate_status = "conditional"
        else:
            candidate_status = "feasible"

        results.append({
            **cand.to_dict(),
            "moq_gate_pass": int(moq_pass),
            "lot_multiple_gate_pass": int(lot_pass),
            "warehouse_gate_pass": int(warehouse_pass),
            "working_capital_gate_result": wc_gate,
            "arrival_timing_gate_result": arrival_gate,
            "projected_max_inventory_after_receipts_ton_base": float(inventory_path["inventory_after_receipts_ton"].max()),
            "projected_max_end_inv_ton_base": float(inventory_path["ending_inv_ton"].max()),
            "projected_total_shortage_ton_base": float(inventory_path["shortage_ton"].sum()),
            "candidate_status": candidate_status,
            "hard_fail_reason": "; ".join(hard_fail_reasons),
            "soft_warning_reason": "; ".join(soft_warnings),
        })

    return pd.DataFrame(results)
