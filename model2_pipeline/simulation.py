from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .config import PipelineConfig
from .scenarios import apply_scenario


def simulate_candidate_under_scenario(
    decision_id: str,
    decision_row: pd.Series,
    flow_subset: pd.DataFrame,
    candidate_row: pd.Series,
    scenario_name: str,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    flow, scenario_arrival_month_idx = apply_scenario(flow_subset, candidate_row, scenario_name, cfg)

    begin_inv = float(decision_row.get("current_inventory_ton", 0.0))
    now_cost = float(decision_row.get(cfg.current_landed_cost_col, 0.0))
    candidate_qty_ton = float(candidate_row["candidate_qty_ton"])

    forced_buy_premium_pct = float(decision_row.get("b_forced_buy_cost_vs_now_pct", 0.08))
    forced_buy_premium_pct = max(forced_buy_premium_pct, 0.08)
    forced_buy_premium_pct *= cfg.scenario_library[scenario_name]["emergency_premium_mult"]

    rows: List[Dict] = []

    for _, month in flow.iterrows():
        month_idx = int(month["month_idx"])
        open_po_ton = float(month["scenario_open_po_ton"])
        candidate_receipt_ton = candidate_qty_ton if (scenario_arrival_month_idx == month_idx and scenario_arrival_month_idx <= cfg.horizon_months) else 0.0
        available_before_demand = begin_inv + open_po_ton + candidate_receipt_ton
        demand_ton = float(month["scenario_demand_ton"])

        shortage_ton = max(0.0, demand_ton - available_before_demand)
        ending_inventory_ton = max(0.0, available_before_demand - demand_ton)
        emergency_buy_ton = shortage_ton

        open_po_cost = open_po_ton * float(month["scenario_unit_cost_per_ton"])
        candidate_cost = candidate_receipt_ton * now_cost
        emergency_buy_cost = emergency_buy_ton * float(month["scenario_unit_cost_per_ton"]) * (1.0 + forced_buy_premium_pct)

        rows.append({
            "decision_id": decision_id,
            "candidate_name": candidate_row["candidate_name"],
            "candidate_qty_ton": candidate_qty_ton,
            "candidate_status": candidate_row["candidate_status"],
            "scenario_name": scenario_name,
            "month_idx": month_idx,
            "begin_inventory_ton": begin_inv,
            "receipt_open_po_ton": open_po_ton,
            "receipt_candidate_ton": candidate_receipt_ton,
            "available_before_demand_ton": available_before_demand,
            "demand_ton": demand_ton,
            "shortage_ton": shortage_ton,
            "ending_inventory_ton": ending_inventory_ton,
            "emergency_buy_ton": emergency_buy_ton,
            "open_po_cost": open_po_cost,
            "candidate_cost": candidate_cost,
            "emergency_buy_cost": emergency_buy_cost,
            "total_month_cost": open_po_cost + candidate_cost + emergency_buy_cost,
        })
        begin_inv = ending_inventory_ton

    return pd.DataFrame(rows)


def run_candidate_simulations(
    decision_master_df: pd.DataFrame,
    baseline_flow_df: pd.DataFrame,
    gated_candidate_df: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    decision_map = decision_master_df.set_index("decision_id")
    frames: List[pd.DataFrame] = []

    for _, candidate_row in gated_candidate_df.iterrows():
        decision_id = candidate_row["decision_id"]
        decision_row = decision_map.loc[decision_id]
        flow_subset = baseline_flow_df[baseline_flow_df["decision_id"] == decision_id].copy()

        for scenario_name in cfg.scenario_library.keys():
            sim_df = simulate_candidate_under_scenario(
                decision_id=decision_id,
                decision_row=decision_row,
                flow_subset=flow_subset,
                candidate_row=candidate_row,
                scenario_name=scenario_name,
                cfg=cfg,
            )
            frames.append(sim_df)

    return pd.concat(frames, ignore_index=True)
