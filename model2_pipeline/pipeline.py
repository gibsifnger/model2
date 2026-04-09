from __future__ import annotations

from typing import Dict

import pandas as pd

from .baseline_flow import build_baseline_flow_df
from .candidates import generate_candidate_df
from .compare_select import build_scenario_compare_summary, select_best_candidate
from .config import PipelineConfig
from .final_action import map_final_action
from .gates import apply_operating_gate
from .simulation import run_candidate_simulations


SELECTED_GATE_COLS = [
    "decision_id",
    "candidate_name",
    "candidate_status",
    "hard_fail_reason",
    "soft_warning_reason",
    "moq_gate_pass",
    "lot_multiple_gate_pass",
    "warehouse_gate_pass",
    "working_capital_gate_result",
    "arrival_timing_gate_result",
    "projected_max_end_inv_ton_base",
    "projected_total_shortage_ton_base",
]


SHORTAGE_ANCHORED_GATE_COLS = [
    "decision_id",
    "candidate_status",
    "hard_fail_reason",
    "soft_warning_reason",
    "candidate_qty_ton",
    "projected_total_shortage_ton_base",
]


def _attach_gate_context(best_candidate_df: pd.DataFrame, gated_candidate_df: pd.DataFrame) -> pd.DataFrame:
    """
    final action 단계에서 '추가확인' 사유를 설명할 수 있도록 gate context를 붙인다.

    1) 실제 선택된 candidate의 gate 상태/사유
    2) shortage_anchored(필요수량 후보)의 gate 상태/사유

    이렇게 두 축을 같이 붙여야,
    - 왜 선택 후보가 conditional 인지
    - 왜 진짜 필요한 큰 후보는 blocked 인지
    를 최종 출력에서 바로 설명할 수 있다.
    """
    out = best_candidate_df.copy()

    selected_gate_df = gated_candidate_df[SELECTED_GATE_COLS].copy()
    selected_gate_df = selected_gate_df.rename(
        columns={
            "candidate_name": "selected_candidate_name",
            "candidate_status": "selected_candidate_gate_status",
            "hard_fail_reason": "selected_hard_fail_reason",
            "soft_warning_reason": "selected_soft_warning_reason",
            "moq_gate_pass": "selected_moq_gate_pass",
            "lot_multiple_gate_pass": "selected_lot_multiple_gate_pass",
            "warehouse_gate_pass": "selected_warehouse_gate_pass",
            "working_capital_gate_result": "selected_working_capital_gate_result",
            "arrival_timing_gate_result": "selected_arrival_timing_gate_result",
            "projected_max_end_inv_ton_base": "selected_projected_max_end_inv_ton_base",
            "projected_total_shortage_ton_base": "selected_projected_total_shortage_ton_base",
        }
    )
    out = out.merge(selected_gate_df, on=["decision_id", "selected_candidate_name"], how="left")

    anchored_gate_df = gated_candidate_df[gated_candidate_df["candidate_name"] == "shortage_anchored"][
        SHORTAGE_ANCHORED_GATE_COLS
    ].copy()
    anchored_gate_df = anchored_gate_df.rename(
        columns={
            "candidate_status": "required_candidate_status",
            "hard_fail_reason": "required_candidate_hard_fail_reason",
            "soft_warning_reason": "required_candidate_soft_warning_reason",
            "candidate_qty_ton": "required_candidate_qty_ton",
            "projected_total_shortage_ton_base": "required_candidate_projected_total_shortage_ton_base",
        }
    )
    out = out.merge(anchored_gate_df, on="decision_id", how="left")

    return out



def run_full_decision_pipeline(decision_master_df: pd.DataFrame, cfg: PipelineConfig) -> Dict[str, pd.DataFrame]:
    baseline_flow_df = build_baseline_flow_df(decision_master_df, cfg)
    candidate_df = generate_candidate_df(decision_master_df, cfg)
    gated_candidate_df = apply_operating_gate(decision_master_df, baseline_flow_df, candidate_df, cfg)
    simulation_result_df = run_candidate_simulations(decision_master_df, baseline_flow_df, gated_candidate_df, cfg)
    scenario_summary_df, robust_summary_df = build_scenario_compare_summary(simulation_result_df)
    best_candidate_df = select_best_candidate(decision_master_df, robust_summary_df)
    best_candidate_df = _attach_gate_context(best_candidate_df, gated_candidate_df)
    final_decision_df = map_final_action(decision_master_df, best_candidate_df)

    return {
        "baseline_flow_df": baseline_flow_df,
        "candidate_df": candidate_df,
        "gated_candidate_df": gated_candidate_df,
        "simulation_result_df": simulation_result_df,
        "scenario_summary_df": scenario_summary_df,
        "robust_summary_df": robust_summary_df,
        "best_candidate_df": best_candidate_df,
        "final_decision_df": final_decision_df,
    }
