from __future__ import annotations

import numpy as np
import pandas as pd


def _pick_first_present_int(row: pd.Series, candidates: list[str], default: int = 0) -> int:
    for col in candidates:
        if col in row.index and pd.notna(row.get(col)):
            return int(row.get(col))
    return int(default)


def _resolve_need_buy_flag(row: pd.Series) -> int:
    a_flag = _pick_first_present_int(
        row,
        ["target_a_final_pred", "target_a_pred", "target_a_rule"],
        default=0,
    )
    b_flag = _pick_first_present_int(
        row,
        ["target_b_final_pred", "target_b_pred", "target_b_rule"],
        default=0,
    )
    return int((a_flag == 1) or (b_flag == 1))


def build_decision_reason(row: pd.Series) -> str:
    parts = []
    if _pick_first_present_int(row, ["target_a_final_pred", "target_a_pred", "target_a_rule"], 0) == 1:
        parts.append("A-risk on")
    if _pick_first_present_int(row, ["target_b_final_pred", "target_b_pred", "target_b_rule"], 0) == 1:
        parts.append("B-risk on")

    parts.append(f"candidate={row['selected_candidate_name']}")
    parts.append(f"status={row['selected_candidate_status']}")
    parts.append(f"robust={int(row['selected_robust_no_shortage_all_scenarios'])}")
    parts.append(f"worst_shortage={row['selected_worst_case_shortage_ton']:.0f}t")
    return " | ".join(parts)


def build_additional_check_reason(row: pd.Series) -> str:
    """
    추가확인 사유를 사람이 읽는 문장으로 분해한다.
    규칙:
    - selected candidate 자체의 conditional/blocked 사유
    - robust 실패 여부
    - residual shortage 존재 여부
    - 실제 필요한 shortage_anchored 후보가 blocked/conditional 인지
    를 같이 묶어서 보여준다.
    """
    reasons: list[str] = []

    selected_name = str(row.get("selected_candidate_name", ""))
    selected_status = str(row.get("selected_candidate_status", ""))
    required_status = str(row.get("required_candidate_status", ""))

    # 1) risk는 있는데 observe만 남은 경우
    if int(row.get("need_buy_flag", 0)) == 1 and selected_name == "observe":
        reasons.append("위험은 있으나 실행 가능한 비관망 후보가 없음")

    # 2) 선택된 후보 자체가 conditional / blocked 인 경우
    if selected_status == "conditional":
        if str(row.get("selected_arrival_timing_gate_result", "")) == "conditional":
            reasons.append("선택후보 도착 타이밍이 타이트함")
        if str(row.get("selected_working_capital_gate_result", "")) == "conditional":
            reasons.append("선택후보 운전자본 압박이 높음")
        if isinstance(row.get("selected_soft_warning_reason"), str) and row.get("selected_soft_warning_reason"):
            reasons.append(f"선택후보 주의사유: {row.get('selected_soft_warning_reason')}")

    if selected_status == "blocked":
        if isinstance(row.get("selected_hard_fail_reason"), str) and row.get("selected_hard_fail_reason"):
            reasons.append(f"선택후보 실행불가: {row.get('selected_hard_fail_reason')}")
        else:
            reasons.append("선택후보 실행불가")

    # 3) robust 실패 / worst shortage 잔존
    if int(row.get("selected_robust_no_shortage_all_scenarios", 0)) == 0:
        reasons.append("전 시나리오 기준 robust하지 않음")

    worst_shortage = float(row.get("selected_worst_case_shortage_ton", 0.0) or 0.0)
    if worst_shortage > 0:
        reasons.append(f"선택후보로도 worst-case shortage {worst_shortage:.0f}톤이 남음")

    # 4) 필요한 큰 후보(shortage_anchored)가 따로 막힌 경우
    required_qty = row.get("required_candidate_qty_ton", np.nan)
    if pd.notna(required_qty):
        if required_status == "blocked":
            hard_reason = str(row.get("required_candidate_hard_fail_reason", "")).strip()
            if hard_reason:
                reasons.append(
                    f"필요수량 후보({float(required_qty):.0f}톤)는 실행불가: {hard_reason}"
                )
            else:
                reasons.append(f"필요수량 후보({float(required_qty):.0f}톤)는 실행불가")
        elif required_status == "conditional":
            soft_reason = str(row.get("required_candidate_soft_warning_reason", "")).strip()
            if soft_reason:
                reasons.append(
                    f"필요수량 후보({float(required_qty):.0f}톤)는 조건부: {soft_reason}"
                )
            else:
                reasons.append(f"필요수량 후보({float(required_qty):.0f}톤)는 조건부")

        # 필요한 후보가 따로 있고 선택후보가 더 작은 경우 설명 추가
        selected_qty = float(row.get("selected_candidate_qty_ton", 0.0) or 0.0)
        if selected_qty > 0 and float(required_qty) > selected_qty and selected_name != "shortage_anchored":
            reasons.append("필요수량 후보 대신 더 작은 실행가능 후보를 선택함")

    # 5) 가격 side 설명은 보조로만 추가
    if _pick_first_present_int(row, ["target_b_final_pred", "target_b_pred", "target_b_rule"], 0) == 1:
        reasons.append("비용 상방 리스크도 함께 확인 필요")

    # 중복 제거
    deduped: list[str] = []
    seen = set()
    for reason in reasons:
        key = reason.strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(key)

    return " | ".join(deduped)



def map_final_action(decision_master_df: pd.DataFrame, best_candidate_df: pd.DataFrame) -> pd.DataFrame:
    final_df = decision_master_df.merge(best_candidate_df, on="decision_id", how="left").copy()
    final_df["need_buy_flag"] = final_df.apply(_resolve_need_buy_flag, axis=1)

    conditions = [
        (final_df["need_buy_flag"] == 1)
        & (final_df["selected_candidate_name"] != "observe")
        & (final_df["selected_candidate_status"].isin(["feasible", "conditional"]))
        & (final_df["selected_robust_no_shortage_all_scenarios"] == 1),
        (final_df["need_buy_flag"] == 0) & (final_df["selected_candidate_name"] == "observe"),
    ]
    choices = ["선매입 검토", "관망"]
    final_df["final_action"] = np.select(conditions, choices, default="추가확인")
    final_df["final_reason"] = final_df.apply(build_decision_reason, axis=1)
    final_df["additional_check_reason"] = np.where(
        final_df["final_action"] == "추가확인",
        final_df.apply(build_additional_check_reason, axis=1),
        "",
    )

    output_cols = [
        "decision_id", "decision_month", "material_code",
        "target_a_rule", "target_b_rule",
        "target_a_pred", "target_b_pred",
        "target_a_final_pred", "target_b_final_pred",
        "need_buy_flag",
        "selected_candidate_name", "selected_candidate_qty_ton", "selected_candidate_status",
        "selected_robust_no_shortage_all_scenarios",
        "selected_worst_case_shortage_ton",
        "selected_worst_case_cost_vs_observe_pct",
        "selected_worst_case_min_ending_inventory_ton",
        "selected_hard_fail_reason", "selected_soft_warning_reason",
        "selected_working_capital_gate_result", "selected_arrival_timing_gate_result",
        "required_candidate_qty_ton", "required_candidate_status",
        "required_candidate_hard_fail_reason", "required_candidate_soft_warning_reason",
        "final_action", "final_reason", "additional_check_reason",
    ]

    # 없는 컬럼은 조용히 무시해서 이전 버전과도 최대한 호환되게 한다.
    output_cols = [col for col in output_cols if col in final_df.columns]
    return final_df[output_cols]
