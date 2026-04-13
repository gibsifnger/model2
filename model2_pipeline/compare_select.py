from __future__ import annotations

import pandas as pd


def infer_need_buy_flag(row: pd.Series) -> int:
    """
    Selection 단계에서도 final prediction을 우선 사용한다.

    우선순위:
    1) target_a_final_pred / target_b_final_pred
    2) target_a_pred / target_b_pred
    3) target_a_rule / target_b_rule
    """
    a_val = row.get("target_a_final_pred", row.get("target_a_pred", row.get("target_a_rule", 0)))
    b_val = row.get("target_b_final_pred", row.get("target_b_pred", row.get("target_b_rule", 0)))
    return int(int(a_val) == 1 or int(b_val) == 1)


def build_scenario_compare_summary(simulation_result_df: pd.DataFrame):
    scenario_summary_df = (
        simulation_result_df.groupby(
            ["decision_id", "candidate_name", "candidate_qty_ton", "candidate_status", "scenario_name"],
            as_index=False,
        )
        .agg(
            total_shortage_ton=("shortage_ton", "sum"),
            any_shortage_flag=("shortage_ton", lambda s: int((s > 0).any())),
            min_ending_inventory_ton=("ending_inventory_ton", "min"),
            total_cost=("total_month_cost", "sum"),
            total_emergency_buy_ton=("emergency_buy_ton", "sum"),
            last_ending_inventory_ton=("ending_inventory_ton", "last"),
        )
    )

    observe_ref = (
        scenario_summary_df[scenario_summary_df["candidate_name"] == "observe"]
        [["decision_id", "scenario_name", "total_cost"]]
        .rename(columns={"total_cost": "observe_total_cost"})
    )

    scenario_summary_df = scenario_summary_df.merge(
        observe_ref,
        on=["decision_id", "scenario_name"],
        how="left",
    )

    scenario_summary_df["cost_vs_observe_pct"] = (
        (scenario_summary_df["total_cost"] - scenario_summary_df["observe_total_cost"])
        / scenario_summary_df["observe_total_cost"]
    )

    robust_summary_df = (
        scenario_summary_df.groupby(
            ["decision_id", "candidate_name", "candidate_qty_ton", "candidate_status"],
            as_index=False,
        )
        .agg(
            scenario_count=("scenario_name", "nunique"),
            robust_no_shortage_all_scenarios=("any_shortage_flag", lambda s: int((s == 0).all())),
            worst_case_shortage_ton=("total_shortage_ton", "max"),
            worst_case_cost_vs_observe_pct=("cost_vs_observe_pct", "max"),
            worst_case_min_ending_inventory_ton=("min_ending_inventory_ton", "min"),
            avg_total_cost=("total_cost", "mean"),
        )
    )

    return scenario_summary_df, robust_summary_df



def select_best_candidate(decision_master_df: pd.DataFrame, robust_summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    need_buy 판단도 final prediction 기준으로 본다.
    """
    decision_map = decision_master_df.set_index("decision_id")
    status_rank_map = {"feasible": 0, "conditional": 1, "blocked": 9}

    picks = []

    for decision_id, group in robust_summary_df.groupby("decision_id"):
        decision_row = decision_map.loc[decision_id]
        need_buy_flag = infer_need_buy_flag(decision_row)

        group = group.copy()
        group["status_rank"] = group["candidate_status"].map(status_rank_map).fillna(9)
        group["qty_rank"] = group["candidate_qty_ton"]

        if need_buy_flag == 1:
            eligible = group[
                (group["candidate_status"] != "blocked")
                & (group["candidate_qty_ton"] > 0)
            ].copy()

            if eligible.empty:
                chosen = group.sort_values(
                    ["status_rank", "worst_case_shortage_ton", "worst_case_cost_vs_observe_pct", "qty_rank"],
                    ascending=[True, True, True, True],
                ).iloc[0]
            else:
                robust_eligible = eligible[eligible["robust_no_shortage_all_scenarios"] == 1].copy()
                pool = robust_eligible if not robust_eligible.empty else eligible

                chosen = pool.sort_values(
                    ["status_rank", "worst_case_cost_vs_observe_pct", "worst_case_shortage_ton", "qty_rank"],
                    ascending=[True, True, True, True],
                ).iloc[0]
        else:
            observe = group[group["candidate_name"] == "observe"]
            chosen = (
                observe.iloc[0]
                if not observe.empty
                else group.sort_values(["status_rank", "qty_rank"]).iloc[0]
            )

        picks.append(
            {
                "decision_id": decision_id,
                "selected_candidate_name": chosen["candidate_name"],
                "selected_candidate_qty_ton": chosen["candidate_qty_ton"],
                "selected_candidate_status": chosen["candidate_status"],
                "selected_robust_no_shortage_all_scenarios": chosen["robust_no_shortage_all_scenarios"],
                "selected_worst_case_shortage_ton": chosen["worst_case_shortage_ton"],
                "selected_worst_case_cost_vs_observe_pct": chosen["worst_case_cost_vs_observe_pct"],
                "selected_worst_case_min_ending_inventory_ton": chosen["worst_case_min_ending_inventory_ton"],
            }
        )

    return pd.DataFrame(picks)
