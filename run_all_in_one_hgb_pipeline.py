"""모델2 올인원 실행 진입점.

핵심 목적
---------
1) 외생 3개 시계열을 준비한다.
   - demo 생성
   - 기존 CSV 로드
   - build_external_inputs.py 로 실데이터 월별 CSV 생성
2) hybrid decision master historical panel을 만든다.
3) target A/B용 HGB를 학습하거나, 저장된 artifact를 로드한다.
4) 최신 decision row를 score한다.
5) gate -> candidate -> scenario simulation -> final action까지 한 번에 실행한다.

중요
----
- '올인원'은 실행 파일이 1개라는 뜻이다.
- 내부 계산 로직은 model2_pipeline/* 분리 구조를 그대로 재사용한다.
- 따라서 모델1처럼 코드가 한 파일에 전부 뭉개지지 않는다.
"""
# 실행 코드  : python run_all_in_one_hgb_pipeline.py --start-month 2023-01-01 --demo-months 44 --decision-month 2026-04-01 --save-outputs

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import math


from build_external_inputs import ExternalBuildConfig, build_external_inputs_monthly
from model2_pipeline.config import PipelineConfig
from model2_pipeline.decision_generator import (
    _ensure_monthly_exogenous_df,
    build_hybrid_decision_master_df,
    make_demo_exogenous_df,
)
from model2_pipeline.model_inference import (
    ModelBundle,
    attach_target_predictions,
    fit_demo_hgb_bundle,
    load_model_bundle,
    save_model_bundle,
)
from model2_pipeline.pipeline import run_full_decision_pipeline


# =========================================================
# 1) CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model2 all-in-one HGB decision pipeline")

    # external data mode
    parser.add_argument("--external-mode", choices=["demo", "csv", "build"], default="demo")
    parser.add_argument("--external-csv-path", default=None)
    parser.add_argument("--start-month", default="2019-01-01")
    parser.add_argument("--end-month", default=None)
    parser.add_argument("--demo-months", type=int, default=40)

    # build_external_inputs.py passthrough
    parser.add_argument("--ecos-api-key", default=None)
    parser.add_argument("--ecos-stat-code", default=None)
    parser.add_argument("--ecos-cycle", default="M")
    parser.add_argument("--ecos-item-code-1", default=None)
    parser.add_argument("--ecos-item-code-2", default=None)
    parser.add_argument("--ecos-item-code-3", default=None)
    parser.add_argument("--freight-mode", choices=["synthetic", "csv"], default="synthetic")
    parser.add_argument("--freight-csv-path", default=None)
    parser.add_argument("--freight-date-col", default="Date")
    parser.add_argument("--freight-value-col", default="freight_index")
    parser.add_argument("--built-external-output-csv", default=None)

    # model artifact mode
    parser.add_argument("--use-saved-artifacts", action="store_true")
    parser.add_argument("--model-a-path", default=None)
    parser.add_argument("--model-b-path", default=None)
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--artifact-dir", default="./artifacts")

    # prediction combine mode
    parser.add_argument(
        "--prediction-combine-mode",
        choices=["auto", "model_only", "rule_floor", "rule_only"],
        default="auto",
        help=(
            "auto: fresh-fit demo는 rule_floor, saved artifact는 model_only. "
            "model_only: 모델 결과만 사용. "
            "rule_floor: max(rule, model). "
            "rule_only: 모델 무시하고 rule만 사용."
        ),
    )

    # output
    parser.add_argument("--save-outputs", action="store_true")
    parser.add_argument("--output-dir", default="./outputs")

    # misc
    parser.add_argument("--decision-month", default="2026-04-01") #월고정 
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# =========================================================
# 2) External data loader / builder
# =========================================================
def load_external_inputs_from_csv(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"external csv not found: {path}")

    df = pd.read_csv(path)
    df = _ensure_monthly_exogenous_df(df)
    return df


def prepare_external_inputs(args: argparse.Namespace) -> Tuple[pd.DataFrame, Optional[Path]]:
    """외생 3개 시계열 준비.

    Returns
    -------
    exogenous_df, built_csv_path
    """
    built_csv_path: Optional[Path] = None

    if args.external_mode == "demo":
        exogenous_df = make_demo_exogenous_df(
            start_month=args.start_month,
            n_months=args.demo_months,
            seed=args.seed,
        )
        exogenous_df = _ensure_monthly_exogenous_df(exogenous_df)
        return exogenous_df, built_csv_path

    if args.external_mode == "csv":
        if not args.external_csv_path:
            raise ValueError("--external-mode csv requires --external-csv-path")
        exogenous_df = load_external_inputs_from_csv(args.external_csv_path)
        return exogenous_df, built_csv_path

    if args.external_mode == "build":
        build_cfg = ExternalBuildConfig(
            start_month=args.start_month,
            end_month=args.end_month,
            output_csv=args.built_external_output_csv or "external_inputs_monthly.csv",
            ecos_api_key=args.ecos_api_key,
            ecos_stat_code=args.ecos_stat_code,
            ecos_cycle=args.ecos_cycle,
            ecos_item_code_1=args.ecos_item_code_1,
            ecos_item_code_2=args.ecos_item_code_2,
            ecos_item_code_3=args.ecos_item_code_3,
            freight_mode=args.freight_mode,
            freight_csv_path=args.freight_csv_path,
            freight_date_col=args.freight_date_col,
            freight_value_col=args.freight_value_col,
            seed=args.seed,
        )
        exogenous_df = build_external_inputs_monthly(build_cfg)
        built_csv_path = Path(build_cfg.output_csv)
        exogenous_df.to_csv(built_csv_path, index=False, encoding="utf-8-sig")
        return exogenous_df, built_csv_path

    raise ValueError(f"unsupported external mode: {args.external_mode}")


# =========================================================
# 3) HGB fit / load
# =========================================================
def fit_hgb_bundles_from_historical_panel(
    historical_master_df: pd.DataFrame,
    seed: int,
) -> Tuple[ModelBundle, ModelBundle]:
    """historical decision panel에서 target A/B용 HGB를 학습한다.

    학습 기준:
    - X: build_model_feature_frame(historical_master_df)
    - y_a: target_a_rule
    - y_b: target_b_rule

    주의:
    - 이 모델은 synthetic/hybrid historical row 위에 quick-fit 하는 데모/개발용 기본기다.
    - 나중에 실전 artifact가 있으면 --use-saved-artifacts 로 교체하면 된다.
    """
    if len(historical_master_df) < 5:
        raise ValueError("historical_master_df too short to fit HGB. Need at least 5 rows.")

    # 최신 row 1개는 live scoring 용도로 남겨두고, 앞 구간으로 학습
    train_df = historical_master_df.iloc[:-1].reset_index(drop=True).copy()

    from model2_pipeline.model_features import build_model_feature_frame

    X_train = build_model_feature_frame(train_df, cfg=PipelineConfig())
    y_a = train_df["target_a_rule"].astype(int)
    y_b = train_df["target_b_rule"].astype(int)

    bundle_a = fit_demo_hgb_bundle(
        X=X_train,
        y=y_a,
        name="target_a_hgb",
        threshold=0.50,
        random_state=seed,
    )
    bundle_b = fit_demo_hgb_bundle(
        X=X_train,
        y=y_b,
        name="target_b_hgb",
        threshold=0.50,
        random_state=seed,
    )
    return bundle_a, bundle_b



def resolve_model_bundles(
    historical_master_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[ModelBundle, ModelBundle, str]:
    """저장 artifact 재사용 또는 quick-fit 중 하나를 선택."""
    if args.use_saved_artifacts:
        if not args.model_a_path or not args.model_b_path:
            raise ValueError("--use-saved-artifacts requires both --model-a-path and --model-b-path")
        bundle_a = load_model_bundle(args.model_a_path)
        bundle_b = load_model_bundle(args.model_b_path)
        return bundle_a, bundle_b, "loaded_saved_artifacts"

    bundle_a, bundle_b = fit_hgb_bundles_from_historical_panel(
        historical_master_df=historical_master_df,
        seed=args.seed,
    )

    if args.save_artifacts:
        artifact_dir = Path(args.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        save_model_bundle(bundle_a, artifact_dir / "target_a_hgb.joblib")
        save_model_bundle(bundle_b, artifact_dir / "target_b_hgb.joblib")

    return bundle_a, bundle_b, "fresh_fit_from_historical_panel"


# =========================================================
# 4) prediction combine mode
# =========================================================
def resolve_prediction_combine_mode(args: argparse.Namespace, model_mode: str) -> str:
    if args.prediction_combine_mode != "auto":
        return args.prediction_combine_mode

    if model_mode == "fresh_fit_from_historical_panel":
        return "rule_floor"
    return "model_only"

# =========================================================
# 4-1) Fixed canonical case: 2026-04
# =========================================================
def _ceil_to_multiple(value: float, multiple: float) -> float:
    if multiple <= 0:
        return float(value)
    return float(math.ceil(float(value) / float(multiple)) * float(multiple))


def _recompute_fixed_case_helpers(decision_df: pd.DataFrame) -> pd.DataFrame:
    """
    고정 케이스 row의 helper / rule을 다시 맞춘다.
    축:
    - 판단시점: 2026-04 말 구매회의
    - 영향 구간: 2026-07 ~ 2026-08 ~ 2026-09 초
    - 따라서 arrival_month_idx = 3 으로 둔다
      (m1=5월, m2=6월, m3=7월, m4=8월 기준)
    """
    out = decision_df.copy()
    if len(out) != 1:
        raise ValueError("_recompute_fixed_case_helpers expects exactly 1 row")

    row = out.iloc[0]

    current_inv = float(row["current_inventory_ton"])
    moq = float(row["moq_ton"])
    lot = float(row["lot_multiple_ton"])
    now_cost = float(row["now_landed_cost_per_ton"])
    arrival_idx = int(row["candidate_arrival_month_idx"])

    usage = [
        float(row["usage_m1_ton"]),
        float(row["usage_m2_ton"]),
        float(row["usage_m3_ton"]),
        float(row["usage_m4_ton"]),
    ]
    open_po = [
        float(row["open_po_m1_ton"]),
        float(row["open_po_m2_ton"]),
        float(row["open_po_m3_ton"]),
        float(row["open_po_m4_ton"]),
    ]
    costs = [
        float(row["expected_landed_cost_m1_per_ton"]),
        float(row["expected_landed_cost_m2_per_ton"]),
        float(row["expected_landed_cost_m3_per_ton"]),
        float(row["expected_landed_cost_m4_per_ton"]),
    ]

    # -------------------------
    # baseline no-buy world
    # -------------------------
    inv = current_inv
    ending_inv_path = []
    shortage_path = []

    for i in range(4):
        available = inv + open_po[i]
        shortage = max(usage[i] - available, 0.0)
        end_inv = max(available - usage[i], 0.0)

        shortage_path.append(shortage)
        ending_inv_path.append(end_inv)
        inv = end_inv

    baseline_total_shortage_ton = float(sum(shortage_path))

    first_shortage_month_idx = None
    for i, s in enumerate(shortage_path, start=1):
        if s > 0:
            first_shortage_month_idx = i
            break

    # arrival 직전 재고
    if arrival_idx <= 1:
        begin_inv_at_arrival_ton = current_inv
    else:
        begin_inv_at_arrival_ton = float(ending_inv_path[arrival_idx - 2])

    # -------------------------
    # arrival 시점에 한 번에 사서 막으려면
    # 얼마가 필요한지 계산
    # -------------------------
    cum_usage = 0.0
    cum_open_po = 0.0
    max_cum_gap_arrival_ton = 0.0

    start_idx = max(arrival_idx - 1, 0)
    for i in range(start_idx, 4):
        cum_usage += usage[i]
        cum_open_po += open_po[i]
        gap = cum_usage - cum_open_po - begin_inv_at_arrival_ton
        max_cum_gap_arrival_ton = max(max_cum_gap_arrival_ton, gap)

    max_cum_gap_arrival_ton = max(0.0, float(max_cum_gap_arrival_ton))

    if max_cum_gap_arrival_ton > 0:
        required_buy_qty_arrival_ton = max_cum_gap_arrival_ton + moq
    else:
        required_buy_qty_arrival_ton = 0.0

    if first_shortage_month_idx is not None:
        if first_shortage_month_idx == 1:
            begin_inv_first_shortage = current_inv
        else:
            begin_inv_first_shortage = float(ending_inv_path[first_shortage_month_idx - 2])

        first_idx0 = first_shortage_month_idx - 1
        first_gap = max(
            usage[first_idx0] - open_po[first_idx0] - begin_inv_first_shortage,
            0.0,
        )
        required_buy_qty_first_shortage_ton = first_gap + moq if first_gap > 0 else 0.0
    else:
        required_buy_qty_first_shortage_ton = 0.0

    # -------------------------
    # B helper (원가상승 압력)
    # -------------------------
    peak_cost = max(costs)
    b_peak_cost_vs_now_pct = (peak_cost / now_cost - 1.0) if now_cost > 0 else 0.0
    b_peak_cost_vs_now_pct = float(max(b_peak_cost_vs_now_pct, 0.0))

    b_forced_buy_flag = int(b_peak_cost_vs_now_pct >= 0.06)
    b_forced_buy_cost_vs_now_pct = b_peak_cost_vs_now_pct
    b_emergency_premium_score = round(b_peak_cost_vs_now_pct * 1000.0, 6)
    b_high_cost_month_count = int(sum(1 for c in costs if c > now_cost * 1.03))

    # -------------------------
    # 최종 helper / rule overwrite
    # -------------------------
    out.loc[:, "baseline_total_shortage_ton"] = baseline_total_shortage_ton
    out.loc[:, "a_first_shortage_month_idx"] = (
        float(first_shortage_month_idx) if first_shortage_month_idx is not None else float("nan")
    )
    out.loc[:, "a_emergency_buy_needed_flag"] = int(baseline_total_shortage_ton > 0)
    out.loc[:, "a_min_end_inv_ton"] = float(min(ending_inv_path))
    out.loc[:, "a_min_cover_months"] = 0.0

    out.loc[:, "begin_inv_at_arrival_ton"] = begin_inv_at_arrival_ton
    out.loc[:, "max_cum_gap_arrival_ton"] = max_cum_gap_arrival_ton
    out.loc[:, "required_buy_qty_arrival_ton"] = required_buy_qty_arrival_ton
    out.loc[:, "required_buy_qty_first_shortage_ton"] = required_buy_qty_first_shortage_ton

    out.loc[:, "b_peak_cost_vs_now_pct"] = b_peak_cost_vs_now_pct
    out.loc[:, "b_forced_buy_flag"] = b_forced_buy_flag
    out.loc[:, "b_forced_buy_cost_vs_now_pct"] = b_forced_buy_cost_vs_now_pct
    out.loc[:, "b_emergency_premium_score"] = b_emergency_premium_score
    out.loc[:, "b_high_cost_month_count"] = b_high_cost_month_count

    # rule은 fixed case 기준으로 다시 부여
    out.loc[:, "target_a_rule"] = int(baseline_total_shortage_ton > 0)
    out.loc[:, "target_b_rule"] = int(b_peak_cost_vs_now_pct >= 0.08)

    return out


def apply_fixed_case_2026_04(decision_df: pd.DataFrame) -> pd.DataFrame:
    """
    최초 가정사항 고정:
    - 구매회의: 2026년 4월 말
    - 지금 발주 시 실질 영향: 7월 ~ 8월 ~ 9월 초
    - 따라서 arrival_month_idx = 3 (7월 도착 영향으로 해석)
    """
    out = decision_df.copy()
    if len(out) != 1:
        raise ValueError("apply_fixed_case_2026_04 expects exactly 1 row")

    # -------------------------
    # 시점 고정
    # -------------------------
    out.loc[:, "decision_id"] = "RAW_SUGAR_2026-04"
    out.loc[:, "decision_month"] = pd.Timestamp("2026-04-01")
    out.loc[:, "as_of_month"] = pd.Timestamp("2026-04-01")
    out.loc[:, "decision_date"] = pd.Timestamp("2026-04-30")

    # -------------------------
    # 리드타임 / 도착축
    # 4월말 회의 -> 7월부터 본격 영향
    # -------------------------
    out.loc[:, "lt_months"] = 3
    out.loc[:, "candidate_arrival_month_idx"] = 3

    # -------------------------
    # 재고 / 운영 제약
    # -------------------------
    out.loc[:, "on_hand_inventory_ton"] = 50000.0
    out.loc[:, "blocked_inventory_ton"] = 2000.0
    out.loc[:, "usable_inventory_ton"] = 48000.0
    out.loc[:, "current_inventory_ton"] = 48000.0

    out.loc[:, "moq_ton"] = 5000.0
    out.loc[:, "lot_multiple_ton"] = 2500.0
    out.loc[:, "warehouse_capacity_ton"] = 55000.0
    out.loc[:, "working_capital_pressure_score"] = 62.0

    # -------------------------
    # 현재 원가 / horizon 원가 경로
    # -------------------------
    out.loc[:, "now_landed_cost_per_ton"] = 648.0
    out.loc[:, "expected_landed_cost_m1_per_ton"] = 652.0
    out.loc[:, "expected_landed_cost_m2_per_ton"] = 658.0
    out.loc[:, "expected_landed_cost_m3_per_ton"] = 669.0
    out.loc[:, "expected_landed_cost_m4_per_ton"] = 676.0

    # -------------------------
    # 수요: 5월, 6월, 7월, 8월
    # 7~8월을 높게 둬서 네 최초 가정과 맞춘다
    # -------------------------
    out.loc[:, "usage_m1_ton"] = 23000.0
    out.loc[:, "usage_m2_ton"] = 24000.0
    out.loc[:, "usage_m3_ton"] = 29000.0
    out.loc[:, "usage_m4_ton"] = 28000.0

    # -------------------------
    # 기존 오더: 6월/7월 일부만 들어오는 상황
    # -------------------------
    out.loc[:, "open_po_m1_ton"] = 0.0
    out.loc[:, "open_po_m2_ton"] = 5000.0
    out.loc[:, "open_po_m3_ton"] = 2500.0
    out.loc[:, "open_po_m4_ton"] = 0.0

    # shock flag 현재월은 off
    if "shock_event_flag_now" in out.columns:
        out.loc[:, "shock_event_flag_now"] = 0

    # helper / rule 재계산
    out = _recompute_fixed_case_helpers(out)
    return out

# =========================================================
# 5) Main all-in-one runner
# =========================================================
def run_all_in_one_pipeline(args: argparse.Namespace) -> Dict[str, pd.DataFrame]:
    cfg = PipelineConfig()

    # 1. 외생 3개 준비
    exogenous_df, built_csv_path = prepare_external_inputs(args)

    # 2. historical decision master 생성
    historical_master_df = build_hybrid_decision_master_df(
        exogenous_df=exogenous_df,
        cfg=cfg,
        seed=args.seed,
        keep_latest_only=False,
    )
    if historical_master_df.empty:
        raise ValueError("historical_master_df is empty after generation.")

    # 3. HGB bundle 확보 (load or fit)
    bundle_a, bundle_b, model_mode = resolve_model_bundles(
        historical_master_df=historical_master_df,
        args=args,
    )

    combine_mode = resolve_prediction_combine_mode(args, model_mode)

    # 4. 지정 판단월 row 선택 + fixed case overwrite + score
    decision_month = pd.to_datetime(args.decision_month).to_period("M").to_timestamp()

    latest_decision_df = (
        historical_master_df[
            pd.to_datetime(historical_master_df["decision_month"]).dt.to_period("M").dt.to_timestamp()
            == decision_month
        ]
        .tail(1)
        .reset_index(drop=True)
    )

    if latest_decision_df.empty:
        available_months = (
            pd.to_datetime(historical_master_df["decision_month"])
            .dt.to_period("M")
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            f"decision_month {decision_month.date()} not found in historical_master_df. "
            f"available decision_months tail={available_months[-12:]}"
        )

    latest_decision_df = apply_fixed_case_2026_04(latest_decision_df)

    scored_latest_df = attach_target_predictions(
        decision_master_df=latest_decision_df,
        cfg=cfg,
        model_a_bundle=bundle_a,
        model_b_bundle=bundle_b,
        fallback_to_rule=False,
        combine_mode=combine_mode,
    )

    # 5. final decision pipeline
    pipeline_outputs = run_full_decision_pipeline(
        decision_master_df=scored_latest_df,
        cfg=cfg,
    )

    # 6. 메타 정보
    meta_df = pd.DataFrame([
        {
            "external_mode": args.external_mode,
            "model_mode": model_mode,
            "prediction_combine_mode": combine_mode,
            "historical_rows": len(historical_master_df),
            "latest_decision_month": scored_latest_df.loc[0, "decision_month"],
            "built_external_csv": str(built_csv_path) if built_csv_path else "",
            "save_artifacts": int(bool(args.save_artifacts)),
            "use_saved_artifacts": int(bool(args.use_saved_artifacts)),
        }
    ])

    outputs: Dict[str, pd.DataFrame] = {
        "meta_df": meta_df,
        "exogenous_df": exogenous_df,
        "historical_master_df": historical_master_df,
        "scored_latest_df": scored_latest_df,
        **pipeline_outputs,
    }

    if args.save_outputs:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, df in outputs.items():
            df.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    return outputs


# =========================================================
# 6) Console display
# =========================================================
def _fmt_value(v):
    if pd.isna(v):
        return "-"
    if isinstance(v, (int, float)):
        if abs(float(v)) >= 1000:
            return f"{float(v):,.0f}"
        if float(v).is_integer():
            return f"{int(v)}"
        return f"{float(v):,.4f}"
    return str(v)


def _print_kv_block(title: str, items: Dict[str, object]) -> None:
    print(f"\n[{title}]")
    for k, v in items.items():
        print(f"- {k}: {_fmt_value(v)}")


def _safe_row(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=object)
    return df.iloc[0]


def _print_df(title: str, df: pd.DataFrame, columns: list[str] | None = None, sort_by: list[str] | None = None) -> None:
    print(f"\n[{title}]")
    if df is None or df.empty:
        print("(empty)")
        return

    out = df.copy()
    if sort_by:
        existing_sort = [c for c in sort_by if c in out.columns]
        if existing_sort:
            out = out.sort_values(existing_sort).reset_index(drop=True)

    if columns:
        existing_cols = [c for c in columns if c in out.columns]
        out = out[existing_cols]

    print(out.to_string(index=False))


def print_key_outputs(outputs: Dict[str, pd.DataFrame]) -> None:
    meta = _safe_row(outputs.get("meta_df", pd.DataFrame()))
    latest = _safe_row(outputs.get("scored_latest_df", pd.DataFrame()))
    best = _safe_row(outputs.get("best_candidate_df", pd.DataFrame()))
    final = _safe_row(outputs.get("final_decision_df", pd.DataFrame()))

    _print_kv_block(
        "run_summary",
        {
            "external_mode": meta.get("external_mode"),
            "model_mode": meta.get("model_mode"),
            "prediction_combine_mode": meta.get("prediction_combine_mode"),
            "latest_decision_month": meta.get("latest_decision_month"),
            "historical_rows": meta.get("historical_rows"),
        },
    )

    _print_df(
        "exogenous_tail",
        outputs.get("exogenous_df", pd.DataFrame()).tail(4),
        columns=["as_of_month", "global_raw_sugar_price", "usdkrw", "freight_index"],
    )

    _print_kv_block(
        "latest_purchase_state",
        {
            "decision_month": latest.get("decision_month"),
            "decision_date": latest.get("decision_date"),
            "lt_months": latest.get("lt_months"),
            "candidate_arrival_month_idx": latest.get("candidate_arrival_month_idx"),
            "usable_inventory_ton": latest.get("usable_inventory_ton"),
            "now_landed_cost_per_ton": latest.get("now_landed_cost_per_ton"),
            "working_capital_pressure_score": latest.get("working_capital_pressure_score"),
            "a_first_shortage_month_idx": latest.get("a_first_shortage_month_idx"),
            "baseline_total_shortage_ton": latest.get("baseline_total_shortage_ton"),
            "max_cum_gap_arrival_ton": latest.get("max_cum_gap_arrival_ton"),
            "required_buy_qty_arrival_ton": latest.get("required_buy_qty_arrival_ton"),
            "required_buy_qty_first_shortage_ton": latest.get("required_buy_qty_first_shortage_ton"),
            "target_a_final_pred": latest.get("target_a_final_pred"),
            "target_b_final_pred": latest.get("target_b_final_pred"),
        },
    )

    _print_df(
        "month_path_m1_to_m4",
        pd.DataFrame([
            {
                "usage_m1_ton": latest.get("usage_m1_ton"),
                "open_po_m1_ton": latest.get("open_po_m1_ton"),
                "expected_landed_cost_m1_per_ton": latest.get("expected_landed_cost_m1_per_ton"),
                "usage_m2_ton": latest.get("usage_m2_ton"),
                "open_po_m2_ton": latest.get("open_po_m2_ton"),
                "expected_landed_cost_m2_per_ton": latest.get("expected_landed_cost_m2_per_ton"),
                "usage_m3_ton": latest.get("usage_m3_ton"),
                "open_po_m3_ton": latest.get("open_po_m3_ton"),
                "expected_landed_cost_m3_per_ton": latest.get("expected_landed_cost_m3_per_ton"),
                "usage_m4_ton": latest.get("usage_m4_ton"),
                "open_po_m4_ton": latest.get("open_po_m4_ton"),
                "expected_landed_cost_m4_per_ton": latest.get("expected_landed_cost_m4_per_ton"),
            }
        ]),
    )

    _print_df(
        "candidate_summary",
        outputs.get("candidate_df", pd.DataFrame()),
        columns=[
            "candidate_name",
            "candidate_qty_ton",
            "candidate_arrival_month_idx",
            "candidate_unit_cost_per_ton_now",
            "candidate_po_value_now",
            "required_buy_qty_arrival_ton",
            "baseline_total_shortage_ton",
        ],
    )

    _print_df(
        "gated_candidate_summary",
        outputs.get("gated_candidate_df", pd.DataFrame()),
        columns=[
            "candidate_name",
            "candidate_qty_ton",
            "candidate_status",
            "moq_gate_pass",
            "lot_multiple_gate_pass",
            "warehouse_gate_pass",
            "working_capital_gate_result",
            "arrival_timing_gate_result",
            "projected_total_shortage_ton_base",
            "hard_fail_reason",
            "soft_warning_reason",
        ],
    )

    scenario_df = outputs.get("scenario_summary_df", pd.DataFrame()).copy()
    if not scenario_df.empty:
        scenario_df = scenario_df[
            [
                "candidate_name",
                "scenario_name",
                "candidate_status",
                "total_shortage_ton",
                "min_ending_inventory_ton",
                "total_cost",
                "cost_vs_observe_pct",
            ]
        ].sort_values(["candidate_name", "scenario_name"]).reset_index(drop=True)
    _print_df("scenario_summary_compact", scenario_df)

    _print_kv_block(
        "selected_candidate",
        {
            "selected_candidate_name": best.get("selected_candidate_name"),
            "selected_candidate_qty_ton": best.get("selected_candidate_qty_ton"),
            "selected_candidate_status": best.get("selected_candidate_status"),
            "selected_worst_case_shortage_ton": best.get("selected_worst_case_shortage_ton"),
            "selected_worst_case_cost_vs_observe_pct": best.get("selected_worst_case_cost_vs_observe_pct"),
            "required_candidate_qty_ton": best.get("required_candidate_qty_ton"),
            "required_candidate_status": best.get("required_candidate_status"),
            "required_candidate_hard_fail_reason": best.get("required_candidate_hard_fail_reason"),
        },
    )

    _print_kv_block(
        "final_action_summary",
        {
            "need_buy_flag": final.get("need_buy_flag"),
            "selected_candidate_name": final.get("selected_candidate_name"),
            "selected_candidate_qty_ton": final.get("selected_candidate_qty_ton"),
            "final_action": final.get("final_action"),
            "final_reason": final.get("final_reason"),
            "additional_check_reason": final.get("additional_check_reason"),
        },
    )



if __name__ == "__main__":
    args = parse_args()
    outputs = run_all_in_one_pipeline(args)
    print_key_outputs(outputs)
