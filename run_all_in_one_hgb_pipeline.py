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

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

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

    # 4. 최신 row score
    latest_decision_df = historical_master_df.iloc[[-1]].reset_index(drop=True)
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
def print_key_outputs(outputs: Dict[str, pd.DataFrame]) -> None:
    print("\n[meta_df]")
    print(outputs["meta_df"].to_string(index=False))

    print("\n[exogenous_df.tail(8)]")
    print(outputs["exogenous_df"].tail(8).to_string(index=False))

    print("\n[historical_master_df.tail(3)]")
    print(outputs["historical_master_df"].tail(3).to_string(index=False))

    print("\n[scored_latest_df]")
    print(outputs["scored_latest_df"].to_string(index=False))

    print("\n[candidate_df]")
    print(outputs["candidate_df"].to_string(index=False))

    print("\n[gated_candidate_df]")
    print(outputs["gated_candidate_df"].to_string(index=False))

    print("\n[scenario_summary_df]")
    print(outputs["scenario_summary_df"].to_string(index=False))

    print("\n[best_candidate_df]")
    print(outputs["best_candidate_df"].to_string(index=False))

    print("\n[final_decision_df]")
    print(outputs["final_decision_df"].to_string(index=False))


if __name__ == "__main__":
    args = parse_args()
    outputs = run_all_in_one_pipeline(args)
    print_key_outputs(outputs)
