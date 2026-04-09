"""포함된 demo HGB artifact로 바로 실행하는 파일.

사용:
    python run_with_demo_hgb_artifacts.py
"""

from pathlib import Path

from model2_pipeline.config import PipelineConfig
from model2_pipeline.end_to_end import run_end_to_end_demo


def main() -> None:
    cfg = PipelineConfig()

    outputs = run_end_to_end_demo(
        cfg=cfg,
        model_a_path=Path("./demo_model_artifacts/target_a_hgb_demo.joblib"),
        model_b_path=Path("./demo_model_artifacts/target_b_hgb_demo.joblib"),
    )

    print("\n[scored_latest_df]")
    print(outputs["scored_latest_df"].to_string(index=False))

    print("\n[final_decision_df]")
    print(outputs["final_decision_df"].to_string(index=False))


if __name__ == "__main__":
    main()
