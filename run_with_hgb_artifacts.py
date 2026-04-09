"""실제 HGB artifact 연결용 실행 예시.

사용 예시:
    python run_with_hgb_artifacts.py

전제:
- 아래 경로에 target_a / target_b HGB artifact가 존재
- artifact는 model_inference.load_model_bundle 형식을 따른다
"""

from pathlib import Path

from model2_pipeline.config import PipelineConfig
from model2_pipeline.end_to_end import run_end_to_end_demo


def main() -> None:
    cfg = PipelineConfig()

    outputs = run_end_to_end_demo(
        cfg=cfg,
        model_a_path=Path("./model_artifacts/target_a_hgb.joblib"),
        model_b_path=Path("./model_artifacts/target_b_hgb.joblib"),
    )

    print("\n[scored_latest_df]")
    print(outputs["scored_latest_df"].to_string(index=False))

    print("\n[final_decision_df]")
    print(outputs["final_decision_df"].to_string(index=False))


if __name__ == "__main__":
    main()
