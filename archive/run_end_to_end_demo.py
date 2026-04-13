"""모델2 완성형 demo 실행 파일.

실행 목적:
- 분리된 파일 구조가 실제로 어디까지 이어지는지 한 번에 확인
- HGB load/score 자리가 어디인지 확인
- HGB artifact가 없어도 fallback rule로 최종 action까지 먼저 확인
"""

from model2_pipeline.config import PipelineConfig
from model2_pipeline.end_to_end import run_end_to_end_demo


def main() -> None:
    cfg = PipelineConfig()
    outputs = run_end_to_end_demo(cfg=cfg)

    print("\n[exogenous_df.tail()]")
    print(outputs["exogenous_df"].tail(8).to_string(index=False))

    print("\n[historical_master_df.tail()]")
    print(outputs["historical_master_df"].tail(5).to_string(index=False))

    print("\n[scored_latest_df]")
    print(outputs["scored_latest_df"].to_string(index=False))

    print("\n[final_decision_df]")
    print(outputs["final_decision_df"].to_string(index=False))

    print("\nNOTE: 현재 demo는 HGB artifact가 없으면 target_a_rule / target_b_rule을 fallback prediction으로 사용한다.")
    print("NOTE: 실제 HGB artifact를 연결할 때는 run_with_hgb_artifacts.py 또는 end_to_end.run_end_to_end_demo(model_a_path=..., model_b_path=...)를 사용하면 된다.")


if __name__ == "__main__":
    main()
