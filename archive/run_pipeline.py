"""모델2 최종 의사결정 파이프라인 실행 예시 파일.

이 파일은:
1) dummy decision master 생성
2) 전체 pipeline 실행
3) 주요 출력표를 콘솔에 표시

실제 데이터 연결 전, 파일 분리 구조가 정상 동작하는지 확인하는 진입점이다.
"""

from model2_pipeline.config import PipelineConfig
from model2_pipeline.data_examples import make_dummy_decision_master_df
from model2_pipeline.pipeline import run_full_decision_pipeline


def main() -> None:
    cfg = PipelineConfig()
    decision_master_df = make_dummy_decision_master_df(cfg=cfg)

    outputs = run_full_decision_pipeline(
        decision_master_df=decision_master_df,
        cfg=cfg,
    )

    print("\n[decision_master_df]")
    print(outputs["decision_master_df"].to_string(index=False))

    print("\n[baseline_flow_df]")
    print(outputs["baseline_flow_df"].to_string(index=False))

    print("\n[candidate_df]")
    print(outputs["candidate_df"].to_string(index=False))

    print("\n[gated_candidate_df]")
    print(outputs["gated_candidate_df"].to_string(index=False))

    print("\n[scenario_summary_df]")
    print(outputs["scenario_summary_df"].to_string(index=False))

    print("\n[robust_summary_df]")
    print(outputs["robust_summary_df"].to_string(index=False))

    print("\n[best_candidate_df]")
    print(outputs["best_candidate_df"].to_string(index=False))

    print("\n[final_decision_df]")
    print(outputs["final_decision_df"].to_string(index=False))


if __name__ == "__main__":
    main()
