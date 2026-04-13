# Model 2: 대한제당형 원당 구매 의사결정 MVP

모델1처럼 하나의 파일에 전부 넣지 않고, **역할별로 파일을 분리한 mini decision pipeline** 구조입니다.

## 폴더 구조

```text
model2_decision_pipeline/
├── README.md
├── requirements.txt
├── run_pipeline.py
└── model2_pipeline/
    ├── __init__.py
    ├── config.py
    ├── schemas.py
    ├── utils.py
    ├── baseline_flow.py
    ├── candidates.py
    ├── gates.py
    ├── scenarios.py
    ├── simulation.py
    ├── compare_select.py
    ├── final_action.py
    ├── data_examples.py
    └── pipeline.py
```

## 핵심 구조

1. `decision_master_df`
   - row unit = 원재료-월-의사결정시점
2. `baseline_flow_df`
   - baseline no-buy world를 horizon long table로 펼침
3. `candidate_df`
   - observe / MOQ / MOQ+1lot / shortage_anchored 생성
4. `gated_candidate_df`
   - blocked / conditional / feasible 판정
5. `simulation_result_df`
   - candidate x scenario x month 결과
6. `scenario_summary_df`, `robust_summary_df`
   - 시나리오 비교 요약
7. `best_candidate_df`
   - 최종 후보 1개 선택
8. `final_decision_df`
   - 선매입 검토 / 추가확인 / 관망

## 실행 방법

```bash
pip install -r requirements.txt
python run_pipeline.py
```

## 지금 단계에서 일부러 단순화한 것

- scenario multiplier는 skeleton용 기본값
- working capital gate는 score threshold 기반
- shortage anchored 수량은 `a_min_end_inv_ton` 중심 보수 로직
- final action mapping은 mini pipeline용 단순 rule

## 실제 데이터 붙일 때 가장 먼저 확인할 것

- `decision_id` 유일성
- `usage_m1~m4`, `open_po_m1~m4`, `expected_landed_cost_m1~m4`의 month axis 일치
- helper가 baseline no-buy world에서 계산되었는지
- current_inventory / open PO 중복 반영 없는지
- landed cost 기준 범위 통일(CIF/도착원가/부대비 포함범위)
