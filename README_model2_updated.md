# model2: 대한제당형 원당 구매의사결정 파이프라인

원당 구매 의사결정을 작은 end-to-end 파이프라인으로 묶은 프로젝트입니다.

이 프로젝트는 단순 가격 예측기가 아니라,

- 외생 데이터(원당 가격 / 환율 / 운임)
- 월별 운영 가정
- baseline no-buy 기준 shortage helper
- 후보안 생성
- 운영 게이트 판정
- 시나리오 시뮬레이션
- 후보 비교 / 선택
- 최종 action 결정

을 한 흐름으로 연결하는 구조입니다.

---

## 1. 현재 저장소 기준 핵심 실행 스크립트

### `run_all_in_one_hgb_pipeline.py`
현재 저장소의 메인 실행 스크립트입니다.

역할:
1. 외생 데이터 준비
2. historical decision master 생성
3. HGB bundle load 또는 fresh fit
4. 최신 row scoring
5. final decision pipeline 실행
6. 주요 DF 출력 및 CSV 저장

지원 모드:
- `--external-mode demo`
- `--external-mode csv`
- `--external-mode build`

추가 옵션:
- `--use-saved-artifacts`
- `--save-artifacts`
- `--prediction-combine-mode {auto, model_only, rule_floor, rule_only}`
- `--save-outputs`

### `build_external_inputs.py`
외생 3개 시계열을 월단위로 만드는 스크립트입니다.

구성:
- sugar: FRED 공식 CSV pull
- usdkrw: 한국은행 ECOS API
- freight: 로컬 CSV 또는 synthetic fallback

### `run_with_hgb_artifacts.py`
저장된 HGB artifact를 불러와 추론하는 실행 스크립트입니다.

### 기타 스크립트
- `run_end_to_end_demo.py`
- `run_pipeline.py`
- `run_with_demo_hgb_artifacts.py`

이 스크립트들은 보조 실행용이며, 현재 구조를 한 번에 보기에는 `run_all_in_one_hgb_pipeline.py`가 중심입니다.

---

## 2. 폴더 구조

```text
model2/
├── README.md
├── START.md
├── build_external_inputs.py
├── run_all_in_one_hgb_pipeline.py
├── run_end_to_end_demo.py
├── run_pipeline.py
├── run_with_demo_hgb_artifacts.py
├── run_with_hgb_artifacts.py
├── synthetic_rules_v1.xlsx
├── outputs/
└── model2_pipeline/
    ├── __init__.py
    ├── baseline_flow.py
    ├── candidates.py
    ├── compare_select.py
    ├── config.py
    ├── decision_generator.py
    ├── final_action.py
    ├── gates.py
    ├── model_features.py
    ├── model_inference.py
    ├── pipeline.py
    ├── scenarios.py
    └── simulation.py
```

---

## 3. 파이프라인 흐름

### Step 1. 외생 데이터 준비
외생 3개 시계열을 월단위로 정렬합니다.

필수 컬럼:
- `as_of_month`
- `global_raw_sugar_price`
- `usdkrw`
- `freight_index`

### Step 2. decision master 생성
`decision_generator.py`에서 historical / latest decision row를 생성합니다.

row unit:
- 원재료 - 월 - 의사결정시점

여기서 붙는 주요 값:
- 현재 재고 / blocked inventory / usable inventory
- usage path / open PO path
- helper
- target rule
- expected landed cost path

### Step 3. HGB scoring
`model_features.py`와 `model_inference.py`를 통해
Target A / Target B를 score 합니다.

combine mode:
- `model_only`
- `rule_floor`
- `rule_only`
- `auto`

현재 fresh-fit 기본은 `rule_floor`입니다.

### Step 4. baseline no-buy world 계산
`baseline_flow.py`

의미:
- 신규 발주를 하지 않는 세계를 먼저 깔고
- 현재 shortage 구조와 helper를 해석하기 위한 기준선을 만듭니다.

### Step 5. 후보안 생성
`candidates.py`

기본 후보:
- `observe`
- `MOQ`
- `MOQ+1lot`
- `shortage_anchored`

주의:
- `shortage_anchored`는 “필요수량 후보” 성격입니다.
- 실행가능성은 여기서 자르지 않고, 뒤 gate에서 판정합니다.

### Step 6. 운영 게이트
`gates.py`

주요 판정:
- MOQ gate
- lot multiple gate
- warehouse gate
- working capital gate
- arrival timing gate

candidate status:
- `feasible`
- `conditional`
- `blocked`

핵심 개념:
- “필요수량”과 “실행가능수량”은 다를 수 있습니다.

### Step 7. 시나리오 시뮬레이션
`scenarios.py`, `simulation.py`

기본 시나리오:
- `base`
- `stress`
- `shock`

개념:
- 수요 multiplier
- 비용 multiplier
- open PO 지연
- candidate 도착 지연
- emergency premium 반영

### Step 8. 후보 비교 / 선택
`compare_select.py`

출력:
- `scenario_summary_df`
- `robust_summary_df`
- `best_candidate_df`

의미:
- 후보별 shortage / cost / robust 여부를 비교해
- 현재 row 기준 최종 후보 1개를 선택합니다.

### Step 9. 최종 action
`final_action.py`

최종 액션:
- `선매입 검토`
- `관망`
- `추가확인`

중요:
- 이 단계는 단순 추천이 아니라
  선택후보 상태, robust 여부, residual shortage, 필요수량 후보의 blocked 여부까지 같이 설명합니다.

---

## 4. 주요 출력 DF

`run_all_in_one_hgb_pipeline.py`는 다음 DF를 주요 산출물로 다룹니다.

- `meta_df`
- `exogenous_df`
- `historical_master_df`
- `scored_latest_df`
- `baseline_flow_df`
- `candidate_df`
- `gated_candidate_df`
- `simulation_result_df`
- `scenario_summary_df`
- `robust_summary_df`
- `best_candidate_df`
- `final_decision_df`

`--save-outputs` 옵션을 사용하면 위 DF들이 CSV로 저장됩니다.

---

## 5. 빠른 실행 예시

### A. CSV 외생 데이터로 실행
```bash
python run_all_in_one_hgb_pipeline.py ^
  --external-mode csv ^
  --external-csv-path actual_external_inputs_monthly.csv ^
  --save-outputs
```

### B. 외생 데이터 직접 build
```bash
python run_all_in_one_hgb_pipeline.py ^
  --external-mode build ^
  --start-month 2023-01-01 ^
  --end-month 2026-08-01 ^
  --ecos-api-key YOUR_KEY ^
  --ecos-stat-code YOUR_STAT_CODE ^
  --save-outputs
```

### C. 저장된 HGB artifact 사용
```bash
python run_all_in_one_hgb_pipeline.py ^
  --external-mode csv ^
  --external-csv-path actual_external_inputs_monthly.csv ^
  --use-saved-artifacts ^
  --model-a-path ./artifacts/target_a_hgb.joblib ^
  --model-b-path ./artifacts/target_b_hgb.joblib ^
  --save-outputs
```

---

## 6. 이 프로젝트를 볼 때 핵심 해석 포인트

이 프로젝트는 “정답 수량 하나 추천기”가 아닙니다.

핵심은 아래를 분리해서 보여주는 데 있습니다.

1. baseline 기준 shortage 구조가 무엇인가
2. 실제 필요한 물량은 얼마인가
3. 그 물량이 실제로 실행 가능한가
4. 실행 가능한 후보들 중 어떤 안이 상대적으로 덜 나쁜가
5. 그래도 최종 확정이 가능한지, 아니면 추가확인이 필요한지

즉,
- 구매언어
- 운영언어
- 데이터언어

를 한 테이블 흐름으로 연결하는 미니 의사결정 파이프라인입니다.
