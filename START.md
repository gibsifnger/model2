# 최초실행시

터미널에서 폴더 위치가 model2_decision_pipeline인지 먼저 확인하고 실행: 
4단계. 가상환경 만들기
python -m venv .venv
.\.venv\Scripts\Activate.ps1

5단계. 패키지 설치
pip install -r requirements.txt

6단계. 데모 모드로 일단 끝까지 돌려보기
python run_all_in_one_hgb_pipeline.py --external-mode demo
# 결과를 파일로 보고싶으면
python run_all_in_one_hgb_pipeline.py --external-mode demo --save-outputs --output-dir .\outputs
- 이때 돌아가는 흐름
- 외생 3개 demo 생성
- historical decision master 생성
- HGB quick-fit
- 최신 row scoring
- gate
- candidate
- scenario simulation
- final action 출력

# 다시 실행
python run_all_in_one_hgb_pipeline.py --external-mode demo --save-outputs --output-dir .\outputs

# demo에서 강제로 rule-floor
python run_all_in_one_hgb_pipeline.py --external-mode demo --prediction-combine-mode rule_floor --save-outputs --output-dir .\outputs

# 실artifact 붙였을 때 model only
python run_all_in_one_hgb_pipeline.py --external-mode demo --use-saved-artifacts --model-a-path .\artifacts\target_a_hgb.joblib --model-b-path .\artifacts\target_b_hgb.joblib --prediction-combine-mode model_only --save-outputs --output-dir .\outputs