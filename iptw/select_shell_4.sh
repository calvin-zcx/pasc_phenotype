mkdir log
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity less65 2>&1 | tee  log/screen_dx_oneflorida_all_less65-competingRiskCumInc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity 65to75 2>&1 | tee  log/screen_dx_oneflorida_all_65to75-competingRiskCumInc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity 75above 2>&1 | tee  log/screen_dx_oneflorida_all_75above-competingRiskCumInc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity white 2>&1 | tee  log/screen_dx_oneflorida_all_white-competingRiskCumInc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity black 2>&1 | tee  log/screen_dx_oneflorida_all_black-competingRiskCumInc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity female 2>&1 | tee  log/screen_dx_oneflorida_all_female-competingRiskCumInc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity male 2>&1 | tee  log/screen_dx_oneflorida_all_male-competingRiskCumInc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity CPD-COPD 2>&1 | tee  log/screen_dx_oneflorida_all_CPD-COPD-competingRiskCumInc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity CAD 2>&1 | tee  log/screen_dx_oneflorida_all_CAD-competingRiskCumInc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity Hypertension 2>&1 | tee  log/screen_dx_oneflorida_all_Hypertension-competingRiskCumInc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity Corticosteroids 2>&1 | tee  log/screen_dx_oneflorida_all_Corticosteroids-competingRiskCumInc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity T2D-Obesity 2>&1 | tee  log/screen_dx_oneflorida_all_T2D-Obesity-competingRiskCumInc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity Mental-substance 2>&1 | tee  log/screen_dx_oneflorida_all_Mental-substance-competingRiskCumInc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity Anemia 2>&1 | tee  log/screen_dx_oneflorida_all_Anemia-competingRiskCumInc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity Arrythmia 2>&1 | tee  log/screen_dx_oneflorida_all_Arrythmia-competingRiskCumInc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity CKD 2>&1 | tee  log/screen_dx_oneflorida_all_CKD-competingRiskCumInc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity healthy 2>&1 | tee  log/screen_dx_oneflorida_ALL_healthy-competingRiskCumInc.txt