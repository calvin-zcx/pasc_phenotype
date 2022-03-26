mkdir log
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity 65to75 2>&1 | tee  log/screen_dx_insight_ALL_65to75-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity 75above 2>&1 | tee  log/screen_dx_insight_ALL_75above-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity male 2>&1 | tee  log/screen_dx_insight_ALL_male-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity female 2>&1 | tee  log/screen_dx_insight_ALL_female-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity white 2>&1 | tee  log/screen_dx_insight_ALL_white-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity black 2>&1 | tee  log/screen_dx_insight_ALL_black-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity T2D-Obesity 2>&1 | tee  log/screen_dx_insight_ALL_T2D-Obesity-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity Mental-substance 2>&1 | tee  log/screen_dx_insight_ALL_Mental-substance-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity Anemia 2>&1 | tee  log/screen_dx_insight_ALL_Anemia-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity Arrythmia 2>&1 | tee  log/screen_dx_insight_ALL_Arrythmia-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity CKD 2>&1 | tee  log/screen_dx_insight_ALL_CKD-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity CPD-COPD 2>&1 | tee  log/screen_dx_insight_ALL_CPD-COPD-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity CAD 2>&1 | tee  log/screen_dx_insight_ALL_CAD-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity Hypertension 2>&1 | tee  log/screen_dx_insight_ALL_Hypertension-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity Corticosteroids 2>&1 | tee  log/screen_dx_insight_ALL_Corticosteroids-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity healthy 2>&1 | tee  log/screen_dx_insight_ALL_healthy-competingRiskCumInc.txt
