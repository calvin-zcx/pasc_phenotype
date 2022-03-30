mkdir log
taskset --cpu-list 4-7 python screen_dx_pooled.py --dataset pooled --severity 65to75 2>&1 | tee  log/screen_dx_pooled_65to75-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_pooled.py --dataset pooled --severity 75above 2>&1 | tee  log/screen_dx_pooled_75above-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_pooled.py --dataset pooled --severity male 2>&1 | tee  log/screen_dx_pooled_male-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_pooled.py --dataset pooled --severity female 2>&1 | tee  log/screen_dx_pooled_female-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_pooled.py --dataset pooled --severity white 2>&1 | tee  log/screen_dx_pooled_white-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_pooled.py --dataset pooled --severity black 2>&1 | tee  log/screen_dx_pooled_black-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_pooled.py --dataset pooled --severity T2D-Obesity 2>&1 | tee  log/screen_dx_pooled_T2D-Obesity-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_pooled.py --dataset pooled --severity Mental-substance 2>&1 | tee  log/screen_dx_pooled_Mental-substance-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_pooled.py --dataset pooled --severity Anemia 2>&1 | tee  log/screen_dx_pooled_Anemia-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_pooled.py --dataset pooled --severity Arrythmia 2>&1 | tee  log/screen_dx_pooled_Arrythmia-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_pooled.py --dataset pooled --severity CKD 2>&1 | tee  log/screen_dx_pooled_CKD-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_pooled.py --dataset pooled --severity CPD-COPD 2>&1 | tee  log/screen_dx_pooled_CPD-COPD-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_pooled.py --dataset pooled --severity CAD 2>&1 | tee  log/screen_dx_pooled_CAD-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_pooled.py --dataset pooled --severity Hypertension 2>&1 | tee  log/screen_dx_pooled_Hypertension-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_pooled.py --dataset pooled --severity Corticosteroids 2>&1 | tee  log/screen_dx_pooled_Corticosteroids-competingRiskCumInc.txt
taskset --cpu-list 4-7 python screen_dx_pooled.py --dataset pooled --severity healthy 2>&1 | tee  log/screen_dx_pooled_healthy-competingRiskCumInc.txt
