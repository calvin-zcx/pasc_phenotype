mkdir log
taskset --cpu-list 0-3 python screen_dx_pooled.py --dataset pooled --severity all 2>&1 | tee  log/screen_dx_pooled_all-competingRiskCumInc.txt
taskset --cpu-list 0-3 python screen_dx_pooled.py --dataset pooled --severity icu 2>&1 | tee  log/screen_dx_pooled_icu-competingRiskCumInc.txt
taskset --cpu-list 0-3 python screen_dx_pooled.py --dataset pooled --severity inpatient 2>&1 | tee  log/screen_dx_pooled_inpatient-competingRiskCumInc.txt
taskset --cpu-list 0-3 python screen_dx_pooled.py --dataset pooled --severity outpatient 2>&1 | tee  log/screen_dx_pooled_outpatient-competingRiskCumInc.txt
taskset --cpu-list 0-3 python screen_dx_pooled.py --dataset pooled --severity less65 2>&1 | tee  log/screen_dx_pooled_less65-competingRiskCumInc.txt
taskset --cpu-list 0-3 python screen_dx_pooled.py --dataset pooled --severity 20to40 2>&1 | tee  log/screen_dx_pooled_20to40-competingRiskCumInc.txt
taskset --cpu-list 0-3 python screen_dx_pooled.py --dataset pooled --severity 40to55 2>&1 | tee  log/screen_dx_pooled_40to55-competingRiskCumInc.txt
taskset --cpu-list 0-3 python screen_dx_pooled.py --dataset pooled --severity 55to65 2>&1 | tee  log/screen_dx_pooled_55to65-competingRiskCumInc.txt



