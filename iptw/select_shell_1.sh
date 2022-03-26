mkdir log
taskset --cpu-list 0-3 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity all 2>&1 | tee  log/screen_dx_insight_ALL_all-competingRiskCumInc.txt
taskset --cpu-list 0-3 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity icu 2>&1 | tee  log/screen_dx_insight_ALL_icu-competingRiskCumInc.txt
taskset --cpu-list 0-3 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity inpatient 2>&1 | tee  log/screen_dx_insight_ALL_inpatient-competingRiskCumInc.txt
taskset --cpu-list 0-3 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity outpatient 2>&1 | tee  log/screen_dx_insight_ALL_outpatient-competingRiskCumInc.txt
taskset --cpu-list 0-3 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity less65 2>&1 | tee  log/screen_dx_insight_ALL_less65-competingRiskCumInc.txt





