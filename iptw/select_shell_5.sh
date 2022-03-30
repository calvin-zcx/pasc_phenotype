mkdir log
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity inpatienticu 2>&1 | tee  log/screen_dx_insight_ALL_inpatienticu-competingRiskCumInc.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity inpatienticu 2>&1 | tee  log/screen_dx_oneflorida_ALL_inpatienticu-competingRiskCumInc.txt
