mkdir log
#taskset --cpu-list 8-11 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity inpatienticu 2>&1 | tee  log/screen_dx_insight_ALL_inpatienticu-competingRiskCumInc.txt
#taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity inpatienticu 2>&1 | tee  log/screen_dx_oneflorida_ALL_inpatienticu-competingRiskCumInc.txt
#taskset --cpu-list 8-11 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity above65 2>&1 | tee  log/screen_dx_insight_ALL_above65-competingRiskCumInc.txt
#taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity above65 2>&1 | tee  log/screen_dx_oneflorida_ALL_above65-competingRiskCumInc.txt
#
#taskset --cpu-list 8-11 python screen_dx_v3.py --dataset V15_COVID19 --site ALL --severity 03-20-06-20 2>&1 | tee  log/screen_dx_insight_ALL_03-20-06-20-competingRiskCumInc.txt
#taskset --cpu-list 8-11 python screen_dx_v3.py --dataset V15_COVID19 --site ALL --severity 07-20-10-20 2>&1 | tee  log/screen_dx_insight_ALL_07-20-10-20-competingRiskCumInc.txt
#taskset --cpu-list 8-11 python screen_dx_v3.py --dataset V15_COVID19 --site ALL --severity 11-20-02-21 2>&1 | tee  log/screen_dx_insight_ALL_11-20-02-21-competingRiskCumInc.txt
#taskset --cpu-list 8-11 python screen_dx_v3.py --dataset V15_COVID19 --site ALL --severity 03-21-06-21 2>&1 | tee  log/screen_dx_insight_ALL_03-21-06-21-competingRiskCumInc.txt
#taskset --cpu-list 8-11 python screen_dx_v3.py --dataset V15_COVID19 --site ALL --severity 07-21-11-21 2>&1 | tee  log/screen_dx_insight_ALL_07-21-11-21-competingRiskCumInc.txt
#
# taskset --cpu-list 8-11 python screen_dx_v3.py --dataset V15_COVID19 --site ALL --severity 1stwave 2>&1 | tee  log/screen_dx_insight_ALL_1stWave-competingRiskCumInc.txt
#taskset --cpu-list 8-11 python screen_dx_v3.py --dataset V15_COVID19 --site ALL --severity delta 2>&1 | tee  log/screen_dx_insight_ALL_Delta-competingRiskCumInc.txt
taskset --cpu-list 8-11 python screen_dx_pooled.py --dataset pooled --severity 1stwave 2>&1 | tee  log/screen_dx_POOLED_ALL_1stWave-competingRiskCumInc.txt
taskset --cpu-list 8-11 python screen_dx_pooled.py --dataset pooled --severity delta 2>&1 | tee  log/screen_dx_POOLED_ALL_Delta-competingRiskCumInc.txt
