mkdir log
#taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity all 2>&1 | tee  log/screen_dx_oneflorida_ALL_all-competingRiskCumInc.txt
#taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity icu 2>&1 | tee  log/screen_dx_oneflorida_ALL_icu-competingRiskCumInc.txt
#taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity inpatient 2>&1 | tee  log/screen_dx_oneflorida_ALL_inpatient-competingRiskCumInc.txt
#taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity outpatient 2>&1 | tee  log/screen_dx_oneflorida_ALL_outpatient-competingRiskCumInc.txt
#
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity 20to40 2>&1 | tee  log/screen_dx_oneflorida_all_20to40-competingRiskCumInc.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity 40to55 2>&1 | tee  log/screen_dx_oneflorida_all_40to55-competingRiskCumInc.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity 55to65 2>&1 | tee  log/screen_dx_oneflorida_all_55to65-competingRiskCumInc.txt
