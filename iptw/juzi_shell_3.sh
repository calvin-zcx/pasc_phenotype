mkdir log
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity all 2>&1 | tee  log/screen_dx_oneflorida_ALL_all.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity icu 2>&1 | tee  log/screen_dx_oneflorida_ALL_icu.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity inpatient 2>&1 | tee  log/screen_dx_oneflorida_ALL_inpatient.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity outpatient 2>&1 | tee  log/screen_dx_oneflorida_ALL_outpatient.txt
