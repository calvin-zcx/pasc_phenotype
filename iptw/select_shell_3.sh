mkdir log
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity all --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_ALL_all-selectpasc.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity icu --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_ALL_icu-selectpasc.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity inpatient --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_ALL_inpatient-selectpasc.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity outpatient --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_ALL_outpatient-selectpasc.txt
