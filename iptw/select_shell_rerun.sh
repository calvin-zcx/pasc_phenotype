taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity icu --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_ALL_icu-selectpasc.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity CPD-COPD --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_all_CPD-COPD-selectpasc.txt
