mkdir log
taskset --cpu-list 0-3 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity all --selectpasc 2>&1 | tee  log/screen_dx_insight_ALL_all-selectpasc.txt
taskset --cpu-list 0-3 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity icu --selectpasc 2>&1 | tee  log/screen_dx_insight_ALL_icu-selectpasc.txt
taskset --cpu-list 0-3 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity inpatient --selectpasc 2>&1 | tee  log/screen_dx_insight_ALL_inpatient-selectpasc.txt
taskset --cpu-list 0-3 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity outpatient --selectpasc 2>&1 | tee  log/screen_dx_insight_ALL_outpatient-selectpasc.txt





