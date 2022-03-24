mkdir log
#taskset --cpu-list 0-3 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity all 2>&1 | tee  log/screen_dx_insight_ALL_all.txt
#taskset --cpu-list 0-3 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity icu 2>&1 | tee  log/screen_dx_insight_ALL_icu.txt
taskset --cpu-list 0-3 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity inpatient 2>&1 | tee  log/screen_dx_insight_ALL_inpatient.txt
taskset --cpu-list 0-3 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity outpatient 2>&1 | tee  log/screen_dx_insight_ALL_outpatient.txt





