mkdir log
taskset --cpu-list 0-3 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity less65 2>&1 | tee  log/screen_dx_insight_ALL_less65.txt
taskset --cpu-list 0-3 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity male 2>&1 | tee  log/screen_dx_insight_ALL_male.txt
taskset --cpu-list 0-3 python screen_dx_v2.py --dataset oneflorida --site all --severity 65to75 2>&1 | tee  log/screen_dx_oneflorida_all_65to75.txt
taskset --cpu-list 0-3 python screen_dx_v2.py --dataset oneflorida --site all --severity white 2>&1 | tee  log/screen_dx_oneflorida_all_white.txt






