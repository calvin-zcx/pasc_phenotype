mkdir log
#taskset --cpu-list 4-7 python screen_dx_v4.py --dataset V15_COVID19 --site ALL --severity all 2>&1 | tee  log/screen_dx_insight_ALL_all_V4.txt
#taskset --cpu-list 4-7 python screen_dx_v4.py --dataset oneflorida --site all --severity all 2>&1 | tee  log/screen_dx_oneflorida_all_all_V4.txt
taskset --cpu-list 4-7 python screen_dx_vtrim.py --dataset V15_COVID19 --site ALL --severity all 2>&1 | tee  log/screen_dx_insight_ALL_all_Vtrim.txt
taskset --cpu-list 4-7 python screen_dx_vtrim.py --dataset oneflorida --site all --severity all 2>&1 | tee  log/screen_dx_oneflorida_all_all_Vtrim.txt

