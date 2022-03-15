mkdir log
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity female 2>&1 | tee  log/screen_dx_insight_ALL_female.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity less65 2>&1 | tee  log/screen_dx_oneflorida_all_less65.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity male 2>&1 | tee  log/screen_dx_oneflorida_all_male.txt

