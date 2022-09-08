mkdir log
taskset --cpu-list 8-11 python screen_dx_vnegk.py --dataset V15_COVID19 --site ALL --severity all --negative_ratio 2 2>&1 | tee  log/screen_dx_insight_ALL_all_negk2.txt
taskset --cpu-list 8-11 python screen_dx_vnegk.py --dataset V15_COVID19 --site ALL --severity all --negative_ratio 4 2>&1 | tee  log/screen_dx_insight_ALL_all_negk4.txt
taskset --cpu-list 8-11 python screen_dx_vnegk.py --dataset V15_COVID19 --site ALL --severity all --negative_ratio 6 2>&1 | tee  log/screen_dx_insight_ALL_all_negk6.txt
taskset --cpu-list 8-11 python screen_dx_vnegk.py --dataset V15_COVID19 --site ALL --severity all --negative_ratio 8 2>&1 | tee  log/screen_dx_insight_ALL_all_negk8.txt


