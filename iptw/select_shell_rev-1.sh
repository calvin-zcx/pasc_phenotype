mkdir log
taskset --cpu-list 0-3 python screen_med_rxnorm_v3.py --dataset V15_COVID19 --site ALL --severity all 2>&1 | tee  log/screen_med_rxnorm_insight_ALL_all_V3.txt
taskset --cpu-list 0-3 python screen_med_rxnorm_v3.py --dataset oneflorida --site all --severity all 2>&1 | tee  log/screen_med_rxnorm_oneflorida_all_all_V3.txt


