mkdir log
python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity 75above 2>&1 | tee  log/screen_dx_insight_ALL_75above.txt
python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity black 2>&1 | tee  log/screen_dx_insight_ALL_black.txt
python screen_dx_v2.py --dataset oneflorida --site all --severity female 2>&1 | tee  log/screen_dx_oneflorida_all_female.txt
