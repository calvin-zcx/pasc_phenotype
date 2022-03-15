mkdir log
python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity 65to75 2>&1 | tee  log/screen_dx_insight_ALL_65to75.txt
python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity white 2>&1 | tee  log/screen_dx_insight_ALL_white.txt
python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity female 2>&1 | tee  log/screen_dx_insight_ALL_female.txt
python screen_dx_v2.py --dataset oneflorida --site all --severity 75above 2>&1 | tee  log/screen_dx_oneflorida_all_75above.txt
python screen_dx_v2.py --dataset oneflorida --site all --severity less65 2>&1 | tee  log/screen_dx_oneflorida_all_less65.txt
python screen_dx_v2.py --dataset oneflorida --site all --severity black 2>&1 | tee  log/screen_dx_oneflorida_all_black.txt
python screen_dx_v2.py --dataset oneflorida --site all --severity male 2>&1 | tee  log/screen_dx_oneflorida_all_male.txt
