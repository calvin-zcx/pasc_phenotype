# taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity icu --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_ALL_icu-selectpasc.txt
# taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity CPD-COPD --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_all_CPD-COPD-selectpasc.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity Arrythmia --selectpasc 2>&1 | tee  log/screen_dx_insight_ALL_Arrythmia-selectpasc.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity CKD --selectpasc 2>&1 | tee  log/screen_dx_insight_ALL_CKD-selectpasc.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity CPD-COPD --selectpasc 2>&1 | tee  log/screen_dx_insight_ALL_CPD-COPD-selectpasc.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity CAD --selectpasc 2>&1 | tee  log/screen_dx_insight_ALL_CAD-selectpasc.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity Hypertension --selectpasc 2>&1 | tee  log/screen_dx_insight_ALL_Hypertension-selectpasc.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity Corticosteroids --selectpasc 2>&1 | tee  log/screen_dx_insight_ALL_Corticosteroids-selectpasc.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity healthy --selectpasc 2>&1 | tee  log/screen_dx_insight_ALL_healthy-selectpasc.txt
