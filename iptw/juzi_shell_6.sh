mkdir log
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity CPD-COPD 2>&1 | tee  log/screen_dx_insight_ALL_CPD-COPD.txt
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity CAD 2>&1 | tee  log/screen_dx_insight_ALL_CAD.txt
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity Hypertension 2>&1 | tee  log/screen_dx_insight_ALL_Hypertension.txt
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity Corticosteroids 2>&1 | tee  log/screen_dx_insight_ALL_Corticosteroids.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity T2D-Obesity 2>&1 | tee  log/screen_dx_oneflorida_all_T2D-Obesity.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity Mental-substance 2>&1 | tee  log/screen_dx_oneflorida_all_Mental-substance.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity Anemia 2>&1 | tee  log/screen_dx_oneflorida_all_Anemia.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity Arrythmia 2>&1 | tee  log/screen_dx_oneflorida_all_Arrythmia.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity CKD 2>&1 | tee  log/screen_dx_oneflorida_all_CKD.txt
