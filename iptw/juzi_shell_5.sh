mkdir log
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity T2D-Obesity 2>&1 | tee  log/screen_dx_insight_ALL_T2D-Obesity.txt
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity Mental-substance 2>&1 | tee  log/screen_dx_insight_ALL_Mental-substance.txt
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity Anemia 2>&1 | tee  log/screen_dx_insight_ALL_Anemia.txt
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity Arrythmia 2>&1 | tee  log/screen_dx_insight_ALL_Arrythmia.txt
taskset --cpu-list 4-7 python screen_dx_v2.py --dataset V15_COVID19 --site ALL --severity CKD 2>&1 | tee  log/screen_dx_insight_ALL_CKD.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity CPD-COPD 2>&1 | tee  log/screen_dx_oneflorida_all_CPD-COPD.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity CAD 2>&1 | tee  log/screen_dx_oneflorida_all_CAD.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity Hypertension 2>&1 | tee  log/screen_dx_oneflorida_all_Hypertension.txt
taskset --cpu-list 8-11 python screen_dx_v2.py --dataset oneflorida --site all --severity Corticosteroids 2>&1 | tee  log/screen_dx_oneflorida_all_Corticosteroids.txt