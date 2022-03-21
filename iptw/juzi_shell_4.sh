mkdir log
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity less65 2>&1 | tee  log/screen_dx_oneflorida_all_less65.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity 65to75 2>&1 | tee  log/screen_dx_oneflorida_all_65to75.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity 75above 2>&1 | tee  log/screen_dx_oneflorida_all_75above.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity white 2>&1 | tee  log/screen_dx_oneflorida_all_white.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity black 2>&1 | tee  log/screen_dx_oneflorida_all_black.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity female 2>&1 | tee  log/screen_dx_oneflorida_all_female.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity male 2>&1 | tee  log/screen_dx_oneflorida_all_male.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity CPD-COPD 2>&1 | tee  log/screen_dx_oneflorida_all_CPD-COPD.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity CAD 2>&1 | tee  log/screen_dx_oneflorida_all_CAD.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity Hypertension 2>&1 | tee  log/screen_dx_oneflorida_all_Hypertension.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity Corticosteroids 2>&1 | tee  log/screen_dx_oneflorida_all_Corticosteroids.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity T2D-Obesity 2>&1 | tee  log/screen_dx_oneflorida_all_T2D-Obesity.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity Mental-substance 2>&1 | tee  log/screen_dx_oneflorida_all_Mental-substance.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity Anemia 2>&1 | tee  log/screen_dx_oneflorida_all_Anemia.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity Arrythmia 2>&1 | tee  log/screen_dx_oneflorida_all_Arrythmia.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity CKD 2>&1 | tee  log/screen_dx_oneflorida_all_CKD.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity healthy 2>&1 | tee  log/screen_dx_oneflorida_ALL_healthy.txt