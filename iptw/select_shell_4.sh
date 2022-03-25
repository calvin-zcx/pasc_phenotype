mkdir log
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity less65 --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_all_less65-selectpasc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity 65to75 --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_all_65to75-selectpasc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity 75above --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_all_75above-selectpasc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity white --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_all_white-selectpasc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity black --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_all_black-selectpasc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity female --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_all_female-selectpasc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity male --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_all_male-selectpasc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity CPD-COPD --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_all_CPD-COPD-selectpasc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity CAD --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_all_CAD-selectpasc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity Hypertension --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_all_Hypertension-selectpasc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity Corticosteroids --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_all_Corticosteroids-selectpasc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity T2D-Obesity --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_all_T2D-Obesity-selectpasc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity Mental-substance --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_all_Mental-substance-selectpasc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity Anemia --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_all_Anemia-selectpasc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity Arrythmia --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_all_Arrythmia-selectpasc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity CKD --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_all_CKD-selectpasc.txt
taskset --cpu-list 12-15 python screen_dx_v2.py --dataset oneflorida --site all --severity healthy --selectpasc 2>&1 | tee  log/screen_dx_oneflorida_ALL_healthy-selectpasc.txt