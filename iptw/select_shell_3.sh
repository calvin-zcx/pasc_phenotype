mkdir log
taskset --cpu-list 8-11 python screen_dx_vtrim.py --dataset oneflorida --site all --severity outpatient 2>&1 | tee  log/screen_dx_vtrim_oneflorida_all_outpatient.txt
taskset --cpu-list 8-11 python screen_dx_vtrim.py --dataset oneflorida --site all --severity inpatienticu 2>&1 | tee  log/screen_dx_vtrim_oneflorida_all_inpatienticu.txt
taskset --cpu-list 8-11 python screen_dx_vtrim.py --dataset oneflorida --site all --severity above65 2>&1 | tee  log/screen_dx_vtrim_oneflorida_all_above65.txt
taskset --cpu-list 8-11 python screen_dx_vtrim.py --dataset oneflorida --site all --severity less65 2>&1 | tee  log/screen_dx_vtrim_oneflorida_all_less65.txt
taskset --cpu-list 8-11 python screen_dx_vtrim.py --dataset oneflorida --site all --severity female 2>&1 | tee  log/screen_dx_vtrim_oneflorida_all_female.txt
taskset --cpu-list 8-11 python screen_dx_vtrim.py --dataset oneflorida --site all --severity male 2>&1 | tee  log/screen_dx_vtrim_oneflorida_all_male.txt
taskset --cpu-list 8-11 python screen_dx_vtrim.py --dataset oneflorida --site all --severity white 2>&1 | tee  log/screen_dx_vtrim_oneflorida_all_white.txt
taskset --cpu-list 8-11 python screen_dx_vtrim.py --dataset oneflorida --site all --severity black 2>&1 | tee  log/screen_dx_vtrim_oneflorida_all_black.txt

taskset --cpu-list 8-11 python screen_dx_vtrim.py --dataset oneflorida --site all --severity Arrythmia 2>&1 | tee  log/screen_dx_vtrim_oneflorida_all_Arrythmia.txt
taskset --cpu-list 8-11 python screen_dx_vtrim.py --dataset oneflorida --site all --severity CAD 2>&1 | tee  log/screen_dx_vtrim_oneflorida_all_CAD.txt
taskset --cpu-list 8-11 python screen_dx_vtrim.py --dataset oneflorida --site all --severity CKD 2>&1 | tee  log/screen_dx_vtrim_oneflorida_all_CKD.txt
taskset --cpu-list 8-11 python screen_dx_vtrim.py --dataset oneflorida --site all --severity CPD-COPD 2>&1 | tee  log/screen_dx_vtrim_oneflorida_all_CPD-COPD.txt
taskset --cpu-list 8-11 python screen_dx_vtrim.py --dataset oneflorida --site all --severity Hypertension 2>&1 | tee  log/screen_dx_vtrim_oneflorida_all_Hypertension.txt
taskset --cpu-list 8-11 python screen_dx_vtrim.py --dataset oneflorida --site all --severity Mental-substance 2>&1 | tee  log/screen_dx_vtrim_oneflorida_all_Mental-substance.txt
taskset --cpu-list 8-11 python screen_dx_vtrim.py --dataset oneflorida --site all --severity T2D-Obesity 2>&1 | tee  log/screen_dx_vtrim_oneflorida_all_T2D-Obesity.txt
taskset --cpu-list 8-11 python screen_dx_vtrim.py --dataset oneflorida --site all --severity healthy 2>&1 | tee  log/screen_dx_vtrim_oneflorida_all_healthy.txt
taskset --cpu-list 8-11 python screen_dx_vtrim.py --dataset oneflorida --site all --severity Anemia 2>&1 | tee  log/screen_dx_vtrim_oneflorida_all_Anemia.txt
taskset --cpu-list 8-11 python screen_dx_vtrim.py --dataset oneflorida --site all --severity Corticosteroids 2>&1 | tee  log/screen_dx_vtrim_oneflorida_all_Corticosteroids.txt

taskset --cpu-list 8-11 python screen_dx_vtrim.py --dataset oneflorida --site all --severity 1stwave 2>&1 | tee  log/screen_dx_vtrim_oneflorida_all_1stWave.txt
taskset --cpu-list 8-11 python screen_dx_vtrim.py --dataset oneflorida --site all --severity delta 2>&1 | tee  log/screen_dx_vtrim_oneflorida_all_delta.txt
taskset --cpu-list 8-11 python screen_dx_vtrim.py --dataset oneflorida --site all --severity alpha 2>&1 | tee  log/screen_dx_vtrim_oneflorida_all_alpha.txt


