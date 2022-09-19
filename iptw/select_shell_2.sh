mkdir log
#taskset --cpu-list 0-3 python screen_dx_vtrim.py --dataset V15_COVID19 --site ALL --severity outpatient 2>&1 | tee  log/screen_dx_vtrim_insight_ALL_outpatient.txt
#taskset --cpu-list 0-3 python screen_dx_vtrim.py --dataset V15_COVID19 --site ALL --severity inpatienticu 2>&1 | tee  log/screen_dx_vtrim_insight_ALL_inpatienticu.txt
#taskset --cpu-list 0-3 python screen_dx_vtrim.py --dataset V15_COVID19 --site ALL --severity above65 2>&1 | tee  log/screen_dx_vtrim_insight_ALL_above65.txt
#taskset --cpu-list 0-3 python screen_dx_vtrim.py --dataset V15_COVID19 --site ALL --severity less65 2>&1 | tee  log/screen_dx_vtrim_insight_ALL_less65.txt
#taskset --cpu-list 0-3 python screen_dx_vtrim.py --dataset V15_COVID19 --site ALL --severity female 2>&1 | tee  log/screen_dx_vtrim_insight_ALL_female.txt
#taskset --cpu-list 0-3 python screen_dx_vtrim.py --dataset V15_COVID19 --site ALL --severity male 2>&1 | tee  log/screen_dx_vtrim_insight_ALL_male.txt
#taskset --cpu-list 0-3 python screen_dx_vtrim.py --dataset V15_COVID19 --site ALL --severity white 2>&1 | tee  log/screen_dx_vtrim_insight_ALL_white.txt
#taskset --cpu-list 0-3 python screen_dx_vtrim.py --dataset V15_COVID19 --site ALL --severity black 2>&1 | tee  log/screen_dx_vtrim_insight_ALL_black.txt

taskset --cpu-list 4-7 python screen_dx_vtrim.py --dataset V15_COVID19 --site ALL --severity Arrythmia 2>&1 | tee  log/screen_dx_vtrim_insight_ALL_Arrythmia.txt
taskset --cpu-list 4-7 python screen_dx_vtrim.py --dataset V15_COVID19 --site ALL --severity CAD 2>&1 | tee  log/screen_dx_vtrim_insight_ALL_CAD.txt
taskset --cpu-list 4-7 python screen_dx_vtrim.py --dataset V15_COVID19 --site ALL --severity CKD 2>&1 | tee  log/screen_dx_vtrim_insight_ALL_CKD.txt
taskset --cpu-list 4-7 python screen_dx_vtrim.py --dataset V15_COVID19 --site ALL --severity CPD-COPD 2>&1 | tee  log/screen_dx_vtrim_insight_ALL_CPD-COPD.txt
taskset --cpu-list 4-7 python screen_dx_vtrim.py --dataset V15_COVID19 --site ALL --severity Hypertension 2>&1 | tee  log/screen_dx_vtrim_insight_ALL_Hypertension.txt
taskset --cpu-list 4-7 python screen_dx_vtrim.py --dataset V15_COVID19 --site ALL --severity Mental-substance 2>&1 | tee  log/screen_dx_vtrim_insight_ALL_Mental-substance.txt
taskset --cpu-list 4-7 python screen_dx_vtrim.py --dataset V15_COVID19 --site ALL --severity T2D-Obesity 2>&1 | tee  log/screen_dx_vtrim_insight_ALL_T2D-Obesity.txt
taskset --cpu-list 4-7 python screen_dx_vtrim.py --dataset V15_COVID19 --site ALL --severity healthy 2>&1 | tee  log/screen_dx_vtrim_insight_ALL_healthy.txt
taskset --cpu-list 4-7 python screen_dx_vtrim.py --dataset V15_COVID19 --site ALL --severity Anemia 2>&1 | tee  log/screen_dx_vtrim_insight_ALL_Anemia.txt
taskset --cpu-list 4-7 python screen_dx_vtrim.py --dataset V15_COVID19 --site ALL --severity Corticosteroids 2>&1 | tee  log/screen_dx_vtrim_insight_ALL_Corticosteroids.txt

taskset --cpu-list 4-7 python screen_dx_vtrim.py --dataset V15_COVID19 --site ALL --severity 1stwave 2>&1 | tee  log/screen_dx_vtrim_insight_ALL_1stWave.txt
taskset --cpu-list 4-7 python screen_dx_vtrim.py --dataset V15_COVID19 --site ALL --severity delta 2>&1 | tee  log/screen_dx_vtrim_insight_ALL_delta.txt
taskset --cpu-list 4-7 python screen_dx_vtrim.py --dataset V15_COVID19 --site ALL --severity alpha 2>&1 | tee  log/screen_dx_vtrim_insight_ALL_alpha.txt


