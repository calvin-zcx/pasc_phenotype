taskset --cpu-list 0-3 python compare_dx_twodata.py --dataset V15_COVID19 --site ALL --severity all --selectpasc 2>&1 | tee  log/compare_dx_twodata_insightVSflorida_ALL_all-selectpasc.txt
taskset --cpu-list 0-3 python compare_dx_twodata.py --dataset V15_COVID19 --site ALL --severity icu --selectpasc 2>&1 | tee  log/compare_dx_twodata_insightVSflorida_ALL_icu-selectpasc.txt
taskset --cpu-list 0-3 python compare_dx_twodata.py --dataset V15_COVID19 --site ALL --severity inpatient --selectpasc 2>&1 | tee  log/compare_dx_twodata_insightVSflorida_ALL_inpatient-selectpasc.txt
taskset --cpu-list 0-3 python compare_dx_twodata.py --dataset V15_COVID19 --site ALL --severity outpatient --selectpasc 2>&1 | tee  log/compare_dx_twodata_insightVSflorida_ALL_outpatient-selectpasc.txt
taskset --cpu-list 0-3 python compare_dx_twodata.py --dataset V15_COVID19 --site ALL --severity less65 --selectpasc 2>&1 | tee  log/compare_dx_twodata_insightVSflorida_ALL_less65-selectpasc.txt
taskset --cpu-list 0-3 python compare_dx_twodata.py --dataset V15_COVID19 --site ALL --severity 65to75 --selectpasc 2>&1 | tee  log/compare_dx_twodata_insightVSflorida_ALL_65to75-selectpasc.txt
taskset --cpu-list 0-3 python compare_dx_twodata.py --dataset V15_COVID19 --site ALL --severity 75above --selectpasc 2>&1 | tee  log/compare_dx_twodata_insightVSflorida_ALL_75above-selectpasc.txt
taskset --cpu-list 0-3 python compare_dx_twodata.py --dataset V15_COVID19 --site ALL --severity male --selectpasc 2>&1 | tee  log/compare_dx_twodata_insightVSflorida_ALL_male-selectpasc.txt
taskset --cpu-list 0-3 python compare_dx_twodata.py --dataset V15_COVID19 --site ALL --severity female --selectpasc 2>&1 | tee  log/compare_dx_twodata_insightVSflorida_ALL_female-selectpasc.txt
taskset --cpu-list 0-3 python compare_dx_twodata.py --dataset V15_COVID19 --site ALL --severity white --selectpasc 2>&1 | tee  log/compare_dx_twodata_insightVSflorida_ALL_white-selectpasc.txt
taskset --cpu-list 0-3 python compare_dx_twodata.py --dataset V15_COVID19 --site ALL --severity black --selectpasc 2>&1 | tee  log/compare_dx_twodata_insightVSflorida_ALL_black-selectpasc.txt
taskset --cpu-list 0-3 python compare_dx_twodata.py --dataset V15_COVID19 --site ALL --severity T2D-Obesity --selectpasc 2>&1 | tee  log/compare_dx_twodata_insightVSflorida_ALL_T2D-Obesity-selectpasc.txt
taskset --cpu-list 0-3 python compare_dx_twodata.py --dataset V15_COVID19 --site ALL --severity Mental-substance --selectpasc 2>&1 | tee  log/compare_dx_twodata_insightVSflorida_ALL_Mental-substance-selectpasc.txt
taskset --cpu-list 0-3 python compare_dx_twodata.py --dataset V15_COVID19 --site ALL --severity Anemia --selectpasc 2>&1 | tee  log/compare_dx_twodata_insightVSflorida_ALL_Anemia-selectpasc.txt
taskset --cpu-list 0-3 python compare_dx_twodata.py --dataset V15_COVID19 --site ALL --severity Arrythmia --selectpasc 2>&1 | tee  log/compare_dx_twodata_insightVSflorida_ALL_Arrythmia-selectpasc.txt
taskset --cpu-list 0-3 python compare_dx_twodata.py --dataset V15_COVID19 --site ALL --severity CKD --selectpasc 2>&1 | tee  log/compare_dx_twodata_insightVSflorida_ALL_CKD-selectpasc.txt
taskset --cpu-list 0-3 python compare_dx_twodata.py --dataset V15_COVID19 --site ALL --severity CPD-COPD --selectpasc 2>&1 | tee  log/compare_dx_twodata_insightVSflorida_ALL_CPD-COPD-selectpasc.txt
taskset --cpu-list 0-3 python compare_dx_twodata.py --dataset V15_COVID19 --site ALL --severity CAD --selectpasc 2>&1 | tee  log/compare_dx_twodata_insightVSflorida_ALL_CAD-selectpasc.txt
taskset --cpu-list 0-3 python compare_dx_twodata.py --dataset V15_COVID19 --site ALL --severity Hypertension --selectpasc 2>&1 | tee  log/compare_dx_twodata_insightVSflorida_ALL_Hypertension-selectpasc.txt
taskset --cpu-list 0-3 python compare_dx_twodata.py --dataset V15_COVID19 --site ALL --severity Corticosteroids --selectpasc 2>&1 | tee  log/compare_dx_twodata_insightVSflorida_ALL_Corticosteroids-selectpasc.txt
taskset --cpu-list 0-3 python compare_dx_twodata.py --dataset V15_COVID19 --site ALL --severity healthy --selectpasc 2>&1 | tee  log/compare_dx_twodata_insightVSflorida_ALL_healthy-selectpasc.txt


