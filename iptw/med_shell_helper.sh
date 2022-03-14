python screen_med_rxnorm_v2.py --dataset V15_COVID19 --site ALL --severity icu 2>&1 | tee  log/screen_med_rxnorm_insight_ALL_icu.txt
python screen_med_rxnorm_v2.py --dataset V15_COVID19 --site ALL --severity inpatient 2>&1 | tee  log/screen_med_rxnorm_insight_ALL_inpatient.txt
python screen_med_rxnorm_v2.py --dataset V15_COVID19 --site ALL --severity outpatient 2>&1 | tee  log/screen_med_rxnorm_insight_ALL_outpatient.txt
   