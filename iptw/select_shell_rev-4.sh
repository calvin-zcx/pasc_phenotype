mkdir log
taskset --cpu-list 12-15 python screen_med_rxnorm_vtrim_vac.py --dataset V15_COVID19 --site ALL --severity all 2>&1 | tee  log/screen_med_rxnorm_insight_ALL_all_Vtrim_Vaccine.txt
taskset --cpu-list 12-15 python screen_med_rxnorm_vtrim_vac.py --dataset oneflorida --site all --severity all 2>&1 | tee  log/screen_med_rxnorm_oneflorida_all_all_Vtrim_Vaccine.txt

