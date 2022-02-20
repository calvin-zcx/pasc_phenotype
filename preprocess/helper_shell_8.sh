python query_12_cdc.py --dataset ALL --cohorts pasc_incidence 2>&1 | tee  log/query_12_cdc_ALL_pasc_incidence_update3yrbase.txt
python query_12_cdc.py --dataset ALL --cohorts pasc_prevalence 2>&1 | tee  log/query_12_cdc_ALL_pasc_prevalence_update3yrbase.txt
python query_12_cdc.py --dataset ALL --cohorts covid 2>&1 | tee  log/query_12_cdc_ALL_covid_update3yrbase.txt