python pre_data_manuscript.py --dataset ALL --cohorts covid_4manuscript 2>&1 | tee  log/pre_data_manuscript.txt
python pre_data_manuscript.py --dataset ALL --cohorts covid_4manuNegNoCovid 2>&1 | tee  log/pre_data_manuscript_NegNoCovid.txt
