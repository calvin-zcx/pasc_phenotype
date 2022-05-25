python pre_data_manuscript.py --dataset ALL --cohorts covid_4manuscript 2>&1 | tee  log/pre_data_manuscript.txt
python pre_data_manuscript.py --dataset ALL --cohorts covid_4manuNegNoCovid 2>&1 | tee  log/pre_data_manuscript_NegNoCovid.txt
python pre_data_manuscript_negctrl.py --dataset ALL --cohorts covid_4manuNegNoCovid 2>&1 | tee  log/pre_data_manuscript_negctrl_NegNoCovid.txt
python pre_data_manuscript.py --dataset ALL --cohorts covid_4manuNegNoCovidV2 2>&1 | tee  log/pre_data_manuscript_NegNoCovidV2.txt
python pre_data_manuscript.py --dataset ALL --cohorts covid_4manuNegNoCovidV2 2>&1 | tee  log/pre_data_manuscript_NegNoCovidV2_moreRiskAndVaccine.txt
