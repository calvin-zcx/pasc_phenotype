echo helper_2dx.sh
python pre_data_manuscript_2dx.py --dataset ALL --cohorts covid_4manuNegNoCovidV2 --ndays 30 2>&1 | tee  log/pre_data_manuscript_2dx30daysApart_covid_4manuNegNoCovidV2.txt
python pre_data_manuscript_2dx.py --dataset ALL --cohorts covid_4manuNegNoCovidV2 --ndays 1 2>&1 | tee  log/pre_data_manuscript_2dx1daysApart_covid_4manuNegNoCovidV2.txt
