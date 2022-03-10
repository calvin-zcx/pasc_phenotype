
python pre_lab.py --dataset covid_database 2>&1 | tee  log/pre_lab_COL.txt
python pre_lab.py --dataset main_database 2>&1 | tee  log/pre_lab_WCM.txt
python pre_demo.py --dataset all 2>&1 | tee  log/pre_demo_allcombined.txt
python pre_covid_lab.py --dataset all 2>&1 | tee  log/pre_covid_lab_allcombined.txt