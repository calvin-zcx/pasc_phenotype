python pre_vital.py --dataset COL 2>&1 | tee  log/pre_vital_COL.txt
python pre_vital.py --dataset WCM 2>&1 | tee  log/pre_vital_WCM.txt
python pre_vital.py --dataset NYU 2>&1 | tee  log/pre_vital_NYU.txt
python pre_vital.py --dataset MONTE 2>&1 | tee  log/pre_vital_MONTE.txt
python pre_vital.py --dataset MSHS 2>&1 | tee  log/pre_vital_MSHS.txt