python pre_procedure.py --dataset COL 2>&1 | tee  log/pre_procedure_COL.txt
python pre_procedure.py --dataset WCM 2>&1 | tee  log/pre_procedure_WCM.txt
python pre_procedure.py --dataset NYU 2>&1 | tee  log/pre_procedure_NYU.txt
python pre_procedure.py --dataset MONTE 2>&1 | tee  log/pre_procedure_MONTE.txt
python pre_procedure.py --dataset MSHS 2>&1 | tee  log/pre_procedure_MSHS.txt