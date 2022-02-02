#!/usr/bin/env bash
python pre_medication.py --dataset COL 2>&1 | tee  log/pre_medication_COL.txt
python pre_medication.py --dataset WCM 2>&1 | tee  log/pre_medication_WCM.txt
python pre_medication.py --dataset NYU 2>&1 | tee  log/pre_medication_NYU.txt
python pre_medication.py --dataset MONTE 2>&1 | tee  log/pre_medication_MONTE.txt
python pre_medication.py --dataset MSHS 2>&1 | tee  log/pre_medication_MSHS.txt