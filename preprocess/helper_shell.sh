#!/usr/bin/env bash
python pre_diagnosis.py --dataset COL 2>&1 | tee  log/pre_diagnosis_COL.txt
python pre_diagnosis.py --dataset WCM 2>&1 | tee  log/pre_diagnosis_WCM.txt
python pre_diagnosis.py --dataset NYU 2>&1 | tee  log/pre_diagnosis_NYU.txt
python pre_diagnosis.py --dataset MONTE 2>&1 | tee  log/pre_diagnosis_MONTE.txt
python pre_diagnosis.py --dataset MSHS 2>&1 | tee  log/pre_diagnosis_MSHS.txt