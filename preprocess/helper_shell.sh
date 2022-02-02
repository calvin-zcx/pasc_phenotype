#!/usr/bin/env bash
python pre_covid_lab.py --dataset COL 2>&1 | tee  log/pre_covid_lab_COL.txt
python pre_covid_lab.py --dataset WCM 2>&1 | tee  log/pre_covid_lab_WCM.txt
python pre_covid_lab.py --dataset NYU 2>&1 | tee  log/pre_covid_lab_NYU.txt
python pre_covid_lab.py --dataset MONTE 2>&1 | tee  log/pre_covid_lab_MONTE.txt
python pre_covid_lab.py --dataset MSHS 2>&1 | tee  log/pre_covid_lab_MSHS.txt