#!/usr/bin/env bash

python pre_cohort.py --dataset COL 2>&1 | tee  log/pre_cohort_INCIDENCE_COL.txt
python pre_cohort.py --dataset WCM 2>&1 | tee  log/pre_cohort_INCIDENCE_WCM.txt
python pre_cohort.py --dataset NYU 2>&1 | tee  log/pre_cohort_INCIDENCE_NYU.txt
python pre_cohort.py --dataset MONTE 2>&1 | tee  log/pre_cohort_INCIDENCE_MONTE.txt
python pre_cohort.py --dataset MSHS 2>&1 | tee  log/pre_cohort_INCIDENCE_MSHS.txt