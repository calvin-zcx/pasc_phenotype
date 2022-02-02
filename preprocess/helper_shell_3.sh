#!/usr/bin/env bash
python pre_encounter.py --dataset COL 2>&1 | tee  log/pre_encounter_COL.txt
python pre_encounter.py --dataset WCM 2>&1 | tee  log/pre_encounter_WCM.txt
python pre_encounter.py --dataset NYU 2>&1 | tee  log/pre_encounter_NYU.txt
python pre_encounter.py --dataset MONTE 2>&1 | tee  log/pre_encounter_MONTE.txt
python pre_encounter.py --dataset MSHS 2>&1 | tee  log/pre_encounter_MSHS.txt