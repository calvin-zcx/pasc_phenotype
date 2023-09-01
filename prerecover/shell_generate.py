import sys

# for linux env.
sys.path.insert(0, '..')
import os
import pickle
import numpy as np
from collections import defaultdict, OrderedDict
import pandas as pd
import requests
import functools
from misc import utils
import re
from tqdm import tqdm
import psycopg2
import urllib
import time
from sqlalchemy import create_engine
import json
from datetime import datetime
from misc import utils
print = functools.partial(print, flush=True)

def shell_for_each():
    # python pre_codemapping.py 2>&1 | tee  log/pre_codemapping_zip_adi.txt
    start_time = time.time()

    df_site = pd.read_excel('RECOVER Adult Site schemas_edit.xlsx')

    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']

    for i, site in enumerate(site_list):
        site = site.strip()
        cmdstr = """python pre_lab_4covid.py --dataset nyu 2>&1 | tee  log\pre_lab_nyu.txt
    python pre_demo.py --dataset nyu 2>&1 | tee  log\pre_demo_nyu.txt
    python pre_covid_lab.py --dataset nyu 2>&1 | tee  log\pre_covid_lab_nyu.txt
    python pre_diagnosis.py --dataset nyu 2>&1 | tee  log/pre_diagnosis_nyu.txt
    python pre_medication.py --dataset nyu 2>&1 | tee  log/pre_medication_nyu.txt
    python pre_encounter.py --dataset nyu 2>&1 | tee  log/pre_encounter_nyu.txt
    python pre_procedure.py --dataset nyu 2>&1 | tee  log/pre_procedure_nyu.txt
    python pre_immun.py --dataset nyu 2>&1 | tee  log/pre_immun_nyu.txt
    python pre_death.py --dataset nyu 2>&1 | tee  log/pre_death_nyu.txt
    python pre_vital.py --dataset nyu 2>&1 | tee  log/pre_vital_nyu.txt
    python pre_cohort_manuscript.py --dataset nyu 2>&1 | tee  log/pre_cohort_manuscript_nyu.txt
    python pre_data_manuscript_withAllDays.py --dataset nyu 2>&1 | tee  log\pre_data_manuscript_withAllDays_nyu.txt
            """.replace('nyu', site)
        with open(r'output\shells\shell_for_{}.ps1'.format(site), 'wt') as f:
            f.write(cmdstr)

        print(i, site, 'done')
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


cmdstr = """python pre_cohort_manuscript_age18.py --dataset nyu 2>&1 | tee  log/pre_cohort_manuscript_age18_nyu.txt
python pre_data_manuscript_withAllDays.py --cohorts covid_4manuNegNoCovidV2age18 --dataset nyu 2>&1 | tee  log\pre_data_manuscript_withAllDays_nyu.txt
"""

def last_two_steps():
    # python pre_codemapping.py 2>&1 | tee  log/pre_codemapping_zip_adi.txt
    start_time = time.time()

    df_site = pd.read_excel('RECOVER Adult Site schemas_edit.xlsx')

    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']

    with open(r'output\shells\shell_age18_last2step-all.ps1', 'wt') as f:
        for i, site in enumerate(site_list):
            site = site.strip()
            cmdstr = """python pre_cohort_manuscript_age18.py --dataset nyu 2>&1 | tee  log/pre_cohort_manuscript_age18_nyu.txt
    python pre_data_manuscript_withAllDays.py --cohorts covid_4manuNegNoCovidV2age18 --dataset nyu 2>&1 | tee  log\pre_data_manuscript_withAllDays_nyu.txt
    """.replace('nyu', site)
            f.write(cmdstr)

            print(i, site, 'done')

    utils.split_shell_file("output\shells\shell_age18_last2step-all.ps1", divide=4, skip_first=0)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


def shell_2023_4_6():
    # python pre_codemapping.py 2>&1 | tee  log/pre_codemapping_zip_adi.txt
    start_time = time.time()

    df_site = pd.read_excel('RECOVER Adult Site schemas_edit.xlsx')

    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']

    print('site_list:', len(site_list), site_list)

    with open(r'output\shells\shell_all_rerun.ps1', 'wt') as f:
        for i, site in enumerate(site_list):
            site = site.strip()
            cmdstr = """python pre_lab_4covid.py --dataset nyu 2>&1 | tee  log\pre_lab_nyu.txt
    python pre_demo.py --dataset nyu 2>&1 | tee  log\pre_demo_nyu.txt
    python pre_covid_lab.py --dataset nyu 2>&1 | tee  log\pre_covid_lab_nyu.txt
    python pre_diagnosis.py --dataset nyu 2>&1 | tee  log/pre_diagnosis_nyu.txt
    python pre_medication.py --dataset nyu 2>&1 | tee  log/pre_medication_nyu.txt
    python pre_encounter.py --dataset nyu 2>&1 | tee  log/pre_encounter_nyu.txt
    python pre_procedure.py --dataset nyu 2>&1 | tee  log/pre_procedure_nyu.txt
    python pre_immun.py --dataset nyu 2>&1 | tee  log/pre_immun_nyu.txt
    python pre_death.py --dataset nyu 2>&1 | tee  log/pre_death_nyu.txt
    python pre_vital.py --dataset nyu 2>&1 | tee  log/pre_vital_nyu.txt
    python pre_cohort_manuscript_age18.py --dataset nyu 2>&1 | tee  log/pre_cohort_manuscript_age18_nyu.txt
    python pre_data_manuscript_withAllDays.py --cohorts covid_4manuNegNoCovidV2age18 --dataset nyu 2>&1 | tee  log\pre_data_manuscript_withAllDays_nyu.txt
    """.replace('nyu', site)
            f.write(cmdstr)

            print(i, site, 'done')

    utils.split_shell_file(r"output\shells\shell_all_rerun.ps1", divide=5, skip_first=0)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

def shell_2023_6_2():
    # python pre_codemapping.py 2>&1 | tee  log/pre_codemapping_zip_adi.txt
    start_time = time.time()

    df_site = pd.read_excel('RECOVER Adult Site schemas_edit.xlsx')

    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']

    print('site_list:', len(site_list), site_list)

    with open(r'output\shells\shell_all_pregcoexist.ps1', 'wt') as f:
        for i, site in enumerate(site_list):
            site = site.strip()
            cmdstr = """python pre_data_manuscript_withAllDays_preg_addCoexist.py --dataset nyu --cohorts covid_4manuNegNoCovidV2age18 2>&1 | tee  log/pre_data_manuscript_withAllDays_preg_addCoexist_nyu.txt
    """.replace('nyu', site)
            f.write(cmdstr)

            print(i, site, 'done')

    utils.split_shell_file(r"output\shells\shell_all_pregcoexist.ps1", divide=6, skip_first=0)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

def backup_9_1():
    # python pre_codemapping.py 2>&1 | tee  log/pre_codemapping_zip_adi.txt
    start_time = time.time()

    df_site = pd.read_excel('RECOVER Adult Site schemas_edit.xlsx')

    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']

    print('site_list:', len(site_list), site_list)

    with open(r'output\shells\shell_all_2023-6.ps1', 'wt') as f:
        for i, site in enumerate(site_list):
            site = site.strip()
            cmdstr = """python pre_lab_4covid.py --dataset emory 2>&1 | tee  log\pre_lab_emory_2023_6.txt
    python pre_demo.py --dataset emory 2>&1 | tee  log\pre_demo_emory_2023_6.txt
    python pre_covid_lab.py --dataset emory 2>&1 | tee  log\pre_covid_lab_emory_2023_6.txt
    python pre_diagnosis.py --dataset emory 2>&1 | tee  log/pre_diagnosis_emory_2023_6.txt
    python pre_medication.py --dataset emory 2>&1 | tee  log/pre_medication_emory_2023_6.txt
    python pre_encounter.py --dataset emory 2>&1 | tee  log/pre_encounter_emory_2023_6.txt
    python pre_procedure.py --dataset emory 2>&1 | tee  log/pre_procedure_emory_2023_6.txt
    python pre_immun.py --dataset emory 2>&1 | tee  log/pre_immun_emory_2023_6.txt
    python pre_death.py --dataset emory 2>&1 | tee  log/pre_death_emory_2023_6.txt
    python pre_vital.py --dataset emory 2>&1 | tee  log/pre_vital_emory_2023_6.txt
    python pre_cohort_manuscript_age18.py --dataset emory 2>&1 | tee  log/pre_cohort_manuscript_age18_emory_2023_6.txt
    python pre_data_manuscript_withAllDays.py --cohorts covid_4manuNegNoCovidV2age18 --dataset emory 2>&1 | tee  log\pre_data_manuscript_withAllDays_emory_2023_6.txt
    python pre_data_manuscript_withAllDays_preg_addCoexist.py --dataset emory --cohorts covid_4manuNegNoCovidV2age18 2>&1 | tee  log/pre_data_manuscript_withAllDays_preg_addCoexist_emory_2023_6.txt
    """.replace('emory', site)
            f.write(cmdstr)

            print(i, site, 'done')

    # utils.split_shell_file(r"output\shells\shell_all_2023-6.ps1", divide=4, skip_first=0)


def shell_lab_dx_med_4covid():
    # python pre_codemapping.py 2>&1 | tee  log/pre_codemapping_zip_adi.txt
    start_time = time.time()

    df_site = pd.read_excel('RECOVER Adult Site schemas_edit.xlsx')

    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']

    print('site_list:', len(site_list), site_list)

    with open(r'shell_all_202309.ps1', 'wt') as f:
        for i, site in enumerate(site_list):
            site = site.strip()
            cmdstr = """python pre_lab_4covid.py --dataset nyu 2>&1 | tee  log\pre_lab_4covid_nyu.txt
python pre_dx_4covid.py --dataset nyu 2>&1 | tee  log\pre_dx_4covid_nyu.txt
python pre_med_4covid.py --dataset nyu 2>&1 | tee  log\pre_med_4covid_nyu.txt
python pre_demo.py --dataset nyu 2>&1 | tee  log\pre_demo_nyu.txt
""".replace('nyu', site)
            f.write(cmdstr)
            print(i, site, 'done')

    utils.split_shell_file(r"shell_all_202309.ps1", divide=5, skip_first=0)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    """
    python pre_covid_lab.py --dataset nyu 2>&1 | tee  log\pre_covid_lab_nyu.txt
    python pre_diagnosis.py --dataset nyu 2>&1 | tee  log/pre_diagnosis_nyu.txt
    python pre_medication.py --dataset nyu 2>&1 | tee  log/pre_medication_nyu.txt
    python pre_encounter.py --dataset nyu 2>&1 | tee  log/pre_encounter_nyu.txt
    python pre_procedure.py --dataset nyu 2>&1 | tee  log/pre_procedure_nyu.txt
    python pre_immun.py --dataset nyu 2>&1 | tee  log/pre_immun_nyu.txt
    python pre_death.py --dataset nyu 2>&1 | tee  log/pre_death_nyu.txt
    python pre_vital.py --dataset nyu 2>&1 | tee  log/pre_vital_nyu.txt
    python pre_cohort_manuscript_age18.py --dataset nyu 2>&1 | tee  log/pre_cohort_manuscript_age18_nyu.txt
    python pre_data_manuscript_withAllDays.py --cohorts covid_4manuNegNoCovidV2age18 --dataset nyu 2>&1 | tee  log\pre_data_manuscript_withAllDays_nyu.txt
    """


if __name__ == '__main__':
    start_time = time.time()

    shell_lab_dx_med_4covid()

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
