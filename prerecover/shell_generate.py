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

    #     with open(r'shell_all_202311.ps1', 'wt') as f:
    #         for i, site in enumerate(site_list):
    #             site = site.strip()
    #             cmdstr = """python pre_data_matrix_alldays_labdxmed.py --cohorts covid_posOnly18base --dataset nyu 2>&1 | tee  log\pre_data_matrix_alldays_labdxmed_nyu-covid_posOnly18base.txt
    # """.replace('nyu', site)
    #             f.write(cmdstr)
    #             print(i, site, 'done')

    with open(r'shell_all_202402.ps1', 'wt') as f:
        for i, site in enumerate(site_list):
            site = site.strip()
            cmdstr = """python pre_data_matrix_alldays_labdxmed.py --cohorts covid_posneg18base --dataset nyu 2>&1 | tee  log\pre_data_matrix_alldays_labdxmed_nyu-covid_posneg18base.txt
""".replace('nyu', site)
            f.write(cmdstr)
            print(i, site, 'done')

    # be cautious: pre_covid_records should be after pre_med_4covid finish. However, split might break the order
    # of shells
    divide = 6  # 9
    npersite = cmdstr.count('\n')
    siteperdivide = int(np.ceil(len(site_list) / divide))
    ndelta = npersite * siteperdivide
    print('len(site_list):', len(site_list), 'divide:', divide,
          'cmds/site:', npersite, 'total cmds:', len(site_list) * npersite,
          'siteperdivide:', siteperdivide, 'ndelta:', ndelta)
    # utils.split_shell_file_bydelta(r"shell_all_202311.ps1", delta=ndelta, skip_first=0)
    utils.split_shell_file_bydelta(r"shell_all_202402.ps1", delta=ndelta, skip_first=0)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    # python pre_covid_lab.py --dataset nyu 2>&1 | tee  log\pre_covid_lab_nyu.txt
    # not using this, change to pre_covid_records.py

    """
    #python pre_lab_4covid.py --dataset nyu 2>&1 | tee  log\pre_lab_4covid_nyu.txt
#python pre_dx_4covid.py --dataset nyu 2>&1 | tee  log\pre_dx_4covid_nyu.txt
#python pre_med_4covid.py --dataset nyu 2>&1 | tee  log\pre_med_4covid_nyu.txt
#python pre_demo.py --dataset nyu 2>&1 | tee  log\pre_demo_nyu.txt
#python pre_covid_records.py --dataset nyu 2>&1 | tee  log\pre_covid_records_nyu.txt
#python pre_diagnosis.py --dataset nyu 2>&1 | tee  log/pre_diagnosis_nyu.txt
#python pre_medication.py --dataset nyu 2>&1 | tee  log/pre_medication_nyu.txt
#python pre_encounter.py --dataset nyu 2>&1 | tee  log/pre_encounter_nyu.txt
#python pre_procedure.py --dataset nyu 2>&1 | tee  log/pre_procedure_nyu.txt
#python pre_immun.py --dataset nyu 2>&1 | tee  log/pre_immun_nyu.txt
#python pre_death.py --dataset nyu 2>&1 | tee  log/pre_death_nyu.txt
#python pre_vital.py --dataset nyu 2>&1 | tee  log/pre_vital_nyu.txt
##python pre_ckd_lab.py --dataset nyu 2>&1 | tee  log/pre_ckd_lab_nyu.txt
#python pre_lab_select.py --dataset nyu 2>&1 | tee  log/pre_lab_select_nyu.txt
python pre_cohort_labdxmed.py --dataset nyu 2>&1 | tee  log/pre_cohort_labdxmed_nyu.txt
#python pre_data_matrix_alldays_labdxmed.py --cohorts covid_posOnly18base --dataset nyu 2>&1 | tee  log\pre_data_matrix_alldays_labdxmed_nyu-covid_posOnly18base.txt
    """


"""
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


def shell_lab_dx_med_4covid_addcolumnes():
    # python pre_codemapping.py 2>&1 | tee  log/pre_codemapping_zip_adi.txt
    start_time = time.time()

    df_site = pd.read_excel('RECOVER Adult Site schemas_edit.xlsx')

    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']

    print('site_list:', len(site_list), site_list)

    with open(r'shell_all_addCFS_CVD_V6.ps1', 'wt') as f:
        for i, site in enumerate(site_list):
            site = site.strip()
            #             cmdstr = """python pre_data_matrix_alldays_labdxmed_addcolumns.py --cohorts covid_posOnly18base --dataset nyu 2>&1 | tee  log\pre_data_matrix_alldays_labdxmed_nyu-covid_posOnly18base_addCFR-addPaxRisk.txt
            # """.replace('nyu', site)
            #             cmdstr = """python pre_data_matrix_alldays_labdxmed_addcolumns.py --cohorts covid_posOnly18base --dataset nyu 2>&1 | tee  log\pre_data_matrix_alldays_labdxmed_nyu-covid_posOnly18base_addCFR-PaxRisk-acuteU099-hospita-negctrl.txt
            # """.replace('nyu', site)
            #             cmdstr = """python pre_data_matrix_alldays_labdxmed_addcolumns.py --cohorts covid_posOnly18base --dataset nyu 2>&1 | tee  log_addcol\pre_data_matrix_alldays_labdxmed_nyu-covid_posOnly18base_addCFR-PaxRisk-acuteU099-hospita-SSRI-v3.txt
            # """.replace('nyu', site)
            #             cmdstr = """python pre_data_matrix_alldays_labdxmed_addcolumns.py --cohorts covid_posOnly18base --dataset nyu 2>&1 | tee  log_addcol\pre_data_matrix_alldays_labdxmed_nyu-covid_posOnly18base_addCFR-PaxRisk-acuteU099-hospita-SSRI-v5withmental.txt
            # """.replace('nyu', site)
            cmdstr = """python pre_data_matrix_alldays_labdxmed_addcolumns.py --cohorts covid_posOnly18base --dataset nyu 2>&1 | tee  log_addcol\pre_data_matrix_alldays_labdxmed_nyu-covid_posOnly18base_addCFR-PaxRisk-acuteU099-hospita-SSRI-v6withmentalCFSCVD.txt
""".replace('nyu', site)
            f.write(cmdstr)
            print(i, site, 'done')

    # be cautious: pre_covid_records should be after pre_med_4covid finish. However, split might break the order
    # of shells
    divide = 4
    npersite = cmdstr.count('\n')
    siteperdivide = int(np.ceil(len(site_list) / divide))
    ndelta = npersite * siteperdivide
    print('len(site_list):', len(site_list), 'divide:', divide,
          'cmds/site:', npersite, 'total cmds:', len(site_list) * npersite,
          'siteperdivide:', siteperdivide, 'ndelta:', ndelta)
    utils.split_shell_file_bydelta(r"shell_all_addCFS_CVD_V6.ps1", delta=ndelta, skip_first=0)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    # python pre_covid_lab.py --dataset nyu 2>&1 | tee  log\pre_covid_lab_nyu.txt
    # not using this, change to pre_covid_records.py

    """
    #python pre_lab_4covid.py --dataset nyu 2>&1 | tee  log\pre_lab_4covid_nyu.txt
#python pre_dx_4covid.py --dataset nyu 2>&1 | tee  log\pre_dx_4covid_nyu.txt
#python pre_med_4covid.py --dataset nyu 2>&1 | tee  log\pre_med_4covid_nyu.txt
#python pre_demo.py --dataset nyu 2>&1 | tee  log\pre_demo_nyu.txt
#python pre_covid_records.py --dataset nyu 2>&1 | tee  log\pre_covid_records_nyu.txt
#python pre_diagnosis.py --dataset nyu 2>&1 | tee  log/pre_diagnosis_nyu.txt
#python pre_medication.py --dataset nyu 2>&1 | tee  log/pre_medication_nyu.txt
#python pre_encounter.py --dataset nyu 2>&1 | tee  log/pre_encounter_nyu.txt
#python pre_procedure.py --dataset nyu 2>&1 | tee  log/pre_procedure_nyu.txt
#python pre_immun.py --dataset nyu 2>&1 | tee  log/pre_immun_nyu.txt
#python pre_death.py --dataset nyu 2>&1 | tee  log/pre_death_nyu.txt
#python pre_vital.py --dataset nyu 2>&1 | tee  log/pre_vital_nyu.txt
##python pre_ckd_lab.py --dataset nyu 2>&1 | tee  log/pre_ckd_lab_nyu.txt
#python pre_lab_select.py --dataset nyu 2>&1 | tee  log/pre_lab_select_nyu.txt
python pre_cohort_labdxmed.py --dataset nyu 2>&1 | tee  log/pre_cohort_labdxmed_nyu.txt
#python pre_data_matrix_alldays_labdxmed.py --cohorts covid_posOnly18base --dataset nyu 2>&1 | tee  log\pre_data_matrix_alldays_labdxmed_nyu-covid_posOnly18base.txt
    """


def shell_lab_dx_med_4covid_addcolumnes4CNSLDN():
    # python pre_codemapping.py 2>&1 | tee  log/pre_codemapping_zip_adi.txt
    start_time = time.time()

    df_site = pd.read_excel('RECOVER Adult Site schemas_edit.xlsx')

    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']

    print('site_list:', len(site_list), site_list)

    with open(r'shell_all_add_CNS_LDN.ps1', 'wt') as f:
        for i, site in enumerate(site_list):
            site = site.strip()
            cmdstr = """python pre_data_matrix_alldays_labdxmed_addcolumns4cnsldn.py --cohorts covid_posOnly18base --dataset nyu 2>&1 | tee  log_addcol\pre_data_matrix_alldays_labdxmed_nyu-covid_posOnly18base_addCFR-PaxRisk-acuteU099-hospita-SSRI-v7CNSLDN.txt
""".replace('nyu', site)
            f.write(cmdstr)
            print(i, site, 'done')

    # be cautious: pre_covid_records should be after pre_med_4covid finish. However, split might break the order
    # of shells
    divide = 4
    npersite = cmdstr.count('\n')
    siteperdivide = int(np.ceil(len(site_list) / divide))
    ndelta = npersite * siteperdivide
    print('len(site_list):', len(site_list), 'divide:', divide,
          'cmds/site:', npersite, 'total cmds:', len(site_list) * npersite,
          'siteperdivide:', siteperdivide, 'ndelta:', ndelta)
    utils.split_shell_file_bydelta(r"shell_all_add_CNS_LDN.ps1", delta=ndelta, skip_first=0)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    # python pre_covid_lab.py --dataset nyu 2>&1 | tee  log\pre_covid_lab_nyu.txt
    # not using this, change to pre_covid_records.py

    """
    #python pre_lab_4covid.py --dataset nyu 2>&1 | tee  log\pre_lab_4covid_nyu.txt
#python pre_dx_4covid.py --dataset nyu 2>&1 | tee  log\pre_dx_4covid_nyu.txt
#python pre_med_4covid.py --dataset nyu 2>&1 | tee  log\pre_med_4covid_nyu.txt
#python pre_demo.py --dataset nyu 2>&1 | tee  log\pre_demo_nyu.txt
#python pre_covid_records.py --dataset nyu 2>&1 | tee  log\pre_covid_records_nyu.txt
#python pre_diagnosis.py --dataset nyu 2>&1 | tee  log/pre_diagnosis_nyu.txt
#python pre_medication.py --dataset nyu 2>&1 | tee  log/pre_medication_nyu.txt
#python pre_encounter.py --dataset nyu 2>&1 | tee  log/pre_encounter_nyu.txt
#python pre_procedure.py --dataset nyu 2>&1 | tee  log/pre_procedure_nyu.txt
#python pre_immun.py --dataset nyu 2>&1 | tee  log/pre_immun_nyu.txt
#python pre_death.py --dataset nyu 2>&1 | tee  log/pre_death_nyu.txt
#python pre_vital.py --dataset nyu 2>&1 | tee  log/pre_vital_nyu.txt
##python pre_ckd_lab.py --dataset nyu 2>&1 | tee  log/pre_ckd_lab_nyu.txt
#python pre_lab_select.py --dataset nyu 2>&1 | tee  log/pre_lab_select_nyu.txt
python pre_cohort_labdxmed.py --dataset nyu 2>&1 | tee  log/pre_cohort_labdxmed_nyu.txt
#python pre_data_matrix_alldays_labdxmed.py --cohorts covid_posOnly18base --dataset nyu 2>&1 | tee  log\pre_data_matrix_alldays_labdxmed_nyu-covid_posOnly18base.txt
    """


def shell_lab_dx_med_4covid_aux():
    # python pre_codemapping.py 2>&1 | tee  log/pre_codemapping_zip_adi.txt
    start_time = time.time()

    df_site = pd.read_excel('RECOVER Adult Site schemas_edit.xlsx')

    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']

    # site_list = ['wcm', 'nch', 'pitt', 'osu', 'utah', 'utsw'] # sites need update due to geo updaing adi significantly
    print('site_list:', len(site_list), site_list)

    with open(r'shell_all_aux.ps1', 'wt') as f:
        for i, site in enumerate(site_list):
            site = site.strip()
            cmdstr = """python pre_cohort_labdxmed.py --dataset nyu 2>&1 | tee  log/pre_cohort_labdxmed_nyu.txt
""".replace('nyu', site)
            f.write(cmdstr)
            print(i, site, 'done')

    # be cautious: pre_covid_records should be after pre_med_4covid finish. However, split might break the order
    # of shells
    divide = 2
    npersite = cmdstr.count('\n')
    siteperdivide = int(np.ceil(len(site_list) / divide))
    ndelta = npersite * siteperdivide
    print('len(site_list):', len(site_list), 'divide:', divide,
          'cmds/site:', npersite, 'total cmds:', len(site_list) * npersite,
          'siteperdivide:', siteperdivide, 'ndelta:', ndelta)
    utils.split_shell_file_bydelta(r"shell_all_aux.ps1", delta=ndelta, skip_first=0)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    # python pre_covid_lab.py --dataset nyu 2>&1 | tee  log\pre_covid_lab_nyu.txt
    # not using this, change to pre_covid_records.py


"""
ren ../data/recover/output/nyu/patient_demo_nyu.pkl patient_demo_nyu_old.pkl 
python pre_demo.py --dataset nyu 2>&1 | tee  log\pre_demo_nyu.txt
"""


def shell_iptw_subgroup():
    # python pre_codemapping.py 2>&1 | tee  log/pre_codemapping_zip_adi.txt
    start_time = time.time()

    subgroup_list = ['PaxRisk:Cancer', 'PaxRisk:Chronic kidney disease', 'PaxRisk:Chronic liver disease',
                     'PaxRisk:Chronic lung disease', 'PaxRisk:Cystic fibrosis',
                     'PaxRisk:Dementia or other neurological conditions', 'PaxRisk:Diabetes', 'PaxRisk:Disabilities',
                     'PaxRisk:Heart conditions', 'PaxRisk:Hypertension', 'PaxRisk:HIV infection',
                     'PaxRisk:Immunocompromised condition or weakened immune system',
                     'PaxRisk:Mental health conditions',
                     'PaxRisk:Overweight and obesity', 'PaxRisk:Pregnancy',
                     'PaxRisk:Sickle cell disease or thalassemia',
                     'PaxRisk:Smoking current', 'PaxRisk:Stroke or cerebrovascular disease',
                     'PaxRisk:Substance use disorders', 'PaxRisk:Tuberculosis', ]

    print('subgroup_list:', len(subgroup_list), subgroup_list)

    with open(r'../iptw/shell_iptw_subgroup.ps1', 'wt') as f:
        for i, subgroup in enumerate(subgroup_list):
            cmdstr = """python screen_paxlovid_iptw_pcornet.py  --cohorttype atrisk --severity '{}' 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-atrisk-{}.txt
""".format(subgroup, subgroup.replace(':', '_').replace('/', '-').replace(' ', '_'))
            f.write(cmdstr)
            print(i, subgroup, 'done')

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


def shell_build_lab_dx_4covid_sensitivity():
    # python pre_codemapping.py 2>&1 | tee  log/pre_codemapping_zip_adi.txt
    start_time = time.time()

    df_site = pd.read_excel('RECOVER Adult Site schemas_edit.xlsx')

    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']

    print('site_list:', len(site_list), site_list)

    with open(r'shell_all_labdx-4-sensitivity.ps1', 'wt') as f:
        for i, site in enumerate(site_list):
            site = site.strip()
            cmdstr = """python pre_cohort_labdxmed.py --dataset nyu --cohorttype lab-dx 2>&1 | tee  log/pre_cohort_labdx-4-sensitivity_nyu.txt
""".replace('nyu', site)
            f.write(cmdstr)
            print(i, site, 'done')

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


def shell_lab_dx_med_4covidAndPregnant():
    # python pre_codemapping.py 2>&1 | tee  log/pre_codemapping_zip_adi.txt
    start_time = time.time()

    df_site = pd.read_excel('RECOVER Adult Site schemas_edit.xlsx')

    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']

    site_list = site_list.to_list() + ['northwestern', 'wakeforest',
                                       'chop', 'nemours', 'nationwide', 'seattle', 'colorado', 'lurie',
                                       'cchmc', 'national', 'indiana', 'stanford', ]

    print('site_list:', len(site_list), site_list)

    with open(r'shell_pcornet_all_202403.ps1', 'wt') as f:
        for i, site in enumerate(site_list):
            site = site.strip()
            cmdstr = """python pre_dx_4pregnant.py --dataset nyu_pcornet_all 2>&1 | tee  log_pcornet_all\pre_dx_4pregnant_nyu_pcornet_all.txt
python pre_procedure_4pregnant.py --dataset nyu_pcornet_all 2>&1 | tee  log_pcornet_all\pre_procedure_4pregnant_nyu_pcornet_all.txt
python pre_encounter_4pregnant.py --dataset nyu_pcornet_all 2>&1 | tee  log_pcornet_all\pre_encounter_4pregnant_nyu_pcornet_all.txt
python pre_lab_4covid.py --dataset nyu_pcornet_all 2>&1 | tee  log_pcornet_all\pre_lab_4covid_nyu_pcornet_all.txt
python pre_dx_4covid.py --dataset nyu_pcornet_all 2>&1 | tee  log_pcornet_all\pre_dx_4covid_nyu_pcornet_all.txt
python pre_med_4covid.py --dataset nyu_pcornet_all 2>&1 | tee  log_pcornet_all\pre_med_4covid_nyu_pcornet_all.txt
""".replace('nyu', site)
            f.write(cmdstr)
            print(i, site, 'done')

    # be cautious: pre_covid_records should be after pre_med_4covid finish. However, split might break the order
    # of shells
    divide = 6
    npersite = cmdstr.count('\n')
    siteperdivide = int(np.ceil(len(site_list) / divide))
    ndelta = npersite * siteperdivide
    print('len(site_list):', len(site_list), 'divide:', divide,
          'cmds/site:', npersite, 'total cmds:', len(site_list) * npersite,
          'siteperdivide:', siteperdivide, 'ndelta:', ndelta)
    # utils.split_shell_file_bydelta(r"shell_all_202311.ps1", delta=ndelta, skip_first=0)
    utils.split_shell_file_bydelta(r"shell_pcornet_all_202403.ps1", delta=ndelta, skip_first=0)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    """
    #python pre_lab_4covid.py --dataset nyu 2>&1 | tee  log\pre_lab_4covid_nyu.txt
#python pre_dx_4covid.py --dataset nyu 2>&1 | tee  log\pre_dx_4covid_nyu.txt
#python pre_med_4covid.py --dataset nyu 2>&1 | tee  log\pre_med_4covid_nyu.txt
#python pre_demo.py --dataset nyu 2>&1 | tee  log\pre_demo_nyu.txt
#python pre_covid_records.py --dataset nyu 2>&1 | tee  log\pre_covid_records_nyu.txt
#python pre_diagnosis.py --dataset nyu 2>&1 | tee  log/pre_diagnosis_nyu.txt
#python pre_medication.py --dataset nyu 2>&1 | tee  log/pre_medication_nyu.txt
#python pre_encounter.py --dataset nyu 2>&1 | tee  log/pre_encounter_nyu.txt
#python pre_procedure.py --dataset nyu 2>&1 | tee  log/pre_procedure_nyu.txt
#python pre_immun.py --dataset nyu 2>&1 | tee  log/pre_immun_nyu.txt
#python pre_death.py --dataset nyu 2>&1 | tee  log/pre_death_nyu.txt
#python pre_vital.py --dataset nyu 2>&1 | tee  log/pre_vital_nyu.txt
##python pre_ckd_lab.py --dataset nyu 2>&1 | tee  log/pre_ckd_lab_nyu.txt
#python pre_lab_select.py --dataset nyu 2>&1 | tee  log/pre_lab_select_nyu.txt
python pre_cohort_labdxmed.py --dataset nyu 2>&1 | tee  log/pre_cohort_labdxmed_nyu.txt
#python pre_data_matrix_alldays_labdxmed.py --cohorts covid_posOnly18base --dataset nyu 2>&1 | tee  log\pre_data_matrix_alldays_labdxmed_nyu-covid_posOnly18base.txt
    """


def shell_lab_dx_med_4covid_202407():
    # python pre_codemapping.py 2>&1 | tee  log/pre_codemapping_zip_adi.txt
    start_time = time.time()

    df_site = pd.read_excel('RECOVER Adult Site schemas_edit.xlsx')
    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']
    print('site_list:', len(site_list), site_list)

    site_list = [
        'columbia_pcornet_all', 'duke_pcornet_all', 'emory_pcornet_all', 'intermountain_pcornet_all',
        'iowa_pcornet_all',
        'lsu_pcornet_all', 'mcw_pcornet_all', 'michigan_pcornet_all', 'missouri_pcornet_all', 'montefiore_pcornet_all',
        'mshs_pcornet_all', 'wcm_pcornet_all', 'nch_pcornet_all', 'nebraska_pcornet_all', 'northwestern_pcornet_all',
        'nyu_pcornet_all', 'ochsner_pcornet_all', 'osu_pcornet_all', 'pitt_pcornet_all', 'psu_pcornet_all',
        'temple_pcornet_all', 'ufh_pcornet_all', 'utah_pcornet_all', 'utsw_pcornet_all',
        'vumc_pcornet_all', 'wakeforest_pcornet_all', ]
    print('len(site_list)', len(site_list))

    with open(r'shell_all_202407.ps1', 'wt') as f:
        for i, site in enumerate(site_list):
            site = site.strip()
            cmdstr = """#python pre_lab_4covid.py --dataset nyu 2>&1 | tee  log\pre_lab_4covid_nyu.txt
# python pre_dx_4covid.py --dataset nyu 2>&1 | tee  log\pre_dx_4covid_nyu.txt
# python pre_med_4covid.py --dataset nyu 2>&1 | tee  log\pre_med_4covid_nyu.txt
# python pre_dx_4pregnant.py --dataset nyu 2>&1 | tee  log\pre_dx_4pregnant_nyu.txt
# python pre_procedure_4pregnant.py --dataset nyu 2>&1 | tee  log\pre_procedure_4pregnant_nyu.txt
# python pre_encounter_4pregnant.py --dataset nyu 2>&1 | tee  log\pre_encounter_4pregnant_nyu.txt
# python pre_demo.py --dataset nyu 2>&1 | tee  log\pre_demo_nyu.txt
# python pre_covid_records.py --dataset nyu 2>&1 | tee  log\pre_covid_records_nyu.txt
# python pre_diagnosis.py --dataset nyu 2>&1 | tee  log/pre_diagnosis_nyu.txt
# python pre_medication.py --dataset nyu 2>&1 | tee  log/pre_medication_nyu.txt
# python pre_encounter.py --dataset nyu 2>&1 | tee  log/pre_encounter_nyu.txt
# python pre_procedure.py --dataset nyu 2>&1 | tee  log/pre_procedure_nyu.txt
# python pre_immun.py --dataset nyu 2>&1 | tee  log/pre_immun_nyu.txt
# python pre_death.py --dataset nyu 2>&1 | tee  log/pre_death_nyu.txt
# python pre_vital.py --dataset nyu 2>&1 | tee  log/pre_vital_nyu.txt
# python pre_lab_select.py --dataset nyu 2>&1 | tee  log/pre_lab_select_nyu.txt
python pre_cohort_labdxmedpreg.py --dataset nyu 2>&1 | tee  log/pre_cohort_labdxmedpreg_nyu.txt
python pre_cohort_labdxmedpreg_negInpos.py --dataset nyu 2>&1 | tee  log/pre_cohort_labdxmedpreg_negInpos_nyu.txt
# python pre_data_matrix_alldays_labdxmed.py --cohorts covid_posOnly18base --dataset nyu 2>&1 | tee  log\pre_data_matrix_alldays_labdxmed_nyu-covid_posOnly18base.txt
""".replace('nyu', site)
            f.write(cmdstr)
            print(i, site, 'done')

    # be cautious: pre_covid_records should be after pre_med_4covid finish. However, split might break the order
    # of shells
    divide = 2  # 9
    npersite = cmdstr.count('\n')
    siteperdivide = int(np.ceil(len(site_list) / divide))
    ndelta = npersite * siteperdivide
    print('len(site_list):', len(site_list), 'divide:', divide,
          'cmds/site:', npersite, 'total cmds:', len(site_list) * npersite,
          'siteperdivide:', siteperdivide, 'ndelta:', ndelta)

    utils.split_shell_file_bydelta(r"shell_all_202407.ps1", delta=ndelta, skip_first=0)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    # python pre_covid_lab.py --dataset nyu 2>&1 | tee  log\pre_covid_lab_nyu.txt
    # not using this, change to pre_covid_records.py

    """
    #python pre_lab_4covid.py --dataset nyu 2>&1 | tee  log\pre_lab_4covid_nyu.txt
#python pre_dx_4covid.py --dataset nyu 2>&1 | tee  log\pre_dx_4covid_nyu.txt
#python pre_med_4covid.py --dataset nyu 2>&1 | tee  log\pre_med_4covid_nyu.txt
#python pre_demo.py --dataset nyu 2>&1 | tee  log\pre_demo_nyu.txt
#python pre_covid_records.py --dataset nyu 2>&1 | tee  log\pre_covid_records_nyu.txt
#python pre_diagnosis.py --dataset nyu 2>&1 | tee  log/pre_diagnosis_nyu.txt
#python pre_medication.py --dataset nyu 2>&1 | tee  log/pre_medication_nyu.txt
#python pre_encounter.py --dataset nyu 2>&1 | tee  log/pre_encounter_nyu.txt
#python pre_procedure.py --dataset nyu 2>&1 | tee  log/pre_procedure_nyu.txt
#python pre_immun.py --dataset nyu 2>&1 | tee  log/pre_immun_nyu.txt
#python pre_death.py --dataset nyu 2>&1 | tee  log/pre_death_nyu.txt
#python pre_vital.py --dataset nyu 2>&1 | tee  log/pre_vital_nyu.txt
##python pre_ckd_lab.py --dataset nyu 2>&1 | tee  log/pre_ckd_lab_nyu.txt
#python pre_lab_select.py --dataset nyu 2>&1 | tee  log/pre_lab_select_nyu.txt
python pre_cohort_labdxmed.py --dataset nyu 2>&1 | tee  log/pre_cohort_labdxmed_nyu.txt
#python pre_data_matrix_alldays_labdxmed.py --cohorts covid_posOnly18base --dataset nyu 2>&1 | tee  log\pre_data_matrix_alldays_labdxmed_nyu-covid_posOnly18base.txt
    """


def shell_lab_dx_med_4covid_2025():
    # python pre_codemapping.py 2>&1 | tee  log/pre_codemapping_zip_adi.txt
    start_time = time.time()

    df_site = pd.read_excel('RECOVER Adult Site schemas_edit2025.xlsx')
    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']
    print('site_list:', len(site_list), site_list)

    site_list = [x + '_pcornet_all' for x in site_list]

    # 'columbia_pcornet_all', 'duke_pcornet_all', 'emory_pcornet_all', 'intermountain_pcornet_all',
    # 'iowa_pcornet_all',
    # 'lsu_pcornet_all', 'mcw_pcornet_all', 'michigan_pcornet_all', 'missouri_pcornet_all', 'montefiore_pcornet_all',
    # 'mshs_pcornet_all', 'wcm_pcornet_all', 'nch_pcornet_all', 'nebraska_pcornet_all', 'northwestern_pcornet_all',
    # 'nyu_pcornet_all', 'ochsner_pcornet_all', 'osu_pcornet_all', 'pitt_pcornet_all', 'psu_pcornet_all',
    # 'temple_pcornet_all', 'ufh_pcornet_all', 'utah_pcornet_all', 'utsw_pcornet_all',
    # 'vumc_pcornet_all', 'wakeforest_pcornet_all', ]
    print('len(site_list)', len(site_list), site_list)

    # with open(r'shell_all_2025Q2Afterdemo.ps1', 'wt') as f:
    #with open(r'shell_all_2025Q2CohortandAfter.ps1', 'wt') as f:
    with open(r'shell_all_2025Q2Matrix.ps1', 'wt') as f:
        for i, site in enumerate(site_list):
            site = site.strip()
            cmdstr = """#python pre_lab_4covid.py --dataset nyu 2>&1 | tee  log\pre_lab_4covid_nyu.txt
#python pre_dx_4covid.py --dataset nyu 2>&1 | tee  log\pre_dx_4covid_nyu.txt
#python pre_med_4covid.py --dataset nyu 2>&1 | tee  log\pre_med_4covid_nyu.txt
#python pre_med_4LDN.py --dataset nyu 2>&1 | tee  log\pre_med_4LDN_nyu.txt
#python pre_dx_4pregnant.py --dataset nyu 2>&1 | tee  log\pre_dx_4pregnant_nyu.txt
#python pre_procedure_4pregnant.py --dataset nyu 2>&1 | tee  log\pre_procedure_4pregnant_nyu.txt
#python pre_encounter_4pregnant.py --dataset nyu 2>&1 | tee  log\pre_encounter_4pregnant_nyu.txt
#python pre_demo.py --dataset nyu 2>&1 | tee  log\pre_demo_nyu.txt
python pre_covid_records.py --dataset nyu 2>&1 | tee  log\pre_covid_records_nyu.txt
python pre_diagnosis.py --dataset nyu 2>&1 | tee  log/pre_diagnosis_nyu.txt
python pre_medication.py --dataset nyu 2>&1 | tee  log/pre_medication_nyu.txt
python pre_encounter.py --dataset nyu 2>&1 | tee  log/pre_encounter_nyu.txt
python pre_procedure.py --dataset nyu 2>&1 | tee  log/pre_procedure_nyu.txt
python pre_immun.py --dataset nyu 2>&1 | tee  log/pre_immun_nyu.txt
python pre_death.py --dataset nyu 2>&1 | tee  log/pre_death_nyu.txt
python pre_vital.py --dataset nyu 2>&1 | tee  log/pre_vital_nyu.txt
python pre_lab_select.py --dataset nyu 2>&1 | tee  log/pre_lab_select_nyu.txt
# python pre_cohort_labdxmedpreg.py --dataset nyu 2>&1 | tee  log/pre_cohort_labdxmedpreg_nyu.txt
""".replace('nyu', site)

            cmdstr = """python pre_cohort_labdxmed25Q2.py --dataset nyu 2>&1 | tee  log/pre_cohort_labdxmed25Q2_nyu.txt
""".replace('nyu', site)

            cmdstr = """python pre_data_matrix_alldays_labdxmed25Q2.py --dataset nyu 2>&1 | tee  log/pre_data_matrix_alldays_labdxmed25Q2_nyu.txt
""".replace('nyu', site)


            f.write(cmdstr)
            print(i, site, 'done')

    # be cautious: pre_covid_records should be after pre_med_4covid finish. However, split might break the order
    # of shells
    divide = 6  # 9
    npersite = cmdstr.count('\n')
    siteperdivide = int(np.ceil(len(site_list) / divide))
    ndelta = npersite * siteperdivide
    print('len(site_list):', len(site_list), 'divide:', divide,
          'cmds/site:', npersite, 'total cmds:', len(site_list) * npersite,
          'siteperdivide:', siteperdivide, 'ndelta:', ndelta)

    # utils.split_shell_file_bydelta(r"shell_all_2025Q2Afterdemo.ps1", delta=ndelta, skip_first=0)
    # utils.split_shell_file_bydelta(r"shell_all_2025Q2CohortandAfter.ps1", delta=ndelta, skip_first=0)
    utils.split_shell_file_bydelta(r"shell_all_2025Q2Matrix.ps1", delta=ndelta, skip_first=0)

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    # python pre_covid_lab.py --dataset nyu 2>&1 | tee  log\pre_covid_lab_nyu.txt
    # not using this, change to pre_covid_records.py
    """#python pre_lab_4covid.py --dataset nyu 2>&1 | tee  log\pre_lab_4covid_nyu.txt
    # python pre_dx_4covid.py --dataset nyu 2>&1 | tee  log\pre_dx_4covid_nyu.txt
    # python pre_med_4covid.py --dataset nyu 2>&1 | tee  log\pre_med_4covid_nyu.txt
    # python pre_dx_4pregnant.py --dataset nyu 2>&1 | tee  log\pre_dx_4pregnant_nyu.txt
    # python pre_procedure_4pregnant.py --dataset nyu 2>&1 | tee  log\pre_procedure_4pregnant_nyu.txt
    # python pre_encounter_4pregnant.py --dataset nyu 2>&1 | tee  log\pre_encounter_4pregnant_nyu.txt
    # python pre_demo.py --dataset nyu 2>&1 | tee  log\pre_demo_nyu.txt
    # python pre_covid_records.py --dataset nyu 2>&1 | tee  log\pre_covid_records_nyu.txt
    # python pre_diagnosis.py --dataset nyu 2>&1 | tee  log/pre_diagnosis_nyu.txt
    # python pre_medication.py --dataset nyu 2>&1 | tee  log/pre_medication_nyu.txt
    # python pre_encounter.py --dataset nyu 2>&1 | tee  log/pre_encounter_nyu.txt
    # python pre_procedure.py --dataset nyu 2>&1 | tee  log/pre_procedure_nyu.txt
    # python pre_immun.py --dataset nyu 2>&1 | tee  log/pre_immun_nyu.txt
    # python pre_death.py --dataset nyu 2>&1 | tee  log/pre_death_nyu.txt
    # python pre_vital.py --dataset nyu 2>&1 | tee  log/pre_vital_nyu.txt
    # python pre_lab_select.py --dataset nyu 2>&1 | tee  log/pre_lab_select_nyu.txt
    python pre_cohort_labdxmedpreg.py --dataset nyu 2>&1 | tee  log/pre_cohort_labdxmedpreg_nyu.txt
    python pre_cohort_labdxmedpreg_negInpos.py --dataset nyu 2>&1 | tee  log/pre_cohort_labdxmedpreg_negInpos_nyu.txt
    # python pre_data_matrix_alldays_labdxmed.py --cohorts covid_posOnly18base --dataset nyu 2>&1 | tee  log\pre_data_matrix_alldays_labdxmed_nyu-covid_posOnly18base.txt
    """


def shell_lab_dx_med_4covid_2025_fix():
    # python pre_codemapping.py 2>&1 | tee  log/pre_codemapping_zip_adi.txt
    start_time = time.time()

    df_site = pd.read_excel('RECOVER Adult Site schemas_edit2025.xlsx')
    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']
    print('site_list:', len(site_list), site_list)

    site_list = ['utsw', 'ufh', 'psu']
    site_list = [x + '_pcornet_all' for x in site_list]

    # 'columbia_pcornet_all', 'duke_pcornet_all', 'emory_pcornet_all', 'intermountain_pcornet_all',
    # 'iowa_pcornet_all',
    # 'lsu_pcornet_all', 'mcw_pcornet_all', 'michigan_pcornet_all', 'missouri_pcornet_all', 'montefiore_pcornet_all',
    # 'mshs_pcornet_all', 'wcm_pcornet_all', 'nch_pcornet_all', 'nebraska_pcornet_all', 'northwestern_pcornet_all',
    # 'nyu_pcornet_all', 'ochsner_pcornet_all', 'osu_pcornet_all', 'pitt_pcornet_all', 'psu_pcornet_all',
    # 'temple_pcornet_all', 'ufh_pcornet_all', 'utah_pcornet_all', 'utsw_pcornet_all',
    # 'vumc_pcornet_all', 'wakeforest_pcornet_all', ]
    print('len(site_list)', len(site_list), site_list)

    # with open(r'shell_all_2025Q2Afterdemo.ps1', 'wt') as f:
    #with open(r'shell_all_2025Q2CohortandAfter.ps1', 'wt') as f:
    with open(r'shell_all_2025Q2Fix.ps1', 'wt') as f:
        for i, site in enumerate(site_list):
            site = site.strip()
            cmdstr = """#python pre_lab_4covid.py --dataset nyu 2>&1 | tee  log\pre_lab_4covid_nyu.txt
#python pre_dx_4covid.py --dataset nyu 2>&1 | tee  log\pre_dx_4covid_nyu.txt
#python pre_med_4covid.py --dataset nyu 2>&1 | tee  log\pre_med_4covid_nyu.txt
#python pre_med_4LDN.py --dataset nyu 2>&1 | tee  log\pre_med_4LDN_nyu.txt
#python pre_dx_4pregnant.py --dataset nyu 2>&1 | tee  log\pre_dx_4pregnant_nyu.txt
#python pre_procedure_4pregnant.py --dataset nyu 2>&1 | tee  log\pre_procedure_4pregnant_nyu.txt
#python pre_encounter_4pregnant.py --dataset nyu 2>&1 | tee  log\pre_encounter_4pregnant_nyu.txt
python pre_demo.py --dataset nyu 2>&1 | tee  log\pre_demo_nyu.txt
python pre_covid_records.py --dataset nyu 2>&1 | tee  log\pre_covid_records_nyu.txt
python pre_diagnosis.py --dataset nyu 2>&1 | tee  log/pre_diagnosis_nyu.txt
python pre_medication.py --dataset nyu 2>&1 | tee  log/pre_medication_nyu.txt
python pre_encounter.py --dataset nyu 2>&1 | tee  log/pre_encounter_nyu.txt
python pre_procedure.py --dataset nyu 2>&1 | tee  log/pre_procedure_nyu.txt
python pre_immun.py --dataset nyu 2>&1 | tee  log/pre_immun_nyu.txt
python pre_death.py --dataset nyu 2>&1 | tee  log/pre_death_nyu.txt
python pre_vital.py --dataset nyu 2>&1 | tee  log/pre_vital_nyu.txt
python pre_lab_select.py --dataset nyu 2>&1 | tee  log/pre_lab_select_nyu.txt
python pre_cohort_labdxmed25Q2.py --dataset nyu 2>&1 | tee  log/pre_cohort_labdxmed25Q2_nyu.txt
python pre_data_matrix_alldays_labdxmed25Q2.py --dataset nyu 2>&1 | tee  log/pre_data_matrix_alldays_labdxmed25Q2_nyu.txt
""".replace('nyu', site)


            f.write(cmdstr)
            print(i, site, 'done')

    # be cautious: pre_covid_records should be after pre_med_4covid finish. However, split might break the order
    # of shells
    divide = 3  # 9
    npersite = cmdstr.count('\n')
    siteperdivide = int(np.ceil(len(site_list) / divide))
    ndelta = npersite * siteperdivide
    print('len(site_list):', len(site_list), 'divide:', divide,
          'cmds/site:', npersite, 'total cmds:', len(site_list) * npersite,
          'siteperdivide:', siteperdivide, 'ndelta:', ndelta)

    # utils.split_shell_file_bydelta(r"shell_all_2025Q2Afterdemo.ps1", delta=ndelta, skip_first=0)
    # utils.split_shell_file_bydelta(r"shell_all_2025Q2CohortandAfter.ps1", delta=ndelta, skip_first=0)
    utils.split_shell_file_bydelta(r"shell_all_2025Q2Fix.ps1", delta=ndelta, skip_first=0)

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    # python pre_covid_lab.py --dataset nyu 2>&1 | tee  log\pre_covid_lab_nyu.txt
    # not using this, change to pre_covid_records.py
    """#python pre_lab_4covid.py --dataset nyu 2>&1 | tee  log\pre_lab_4covid_nyu.txt
    # python pre_dx_4covid.py --dataset nyu 2>&1 | tee  log\pre_dx_4covid_nyu.txt
    # python pre_med_4covid.py --dataset nyu 2>&1 | tee  log\pre_med_4covid_nyu.txt
    # python pre_dx_4pregnant.py --dataset nyu 2>&1 | tee  log\pre_dx_4pregnant_nyu.txt
    # python pre_procedure_4pregnant.py --dataset nyu 2>&1 | tee  log\pre_procedure_4pregnant_nyu.txt
    # python pre_encounter_4pregnant.py --dataset nyu 2>&1 | tee  log\pre_encounter_4pregnant_nyu.txt
    # python pre_demo.py --dataset nyu 2>&1 | tee  log\pre_demo_nyu.txt
    # python pre_covid_records.py --dataset nyu 2>&1 | tee  log\pre_covid_records_nyu.txt
    # python pre_diagnosis.py --dataset nyu 2>&1 | tee  log/pre_diagnosis_nyu.txt
    # python pre_medication.py --dataset nyu 2>&1 | tee  log/pre_medication_nyu.txt
    # python pre_encounter.py --dataset nyu 2>&1 | tee  log/pre_encounter_nyu.txt
    # python pre_procedure.py --dataset nyu 2>&1 | tee  log/pre_procedure_nyu.txt
    # python pre_immun.py --dataset nyu 2>&1 | tee  log/pre_immun_nyu.txt
    # python pre_death.py --dataset nyu 2>&1 | tee  log/pre_death_nyu.txt
    # python pre_vital.py --dataset nyu 2>&1 | tee  log/pre_vital_nyu.txt
    # python pre_lab_select.py --dataset nyu 2>&1 | tee  log/pre_lab_select_nyu.txt
    python pre_cohort_labdxmedpreg.py --dataset nyu 2>&1 | tee  log/pre_cohort_labdxmedpreg_nyu.txt
    python pre_cohort_labdxmedpreg_negInpos.py --dataset nyu 2>&1 | tee  log/pre_cohort_labdxmedpreg_negInpos_nyu.txt
    # python pre_data_matrix_alldays_labdxmed.py --cohorts covid_posOnly18base --dataset nyu 2>&1 | tee  log\pre_data_matrix_alldays_labdxmed_nyu-covid_posOnly18base.txt
    """


if __name__ == '__main__':
    start_time = time.time()

    # shell_lab_dx_med_4covid()
    # shell_lab_dx_med_4covid_aux()

    # shell_lab_dx_med_4covid_addcolumnes()
    # shell_iptw_subgroup()
    # shell_build_lab_dx_4covid_sensitivity()
    # shell_lab_dx_med_4covidAndPregnant()

    # shell_lab_dx_med_4covid_202407()

    # 2025-04-08

    # 20250408
    # shell_lab_dx_med_4covid_addcolumnes4CNSLDN()

    # 2025-04-10
    # shell_lab_dx_med_4covid_2025()
    shell_lab_dx_med_4covid_2025_fix()

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
