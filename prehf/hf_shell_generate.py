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

def shell_for_each(site_list = []):
    # python pre_codemapping.py 2>&1 | tee  log/pre_codemapping_zip_adi.txt
    start_time = time.time()


    if site_list == []:
        df_site = pd.read_excel('../prerecover/RECOVER Adult Site schemas_edit.xlsx')
        site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']

    print(len(site_list), 'site list:', site_list)

    with open(r'shell_hr_dxselect.ps1', 'wt') as f:
        for i, site in enumerate(site_list):
            site = site.strip()
            cmdstr = """#python pre_hf_dx.py --dataset nyu 2>&1 | tee  log\pre_hf_dx_nyu.txt
#python pre_demo.py --dataset nyu 2>&1 | tee  log\pre_demo_nyu.txt
#python pre_hf_pat_list.py --dataset nyu 2>&1 | tee  log\pre_hf_pat_list_nyu.txt
python pre_diagnosis.py --dataset nyu 2>&1 | tee  log/pre_diagnosis_nyu.txt
""".replace('nyu', site)

            f.write(cmdstr)
            print(i, site, 'done')

    # remaining shell reminder
    """
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
    """
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


if __name__ == '__main__':
    start_time = time.time()

    #"wcm", "montefiore", "mshs", "columbia", "nyu"
    #"ufh", "emory", "usf", "nch", "miami"

    shell_for_each(site_list = ["wcm", "montefiore", "mshs", "columbia", "nyu"])
    # utils.split_shell_file(r"output\shells\shell_all_2023-6.ps1", divide=4, skip_first=0)

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
