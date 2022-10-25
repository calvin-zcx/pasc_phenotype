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

print = functools.partial(print, flush=True)


if __name__ == '__main__':
    # python pre_codemapping.py 2>&1 | tee  log/pre_codemapping_zip_adi.txt
    start_time = time.time()

    df_site = pd.read_excel('RECOVER Adult Site schemas_edit.xlsx')

    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']

    for i, site in enumerate(site_list):
        site = site.strip()
        cmdstr = """python pre_lab.py --dataset nyu 2>&1 | tee  log\pre_lab_nyu.txt
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
        """.replace('nyu', site)
        with open(r'output\shells\shell_for_{}.ps1'.format(site), 'wt') as f:
            f.write(cmdstr)

        print(i, site, 'done')
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
