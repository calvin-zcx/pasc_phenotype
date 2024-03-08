import sys

# for linux env.
sys.path.insert(0, '..')
import pandas as pd
import time
import pickle
import argparse
from misc import utils
import numpy as np
import functools
from collections import Counter
from tqdm import tqdm

print = functools.partial(print, flush=True)
from collections import defaultdict
from misc.utils import clean_date_str
import datetime


if __name__ == '__main__':
    # python pre_covid_lab.py --dataset WCM 2>&1 | tee  log/pre_covid_lab_WCM.txt
    start_time = time.time()
    # args = parse_args()
    df_site = pd.read_excel('RECOVER Adult Site schemas_edit.xlsx')

    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']
    # ['duke', 'intermountain', 'missouri', 'iowa', 'northwestern', 'ochin', 'osu', 'wakeforest',  'musc']
    site_list = site_list.to_list() + ['northwestern', 'wakeforest',
                                       'chop', 'nemours', 'nationwide', 'seattle', 'colorado', 'lurie',
                                       'cchmc', 'national', 'indiana', 'stanford', ]  # these two sites with label 0
    # Intermountain does not have covid data? From dmi
    #  Indiana is the site that is not present.
    print('len(site_list):', len(site_list), site_list)

    site_list = ['ochin',
                 'intermountain', 'mcw', 'iowa', 'missouri', 'nebraska', 'utah', 'utsw',
                 'wcm', 'montefiore', 'mshs', 'columbia', 'nyu',
                 'ufh', 'emory', 'usf', 'nch', 'miami',
                 'pitt', 'osu', 'psu', 'temple', 'michigan',
                 'ochsner', 'ucsf', 'lsu',
                 'vumc', 'duke', 'musc']
    # site_list = ['wcm', ]
    covid_set = set()
    preg_set = set()
    for site in site_list:
        site = site + '_pcornet_all'
        print('site:', site)

        covid_lab_file = r'../data/recover/output/{}/covid_lab_{}.csv'.format(site, site)
        covid_dx_file = r'../data/recover/output/{}/covid_diagnosis_{}.csv'.format(site, site)

        covid_prescribe_file = r'../data/recover/output/{}/covid_prescribing_{}.csv'.format(site, site)
        covid_med_admin_file = r'../data/recover/output/{}/covid_med_admin_{}.csv'.format(site, site)
        covid_dispensing_file = r'../data/recover/output/{}/covid_dispensing_{}.csv'.format(site, site)

        preg_dx_file = r'../data/recover/output/{}/pregnant_diagnosis_{}.csv'.format(site, site)
        preg_pro_file = r'../data/recover/output/{}/pregnant_procedures_{}.csv'.format(site, site)
        preg_enc_file = r'../data/recover/output/{}/pregnant_encounter_{}.csv'.format(site, site)

        df_covid_lab = pd.read_csv(covid_lab_file, dtype=str, parse_dates=['SPECIMEN_DATE', "RESULT_DATE"])
        df_covid_dx = pd.read_csv(covid_dx_file, dtype=str, parse_dates=['ADMIT_DATE', "DX_DATE"])
        df_covid_prescribe = pd.read_csv(covid_prescribe_file, dtype=str, parse_dates=['RX_ORDER_DATE', 'RX_START_DATE', 'RX_END_DATE'])
        df_covid_dispensing = pd.read_csv(covid_dispensing_file, dtype=str, parse_dates=['DISPENSE_DATE', ])
        df_covid_med_admin = pd.read_csv(covid_med_admin_file, dtype=str, parse_dates=['MEDADMIN_START_DATE', 'MEDADMIN_STOP_DATE', ])

        df_preg_dx = pd.read_csv(preg_dx_file, dtype=str, parse_dates=['ADMIT_DATE', "DX_DATE"])
        df_preg_pro = pd.read_csv(preg_pro_file, dtype=str, parse_dates=['ADMIT_DATE', "PX_DATE"])
        df_preg_enc = pd.read_csv(preg_enc_file, dtype=str, parse_dates=['ADMIT_DATE', "DISCHARGE_DATE"])

        covid_set.update([(site, x) for x in df_covid_lab['PATID']])
        print('len(covid_set)', len(covid_set))
        covid_set.update([(site, x) for x in df_covid_dx['PATID']])
        print('len(covid_set)', len(covid_set))
        covid_set.update([(site, x) for x in df_covid_prescribe['PATID']])
        print('len(covid_set)', len(covid_set))
        covid_set.update([(site, x) for x in df_covid_dispensing['PATID']])
        print('len(covid_set)', len(covid_set))
        covid_set.update([(site, x) for x in df_covid_med_admin['PATID']])
        print('len(covid_set)', len(covid_set))

        preg_set.update([(site, x) for x in
                         df_preg_dx.loc[df_preg_dx['ADMIT_DATE']>= datetime.datetime(2020, 3, 1, 0, 0), 'PATID']])
        print('len(preg_set)', len(preg_set))
        preg_set.update([(site, x) for x in
                         df_preg_pro.loc[df_preg_pro['ADMIT_DATE'] >= datetime.datetime(2020, 3, 1, 0, 0), 'PATID']])
        print('len(preg_set)', len(preg_set))
        preg_set.update([(site, x) for x in
                         df_preg_enc.loc[df_preg_enc['ADMIT_DATE'] >= datetime.datetime(2020, 3, 1, 0, 0), 'PATID']])
        print('len(preg_set)', len(preg_set))


    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
