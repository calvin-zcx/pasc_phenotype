import os
import sys

# for linux env.
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
import argparse
import time
import random
import pickle
import ast
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import functools
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce

print = functools.partial(print, flush=True)
from misc import utils

def before_2023_2_23():
    start_time = time.time()

    df_site = pd.read_excel('RECOVER Adult Site schemas_edit.xlsx')

    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']
    print('len(site_list)', len(site_list), site_list)

    vdfec = []
    vdf = []
    for i, site in tqdm(enumerate(site_list)):
        print(i, site)
        try:
            df_ec = pd.read_csv(
                r'../data/recover/output/{}/cohorts_covid_4manuNegNoCovidV2_{}_info.csv'.format(site, site))
            df_ec['n_site'] = 1
            vdfec.append(df_ec)
            data_file = r'../data/recover/output/{}/matrix_cohorts_covid_4manuNegNoCovidV2_boolbase-nout-withAllDays_{}.csv'.format(
                site,
                site)
            df_site = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str}, parse_dates=['index date'])
            df_site_sub = df_site.iloc[:, :175]
            dsum = df_site_sub.sum()
            dmean = df_site_sub.mean()
            print('')
            vdf.append(df_site_sub)
        except Exception as e:
            print('[ERROR:]', e, file=sys.stderr)
            continue
    dfec_sum = reduce(lambda x, y: x.add(y, fill_value=0), vdfec)
    print(dfec_sum)
    dfec_sum.to_csv(r'output/cohorts_covid_4manuNegNoCovidV2_all_info.csv')
    df = pd.concat(vdf, axis=0, ignore_index=True)
    df.to_csv(r'output/cohorts_covid_4manuNegNoCovidV2_all_covariates.csv')

    dsum = df.sum()
    dsum.to_csv(r'output/cohorts_covid_4manuNegNoCovidV2_all_covariates_sum.csv')

    dmean = df.mean()
    dmean.to_csv(r'output/cohorts_covid_4manuNegNoCovidV2_all_covariates_mean.csv')

    dsmdf = pd.DataFrame({'sum': dsum[2:], 'mean': dmean})
    dsmdf.to_csv(r'output/cohorts_covid_4manuNegNoCovidV2_all_covariates_summary.csv')

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


def before_20231207():
    start_time = time.time()

    df_site = pd.read_excel('RECOVER Adult Site schemas_edit.xlsx')

    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']
    print('len(site_list)', len(site_list), site_list)

    vdfec = []
    for i, site in tqdm(enumerate(site_list)):
        print(i, site)
        try:
            df_ec = pd.read_csv(
                r'../data/recover/output/{}/cohorts_covid_4manuNegNoCovidV2age18_{}_info.csv'.format(site, site))
            df_ec['n_site'] = 1
            vdfec.append(df_ec)

        except Exception as e:
            print('[ERROR:]', e, file=sys.stderr)
            continue
    dfec_sum = reduce(lambda x, y: x.add(y, fill_value=0), vdfec)
    print(dfec_sum)
    dfec_sum.to_csv(r'output/cohorts_covid_4manuNegNoCovidV2age18_all_info.csv')

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


def before_20240809():
    start_time = time.time()

    df_site = pd.read_excel('RECOVER Adult Site schemas_edit.xlsx')

    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']
    print('len(site_list)', len(site_list), site_list)

    vdfec = []
    for i, site in tqdm(enumerate(site_list)):
        print(i, site)
        try:
            df_ec = pd.read_csv(r'../data/recover/output/{}/cohorts_covid_posOnly18base_{}_info.csv'.format(site, site))
            df_ec['n_site'] = 1
            vdfec.append(df_ec)

        except Exception as e:
            print('[ERROR:]', e, file=sys.stderr)
            continue
    dfec_sum = reduce(lambda x, y: x.add(y, fill_value=0), vdfec)
    print(dfec_sum)
    dfec_sum.to_csv(r'output/cohorts_covid_posOnly18base_all_info-20231207.csv')

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


def before20250513():
    start_time = time.time()

    df_site = pd.read_excel('RECOVER Adult Site schemas_edit.xlsx')

    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']
    print('len(site_list)', len(site_list), site_list)

    # 2024-8-9 for pulmonary
    site_list = ['mcw', 'nebraska', 'utah', 'utsw',  # GPC cohort
                 'wcm', 'montefiore', 'mshs', 'columbia', 'nyu',  # insight
                 'ufh', 'nch',  # OneFlorida cohort
                 'pitt', 'psu', 'temple', 'michigan',  # PaTH cohort
                 'ochsner', 'ucsf', 'lsu',  # REACHnet cohort
                 'vumc',  # Star cohort #
                 'duke', 'emory', 'iowa', 'musc', 'osu', 'missouri'
                 ]  # 25 sites,

    # remove     'intermountain', 'ochin', 'miami', 'usf'
    vdfec = []
    for i, site in tqdm(enumerate(site_list)):
        print(i, site)
        try:
            df_ec = pd.read_csv(r'../data/recover/output/{}/cohorts_covid_posOnly18base_{}_info.csv'.format(site, site))
            df_ec['n_site'] = 1
            vdfec.append(df_ec)

        except Exception as e:
            print('[ERROR:]', e, file=sys.stderr)
            continue
    dfec_sum = reduce(lambda x, y: x.add(y, fill_value=0), vdfec)
    print(dfec_sum)
    dfec_sum.to_csv(r'output/cohorts_covid_posOnly18base_all_info-20240809-pulmomnary25.csv')

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


if __name__ == '__main__':
    start_time = time.time()

    site_list = [
                'ochin_pcornet_all', 'northwestern_pcornet_all', 'intermountain_pcornet_all', 'mcw_pcornet_all', 'iowa_pcornet_all',
                'missouri_pcornet_all', 'nebraska_pcornet_all', 'utah_pcornet_all', 'utsw_pcornet_all', 'wcm_pcornet_all',
                'monte_pcornet_all', 'mshs_pcornet_all', 'columbia_pcornet_all', 'nyu_pcornet_all', 'ufh_pcornet_all',
                'emory_pcornet_all', 'nch_pcornet_all', 'pitt_pcornet_all', 'osu_pcornet_all', 'psu_pcornet_all',
                'temple_pcornet_all', 'michigan_pcornet_all', 'stanford_pcornet_all', 'ochsner_pcornet_all',
                'ucsf_pcornet_all', 'lsu_pcornet_all', 'vumc_pcornet_all', 'duke_pcornet_all', 'wakeforest_pcornet_all']

    vdfec = []
    for i, site in tqdm(enumerate(site_list)):
        print(i, site)
        try:
            df_ec = pd.read_csv(r'../data/recover/output/{}/cohorts_covid_posOnly18base_{}_info.csv'.format(site, site))
            df_ec['n_site'] = 1
            vdfec.append(df_ec)

        except Exception as e:
            print('[ERROR:]', e, file=sys.stderr)
            continue
    dfec_sum = reduce(lambda x, y: x.add(y, fill_value=0), vdfec)
    print(dfec_sum)
    dfec_sum.to_csv(r'output/cohorts_covid_posOnly18base-2025Q2.csv')

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

