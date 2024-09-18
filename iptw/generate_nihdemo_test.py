import sys

# for linux env.
sys.path.insert(0, '..')
import time
import pickle
import argparse
from evaluation import *
import os
import random
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from PSModels import ml
from misc import utils
import itertools
import functools
from tqdm import tqdm
import datetime
import seaborn as sns
from sklearn.preprocessing import SplineTransformer
import itertools

print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    # parser.add_argument('--dataset', choices=['oneflorida', 'V15_COVID19'], default='V15_COVID19',
    #                     help='data bases')
    parser.add_argument('--site', default='all',  # choices=['COL', 'MSHS', 'MONTE', 'NYU', 'WCM', 'ALL', 'all'],
                        help='one particular site or all')
    parser.add_argument('--severity', choices=['all',
                                               'outpatient', 'inpatient', 'icu', 'inpatienticu',
                                               'female', 'male',
                                               'white', 'black',
                                               'less65', '65to75', '75above', '20to40', '40to55', '55to65', 'above65',
                                               'Anemia', 'Arrythmia', 'CKD', 'CPD-COPD', 'CAD',
                                               'T2D-Obesity', 'Hypertension', 'Mental-substance', 'Corticosteroids',
                                               'healthy',
                                               '03-20-06-20', '07-20-10-20', '11-20-02-21',
                                               '03-21-06-21', '07-21-11-21',
                                               '1stwave', 'delta', 'alpha', 'preg-pos-neg',
                                               'pospreg-posnonpreg',
                                               'fullyvac', 'partialvac', 'anyvac', 'novacdata',
                                               ],
                        default='all')
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument('--negative_ratio', type=int, default=10)  # 5
    parser.add_argument('--selectpasc', action='store_true')

    parser.add_argument("--kmatch", type=int, default=3)
    parser.add_argument("--usedx", type=int, default=1)  # useacute
    parser.add_argument("--useacute", type=int, default=1)

    args = parser.parse_args()

    # More args

    if args.random_seed < 0:
        from datetime import datetime
        args.random_seed = int(datetime.now())

    # args.save_model_filename = os.path.join(args.output_dir, '_S{}{}'.format(args.random_seed, args.run_model))
    # utils.check_and_mkdir(args.save_model_filename)
    return args


def OLD_MAIN():
    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 1 --usedx 1 --useacute 1 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch1-useSelectdx1-useacute1.txt
    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 3 --usedx 1 --useacute 1 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch3-useSelectdx1-useacute1.txt
    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 5 --usedx 1 --useacute 1 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch5-useSelectdx1-useacute1.txt
    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 10 --usedx 1 --useacute 1 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch10-useSelectdx1-useacute1.txt

    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 1 --usedx 1 --useacute 0 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch1-useSelectdx1-useacute0.txt
    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 3 --usedx 1 --useacute 0 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch3-useSelectdx1-useacute0.txt
    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 5 --usedx 1 --useacute 0 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch5-useSelectdx1-useacute0.txt
    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 10 --usedx 1 --useacute 0 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch10-useSelectdx1-useacute0.txt

    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 1 --usedx 0 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch1-usedx0.txt
    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 3 --usedx 0 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch3-usedx0.txt
    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 5 --usedx 0 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch5-usedx0.txt
    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 10 --usedx 0 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch10-usedx0.txt

    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)
    # print('save_model_filename', args.save_model_filename)

    # %% 1. Load  Data
    print('In cohorts_characterization_build_data...')

    if args.site == 'all':
        sites = ['ochin',
                 'intermountain', 'mcw', 'iowa', 'missouri', 'nebraska', 'utah', 'utsw',
                 'wcm', 'montefiore', 'mshs', 'columbia', 'nyu',
                 'ufh', 'emory', 'usf', 'nch', 'miami',
                 'pitt', 'osu', 'psu', 'temple', 'michigan',
                 'ochsner', 'ucsf', 'lsu',
                 'vumc', 'duke', 'musc']

        insight = ['wcm', 'montefiore', 'mshs', 'columbia', 'nyu', ]
        florida = ['ufh', 'emory', 'usf', 'nch', 'miami', ]

        df_site = pd.read_excel(r'../prerecover/RECOVER Adult Site schemas_edit.xlsx')
        _site_network = df_site[['Schema name', 'pcornet']].values.tolist()
        site_network = {x[0].strip(): x[1].strip() for x in _site_network}
        # sites = ['wcm', 'montefiore', 'mshs', ]
        # sites = ['wcm', ]
        # sites = ['pitt', ]
        sites = insight
        print('len(sites), sites:', len(sites), sites)
    else:
        sites = [args.site, ]

    df_info_list = []
    df_label_list = []
    df_covs_list = []
    df_outcome_list = []

    df_list = []
    for ith, site in tqdm(enumerate(sites), total=len(sites)):
        print('Loading: ', ith, site)
        # matrix_cohorts_covid_posOnly18base-nbaseout-alldays-preg_mshs + pregnancy tag afterwards
        data_file = r'../data/recover/output/pregnancy_data/pregnancy_{}.csv'.format(site)
        # Load Covariates Data
        print('Load data covariates file:', data_file)
        df = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str},
                         parse_dates=['index date', 'dob',
                                      'flag_delivery_date',
                                      'flag_pregnancy_start_date',
                                      'flag_pregnancy_end_date'])
        # because a patid id may occur in multiple sites. patid were site specific
        df['pcornet'] = site_network[site]
        print('df.shape:', df.shape)
        df_list.append(df)

    # combine all sites and select subcohorts
    df = pd.concat(df_list, ignore_index=True)

    # print(r"df['site'].value_counts(sort=False)", df['site'].value_counts(sort=False))
    print(r"df['site'].value_counts()", df['site'].value_counts())
    print('over all: df.shape:', df.shape)
    print('Pregnant in all:',
          len(df),
          df['flag_pregnancy'].sum(),
          df['flag_pregnancy'].mean())
    print('Pregnant in pos:',
          len(df.loc[df['covid'] == 1, :]),
          df.loc[df['covid'] == 1, 'flag_pregnancy'].sum(),
          df.loc[df['covid'] == 1, 'flag_pregnancy'].mean())
    print('Pregnant in neg:',
          len(df.loc[df['covid'] == 0, :]),
          df.loc[df['covid'] == 0, 'flag_pregnancy'].sum(),
          df.loc[df['covid'] == 0, 'flag_pregnancy'].mean())
    print('Pregnant excluded special cases in all:',
          len(df),
          df['flag_exclusion'].sum(),
          df['flag_exclusion'].mean())

    # %% 2. Cohort building
    df = select_subpopulation(df, args.severity)
    df_general, df1, df2 = more_ec_for_cohort_selection(df)
    zz
    df1 = feature_process_additional(df1)
    df2 = feature_process_additional(df2)

    utils.check_and_mkdir(r'../data/recover/output/pregnancy_output/')
    df1.to_csv(r'../data/recover/output/pregnancy_output/covidpos_eligible_pregnant.csv')
    df2.to_csv(r'../data/recover/output/pregnancy_output/covidpos_eligible_Non-pregnant.csv')
    utils.dump((df1, df2), r'../data/recover/output/pregnancy_output/_selected_preg_cohort_1-2.pkl')

    print('Severity cohorts:', args.severity,
          'df.shape:', df.shape,
          'df_general.shape:', df_general.shape,
          'df1.shape:', df1.shape,
          'df2.shape:', df2.shape,
          )
    print('Cohort build Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    zz

    print('Cohort build Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


if __name__ == "__main__":

    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)
    # print('save_model_filename', args.save_model_filename)

    # %% 1. Load  Data
    print('In cohorts_characterization_build_data...')

    # infile = r'../prerecover/output/demo-covid_posOnly18base-florida.csv'
    # outfile = r'../prerecover/output/demo-covid_posOnly18base-both-summarize.csv'
    df_list =[]
    for infile in [r'../prerecover/output/demo-covid_posOnly18base-insight.csv',
                   r'../prerecover/output/demo-covid_posOnly18base-florida.csv']:
        df = pd.read_csv(infile,
                         dtype={'patid': str, 'site': str, 'zip': str},
                         parse_dates=['index date', ])
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    outfile = r'../prerecover/output/demo-covid_posOnly18base-both-summarize.csv'

    race_column_names = ['American Indian/Alaska Native',
                         'Asian',
                         'Native Hawaiian or Other Pacific Islander',
                         'Black or African American',
                         'White',
                         'More than One Race',
                         'Unknown or Not Reported']
    hispanic_column_names = ['Hispanic: No',
                             'Hispanic: Yes',
                             'Hispanic: Other/Missing']

    gender_column_names = ['Female',
                           'Male',
                           'Other/Missing']
    print('df.shape', df.shape)
    data = np.zeros((7, 9))
    for i, ci in enumerate(race_column_names):
        for j, cj in enumerate(hispanic_column_names):
            for k, ck in enumerate(gender_column_names):
                nn = len(df.loc[(df[ci] == 1) & (df[cj] == 1) & (df[ck] == 1), :])
                data[i, j * 3 + k] = nn

    df = pd.DataFrame(data, index=race_column_names,
                      columns=list(itertools.product(hispanic_column_names, gender_column_names)))
    df.to_csv(outfile)
    # if args.site == 'all':
    #     sites = ['ochin',
    #              'intermountain', 'mcw', 'iowa', 'missouri', 'nebraska', 'utah', 'utsw',
    #              'wcm', 'montefiore', 'mshs', 'columbia', 'nyu',
    #              'ufh', 'emory', 'usf', 'nch', 'miami',
    #              'pitt', 'osu', 'psu', 'temple', 'michigan',
    #              'ochsner', 'ucsf', 'lsu',
    #              'vumc', 'duke', 'musc']
    #
    #     insight = ['wcm', 'montefiore', 'mshs', 'columbia', 'nyu',]
    #     florida = ['ufh', 'emory', 'usf', 'nch', 'miami',]
    #
    #     df_site = pd.read_excel(r'../prerecover/RECOVER Adult Site schemas_edit.xlsx')
    #     _site_network = df_site[['Schema name', 'pcornet']].values.tolist()
    #     site_network = {x[0].strip(): x[1].strip() for x in _site_network}
    #     # sites = ['wcm', 'montefiore', 'mshs', ]
    #     # sites = ['wcm', ]
    #     # sites = ['pitt', ]
    #     sites = insight
    #     print('len(sites), sites:', len(sites), sites)
    # else:
    #     sites = [args.site, ]

    # df_info_list = []
    # df_label_list = []
    # df_covs_list = []
    # df_outcome_list = []
    #
    # df_list = []
    # for ith, site in tqdm(enumerate(sites), total=len(sites)):
    #     print('Loading: ', ith, site)
    #     # matrix_cohorts_covid_posOnly18base-nbaseout-alldays-preg_mshs + pregnancy tag afterwards
    #     data_file = r'../data/recover/output/pregnancy_data/pregnancy_{}.csv'.format(site)
    #     # Load Covariates Data
    #     print('Load data covariates file:', data_file)
    #     df = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str},
    #                      parse_dates=['index date', 'dob',
    #                                   'flag_delivery_date',
    #                                   'flag_pregnancy_start_date',
    #                                   'flag_pregnancy_end_date'])
    #     # because a patid id may occur in multiple sites. patid were site specific
    #     df['pcornet'] = site_network[site]
    #     print('df.shape:', df.shape)
    #     df_list.append(df)
    #
    # # combine all sites and select subcohorts
    # df = pd.concat(df_list, ignore_index=True)
    #
    # # print(r"df['site'].value_counts(sort=False)", df['site'].value_counts(sort=False))
    # print(r"df['site'].value_counts()", df['site'].value_counts())
    # print('over all: df.shape:', df.shape)
    # print('Pregnant in all:',
    #       len(df),
    #       df['flag_pregnancy'].sum(),
    #       df['flag_pregnancy'].mean())
    # print('Pregnant in pos:',
    #       len(df.loc[df['covid'] == 1, :]),
    #       df.loc[df['covid'] == 1, 'flag_pregnancy'].sum(),
    #       df.loc[df['covid'] == 1, 'flag_pregnancy'].mean())
    # print('Pregnant in neg:',
    #       len(df.loc[df['covid'] == 0, :]),
    #       df.loc[df['covid'] == 0, 'flag_pregnancy'].sum(),
    #       df.loc[df['covid'] == 0, 'flag_pregnancy'].mean())
    # print('Pregnant excluded special cases in all:',
    #       len(df),
    #       df['flag_exclusion'].sum(),
    #       df['flag_exclusion'].mean())
    #
    # # %% 2. Cohort building
    # df = select_subpopulation(df, args.severity)
    # df_general, df1, df2 = more_ec_for_cohort_selection(df)
    # zz
    # df1 = feature_process_additional(df1)
    # df2 = feature_process_additional(df2)
    #
    # utils.check_and_mkdir(r'../data/recover/output/pregnancy_output/')
    #
    # df2.to_csv(r'../data/recover/output/pregnancy_output/covidpos_eligible_Non-pregnant.csv')
    # utils.dump((df1, df2), r'../data/recover/output/pregnancy_output/_selected_preg_cohort_1-2.pkl')
    #
    # print('Severity cohorts:', args.severity,
    #       'df.shape:', df.shape,
    #       'df_general.shape:', df_general.shape,
    #       'df1.shape:', df1.shape,
    #       'df2.shape:', df2.shape,
    #       )
    # print('Cohort build Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    #
    # zz

    print('Cohort build Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
