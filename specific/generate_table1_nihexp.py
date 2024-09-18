import fnmatch
import sys

# for linux env.
sys.path.insert(0, '..')
import time
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import datetime
from misc import utils
# import eligibility_setting as ecs
import functools
import fnmatch
from lifelines import KaplanMeierFitter, CoxPHFitter
import random

print = functools.partial(print, flush=True)


# from iptw.PSModels import ml
# from iptw.evaluation import *

def add_col(df):
    selected_cols = [x for x in df.columns if (
            x.startswith('DX:') or
            x.startswith('MEDICATION:') or
            x.startswith('CCI:') or
            x.startswith('obc:')
    )]
    df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')

    # baseline part have been binarized already
    selected_cols = [x for x in df.columns if
                     (x.startswith('dx-out@') or
                      x.startswith('dxadd-out@') or
                      x.startswith('dxbrainfog-out@') or
                      x.startswith('covidmed-out@') or
                      x.startswith('smm-out@') or
                      x.startswith('dxdxCFR-out@') or
                      x.startswith('mental-base@') or
                      x.startswith('dxbrainfog-base@')
                      )]

    # pasc_flag = (df['dxbrainfog-out@' + pasc].copy() >= 1).astype('int')
    # pasc_t2e = df['dxbrainfog-t2e@' + pasc].astype('float')
    # pasc_baseline = df['dxbrainfog-base@' + pasc]

    brainfog_encoding = utils.load(r'../data/mapping/brainfog_index_mapping.pkl')
    brainfog_list = list(brainfog_encoding.keys())
    # ['Neurodegenerative', 'Memory-Attention', 'Headache', 'Sleep Disorder', 'Psych', 'Dysautonomia-Orthostatic', 'Stroke']

    df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')

    df['brain_fog-cnt'] = df[['dxbrainfog-out@' + x for x in brainfog_list]].sum(axis=1)
    df['brain_fog-flag'] = (df['brain_fog-cnt'] > 0).astype('int')

    df.loc[df['death t2e'] < 0, 'death'] = np.nan
    df.loc[df['death t2e'] < 0, 'death t2e'] = 9999
    df.loc[df['death t2e'] == 9999, 'death t2e'] = np.nan

    df['death in acute'] = (df['death t2e'] <= 30).astype('int')
    df['death post acute'] = (df['death t2e'] > 30).astype('int')

    return df


def add_any_pasc(df):
    df_pasc_info = pd.read_excel(r'../prediction/output/causal_effects_specific_withMedication_v3.xlsx',
                                 sheet_name='diagnosis')
    addedPASC_encoding = utils.load(r'../data/mapping/addedPASC_index_mapping.pkl')
    addedPASC_list = list(addedPASC_encoding.keys())
    brainfog_encoding = utils.load(r'../data/mapping/brainfog_index_mapping.pkl')
    brainfog_list = list(brainfog_encoding.keys())

    pasc_simname = {}
    pasc_organ = {}
    for index, rows in df_pasc_info.iterrows():
        pasc_simname[rows['pasc']] = (rows['PASC Name Simple'], rows['Organ Domain'])
        pasc_organ[rows['pasc']] = rows['Organ Domain']

    for p in addedPASC_list:
        pasc_simname[p] = (p, 'General-add')
        pasc_organ[p] = 'General-add'

    for p in brainfog_list:
        pasc_simname[p] = (p, 'brainfog')
        pasc_organ[p] = 'brainfog'

    # pasc_list = df_pasc_info.loc[df_pasc_info['selected'] == 1, 'pasc']
    pasc_list_raw = df_pasc_info.loc[df_pasc_info['selected_narrow'] == 1, 'pasc'].to_list()
    _exclude_list = ['Pressure ulcer of skin', 'Fluid and electrolyte disorders' ]
    pasc_list = [x for x in pasc_list_raw if x not in _exclude_list]

    pasc_add = ['smell and taste',]
    print('len(pasc_list)', len(pasc_list), 'len(pasc_add)', len(pasc_add))

    for p in pasc_list:
        df[p + '_pasc_flag'] = 0
    for p in pasc_add:
        df[p + '_pasc_flag'] = 0

    df['any_pasc_flag'] = 0
    df['any_pasc_type'] = np.nan
    df['any_pasc_t2e'] = 180  # np.nan
    df['any_pasc_txt'] = ''
    df['any_pasc_baseline'] = 0  # placeholder for screening, no special meaning, null column
    for index, rows in tqdm(df.iterrows(), total=df.shape[0]):
        # for any 1 pasc
        t2e_list = []
        pasc_1_list = []
        pasc_1_name = []
        pasc_1_text = ''
        for p in pasc_list:
            if (rows['dx-out@' + p] > 0) and (rows['dx-base@' + p] == 0):
                t2e_list.append(rows['dx-t2e@' + p])
                pasc_1_list.append(p)
                pasc_1_name.append(pasc_simname[p])
                pasc_1_text += (pasc_simname[p][0] + ';')

                df.loc[index, p + '_pasc_flag'] = 1

        for p in pasc_add:
            if (rows['dxadd-out@' + p] > 0) and (rows['dxadd-base@' + p] == 0):
                t2e_list.append(rows['dxadd-t2e@' + p])
                pasc_1_list.append(p)
                pasc_1_name.append(pasc_simname[p])
                pasc_1_text += (pasc_simname[p][0] + ';')

                df.loc[index, p + '_pasc_flag'] = 1

        if len(t2e_list) > 0:
            df.loc[index, 'any_pasc_flag'] = 1
            df.loc[index, 'any_pasc_t2e'] = np.min(t2e_list)
            df.loc[index, 'any_pasc_txt'] = pasc_1_text
        else:
            df.loc[index, 'any_pasc_flag'] = 0
            df.loc[index, 'any_pasc_t2e'] = rows[['dx-t2e@' + p for p in pasc_list]].max()  # censoring time

    return df


def read_csv_files(filelist, flag_cols, dtype={'patid': str, 'site': str, 'zip': str}, parse_dates=['index date', 'dob']):
    result = []
    assert len(flag_cols) == len(flag_cols)
    for f, col in zip(filelist,flag_cols):
        print('Read f:', f)
        _df = pd.read_csv(f, dtype=dtype, parse_dates=parse_dates)
        for c in flag_cols:
            _df[c] = 0
        _df[col] = 1
        print('Read f done, _df.shape:', _df.shape)
        result.append(_df)

    df = pd.concat(result, ignore_index=True)
    print('Combined, total df.shape:', df.shape)
    return df




def table1_more_4_analyse(cohort='all', subgroup='all'):
    # %% Step 1. Load  Data
    start_time = time.time()

    np.random.seed(0)
    random.seed(0)
    negative_ratio = 5

    # add matched cohorts later
    # in_file = '../iptw/recover29Nov27_covid_pos_addCFR-PaxRisk-U099-Hospital-Preg_4PCORNet-SSRI-v2-addPaxFeats-addGeneralEC-withexposure.csv'
    # in_file = '../iptw/recover29Nov27_covid_pos_addCFR-PaxRisk-U099-Hospital-Preg_4PCORNet-SSRI-v5-withmental-addPaxFeats-addGeneralEC-withexposure.csv'

    brainfog_encoding = utils.load(r'../data/mapping/brainfog_index_mapping.pkl')
    brainfog_list = list(brainfog_encoding.keys())

    # print('in read: ', in_file)
    df1 = pd.read_csv('../iptw/recover29Nov27_covid_pos_addCFR-PaxRisk-U099-Hospital-Preg_4PCORNet-nihtable1-insight.csv',
                     dtype={'patid': str, 'site': str, 'zip': str},
                     parse_dates=['index date', 'dob',
                                  # 'flag_delivery_date',
                                  # 'flag_pregnancy_start_date',
                                  # 'flag_pregnancy_end_date'
                                  ])
    df0 = pd.read_csv('../iptw/recover29Nov27_covid_pos_addCFR-PaxRisk-U099-Hospital-Preg_4PCORNet-nihtable1-oneflorida.csv',
                      dtype={'patid': str, 'site': str, 'zip': str},
                      parse_dates=['index date', 'dob',
                                   # 'flag_delivery_date',
                                   # 'flag_pregnancy_start_date',
                                   # 'flag_pregnancy_end_date'
                                   ])

    case_label = 'INSIGHT'
    ctrl_label = 'OneFlorida'

    n1 = len(df1)

    print('n1', n1, 'n0', len(df0))
    # df0 = df0.sample(n=min(len(df0), int(negative_ratio * n1)), replace=False, random_state=0)
    print('after sample, n0', len(df0), 'with ratio:', negative_ratio, negative_ratio * n1)
    df1['treated'] = 1
    df1['SSRI'] = 1
    df0['treated'] = 0
    df0['SSRI'] = 0

    df = pd.concat([df1, df0], ignore_index=True)
    df = add_col(df)

    df_pos = df.loc[df['treated'] == 1, :]
    df_neg = df.loc[df['treated'] == 0, :]

    out_file = r'.Table-recover29Nov27_covid_pos_insight-vs-oneflorida-brainFog.xlsx'
    output_columns = ['All', case_label, ctrl_label, 'SMD']

    print('treated df_pos.shape', df_pos.shape,
          'control df_neg.shape', df_neg.shape,
          'combined df.shape', df.shape, )

    def _n_str(n):
        return '{:,}'.format(n)

    def _quantile_str(x):
        v = x.quantile([0.25, 0.5, 0.75]).to_list()
        return '{:.0f} ({:.0f}—{:.0f})'.format(v[1], v[0], v[2])

    def _percentage_str(x):
        n = x.sum()
        per = x.mean()
        return '{:,} ({:.1f})'.format(n, per * 100)

    def _smd(x1, x2):
        m1 = x1.mean()
        m2 = x2.mean()
        v1 = x1.var()
        v2 = x2.var()

        VAR = np.sqrt((v1 + v2) / 2)
        smd = np.divide(
            m1 - m2,
            VAR, out=np.zeros_like(m1), where=VAR != 0)
        return smd

    row_names = []
    records = []

    # N
    row_names.append('N')
    records.append([
        _n_str(len(df)),
        _n_str(len(df_pos)),
        _n_str(len(df_neg)),
        np.nan
    ])

    # Sex
    row_names.append('Sex — no. (%)')
    records.append([])
    sex_col = ['Female', 'Male', 'Other/Missing']
    # sex_col = ['Female', 'Male']

    row_names.extend(sex_col)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in sex_col])

    # row_names.append('Acute severity — no. (%)')
    # records.append([])
    # # col_names = ['outpatient', 'inpatient', 'icu', 'inpatienticu',
    # #              'hospitalized', 'ventilation', 'criticalcare', ]
    # col_names = ['outpatient',  ]
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])

    # age
    row_names.append('Median age (IQR) — yr')
    records.append([
        _quantile_str(df['age']),
        _quantile_str(df_pos['age']),
        _quantile_str(df_neg['age']),
        _smd(df_pos['age'], df_neg['age'])
    ])

    row_names.append('Age group — no. (%)')
    records.append([])

    age_col = ['age@18-24', 'age@25-34', 'age@35-49', 'age@50-64',
               '65-<75 years', 'age@75+', '75-<85 years', '85+ years']  # 'age@65+'
    row_names.extend(age_col)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in age_col])

    # Race
    row_names.append('Race — no. (%)')
    records.append([])
    # col_names = ['Asian', 'Black or African American', 'White', 'Other', 'Missing']
    col_names = ['RE:Asian Non-Hispanic', 'RE:Black or African American Non-Hispanic', 'RE:Hispanic or Latino Any Race',
                 'RE:White Non-Hispanic', 'RE:Other Non-Hispanic', 'RE:Unknown']
    col_names = ['table1-NHW', 'table1-NHB', 'table1-Hispanics', 'table1-Other', 'table1-Unknown']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # Ethnic group
    row_names.append('Ethnic group — no. (%)')
    records.append([])
    col_names = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    col_names = ['brain_fog-flag',
                 'dxbrainfog-out@Neurodegenerative', 'dxbrainfog-out@Memory-Attention',
                 'dxbrainfog-out@Headache', 'dxbrainfog-out@Sleep Disorder',
                 'dxbrainfog-out@Psych',
                 'dxbrainfog-out@Dysautonomia-Orthostatic', 'dxbrainfog-out@Stroke' ]
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    df_out = pd.DataFrame(records, columns=output_columns, index=row_names)
    df_out['SMD'] = df_out['SMD'].astype(float)
    df_out.to_excel(out_file)
    print('Dump done ', df_out)
    return df, df_out


if __name__ == '__main__':
    start_time = time.time()

    table1_more_4_analyse( cohort='all') #

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
