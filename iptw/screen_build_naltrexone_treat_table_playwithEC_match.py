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


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    # parser.add_argument('--dataset', choices=['oneflorida', 'V15_COVID19'], default='V15_COVID19',
    #                     help='data bases')
    # parser.add_argument('--site', default='all',  # choices=['COL', 'MSHS', 'MONTE', 'NYU', 'WCM', 'ALL', 'all'],
    #                     help='one particular site or all')
    # parser.add_argument('--severity', choices=['all',
    #                                            'outpatient', 'inpatient', 'icu', 'inpatienticu',
    #                                            'female', 'male',
    #                                            'white', 'black',
    #                                            'less65', '65to75', '75above', '20to40', '40to55', '55to65', 'above65',
    #                                            'Anemia', 'Arrythmia', 'CKD', 'CPD-COPD', 'CAD',
    #                                            'T2D-Obesity', 'Hypertension', 'Mental-substance', 'Corticosteroids',
    #                                            'healthy',
    #                                            '03-20-06-20', '07-20-10-20', '11-20-02-21',
    #                                            '03-21-06-21', '07-21-11-21',
    #                                            '1stwave', 'delta', 'alpha', 'preg-pos-neg',
    #                                            'pospreg-posnonpreg',
    #                                            'fullyvac', 'partialvac', 'anyvac', 'novacdata',
    #                                            ],
    #                     default='all')
    parser.add_argument("--random_seed", type=int, default=0)
    # parser.add_argument('--negative_ratio', type=int, default=10)  # 5
    # parser.add_argument('--selectpasc', action='store_true')

    parser.add_argument("--kmatch", type=int, default=5)
    parser.add_argument('--replace', action='store_true')

    # parser.add_argument("--usedx", type=int, default=1)  # useacute
    # parser.add_argument("--useacute", type=int, default=1)

    args = parser.parse_args()

    # More args

    if args.random_seed < 0:
        from datetime import datetime
        args.random_seed = int(datetime.now())

    # args.save_model_filename = os.path.join(args.output_dir, '_S{}{}'.format(args.random_seed, args.run_model))
    # utils.check_and_mkdir(args.save_model_filename)
    return args


# from iptw.PSModels import ml
# from iptw.evaluation import *
def _t2eall_to_int_list_dedup(t2eall):
    t2eall = t2eall.strip(';').split(';')
    t2eall = set(map(int, t2eall))
    t2eall = sorted(t2eall)

    return t2eall


def add_col_2(df):
    print('add_col_2: add and revise some covs per the needs. in addition to prior screen_build_naltrexone file and add_col')
    df['cci_quan:3+'] = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):
        if row['score_cci_quan'] >= 3:
            df.loc[index, 'cci_quan:3+'] = 1

    print('add_col_2 done')
    return df

def add_col(df):
    print('add and revise some covs per the needs. in addition to prior screen_build_naltrexone file')
    drug_names = [
        'naltrexone',
    ]

    # step 1: add index date
    # step 2: add ADHD
    df['index_date_drug'] = np.nan
    df['index_date_drug_days'] = np.nan
    df['ADHD_before_drug_onset'] = 0

    df['cci_quan:3+'] = 0

    covNaltrexone_med_names = ['NSAIDs_combined', 'opioid drug']
    for x in covNaltrexone_med_names:
        df['covNaltrexone_med-base@' + x] = 0  # -3yrs to 0
        df['covNaltrexone_med-basedrugonset@' + x] = 0  # target drug, acute Naltrexone time, no left limit

    for index, row in tqdm(df.iterrows(), total=len(df)):
        index_date = pd.to_datetime(row['index date'])

        if row['score_cci_quan'] >= 3:
            df.loc[index, 'cci_quan:3+'] = 1

        # step 1: drug onset day
        drug_onsetday_list = []
        for x in drug_names:
            t2eall = row['cnsldn-t2eall@' + x]
            if pd.notna(t2eall):
                t2eall = _t2eall_to_int_list_dedup(t2eall)
                for t2e in t2eall:
                    if 0 <= t2e < 30:
                        drug_onsetday_list.append(t2e)

        drug_onsetday_list = sorted(drug_onsetday_list)
        if drug_onsetday_list:
            if row['treated'] == 1:
                index_date_drug = index_date + datetime.timedelta(days=drug_onsetday_list[0])
                df.loc[index, 'index_date_drug'] = index_date_drug
                df.loc[index, 'index_date_drug_days'] = drug_onsetday_list[0]
            else:
                index_date_drug = index_date  # or matching on index_date_drug_days

        # step 2: ADHD at drug
        t2eallADHD = row['dxcovCNSLDN-t2eall@ADHD']
        if pd.notna(t2eallADHD):
            t2eallADHD = _t2eall_to_int_list_dedup(t2eallADHD)
            for t in t2eallADHD:
                if (row['treated'] == 1) & (len(drug_onsetday_list) > 0):
                    if t <= drug_onsetday_list[0]:
                        df.loc[index, 'ADHD_before_drug_onset'] = 1
                else:
                    if t <= 0:
                        df.loc[index, 'ADHD_before_drug_onset'] = 1

        # step 3: ['NSAIDs_combined', 'opioid drug'], at covid t0 and target drug index t0
        for x in covNaltrexone_med_names:
            t2eall = row['covNaltrexone_med-t2eall@' + x]
            if pd.notna(t2eall):
                t2eall = _t2eall_to_int_list_dedup(t2eall)

                for t2e in t2eall:
                    if -1095 <= t2e <= -7:
                        df.loc[index, 'covNaltrexone_med-base@' + x] = 1

                    # just before drug index for treated, covid 0 for untreated, no left bound
                    if (row['treated'] == 1) & (len(drug_onsetday_list) > 0):
                        if t2e <= drug_onsetday_list[0]:
                            df.loc[index, 'covNaltrexone_med-basedrugonset@' + x] = 1
                    else:
                        if t2e <= 0:
                            df.loc[index, 'covNaltrexone_med-basedrugonset@' + x] = 1

        # end
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
                      x.startswith('dxcovCNSLDN-base@') or
                      x.startswith('dxcovNaltrexone-base@')  # or
                      # x.startswith('dxcovNaltrexone-base@')
                      )]
    df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')

    df.loc[df['death t2e'] < 0, 'death'] = np.nan
    df.loc[df['death t2e'] < 0, 'death t2e'] = 9999
    df.loc[df['death t2e'] == 9999, 'death t2e'] = np.nan

    df['death in acute'] = (df['death t2e'] <= 30).astype('int')
    df['death post acute'] = (df['death t2e'] > 30).astype('int')

    return df


def exact_match_on(df_case_og, df_ctrl_og, kmatch, cols_to_match, replace=False, random_seed=0):
    print('Matched on columns, len(cols_to_match):', len(cols_to_match), cols_to_match)
    print('len(df_case_og)', len(df_case_og), 'len(df_ctrl_og)', len(df_ctrl_og), 'kmatch:', kmatch)
    print('replace:', replace, 'random_seed:', random_seed)

    df_case_og['exposure_label'] = 1
    df_ctrl_og['exposure_label'] = 0

    df_case_og['match_to'] = np.nan
    df_ctrl_og['match_to'] = np.nan
    df_case_og['match_to_k'] = 0
    df_ctrl_og['match_to_k'] = 0
    df_case_og['match_to_k_unique'] = 0
    df_ctrl_og['match_to_k_unique'] = 0

    if not replace:
        df_case, df_ctrl = df_case_og.copy(), df_ctrl_og.copy()
    else:
        df_case, df_ctrl = df_case_og, df_ctrl_og

    ctrl_list = []
    n_no_match = 0
    n_fewer_match = 0
    for index, rows in tqdm(df_case.iterrows(), total=df_case.shape[0]):
        # if index == 275 or index == 1796:
        #     print(275, 'debug')
        #     print(1796, 'debug')
        # else:
        #     continue
        boolidx = df_ctrl[cols_to_match[0]] == rows[cols_to_match[0]]
        for c in cols_to_match[1:]:
            boolidx &= df_ctrl[c] == rows[c]
        sub_df = df_ctrl.loc[boolidx, :]

        if not replace:
            if len(sub_df) >= kmatch:
                _add_index = sub_df.sample(n=kmatch, replace=False, random_state=random_seed).index
            elif len(sub_df) > 0:
                n_fewer_match += 1
                _add_index = sub_df.sample(frac=1, replace=False, random_state=random_seed).index
                print(len(sub_df), ' match for', index)
            else:
                _add_index = []
                n_no_match += 1
                print('No match for', index)

            # df_ctrl.drop(_add_index, inplace=True)
            if len(_add_index) > 0:
                # df_ctrl = df_ctrl.loc[~df_ctrl.index.isin(_add_index)]
                df_ctrl.drop(index=_add_index, inplace=True)
        else:
            if len(sub_df) > 0:
                _add_index = sub_df.sample(n=kmatch, replace=True, random_state=random_seed).index
            else:
                _add_index = []
                n_no_match += 1
                print('No match for', index)

        ctrl_list.extend(_add_index)

        df_case.loc[index, 'match_to'] = ';'.join(str(x) for x in _add_index)
        df_case.loc[index, 'match_to_k'] = len(_add_index)
        df_case.loc[index, 'match_to_k_unique'] = len(set(_add_index))

        if len(df_ctrl) == 0:
            break

    print('len(ctrl_list)', len(ctrl_list))
    # neg_selected = pd.Series(False, index=df_ctrl_og.index)
    # neg_selected[ctrl_list] = True
    #
    # df_ctrl_matched = df_ctrl_og.loc[neg_selected, :]

    df_ctrl_matched_v2 = df_ctrl_og.loc[ctrl_list]
    print('Done, total {}:{} no match, {} fewer match'.format(len(df_case), n_no_match, n_fewer_match))
    return df_case.copy(), df_ctrl_matched_v2.copy()


def build_matched_control(df_case, df_contrl, kmatch=10, replace=False):  # , usedx=True, useacute=True
    print('In build matched control, before match: len(case), len(ctrl), ratio:, relace',
          len(df_case), len(df_contrl), kmatch, replace)
    start_time = time.time()

    sex_col = ['Female', 'Male', ]
    age_col = ['age@18-24', 'age@25-34', 'age@35-49', 'age@50-64',
               'age@65+', ]  # '65-<75 years', '75-<85 years', '85+ years', ]

    race_eth_col = ['RE:Asian Non-Hispanic',
                    'RE:Black or African American Non-Hispanic',
                    'RE:Hispanic or Latino Any Race', 'RE:White Non-Hispanic',
                    'RE:Other Non-Hispanic', 'RE:Unknown', ]

    acute_col = ['outpatient', 'inpatienticu']  # 'inpatient', 'icu'

    # period_col = ['03/20-06/20', '07/20-10/20', '11/20-02/21',
    #               '03/21-06/21', '07/21-10/21', '11/21-02/22',
    #               '03/22-06/22', '07/22-10/22', '11/22-02/23',
    #               '03/23-06/23', '07/23-10/23', '11/23-02/24',
    #               '03/24-06/24', '07/24-10/24', ]

    dx_col = [
        # 'dxcovNaltrexone-base@pain include',
        'dxcovNaltrexone-basedrugonset@substance use disorder ',
        'PaxRisk:Obesity',
        # 'dxcovNaltrexone-basedrugonset@fibromyalgia',
        'PaxRisk:Chronic kidney disease',
        'PaxRisk:Chronic liver disease',
        'PaxRisk:Mental health conditions',
        'mental-base@Depressive Disorders',
        'mental-base@Anxiety Disorders',
        'PaxRisk:Smoking current',
        # df['PaxRisk:Smoking current'] = ((df['Smoker: current'] >= 1) | (df['Smoker: former'] >= 1)).astype('int')
    ]

    cci_score = ['cci_quan:0', 'cci_quan:1-2', 'cci_quan:3+', ]  # 'cci_quan:3-4', 'cci_quan:5-10', 'cci_quan:11+']

    # # cols_to_match = ['site', ] + age_col + period_col + acute_col + race_col + eth_col
    # cols_to_match = ['pcornet', ] + age_col + period_col  # + acute_col
    # if useacute:
    #     cols_to_match += acute_col
    #
    # if usedx:
    #     cols_to_match += dx_col

    cols_to_match = sex_col + age_col + race_eth_col + acute_col + dx_col + cci_score  # + period_col

    df_case, df_ctrl_matched = exact_match_on(df_case, df_contrl, kmatch, cols_to_match, replace=replace)

    # print('len(ctrl_list)', len(ctrl_list))
    # neg_selected = pd.Series(False, index=df_contrl.index)
    # neg_selected[ctrl_list] = True
    # df_ctrl_matched = df_contrl.loc[neg_selected, :]
    print('len(df_case):', len(df_case),
          'len(df_contrl):', len(df_contrl),
          'len(df_ctrl_matched):', len(df_ctrl_matched),
          'kmatch:', kmatch,
          'actual ratio:', len(df_ctrl_matched)/len(df_case))

    print('build_matched_control Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return df_case, df_ctrl_matched


def ec_anyenc_1yrbefore_baseline(_df):
    print('before ec_anyenc_1yrbefore_baseline, _df.shape', _df.shape)
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    n0 = len(_df)
    _df = _df.loc[(_df['baseline1yr_any_dx_enc'] == 1)]
    n1 = len(_df)
    print('after ec_anyenc_1yrbefore_baseline, _df.shape', _df.shape)
    print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    return _df


def ec_no_U099_baseline(_df):
    print('before ec_no_U099_baseline, _df.shape', _df.shape)
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    n0 = len(_df)
    _df = _df.loc[(_df['dx-base@PASC-General'] == 0)]
    n1 = len(_df)
    print('after ec_no_U099_baseline, _df.shape', _df.shape)
    print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    # print('exclude baseline U099 in df_ec_start', (df_ec_start['dx-base@PASC-General'] > 0).sum())
    return _df


# pain_before_drug_onset
# potential revise pain to pain at drug onset
def ec_pain_baseline(_df):
    # revise: 2025-7-17, not include NSAIDs, use cov at drug on set
    print('before ec_pain_baseline, _df.shape', _df.shape)
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    n0 = len(_df)
    # _df = _df.loc[
    #     (_df['dxcovNaltrexone-base@Pain'] == 1) | (_df['covNaltrexone_med-basedrugonset@NSAIDs_combined'] == 1)]
    _df = _df.loc[
        (_df['dxcovNaltrexone-base@Pain'] == 1) | (_df['dxcovNaltrexone-basedrugonset@Pain'] == 1)]
    n1 = len(_df)
    print('after ec_pain_baseline, _df.shape', _df.shape)
    print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    return _df


def ec_painIncludeAndTbd_baseline(_df):
    print('before ec_painIncludeAndTbd_baseline, _df.shape', _df.shape)
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    n0 = len(_df)
    # _df = _df.loc[
    #     (_df['dxcovNaltrexone-base@Pain'] == 1) | (_df['covNaltrexone_med-basedrugonset@NSAIDs_combined'] == 1)]
    _df = _df.loc[
        (_df['dxcovNaltrexone-base@pain include'] == 1) | (_df['dxcovNaltrexone-basedrugonset@pain include'] == 1) |
        (_df['dxcovNaltrexone-base@pain tbd'] == 1) | (_df['dxcovNaltrexone-basedrugonset@pain tbd'] == 1)]
    n1 = len(_df)
    print('after ec_painIncludeAndTbd_baseline, _df.shape', _df.shape)
    print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    return _df


def ec_painIncludeOnly_baseline(_df):
    # revise: 2025-7-17, not include NSAIDs, use cov at drug on set
    print('before ec_painIncludeOnly_baseline, _df.shape', _df.shape)
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    n0 = len(_df)
    # _df = _df.loc[
    #     (_df['dxcovNaltrexone-base@Pain'] == 1) | (_df['covNaltrexone_med-basedrugonset@NSAIDs_combined'] == 1)]
    _df = _df.loc[
        (_df['dxcovNaltrexone-base@pain include'] == 1) | (_df['dxcovNaltrexone-basedrugonset@pain include'] == 1)]
    n1 = len(_df)
    print('after ec_painIncludeOnly_baseline, _df.shape', _df.shape)
    print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    return _df


# def ec_pain_obesity_OUD_withmed_baseline(_df):
#     print('before ec_pain_baseline, _df.shape', _df.shape)
#     print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())
#
#     n0 = len(_df)
#     _df = _df.loc[
#         (_df['dxcovNaltrexone-base@Pain'] == 1) |
#         (_df['covNaltrexone_med-basedrugonset@NSAIDs_combined'] == 1) |
#         (_df['PaxRisk:Obesity'] == 1) |
#         (_df['dxcovNaltrexone-base@opioid use disorder'] == 1) |
#         (_df['PaxRisk:Obesity'] == 1) |
#         (_df['covNaltrexone_med-basedrugonset@opioid drug'] == 1)]
#     n1 = len(_df)
#     print('after ec_pain_baseline, _df.shape', _df.shape)
#     print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
#     print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())
#
#     return _df

def ec_pain_obesity_OUD_baseline(_df):
    print('before ec_pain_obesity_OUD_baseline, _df.shape', _df.shape)
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    n0 = len(_df)
    _df = _df.loc[
        (_df['dxcovNaltrexone-base@Pain'] == 1) |
        (_df['PaxRisk:Obesity'] == 1) |
        (_df['dxcovNaltrexone-base@opioid use disorder'] == 1) |
        (_df['PaxRisk:Obesity'] == 1) |
        (_df['dxcovNaltrexone-basedrugonset@Pain'] == 1) |
        (_df['dxcovNaltrexone-basedrugonset@opioid use disorder'] == 1) |
        (_df['dxcovNaltrexone-basedrugonset@Obesity'] == 1)
        ]
    n1 = len(_df)
    print('after ec_pain_baseline, _df.shape', _df.shape)
    print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    return _df


def ec_painIncludeAndTbd_obesity_OUD_baseline(_df):
    print('before ec_painIncludeAndTbd_obesity_OUD_baseline, _df.shape', _df.shape)
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    n0 = len(_df)
    _df = _df.loc[
        (_df['dxcovNaltrexone-base@pain include'] == 1) | (_df['dxcovNaltrexone-basedrugonset@pain include'] == 1) |
        (_df['dxcovNaltrexone-base@pain tbd'] == 1) | (_df['dxcovNaltrexone-basedrugonset@pain tbd'] == 1) |
        (_df['PaxRisk:Obesity'] == 1) |
        (_df['dxcovNaltrexone-base@opioid use disorder'] == 1) |
        (_df['PaxRisk:Obesity'] == 1) |
        (_df['dxcovNaltrexone-basedrugonset@opioid use disorder'] == 1) |
        (_df['dxcovNaltrexone-basedrugonset@Obesity'] == 1)
        ]
    n1 = len(_df)
    print('after ec_painIncludeAndTbd_obesity_OUD_baseline, _df.shape', _df.shape)
    print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    return _df


def ec_no_cancer_baseline(_df):
    print('before ec_no_cancer_baseline, _df.shape', _df.shape)
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    n0 = len(_df)
    _df = _df.loc[(_df['PaxRisk:Cancer'] == 0)]
    n1 = len(_df)
    print('after ec_no_cancer_baseline, _df.shape', _df.shape)
    print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    # print('include baseline ADHD in df_ec_start', (df_ec_start['ADHD_before_drug_onset'] == 0).sum())
    return _df


def ec_no_pregnant_baseline(_df):
    print('before ec_no_pregnant_baseline, _df.shape', _df.shape)
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    n0 = len(_df)
    _df = _df.loc[(_df['PaxRisk:Pregnancy'] == 0)]
    n1 = len(_df)
    print('after ec_no_pregnant_baseline, _df.shape', _df.shape)
    print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    # print('include baseline ADHD in df_ec_start', (df_ec_start['ADHD_before_drug_onset'] == 0).sum())
    return _df


def ec_no_HIV_baseline(_df):
    print('before ec_no_HIV_baseline, _df.shape', _df.shape)
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    n0 = len(_df)
    _df = _df.loc[(_df['PaxRisk:HIV infection'] == 0)]
    n1 = len(_df)
    print('after ec_no_HIV_baseline, _df.shape', _df.shape)
    print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    # print('include baseline ADHD in df_ec_start', (df_ec_start['ADHD_before_drug_onset'] == 0).sum())
    return _df


def ec_no_severe_conditions_4_pax(_df):
    print('before ec_no_severe_conditions_4_pax, _df.shape', _df.shape)
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    n0 = len(_df)
    _df = _df.loc[(_df['PaxExclude-Count'] == 0)]
    print('after ec_no_severe_conditions_4_pax, _df.shape', _df.shape)
    n1 = len(_df)
    print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    # print('exclude severe conditions in df_ec_start', (df_ec_start['PaxExclude-Count'] > 0).sum())
    return _df


def ec_timbe_before(_df):
    print('before ec_timbe_before, _df.shape', _df.shape)
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    n0 = len(_df)
    _df = _df.loc[(_df['index date'] <= datetime.datetime(2024, 12, 31, 0, 0))]
    print('after ec_timbe_before, _df.shape', _df.shape)
    n1 = len(_df)
    print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
    print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

    # print('exclude severe conditions in df_ec_start', (df_ec_start['PaxExclude-Count'] > 0).sum())
    return _df


def build_exposure_group_and_table1_less_4_print(exptype='all', debug=False):
    # %% Step 1. Load  Data
    start_time = time.time()
    np.random.seed(0)
    random.seed(0)

    args = parse_args()

    # in_file = 'recover25Q3_covid_pos_naltrexone-withexposure.csv'
    #
    # print('in read: ', in_file)
    # if debug:
    #     nrows = 200000
    # else:
    #     nrows = None
    # df = pd.read_csv(in_file,
    #                  dtype={'patid': str, 'site': str, 'zip': str},
    #                  parse_dates=['index date', 'dob',
    #                               'flag_delivery_date',
    #                               'flag_pregnancy_start_date',
    #                               'flag_pregnancy_end_date'
    #                               ],
    #                  nrows=nrows,
    #                  )
    # print('df.shape:', df.shape)
    # print('Read Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    #
    # """
    # Drug captures
    #     cnsldn_names = [
    #         'naltrexone', 'LDN_name',
    #         'adderall_combo', 'lisdexamfetamine', 'methylphenidate',
    #         'amphetamine', 'amphetamine_nocombo', 'dextroamphetamine', 'dextroamphetamine_nocombo', 'modafinil',
    #         'pitolisant', 'solriamfetol', 'armodafinil', 'atomoxetine', 'benzphetamine',
    #         'azstarys_combo', 'dexmethylphenidate', 'dexmethylphenidate_nocombo', 'diethylpropion', 'methamphetamine',
    #         'phendimetrazine', 'phentermine', 'caffeine', 'fenfluramine_delet', 'oxybate_delet',
    #         'doxapram_delet', 'guanfacine'
    #         ]
    # Drug in different time windows
    #     for x in cnsldn_names:
    #         print('cnsldn-treat-0-30@'+x, df['cnsldn-treat-0-30@' + x].sum())
    #         print('cnsldn-treat--30-30@'+x, df['cnsldn-treat--30-30@'+x].sum())
    #         print('cnsldn-treat-0-15@'+x, df['cnsldn-treat-0-15@'+x].sum())
    #         print('cnsldn-treat-0-5@'+x, df['cnsldn-treat-0-5@'+x].sum())
    #         print('cnsldn-treat-0-7@'+x, df['cnsldn-treat-0-7@'+x].sum())
    #         print('cnsldn-treat--15-15@'+x, df['cnsldn-treat--15-15@'+x].sum())
    #
    #         print('cnsldn-treat--30-0@'+x, df['cnsldn-treat--30-0@'+x].sum())
    #         print('cnsldn-treat--60-0@'+x, df['cnsldn-treat--60-0@'+x].sum())
    #         print('cnsldn-treat--90-0@'+x, df['cnsldn-treat--90-0@'+x].sum())
    #         print('cnsldn-treat--120-0@'+x, df['cnsldn-treat--120-0@'+x].sum())
    #         print('cnsldn-treat--180-0@'+x, df['cnsldn-treat--180-0@'+x].sum())
    #
    #         print('cnsldn-treat--120-120@'+x, df['cnsldn-treat--120-120@'+x].sum())
    #         print('cnsldn-treat--180-180@'+x, df['cnsldn-treat--180-180@'+x].sum())
    #
    #         print('cnsldn-treat--365-0@'+x, df['cnsldn-treat--365-0@'+x].sum())
    #         print('cnsldn-treat--1095-0@'+x, df['cnsldn-treat--1095-0@'+x].sum())
    #         print('cnsldn-treat-30-180@'+x, df['cnsldn-treat-30-180@'+x].sum())
    #         print('cnsldn-treat--1095-30@'+x, df['cnsldn-treat--1095-30@'+x].sum())
    #
    # """
    # # to define exposure strategies below:
    # if exptype == 'naltrexone-acuteIncident-0-30':
    #     df1 = df.loc[(df['cnsldn-treat-0-30@naltrexone'] >= 1) &
    #                  (df['cnsldn-treat--1095-0@naltrexone'] == 0), :]
    #
    #     df0 = df.loc[(df['cnsldn-treat--1095-30@naltrexone'] == 0), :]
    #
    #     case_label = 'naltrexone 0 to 30 Incident'
    #     ctrl_label = 'no naltrexone'
    #
    # else:
    #     raise ValueError
    #
    # print('exposre strategy, exptype:', exptype)
    #
    # n1 = len(df1)
    # n0 = len(df0)
    # print('n1', n1, 'n0', n0)
    # df1['treated'] = 1
    # df0['treated'] = 0
    #
    # df = pd.concat([df1, df0], ignore_index=True)
    # df = add_col(df)
    #
    # df_pos = df.loc[df['treated'] == 1, :]
    # df_neg = df.loc[df['treated'] == 0, :]

    # out_file_df = r'./naltrexone_output/Matrix-naltrexone-{}-25Q3.csv'.format(exptype)
    # out_file_df = r'./naltrexone_output/Matrix-naltrexone-{}-25Q3-naltrexCovAtDrugOnset.csv'.format(exptype)
    in_file_df = r'./naltrexone_output/Matrix-naltrexone-{}-25Q3-naltrexCovAtDrugOnset-Painsub.csv'.format(exptype)

    print('in read: ', in_file_df)
    if debug:
        nrows = 200000
    else:
        nrows = None

    df = pd.read_csv(in_file_df,
                     dtype={'patid': str, 'site': str, 'zip': str},
                     parse_dates=['index date', 'dob',
                                  'flag_delivery_date',
                                  'flag_pregnancy_start_date',
                                  'flag_pregnancy_end_date'
                                  ],
                     nrows=nrows,
                     )
    print('df.shape:', df.shape)
    print('Read Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    df = ec_anyenc_1yrbefore_baseline(df)
    df = ec_timbe_before(df)
    df = ec_no_cancer_baseline(df)
    df = ec_no_U099_baseline(df)
    df = ec_no_severe_conditions_4_pax(df)
    df = ec_no_pregnant_baseline(df)
    df = ec_no_HIV_baseline(df)

    # # # df = ec_pain_obesity_OUD_withmed_baseline(df)
    # # df = ec_pain_obesity_OUD_baseline(df)
    # # df = ec_pain_baseline(df)
    # #
    # # df = ec_painIncludeAndTbd_obesity_OUD_baseline(df)
    # # df = ec_painIncludeAndTbd_baseline(df)

    df = ec_painIncludeOnly_baseline(df)
    df_beforematch = add_col_2(df)

    out_file_df = r'./naltrexone_output/Matrix-naltrexone-{}-25Q3-naltrexCovAtDrugOnset-applyEC-PainIncludeOnly-kmatch-{}-v3{}.csv'.format(
        exptype, args.kmatch, '-replace' if args.replace else '')
    out_file = r'./naltrexone_output/Table-naltrexone-{}-25Q3-naltrexCovAtDrugOnset-applyEC-PainIncludeOnly-kmatch-{}-v3{}.xlsx'.format(
        exptype, args.kmatch, '-replace' if args.replace else '')

    print(out_file_df)
    print(out_file)
    case_label = 'naltrexone 0 to 30 Incident'
    ctrl_label = 'no naltrexone'
    df_pos_beforematch = df_beforematch.loc[df_beforematch['treated'] == 1, :]
    df_neg_beforematch = df_beforematch.loc[df_beforematch['treated'] == 0, :]

    df_pos, df_neg = build_matched_control(df_pos_beforematch, df_neg_beforematch, kmatch=args.kmatch, replace=args.replace)
    # utils.dump(df_neg_matched,
    #            r'./naltrexone_output/_selected_preg_cohort2-matched-k{}-useSelectdx{}-useacute{}V2.pkl'.format(
    #                args.kmatch, args.usedx, args.useacute))

    df = pd.concat([df_pos.reset_index(), df_neg.reset_index()], ignore_index= True)
    print('len(df)', len(df),
          'len(df_pos)', len(df_pos),
          'len(df_neg)', len(df_neg),
          )
    df.to_csv(out_file_df, index=False)

    output_columns = ['All', case_label, ctrl_label, 'SMD']

    print('treated df_pos.shape', df_pos.shape,
          'control df_neg.shape', df_neg.shape,
          'combined df.shape', df.shape, )

    # print('Dump file for updating covs at drug index ', out_file_df)
    # df.to_csv(out_file_df, index=False)
    # print('Dump file done ', out_file_df)

    # step 2: generate initial table!
    # return df, df
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

    # row_names.append('PASC — no. (%)')
    # records.append([])
    # col_names = ['any_pasc_flag', ]
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])

    row_names.append('Drugs — no. (%)')
    records.append([])

    drug_col = ['cnsldn-treat-0-30@adderall_combo', 'cnsldn-treat-0-30@lisdexamfetamine',
                'cnsldn-treat-0-30@methylphenidate', 'cnsldn-treat-0-30@dexmethylphenidate']
    drug_col_output = ['adderall_combo', 'lisdexamfetamine', 'methylphenidate', 'dexmethylphenidate']
    row_names.extend(drug_col_output)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in drug_col])

    # Sex
    row_names.append('Sex — no. (%)')
    records.append([])
    sex_col = ['Female', 'Male', 'Other/Missing']
    # sex_col = ['Female', 'Male']

    row_names.extend(sex_col)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in sex_col])

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

    age_col = ['age@18-24', 'age@25-34', 'age@35-49', 'age@50-64', 'age@65+']
    age_col_output = ['18-24', '25-34', '35-49', '50-64', '65+']
    row_names.extend(age_col_output)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in age_col])

    # Race
    row_names.append('Race — no. (%)')
    records.append([])
    # col_names = ['Asian', 'Black or African American', 'White', 'Other', 'Missing']
    col_names = ['RE:Asian Non-Hispanic', 'RE:Black or African American Non-Hispanic', 'RE:Hispanic or Latino Any Race',
                 'RE:White Non-Hispanic', 'RE:Other Non-Hispanic', 'RE:Unknown']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # # Ethnic group
    # row_names.append('Ethnic group — no. (%)')
    # records.append([])
    # col_names = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])

    row_names.append('Acute severity — no. (%)')
    records.append([])
    col_names = ['outpatient', 'inpatient', 'icu', 'inpatienticu']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # # follow-up
    # row_names.append('Follow-up days (IQR)')
    # records.append([
    #     _quantile_str(df['maxfollowup']),
    #     _quantile_str(df_pos['maxfollowup']),
    #     _quantile_str(df_neg['maxfollowup']),
    #     _smd(df_pos['maxfollowup'], df_neg['maxfollowup'])
    # ])
    #
    # row_names.append('T2 Death days (IQR)')
    # records.append([
    #     _quantile_str(df['death t2e']),
    #     _quantile_str(df_pos['death t2e']),
    #     _quantile_str(df_neg['death t2e']),
    #     _smd(df_pos['death t2e'], df_neg['death t2e'])
    # ])
    # col_names = ['death', 'death in acute', 'death post acute']
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in
    #      col_names])

    # # utilization
    # row_names.append('No. of hospital visits in the past 3 yr — no. (%)')
    # records.append([])
    # # part 1
    # col_names = ['No. of Visits:0', 'No. of Visits:1-3', 'No. of Visits:4-9', 'No. of Visits:10-19',
    #              'No. of Visits:>=20',
    #              'No. of hospitalizations:0', 'No. of hospitalizations:1', 'No. of hospitalizations:>=1']
    # col_names_out = col_names
    # row_names.extend(col_names_out)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in
    #      col_names])

    # utilization
    row_names.append('No. of hospital visits in the past 3 yr — no. (%)')
    records.append([])
    # part 1
    col_names = ['inpatient no.', 'outpatient no.', 'emergency visits no.', 'other visits no.']
    col_names_out = ['No. of Inpatient Visits', 'No. of Outpatient Visits',
                     'No. of Emergency Visits', 'No. of Other Visits']

    row_names.extend(col_names_out)
    records.extend(
        [[_quantile_str(df[c]), _quantile_str(df_pos[c]), _quantile_str(df_neg[c]), _smd(df_pos[c], df_neg[c])] for c in
         col_names])

    # part2
    df_pos['Inpatient >=3'] = df_pos['inpatient visits 3-4'] + df_pos['inpatient visits >=5']
    df_neg['Inpatient >=3'] = df_neg['inpatient visits 3-4'] + df_neg['inpatient visits >=5']
    df_pos['Outpatient >=3'] = df_pos['outpatient visits 3-4'] + df_pos['outpatient visits >=5']
    df_neg['Outpatient >=3'] = df_neg['outpatient visits 3-4'] + df_neg['outpatient visits >=5']
    df_pos['Emergency >=3'] = df_pos['emergency visits 3-4'] + df_pos['emergency visits >=5']
    df_neg['Emergency >=3'] = df_neg['emergency visits 3-4'] + df_neg['emergency visits >=5']

    df['Inpatient >=3'] = df['inpatient visits 3-4'] + df['inpatient visits >=5']
    df['Outpatient >=3'] = df['outpatient visits 3-4'] + df['outpatient visits >=5']
    df['Emergency >=3'] = df['emergency visits 3-4'] + df['emergency visits >=5']

    col_names = ['inpatient visits 0', 'inpatient visits 1-2', 'Inpatient >=3',
                 'outpatient visits 0', 'outpatient visits 1-2', 'Outpatient >=3',
                 'emergency visits 0', 'emergency visits 1-2', 'Emergency >=3']
    col_names_out = ['Inpatient 0', 'Inpatient 1-2', 'Inpatient >=3',
                     'Outpatient 0', 'Outpatient 1-2', 'Outpatient >=3',
                     'Emergency 0', 'Emergency 1-2', 'Emergency >=3']
    row_names.extend(col_names_out)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # ADI
    row_names.append('Median area deprivation index (IQR) — rank')
    records.append([
        _quantile_str(df['adi']),
        _quantile_str(df_pos['adi']),
        _quantile_str(df_neg['adi']),
        _smd(df_pos['adi'], df_neg['adi'])
    ])

    # col_names = ['ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49',
    #              'ADI50-59', 'ADI60-69', 'ADI70-79', 'ADI80-89', 'ADI90-100',
    #              'ADIMissing']
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])

    # BMI
    row_names.append('BMI (IQR)')
    records.append([
        _quantile_str(df['bmi']),
        _quantile_str(df_pos['bmi']),
        _quantile_str(df_neg['bmi']),
        _smd(df_pos['bmi'], df_neg['bmi'])
    ])

    col_names = ['BMI: <18.5 under weight', 'BMI: 18.5-<25 normal weight',
                 'BMI: 25-<30 overweight ', 'BMI: >=30 obese ', 'BMI: missing']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # Smoking:
    # col_names = ['Smoker: never', 'Smoker: current', 'Smoker: former', 'Smoker: missing']
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])

    # Vaccine:
    col_names = ['Fully vaccinated - Pre-index',
                 # 'Fully vaccinated - Post-index',
                 'Partially vaccinated - Pre-index',
                 # 'Partially vaccinated - Post-index',
                 'No evidence - Pre-index',
                 # 'No evidence - Post-index',
                 ]
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # time of index period
    row_names.append('Index time period of patients — no. (%)')
    records.append([])

    # part 1
    col_names = ['03/20-06/20', '07/20-10/20', '11/20-02/21',
                 '03/21-06/21', '07/21-10/21', '11/21-02/22',
                 '03/22-06/22', '07/22-10/22', '11/22-02/23',
                 '03/23-06/23', '07/23-10/23', '11/23-02/24',
                 '03/24-06/24', '07/24-10/24', '11/24-02/25',
                 ]
    # col_names = [
    #              '03/22-06/22', '07/22-10/22', '11/22-02/23',
    #              ]
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # Coexisting coditions
    row_names.append('Coexisting conditions — no. (%)')

    records.append([])
    col_names = (
            ['dxcovNaltrexone-base@MECFS', 'dxcovNaltrexone-base@Pain', 'dxcovNaltrexone-base@substance use disorder ',
             'dxcovNaltrexone-base@opioid use disorder', 'dxcovNaltrexone-base@Opioid induced constipation',
             'dxcovNaltrexone-base@Obesity', 'dxcovNaltrexone-base@Crohn-Inflamm_Bowel',
             'dxcovNaltrexone-base@fibromyalgia', 'dxcovNaltrexone-base@multiple sclerosis',
             'dxcovNaltrexone-base@POTS',
             'covNaltrexone_med-base@NSAIDs_combined', 'covNaltrexone_med-base@opioid drug',
             'covNaltrexone_med-basedrugonset@NSAIDs_combined',
             'covNaltrexone_med-basedrugonset@opioid drug', ] +
            ['dxcovNaltrexone-basedrugonset@MECFS', 'dxcovNaltrexone-basedrugonset@Pain',
             'dxcovNaltrexone-basedrugonset@substance use disorder ',
             'dxcovNaltrexone-basedrugonset@opioid use disorder',
             'dxcovNaltrexone-basedrugonset@Opioid induced constipation',
             'dxcovNaltrexone-basedrugonset@Obesity',
             'dxcovNaltrexone-basedrugonset@Crohn-Inflamm_Bowel',
             'dxcovNaltrexone-basedrugonset@fibromyalgia',
             'dxcovNaltrexone-basedrugonset@multiple sclerosis',
             'dxcovNaltrexone-basedrugonset@POTS'] +
            ['dxcovCNSLDN-base@ADHD', 'ADHD_before_drug_onset', 'dxcovCNSLDN-base@Narcolepsy', 'dxcovCNSLDN-base@MECFS',
             'dxcovCNSLDN-base@Pain',
             'dxcovCNSLDN-base@alcohol opioid other substance '] +
            ['PaxRisk:Cancer', 'PaxRisk:Chronic kidney disease', 'PaxRisk:Chronic liver disease',
             'PaxRisk:Chronic lung disease', 'PaxRisk:Cystic fibrosis',
             'PaxRisk:Dementia or other neurological conditions', 'PaxRisk:Diabetes', 'PaxRisk:Disabilities',
             'PaxRisk:Heart conditions', 'PaxRisk:Hypertension', 'PaxRisk:HIV infection',
             'PaxRisk:Immunocompromised condition or weakened immune system', 'PaxRisk:Mental health conditions',
             'PaxRisk:Overweight and obesity', 'PaxRisk:Obesity', 'PaxRisk:Pregnancy',
             'PaxRisk:Sickle cell disease or thalassemia',
             'PaxRisk:Smoking current', 'PaxRisk:Stroke or cerebrovascular disease',
             'PaxRisk:Substance use disorders', 'PaxRisk:Tuberculosis'] +
            ["DX: Coagulopathy", "DX: Peripheral vascular disorders ", "DX: Seizure/Epilepsy", "DX: Weight Loss",
             'DX: Obstructive sleep apnea', 'DX: Epstein-Barr and Infectious Mononucleosis (Mono)', 'DX: Herpes Zoster',
             'mental-base@Schizophrenia Spectrum and Other Psychotic Disorders',
             'mental-base@Depressive Disorders',
             'mental-base@Bipolar and Related Disorders',
             'mental-base@Anxiety Disorders',
             'mental-base@Obsessive-Compulsive and Related Disorders',
             'mental-base@Post-traumatic stress disorder',
             'mental-base@Bulimia nervosa',
             'mental-base@Binge eating disorder',
             'mental-base@premature ejaculation',
             'mental-base@Autism spectrum disorder',
             'mental-base@Premenstrual dysphoric disorder',
             'mental-base@SMI',
             'mental-base@non-SMI', ] + ['cnsldn-treat--1095-0@' + x for x in [
        'naltrexone', 'LDN_name', 'adderall_combo', 'lisdexamfetamine', 'methylphenidate',
        'amphetamine', 'amphetamine_nocombo', 'dextroamphetamine', 'dextroamphetamine_nocombo', 'modafinil',
        'pitolisant', 'solriamfetol', 'armodafinil', 'atomoxetine', 'benzphetamine',
        'azstarys_combo', 'dexmethylphenidate', 'dexmethylphenidate_nocombo', 'diethylpropion', 'methamphetamine',
        'phendimetrazine', 'phentermine', 'caffeine', 'fenfluramine_delet', 'oxybate_delet',
        'doxapram_delet', 'guanfacine']]
    )

    col_names_out = (['MECFS', 'Pain', 'substance use disorder ', 'opioid use disorder', 'Opioid induced constipation',
                      'Obesity:just code', 'Crohn-Inflamm_Bowel', 'fibromyalgia', 'multiple sclerosis', 'POTS',
                      'NSAIDs_combined@base', 'opioid drug@base',
                      'NSAIDs_combined@basedrugonset',
                      'opioid drug@basedrugonset'
                      ] + ['MECFS@drugonset', 'Pain@drugonset', 'substance use disorder @drugonset',
                           'opioid use disorder@drugonset', 'Opioid induced constipation@drugonset',
                           'Obesity@drugonset', 'Crohn-Inflamm_Bowel@drugonset', 'fibromyalgia@drugonset',
                           'multiple sclerosis@drugonset', 'POTS@drugonset']
                     + ['ADHD', 'ADHD_before_drug_onset', 'Narcolepsy', 'MECFS', 'Pain',
                        'alcohol opioid other substance'] + ['Cancer', 'Chronic kidney disease',
                                                             'Chronic liver disease',
                                                             'Chronic lung disease', 'Cystic fibrosis',
                                                             'Dementia or other neurological conditions', 'Diabetes',
                                                             'Disabilities',
                                                             'Heart conditions', 'Hypertension', 'HIV infection',
                                                             'Immunocompromised condition or weakened immune system',
                                                             'Mental health conditions',
                                                             'Overweight and obesity', 'Obesity', 'Pregnancy',
                                                             'Sickle cell disease or thalassemia',
                                                             'Smoking current or former',
                                                             'Stroke or cerebrovascular disease',
                                                             'Substance use disorders', 'Tuberculosis', ] +
                     ["Coagulopathy", "Peripheral vascular disorders ", "Seizure/Epilepsy", "Weight Loss",
                      'Obstructive sleep apnea', 'Epstein-Barr and Infectious Mononucleosis (Mono)', 'Herpes Zoster',
                      'Schizophrenia Spectrum and Other Psychotic Disorders',
                      'Depressive Disorders',
                      'Bipolar and Related Disorders',
                      'Anxiety Disorders',
                      'Obsessive-Compulsive and Related Disorders',
                      'Post-traumatic stress disorder',
                      'Bulimia nervosa',
                      'Binge eating disorder',
                      'premature ejaculation',
                      'Autism spectrum disorder',
                      'Premenstrual dysphoric disorder',
                      'SMI',
                      'non-SMI', ] + [
                         'naltrexone', 'LDN_name', 'adderall_combo', 'lisdexamfetamine', 'methylphenidate',
                         'amphetamine', 'amphetamine_nocombo', 'dextroamphetamine', 'dextroamphetamine_nocombo',
                         'modafinil',
                         'pitolisant', 'solriamfetol', 'armodafinil', 'atomoxetine', 'benzphetamine',
                         'azstarys_combo', 'dexmethylphenidate', 'dexmethylphenidate_nocombo', 'diethylpropion',
                         'methamphetamine',
                         'phendimetrazine', 'phentermine', 'caffeine', 'fenfluramine_delet', 'oxybate_delet',
                         'doxapram_delet', 'guanfacine']
                     )

    row_names.extend(col_names_out)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # col_names = ['score_cci_charlson', 'score_cci_quan']
    # col_names_out = ['score_cci_charlson', 'score_cci_quan']
    # row_names.extend(col_names_out)
    # records.extend(
    #     [[_quantile_str(df[c]), _quantile_str(df_pos[c]), _quantile_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])
    # row_names.append('CCI Score — no. (%)')
    # records.append([])

    col_names = ['cci_quan:0', 'cci_quan:1-2', 'cci_quan:3+', 'cci_quan:3-4', 'cci_quan:5-10', 'cci_quan:11+']
    col_names_out = ['cci_quan:0', 'cci_quan:1-2', 'cci_quan:3+', 'cci_quan:3-4', 'cci_quan:5-10', 'cci_quan:11+']
    row_names.extend(col_names_out)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    df_out = pd.DataFrame(records, columns=output_columns, index=row_names)
    df_out['SMD'] = df_out['SMD'].astype(float)

    df_out.to_excel(out_file)
    print('Dump done ', df_out)

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return df, df_out


if __name__ == '__main__':
    # python screen_build_naltrexone_treat_table_playwithEC_match.py  --replace --kmatch 10 2>&1 | tee  log/screen_build_naltrexone_treat_table_playwithEC_match-k10-v3-replace.txt
    # python screen_build_naltrexone_treat_table_playwithEC_match.py  --replace --kmatch 5 2>&1 | tee  log/screen_build_naltrexone_treat_table_playwithEC_match-k5-v3-replace.txt
    # python screen_build_naltrexone_treat_table_playwithEC_match.py  --replace --kmatch 15 2>&1 | tee  log/screen_build_naltrexone_treat_table_playwithEC_match-k15-v3-replace.txt
    # python screen_build_naltrexone_treat_table_playwithEC_match.py  --replace --kmatch 3 2>&1 | tee  log/screen_build_naltrexone_treat_table_playwithEC_match-k3-v3-replace.txt
    # python screen_build_naltrexone_treat_table_playwithEC_match.py  --replace --kmatch 1 2>&1 | tee  log/screen_build_naltrexone_treat_table_playwithEC_match-k1-v3-replace.txt

    #  timeout /t 21600;
    # timeout /t 28800; python screen_build_naltrexone_treat_table_playwithEC_match.py  --kmatch 1 2>&1 | tee  log/screen_build_naltrexone_treat_table_playwithEC_match-k1-v2.txt
    # timeout /t 28800; python screen_build_naltrexone_treat_table_playwithEC_match.py  --kmatch 2 2>&1 | tee  log/screen_build_naltrexone_treat_table_playwithEC_match-k2-v2.txt

    start_time = time.time()

    # 2025-07-15
    df, df_out = build_exposure_group_and_table1_less_4_print(exptype='naltrexone-acuteIncident-0-30',
                                                              debug=False)  # 'ssri-base180-acutevsnot'

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
