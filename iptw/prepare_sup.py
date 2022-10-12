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

print = functools.partial(print, flush=True)
import time

if __name__ == '__main__':
    database = 'V15_COVID19'
    type = 'med'
    if type == 'dx':
        if database == 'V15_COVID19':
            df = pd.read_excel(
                r'../data/V15_COVID19/output/character/outcome/DX-all-new-trim/causal_effects_specific_dx_insight-MultiPval-DXMEDALL.xlsx',
                sheet_name='dx')
            outfile = r'../data/V15_COVID19/output/character/outcome/DX-all-new-trim/_causal_effects_specific_dx_insight-MultiPval-DXMEDALL-OnlyCIF180days.xlsx'

        elif database == 'oneflorida':
            df = pd.read_excel(
                r'../data/oneflorida/output/character/outcome/DX-all-new-trim/causal_effects_specific_dx_oneflorida-MultiPval-DXMEDALL.xlsx',
                sheet_name='dx')
            outfile = r'../data/oneflorida/output/character/outcome/DX-all-new-trim/_causal_effects_specific_dx_oneflorida-MultiPval-DXMEDALL-OnlyCIF180days.xlsx'

    elif type == 'med':
        if database == 'oneflorida':
            df = pd.read_excel(
                r'../data/oneflorida/output/character/outcome/MED-all-new-trim/causal_effects_specific_med_oneflorida-MultiPval-DXMEDALL.xlsx',
                sheet_name='med')
            outfile = r'../data/oneflorida/output/character/outcome/MED-all-new-trim/_causal_effects_specific_med_oneflorida-MultiPval-DXMEDALL-OnlyCIF180days.xlsx'
        elif database == 'V15_COVID19':
            df = pd.read_excel(
                r'../data/V15_COVID19/output/character/outcome/MED-all-new-trim/causal_effects_specific_med_insight-MultiPval-DXMEDALL.xlsx',
                sheet_name='med')
            outfile = r'../data/V15_COVID19/output/character/outcome/MED-all-new-trim/_causal_effects_specific_med_insight-MultiPval-DXMEDALL-OnlyCIF180days.xlsx'

    selec_cols = [x for x in df.columns if x.startswith('cif')]

    for col in selec_cols:
        df[col] = df[col].apply(lambda x: utils.stringlist_2_list(x)[-1] * 1000)

    df['cif_1_CI'] = np.nan
    df['cif_0_CI'] = np.nan
    df['cif_1_w_CI'] = np.nan
    df['cif_0_w_CI'] = np.nan

    for key, row in df.iterrows():
        df.loc[key, 'cif_1_CI'] = '[{:.2f}, {:.2f}]'.format(row['cif_1_CILower'], row['cif_1_CIUpper'])
        df.loc[key, 'cif_0_CI'] = '[{:.2f}, {:.2f}]'.format(row['cif_0_CILower'], row['cif_0_CIUpper'])
        df.loc[key, 'cif_1_w_CI'] = '[{:.2f}, {:.2f}]'.format(row['cif_1_w_CILower'], row['cif_1_w_CIUpper'])
        df.loc[key, 'cif_0_w_CI'] = '[{:.2f}, {:.2f}]'.format(row['cif_0_w_CILower'], row['cif_0_w_CIUpper'])

        if pd.notna(row['hr-w-CI']):
            hrwci = utils.stringlist_2_list(row['hr-w-CI'])
            df.loc[key, 'hr-w-CI'] = '[{:.2f}, {:.2f}]'.format(hrwci[0], hrwci[1])

    df.to_excel(outfile)
