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
import statsmodels.stats.multitest as smsmlt
import multipy.fdr as fdr

print = functools.partial(print, flush=True)
import time


def old_version():
    # 2023-3-23
    infile = r'../data/recover/output/results/DX-all-neg1.0/causal_effects_specific-all-neg1_compare_with_others.xlsx'
    outfile = infile.replace('.xlsx', '_aux_sum_selection.xlsx')
    df = pd.read_excel(infile, sheet_name='dx')

    apx_vec = ['', ' deltaAndBefore', ' omicron', ' inpatienticu']

    for apx in apx_vec:
        hr = df['hr-w' + apx]
        bf = df['bool_bonf' + apx]
        by = df['bool_by' + apx]
        top_bf = (((hr) > 1) & (bf == 1)).astype('int')
        top_by = (((hr) > 1) & (by == 1)).astype('int')
        df['risk_bf' + apx] = top_bf
        df['risk_by' + apx] = top_by

    df['risk_bf_sum'] = df[['risk_bf' + apx for apx in apx_vec]].sum(axis=1)
    df['risk_by_sum'] = df[['risk_by' + apx for apx in apx_vec]].sum(axis=1)

    df['risk_bf_sum+narrow'] = df['risk_bf_sum'] + df['selected_narrow_25'].fillna(0)
    df['risk_bf_sum+broad'] = df['risk_bf_sum'] + df['selected_broad44'].fillna(0)
    df['risk_by_sum+narrow'] = df['risk_by_sum'] + df['selected_narrow_25'].fillna(0)
    df['risk_by_sum+broad'] = df['risk_by_sum'] + df['selected_broad44'].fillna(0)

    df.to_excel(outfile, sheet_name='dx')

    print('Done')


if __name__ == '__main__':

    apx_vec = ['deltaAndBeforeoutpatient', 'deltaAndBeforeinpatienticu', 'omicronoutpatient', 'omicroninpatienticu', ]

    for apx in apx_vec:
        infile = r'../data/recover/output/results/DX-{}-neg1.0/causal_effects_specific-{}.xlsx'.format(apx, apx)
        outfile = infile.replace('.xlsx', '_add_selection.xlsx')
        df = pd.read_excel(infile, sheet_name='dx')

        hr = df['hr-w']
        bf = df['bool_bonf']
        by = df['bool_by' ]
        top_bf = (((hr) > 1) & (bf == 1)).astype('int')
        top_by = (((hr) > 1) & (by == 1)).astype('int')
        df['risk_bf' + apx] = top_bf
        df['risk_by' + apx] = top_by
        df.to_excel(outfile, sheet_name='dx')
        print('Done')

    print('Done')
