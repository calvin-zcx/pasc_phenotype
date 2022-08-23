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


def hn(n):
    # H(N) = 1+1/2+1/3+...+1/N
    a = 0
    for i in range(1, n + 1):
        a += 1. / i
    return a


def test():
    df = pd.read_csv('log/multiple_comparision_example.csv')

    fdr_threshold = 0.05
    vbool_bonf, vp_bonf, _, threshold_bon = smsmlt.multipletests(df['P value'], alpha=fdr_threshold,
                                                                 method='bonferroni')
    vbool_bh, vp_bh = smsmlt.fdrcorrection(df['P value'], alpha=fdr_threshold)
    vbool_by, vp_by = smsmlt.fdrcorrection(df['P value'], method="negcorr", alpha=fdr_threshold)
    # vbool_m2 = fdr.lsu(df['P value'], q=fdr_threshold) # the same to bh method, when using <= threshold
    vbool_storey, vq_storey = fdr.qvalue(df['P value'], threshold=fdr_threshold)
    # print((vbool == vbool_m2).mean())
    # print((vp - vq).mean())


def multiple_test_correct(pv_list, fdr_threshold=0.05):
    vbool_bonf, vp_bonf, _, threshold_bon = smsmlt.multipletests(pv_list, alpha=fdr_threshold, method='bonferroni')
    vbool_bh, vp_bh = smsmlt.fdrcorrection(pv_list, alpha=fdr_threshold)
    vbool_by, vp_by = smsmlt.fdrcorrection(pv_list, method="negcorr", alpha=fdr_threshold)
    vbool_storey, vq_storey = fdr.qvalue(pv_list, threshold=fdr_threshold)

    df = pd.DataFrame({'p-value':pv_list,
                       'bool_bonf': vbool_bonf, 'p-bonf':vp_bonf,
                       'bool_bh': vbool_bh, 'p-bh': vp_bh,
                       'bool_by': vbool_by, 'p-by': vp_by,
                       'bool_storey': vbool_storey, 'q-storey': vq_storey,
                       })

    return df


if __name__ == '__main__':
    df = pd.read_excel(
        r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3-multitest.xlsx',
        sheet_name='diagnosis')

    df_select = df.sort_values(by='hr-w', ascending=False)
    df_select = df_select.loc[df_select['hr-w-p'].notna(), :]
    df_p = multiple_test_correct(df_select['hr-w-p'], fdr_threshold=0.05)

    # pvalue = 0.01  # 0.05 / 137
    # df_select = df_select.loc[df_select['hr-w-p'] <= pvalue, :]  #
    # df_select = df_select.loc[df_select['hr-w'] > 1, :]
    # df_select = df_select.loc[df_select['no. pasc in +'] >= 100, :]
    print('df_select.shape:', df_select.shape)
