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

    df = pd.DataFrame({'p-value': pv_list,
                       'bool_bonf': vbool_bonf.astype(int), 'p-bonf': vp_bonf,
                       'bool_by': vbool_by.astype(int), 'p-by': vp_by,
                       'bool_bh': vbool_bh.astype(int), 'p-bh': vp_bh,
                       'bool_storey': vbool_storey.astype(int), 'q-storey': vq_storey,
                       })

    return df


def add_test_to_old_results():
    # DX only
    # df = pd.read_excel(
    #     r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3-multitest.xlsx',
    #     sheet_name='diagnosis')
    #
    # df_select = df.sort_values(by='hr-w', ascending=False)
    # df_select = df_select.loc[df_select['hr-w-p'].notna(), :]
    # df_p = multiple_test_correct(df_select['hr-w-p'], fdr_threshold=0.05)
    #
    # dfm = pd.merge(df, df_p, how='left', left_index=True, right_index=True)
    # dfm.to_excel( r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3-multitest-withMultiPval.xlsx')
    # print('df_select.shape:', df_select.shape)

    # DX + Med
    # ../data/oneflorida/output/character/outcome/DX-all/causal_effects_specific_v3-multitest.xlsx
    # ../data/oneflorida/output/character/outcome/DX-all/causal_effects_specific_v3-multitest.xlsx

    DATA = "FL"
    ## Insight
    if DATA == 'INSIGHT':
        print('Data', DATA)
        df = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3-multitest.xlsx',
            sheet_name='diagnosis')

        df_med = pd.read_excel(
            '../data/V15_COVID19/output/character/outcome/MED/causal_effects_specific_Medication-withName-simplev2-multitest.xlsx',
            sheet_name='causal_effects_specific_med-sna'
        )
    elif DATA == 'FL':
        print('Data', DATA)
        df = pd.read_excel(
            r'../data/oneflorida/output/character/outcome/DX-all/causal_effects_specific_v3-multitest.xlsx',
            sheet_name='diagnosis')

        df_med = pd.read_excel(
            '../data/oneflorida/output/character/outcome/MED-all/causal_effects_specific_med-multitest.xlsx',
            sheet_name='Sheet1'
        )

        df_med_info = pd.read_excel(
            '../data/V15_COVID19/output/character/outcome/MED/causal_effects_specific_Medication-withName-simplev2-multitest.xlsx',
            sheet_name='causal_effects_specific_med-sna'
        )

    df_select = df.loc[df['hr-w-p'].notna(), :]
    df_med_select = df_med.loc[df_med['hr-w-p'].notna(), :]

    p_all = pd.concat([df_select['hr-w-p'], df_med_select['hr-w-p']])
    df_p = multiple_test_correct(p_all, fdr_threshold=0.05)

    df_p_dx = df_p.iloc[:len(df_select['hr-w-p']), :]
    df_p_med = df_p.iloc[len(df_select['hr-w-p']):, :]

    print('p_all.shape', p_all.shape, 'df_p.shape', df_p.shape)
    print('df_p_dx.shape', df_p_dx.shape, 'df_p_med.shape', df_p_med.shape)

    dfm_dx = pd.merge(df, df_p_dx, how='left', left_index=True, right_index=True)
    dfm_med = pd.merge(df_med, df_p_med, how='left', left_index=True, right_index=True)

    if DATA == 'INSIGHT':
        print('Data', DATA)
        dfm_dx.to_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3-multitest-withMultiPval-DXMEDALL.xlsx')
        dfm_med.to_excel(
            r'../data/V15_COVID19/output/character/outcome/MED/causal_effects_specific_Medication-withName-simplev2-multitest-withMultiPval-DXMEDALL.xlsx')
    elif DATA == 'FL':
        print('Data', DATA)
        dfm_med = pd.merge(dfm_med, df_med_info[['pasc', 'name', 'atc-l3', 'atc-l4']],
                           how='left', left_on='pasc', right_on='pasc')

        # dfm_dx.to_excel(
        #     r'../data/oneflorida/output/character/outcome/DX-all/causal_effects_specific_v3-multitest-withMultiPval-DXMEDALL.xlsx')
        dfm_med.to_excel(
            r'../data/oneflorida/output/character/outcome/MED-all/causal_effects_specific_med-multitest-withMultiPval-DXMEDALL.xlsx')

    print('Done')


def add_test_to_paper_2023_2_23():
    DATA = "FL"

    if DATA == 'INSIGHT':
        print('Data', DATA)
        df = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all-new-trim-vaccine/causal_effects_specific_dx_insight.xlsx',
            sheet_name='dx')

        df_med = pd.read_excel(
            '../data/V15_COVID19/output/character/outcome/MED-all-new-trim-vaccine/causal_effects_specific_med_insight.xlsx',
            sheet_name='med'
        )
    elif DATA == 'FL':
        print('Data', DATA)
        df = pd.read_excel(
            r'../data/oneflorida/output/character/outcome/DX-all-new-trim-vaccine/causal_effects_specific_dx_oneflorida.xlsx',
            sheet_name='dx')

        df_med = pd.read_excel(
            '../data/oneflorida/output/character/outcome/MED-all-new-trim-vaccine/causal_effects_specific_med_oneflorida.xlsx',
            sheet_name='med'
        )

    df_select = df.loc[df['hr-w-p'].notna(), :]
    df_med_select = df_med.loc[df_med['hr-w-p'].notna(), :]

    p_all = pd.concat([df_select['hr-w-p'], df_med_select['hr-w-p']])
    df_p = multiple_test_correct(p_all, fdr_threshold=0.05)

    df_p_dx = df_p.iloc[:len(df_select['hr-w-p']), :]
    df_p_med = df_p.iloc[len(df_select['hr-w-p']):, :]

    print('p_all.shape', p_all.shape, 'df_p.shape', df_p.shape)
    print('df_p_dx.shape', df_p_dx.shape, 'df_p_med.shape', df_p_med.shape)

    dfm_dx = pd.merge(df, df_p_dx, how='left', left_index=True, right_index=True)
    dfm_med = pd.merge(df_med, df_p_med, how='left', left_index=True, right_index=True)

    if DATA == 'INSIGHT':
        print('Data', DATA)
        dfm_dx.to_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all-new-trim-vaccine/causal_effects_specific_dx_insight-MultiPval-DXMEDALL.xlsx',
            sheet_name='dx')
        dfm_med.to_excel(
            r'../data/V15_COVID19/output/character/outcome/MED-all-new-trim-vaccine/causal_effects_specific_med_insight-MultiPval-DXMEDALL.xlsx',
            sheet_name='med')
    elif DATA == 'FL':
        print('Data', DATA)
        dfm_dx.to_excel(
            r'../data/oneflorida/output/character/outcome/DX-all-new-trim-vaccine/causal_effects_specific_dx_oneflorida-MultiPval-DXMEDALL.xlsx',
            sheet_name='dx')
        dfm_med.to_excel(
            r'../data/oneflorida/output/character/outcome/MED-all-new-trim-vaccine/causal_effects_specific_med_oneflorida-MultiPval-DXMEDALL.xlsx',
            sheet_name='med')

    print('Done')


def old_multi():
    # df = pd.read_excel(
    #     r'../data/recover/output/results/DX-all-downsample0.33/causal_effects_specific_downsample0.33.xlsx',
    #     sheet_name='dx')

    # df = pd.read_excel(
    #     r'../data/recover/output/results/DX-preg-pos-neg/causal_effects_specific_preg_pos_neg.xlsx',
    #     sheet_name='dx')

    # df = pd.read_excel(
    #     r'../data/recover/output/results/DX-all-downsample0.50/causal_effects_specific_dx-all-downsample0.5.xlsx',
    #     sheet_name='dx')

    # df = pd.read_excel(
    #     r'../data/recover/output/results/DX-pospreg-posnonpreg/causal_effects_specific-pospreg-posnonpreg.xlsx',
    #     sheet_name='dx')

    # infile = r'../data/recover/output/results/DX-all-neg1.0/causal_effects_specific-all-neg1.xlsx'
    # infile = r'../data/recover/output/results/DX-deltaAndBefore-neg1.0/causal_effects_specific-deltaAndBefore-neg1.xlsx'
    # infile = r'../data/recover/output/results/DX-omicron-neg1.0/causal_effects_specific-omicron-neg1.xlsx'
    # infile = r'../data/recover/output/results/DX-pospreg-posnonpreg/causal_effects_specific-pospreg-posnonpreg.xlsx'
    infile = r'../data/recover/output/results/DX-preg-pos-neg/causal_effects_specific-preg-pos-neg.xlsx'

    outfile = infile.replace('.xlsx', '_aux_correctPvalue.xlsx')

    df = pd.read_excel(infile, sheet_name='dx')

    df_select = df.loc[df['hr-w-p'].notna(), :]
    p_all = df_select['hr-w-p']  # pd.concat([df_select['hr-w-p'], df_med_select['hr-w-p']])
    df_p = multiple_test_correct(p_all, fdr_threshold=0.05)

    # df_p_dx = df_p.iloc[:len(df_select['hr-w-p']), :]
    # df_p_med = df_p.iloc[len(df_select['hr-w-p']):, :]

    print('p_all.shape', p_all.shape, 'df_p.shape', df_p.shape)
    # print('df_p_dx.shape', df_p_dx.shape, 'df_p_med.shape', df_p_med.shape)

    dfm_dx = pd.merge(df, df_p, how='left', left_index=True, right_index=True)
    # dfm_med = pd.merge(df_med, df_p_med, how='left', left_index=True, right_index=True)

    # dfm_dx.to_excel(
    #     r'../data/recover/output/results/DX-all-downsample0.33/causal_effects_specific_downsample0.33_aux_correctPvalue.xlsx',
    #     sheet_name='dx')

    # dfm_dx.to_excel(
    #     r'../data/recover/output/results/DX-preg-pos-neg/causal_effects_specific_preg_pos_neg_aux_correctPvalue.xlsx',
    #     sheet_name='dx')

    # dfm_dx.to_excel(
    #     r'../data/recover/output/results/DX-all-downsample0.50/causal_effects_specific_dx-all-downsample0.5_aux_correctPvalue.xlsx',
    #     sheet_name='dx')

    # dfm_dx.to_excel(
    #     r'../data/recover/output/results/DX-inpatienticu/causal_effects_specific-inpatienticu_aux_correctPvalue.xlsx',
    #     sheet_name='dx')

    # dfm_dx.to_excel(
    #     r'../data/recover/output/results/DX-pospreg-posnonpreg/causal_effects_specific-pospreg-posnonpreg_aux_correctPvalue.xlsx',
    #     sheet_name='dx')

    dfm_dx.to_excel(outfile, sheet_name='dx')
    print('Done')


if __name__ == '__main__':
    start_time = time.time()

    infile = r'../data/recover/output/results/DX-omicroninpatienticu-neg1.0/causal_effects_specific-omicroninpatienticu.csv'
    infile = r'../data/recover/output/results/DX-omicronoutpatient-neg1.0/causal_effects_specific-omicronoutpatient.csv'
    infile = r'../data/recover/output/results/DX-deltaAndBeforeoutpatient-neg1.0/causal_effects_specific-deltaAndBeforeoutpatient.csv'
    infile = r'../data/recover/output/results/DX-deltaAndBeforeinpatienticu-neg1.0/causal_effects_specific-deltaAndBeforeinpatienticu.csv'

    outfile = infile.replace('.csv', '_aux_correctPvalue.xlsx')

    # df = pd.read_excel(infile, sheet_name='dx')
    df = pd.read_csv(infile)

    df_select = df.loc[df['hr-w-p'].notna(), :]
    p_all = df_select['hr-w-p']  # pd.concat([df_select['hr-w-p'], df_med_select['hr-w-p']])
    df_p = multiple_test_correct(p_all, fdr_threshold=0.05)

    # df_p_dx = df_p.iloc[:len(df_select['hr-w-p']), :]
    # df_p_med = df_p.iloc[len(df_select['hr-w-p']):, :]

    print('p_all.shape', p_all.shape, 'df_p.shape', df_p.shape)
    # print('df_p_dx.shape', df_p_dx.shape, 'df_p_med.shape', df_p_med.shape)

    dfm_dx = pd.merge(df, df_p, how='left', left_index=True, right_index=True)

    dfm_dx.to_excel(outfile, sheet_name='dx')
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))