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
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
from datetime import datetime
import functools
import seaborn as sns

print = functools.partial(print, flush=True)
from misc import utils
from lifelines import KaplanMeierFitter, CoxPHFitter, AalenJohansenFitter
from lifelines.statistics import survival_difference_at_fixed_point_in_time_test, proportional_hazard_test, logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import k_fold_cross_validation
from PRModels import ml
import matplotlib.pyplot as plt


def add_pasc_encounter_info():
    df = pd.read_excel(r'C:\Users\zangc\Documents\Boston\workshop\2021-PASC\prediction\every_pasc_and_LR_AUC.xlsx')
    df_aux = pd.read_excel(
        r'C:\Users\zangc\PycharmProjects\pasc\data\V15_COVID19\output\character\pasc_diagnosis_encounter_type_cohorts_covid_4manuNegNoCovidV2_ALL-posOnly.xlsx')
    df = pd.merge(df, df_aux, left_on='pasc', right_on='pasc', how='left')
    df.to_csv(r'C:\Users\zangc\Documents\Boston\workshop\2021-PASC\prediction\every_pasc_and_LR_AUC_helper.csv')


if __name__ == '__main__':
    df_pasc_info = pd.read_excel('output/causal_effects_specific_withMedication_v3.xlsx', sheet_name='diagnosis')
    selected_pasc_list = df_pasc_info.loc[df_pasc_info['selected'] == 1, 'pasc']
    print('len(selected_pasc_list)', len(selected_pasc_list))
    print(selected_pasc_list)
    selected_organ_list = df_pasc_info.loc[df_pasc_info['selected'] == 1, 'Organ Domain'].unique()
    print('len(selected_organ_list)', len(selected_organ_list))

    dataset = 'OneFlorida'
    results = []
    for organ in selected_organ_list:
        infile = r'output/factors/{}/{}/every_organ/ORGAN-{}-modeSelection-{}-{}.csv'.format(
            dataset, 'elix', organ, dataset, 'all')
        df = pd.read_csv(infile)
        df['organ'] = organ
        results.append(df.loc[0, :])

    df_sum = pd.DataFrame(results)
    df_sum.to_csv(r'output/factors/{}/{}/every_organ/ORGAN-summary-{}.csv'.format(
        dataset, 'elix', dataset))
    print('Dump done', organ)

    results = []
    for pasc in selected_pasc_list:
        pasc = pasc.replace('/', '_')
        infile = r'output/factors/{}/{}/every_pasc/PASC-{}-modeSelection-{}-{}.csv'.format(
            dataset, 'elix', pasc, dataset, 'all')
        infile2 = r'output/factors/{}/{}/every_pasc/PASC-{}-riskFactor-{}-{}.csv'.format(
            dataset, 'elix', pasc, dataset, 'all')
        df = pd.read_csv(infile)
        df_risk = pd.read_csv(infile2)
        cov = df_risk.loc[(df_risk['p-Value'] < 0.05) & (df_risk['HR'] > 1), 'covariate']
        hr = df_risk.loc[(df_risk['p-Value'] < 0.05) & (df_risk['HR'] > 1), 'HR']
        df['pasc'] = pasc.replace('_', '/')
        df['risk'] = ';'.join(cov)
        df['hr'] = ';'.join(['{:.2f}'.format(x) for x in hr])
        results.append(df.loc[0, :])

    df_sum = pd.DataFrame(results).reset_index()
    df_sum.to_csv(r'output/factors/{}/{}/every_pasc/pasc-summary-{}.csv'.format(
        dataset, 'elix', dataset))
    print('Dump done')
