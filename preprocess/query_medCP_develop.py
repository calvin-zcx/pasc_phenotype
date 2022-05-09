import sys

# for linux env.
sys.path.insert(0, '..')
import time
import pickle
import argparse
import os
import random
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from misc import utils
import itertools
import functools
from tqdm import tqdm
import requests

print = functools.partial(print, flush=True)


def add_encounter_type():
    dir_name = r'C:/Users/zangc/Documents/Boston/workshop/2021-PASC/query_specific/'
    # med_file = 'Neurologic_Medications_ASN_v0.xlsx'
    med_file = 'Pulmonary_Medications_es_v0.xlsx'

    df_enc = pd.read_csv(dir_name + 'med_encounter_zilong.csv')
    df_med = pd.read_excel(dir_name + med_file)
    df = pd.merge(df_med, df_enc, left_on='rxnorm', right_on='rxnorm', how='left')

    df.to_csv(dir_name + med_file + '-aux.csv', index=False)


def api_rxnorm_from_string(name):
    r = requests.get('https://rxnav.nlm.nih.gov/REST/rxcui.json?name={}'.format(name))
    data = r.json()
    results = set()
    if ('idGroup' in data) and ('rxnormId' in data['idGroup']):
        for x in data['idGroup']['rxnormId']:
            results.add(x)
    return list(results)


if __name__ == "__main__":
    dir_name = r'C:/Users/zangc/Documents/Boston/workshop/2021-PASC/query_specific/'
    dir_name2 = r'C:/Users/zangc/Documents/Boston/workshop/2021-PASC/query_specific/drug_from_KG/'

    # med_file = 'Pulmonary_Medications_es_v1.xlsx'
    # kd_file = "Asthma_COPD_drugs_selected.csv"  # "COPD_drugs.csv"  #'Asthma_drugs.csv'

    med_file = 'Neurologic_Medications_ASN_v1.xlsx'
    kd_file = 'AD_Depress_drugs_selected.csv'

    df_kg = pd.read_csv(dir_name2 + kd_file, dtype=str)
    print('df_kg.shape:', df_kg.shape)
    df_kg = df_kg.loc[df_kg['Relation'].isin(['Treats_DDi', 'Palliates_DDi', 'Effect_DDi']), :].copy()
    print('df_kg.shape:', df_kg.shape)
    df_kg = df_kg.drop_duplicates(subset=['Head'])
    print('df_kg.shape:', df_kg.shape)
    df_kg['name'] = df_kg['Head'].apply(lambda x: x.lower())
    df_kg['rxnorm'] = df_kg['Head'].apply(lambda x: ';'.join(api_rxnorm_from_string(x)))
    df_kg.to_csv(dir_name2 + kd_file + '-drop_duplicates.csv', index=False)

    df_med = pd.read_excel(dir_name + med_file, dtype={'rxnorm': str})
    df_med['name'] = df_med['name'].apply(lambda x: x.lower())

    df = pd.merge(df_kg, df_med, left_on='rxnorm', right_on='rxnorm', how='left')
    df.to_csv(dir_name2 + kd_file + '-aux-V2.csv', index=False)
