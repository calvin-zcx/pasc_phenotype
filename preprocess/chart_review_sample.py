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
import warnings

warnings.filterwarnings("ignore")


def write_race_eth_combo(row):
    if (row['White'] == 1) and (row['Hispanic: No'] == 1):
        return "White_NH"
    elif (row['Black or African American'] == 1) or (row['Hispanic: Yes'] == 1):
        return "Black_AfAm_or_H"
    else:
        return "Other_Race_Eth"


if __name__ == '__main__':
    start_time = time.time()

    # SET SEED
    seed = 3

    np.random.seed(seed=seed)

    cohort_df = pd.read_excel(
        r'../data/V15_COVID19/output/character/cp_dm/diabetes_incidence_cases-Sep2.xlsx')  # , index_col=0)
    print(list(cohort_df.columns))
    cohort_df.head()

    group_cols_dict = dict({"Female, Black/African American and/or Hispanic": ['Female', 'race_eth_combo', 'covid'],
                            "Male, Black/African American and/or Hispanic": ['Male', 'race_eth_combo', 'covid'],
                            "Female, White Non-Hispanic": ['Female', 'race_eth_combo', 'covid'],
                            "Male, White Non-Hispanic": ['Male', 'race_eth_combo', 'covid'],
                            "Female, Other Race and Ethnicity": ['Female', 'race_eth_combo', 'covid'],
                            "Male, Other Race and Ethnicity": ['Male', 'race_eth_combo', 'covid']
                            })

    race_eth_list = ["Black_AfAm_or_H", "Black_AfAm_or_H",
                     "White_NH", "White_NH",
                     "Other_Race_Eth", "Other_Race_Eth"]
    count_list = [[3, 3], [2, 2], [3, 3], [2, 2], [2, 1], [1, 1]]

    print("EXCELL INDEX NUMBERS TO INCLUDE IN CHART REVIEW")

    selected_list = []
    age_distribution = []
    for site in ['WCM', 'NYU', 'MSHS', 'COL', 'MONTE']:
        print('Site:', site)
        site_selected_list = []
        for i, key in enumerate(group_cols_dict.keys()):
            print(key)

            relevant_cols = ['Female', 'Male', 'Black or African American', 'White', 'Hispanic: Yes', 'Hispanic: No',
                             'covid']
            sub_df = cohort_df.loc[cohort_df['site'] == site, relevant_cols]
            sub_df['race_eth_combo'] = sub_df.apply(write_race_eth_combo, axis=1)

            # covid positive
            print("  Covid (+)")
            for j, col_name in enumerate(group_cols_dict[key]):
                if j == 0:
                    pos_df = sub_df[sub_df[col_name] == 1]
                elif j == 1:
                    pos_df = pos_df[pos_df[col_name] == race_eth_list[i]]
                elif j == 2:
                    pos_df = pos_df[pos_df[col_name] == 1]

            # shuffle
            pos_df = pos_df.sample(frac=1)
            excel_index_keep = pos_df.index.values[:count_list[i][0]]
            print(excel_index_keep)
            site_selected_list.extend(excel_index_keep)

            # covid negative
            print("  Covid (-)")
            for j, col_name in enumerate(group_cols_dict[key]):
                if j == 0:
                    neg_df = sub_df[sub_df[col_name] == 1]
                elif j == 1:
                    neg_df = neg_df[neg_df[col_name] == race_eth_list[i]]
                elif j == 2:
                    neg_df = neg_df[neg_df[col_name] == 0]

            # shuffle
            neg_df = neg_df.sample(frac=1)
            excel_index_keep = neg_df.index.values[:count_list[i][1]]
            print(excel_index_keep)
            site_selected_list.extend(excel_index_keep)
            print()

        selected_list.extend(site_selected_list)
        site_select_df = cohort_df.loc[site_selected_list, :]

        age_distribution.append(cohort_df.loc[cohort_df['site'] == site, ["index age"]].describe().rename(columns = {"index age": site+' all'}))
        age_distribution.append(site_select_df[["index age"]].describe().rename(columns = {"index age": site+' sample'}))

        print(pd.concat(age_distribution, axis=1))
        print()

    print('len(selected_list):', len(selected_list))
    cohort_df['race_eth_combo'] = cohort_df.apply(write_race_eth_combo, axis=1)
    selected_df = cohort_df.loc[selected_list, :]
    selected_df.to_excel(
        r'../data/V15_COVID19/output/character/cp_dm/diabetes_incidence_cases-sampled-seed{}-withDOB.xlsx'.format(seed))
    age_distribution_df = pd.concat(age_distribution, axis=1)
    age_distribution_df.to_excel(
        r'../data/V15_COVID19/output/character/cp_dm/diabetes_incidence_cases-sampled-seed{}-agedist-withDOB.xlsx'.format(
            seed))

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
