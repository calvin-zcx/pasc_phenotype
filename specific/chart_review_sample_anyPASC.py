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
import datetime
import zipfile


warnings.filterwarnings("ignore")


def write_race_eth_combo(row):
    # if (row['White'] == 1) and (row['Hispanic: No'] == 1):
    #     return "White_NH"
    # elif (row['Black or African American'] == 1) or (row['Hispanic: Yes'] == 1):
    #     return "Black_AfAm_or_H"
    # else:
    #     return "Other_Race_Eth"

    if (row['Black or African American'] == 1) or (row['Hispanic: Yes'] == 1):
        return "Black_AfAm_or_H"
    else:
        return "Other_Race_Eth"


if __name__ == '__main__':
    start_time = time.time()

    # #
    # # select covid+ if necessary
    # data_file = r'..\data\recover\output\results\anyPASC2DX\anyPASC_2DX_stratified_period_severity_nsites-22.csv'
    # cohort_df = pd.read_csv(data_file,
    #                         dtype={'patid': str, 'site': str, 'zip': str},
    #                         parse_dates=['index date'],
    #                         )
    # print(cohort_df.shape)
    # col_names = pd.DataFrame(cohort_df.columns)
    # cohort_df = cohort_df.loc[(cohort_df['any_2dx_pasc_flag'] == 1) | (cohort_df['any_2dx30day_pasc_flag'] == 1), :]
    # print(cohort_df.shape)
    # # cohort_df = cohort_df.loc[cohort_df['site'] == 'vumc', :]
    # # print(cohort_df.shape)
    # cohort_df.drop([x for x in cohort_df.columns if (x.startswith('med') or x.startswith('covidmed')
    #                                                  or x.startswith('dx-t2e@')
    #                                                  or x.startswith('dx-base@')
    #                                                  or x.startswith('dx-out@')
    #                                                  or x.startswith('dx-t2eall@')
    #
    #                                                  )
    #                 ], axis=1, inplace=True)  #
    # print(cohort_df.shape)
    # cohort_df.to_csv(r'..\data\recover\output\results\anyPASC2DX\anyPASC_2DX_stratified_period_severity_nsites-22-simple.csv')
    # zz

    # SET SEED
    anypasc_type = 'sustained2dx30d'  # 'persistent2dx'  #
    seed = 0
    for seed in [0, 1, 2, 3, 4, 5,]:  # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 150, 155

        np.random.seed(seed=seed)

        cohort_df = pd.read_excel(
            r'..\data\recover\output\results\anyPASC2DX\anyPASC_2DX_stratified_period_severity_nsites-22-simple-revised.xlsx')
        # cohort_df = cohort_df.loc[(cohort_df['index date'] < datetime.datetime(2022, 3, 1, 0, 0)), :].copy()
        print('all', cohort_df.shape)
        cohort_df = cohort_df.loc[(cohort_df['age'] >= 20), :].copy()
        print('age>=20', cohort_df.shape)

        if anypasc_type == 'persistent2dx':
            cohort_df = cohort_df.loc[(cohort_df['any_2dx_pasc_flag'] == 1) & (cohort_df['any_2dx30day_pasc_flag'] != 1), :].copy()
            # should we exclude any_2dx30day_pasc_flag==1 case? Yes
            print('persistent2dx but not sustained2dx30d', cohort_df.shape)
        elif anypasc_type == 'sustained2dx30d':
            cohort_df = cohort_df.loc[(cohort_df['any_2dx30day_pasc_flag'] == 1), :].copy()
            print('sustained2dx30d', cohort_df.shape)
        else:
            raise ValueError

        print(list(cohort_df.columns))
        cohort_df.head()

        group_cols_dict = dict({"Female, Black/African American and/or Hispanic": ['Female', 'race_eth_combo', 'covid'],
                                "Male, Black/African American and/or Hispanic": ['Male', 'race_eth_combo', 'covid'],
                                # "Female, White Non-Hispanic": ['Female', 'race_eth_combo', 'covid'],
                                # "Male, White Non-Hispanic": ['Male', 'race_eth_combo', 'covid'],
                                "Female, Other Race and Ethnicity": ['Female', 'race_eth_combo', 'covid'],
                                "Male, Other Race and Ethnicity": ['Male', 'race_eth_combo', 'covid']
                                })

        # race_eth_list = ["Black_AfAm_or_H", "Black_AfAm_or_H",
        #                  "White_NH", "White_NH",
        #                  "Other_Race_Eth", "Other_Race_Eth"]
        # count_list = [[3, 3], [2, 2], [3, 3], [2, 2], [2, 1], [1, 1]]
        race_eth_list = ["Black_AfAm_or_H", "Black_AfAm_or_H",
                         "Other_Race_Eth", "Other_Race_Eth"]
        count_list = [[2, 1], [2, 1], [2, 2], [1, 1]]

        print("EXCELL INDEX NUMBERS TO INCLUDE IN CHART REVIEW")

        selected_list = []
        age_distribution = []
        NO_RACE_SITE = []
        add_info = []
        for site in ['emory', 'miami', 'usf', 'nebraska', 'lsu', ]:
            # 'mcw', 'ochsner', 'pitt', 'ufh', # miami has no race, all missing. once updated, changed
            # emory has been updated also
            site_selected_list_4period = []
            for period in ['preOmiOut', 'preOmiIn', 'omiOut', 'omiIn']:
                print('period:', period, 'Site:', site)
                site_selected_list = []
                for i, key in enumerate(group_cols_dict.keys()):
                    print(key)

                    # relevant_cols = ['Female', 'Male', 'Black or African American', 'White', 'Hispanic: Yes',
                    #                  'Hispanic: No',
                    #                  'covid']
                    sub_df = cohort_df.loc[cohort_df['site'] == site, :]  # relevant_cols
                    sub_df['race_eth_combo'] = sub_df.apply(write_race_eth_combo, axis=1)
                    sub_df = sub_df.loc[sub_df['any_pasc_type'] == period, :]

                    # covid positive
                    print("  Covid (+)", count_list[i][0])
                    for j, col_name in enumerate(group_cols_dict[key]):
                        if j == 0:
                            pos_df = sub_df[sub_df[col_name] == 1]
                        elif j == 2:
                            pos_df = pos_df[pos_df[col_name] == 1]

                        if site not in NO_RACE_SITE:
                            if j == 1:
                                pos_df = pos_df[pos_df[col_name] == race_eth_list[i]]

                    # shuffle
                    pos_df = pos_df.sample(frac=1)
                    if pos_df.shape[0] < count_list[i][0]:
                        add_info.append([site, key, 'covid+', pos_df.shape[0], count_list[i][0],
                                         'add:{}'.format(count_list[i][0] - pos_df.shape[0])])
                        print('pos_df.shape[0], count_list[i][0]', pos_df.shape[0], count_list[i][0])
                        # print('Need add more patients in covid+, only consider gender')
                        print('Need add more patients in covid+, no gender or race constraints')

                        for j, col_name in enumerate(group_cols_dict[key]):
                            # if j == 2: # just  covid
                            #     pos_df_add = sub_df[sub_df[col_name] == 1]

                            # if j == 0:  # just gender and covid
                            #     pos_df_add = sub_df[sub_df[col_name] == 1]
                            # elif j == 2:
                            #     pos_df_add = pos_df_add[pos_df_add[col_name] == 1]

                            # just consider covid
                            if j == 2:
                                pos_df_add = sub_df[sub_df[col_name] == 1]

                        pos_df_add = pos_df_add.loc[~pos_df_add.index.isin(list(pos_df.index) + site_selected_list),
                                     :].sample(frac=1)  # remove dumplicate from add
                        pos_df = pd.concat([pos_df, pos_df_add], axis=1)

                    excel_index_keep = pos_df.index.values[:count_list[i][0]]
                    print(excel_index_keep)
                    site_selected_list.extend(excel_index_keep)
                    site_selected_list_4period.extend(excel_index_keep)

                    # covid negative
                    print("  Covid (-)", count_list[i][1])
                    for j, col_name in enumerate(group_cols_dict[key]):
                        if j == 0:
                            neg_df = sub_df[sub_df[col_name] == 1]
                        elif j == 2:
                            neg_df = neg_df[neg_df[col_name] == 0]
                        if site not in NO_RACE_SITE:
                            if j == 1:
                                neg_df = neg_df[neg_df[col_name] == race_eth_list[i]]

                    # shuffle
                    neg_df = neg_df.sample(frac=1)
                    if neg_df.shape[0] < count_list[i][1]:
                        add_info.append([site, key, 'covid-', neg_df.shape[0], count_list[i][1],
                                         'add:{}'.format(count_list[i][1] - neg_df.shape[0])])
                        print('neg_df.shape[0], count_list[i][0]', neg_df.shape[0], count_list[i][1])
                        # print('Need add more patients in covid-, only consider gender')
                        print('Need add more patients in covid-, no gender or race constraints')

                        for j, col_name in enumerate(group_cols_dict[key]):
                            # if j == 2:  # just gender and covid
                            #     neg_df_add = sub_df[sub_df[col_name] == 1]
                            # if j == 0:  # just gender and covid
                            #     neg_df_add = sub_df[sub_df[col_name] == 1]
                            # elif j == 2:
                            #     neg_df_add = neg_df_add[neg_df_add[col_name] == 0]

                            if j == 2:  # just consider covid
                                neg_df_add = sub_df[sub_df[col_name] == 0]

                        neg_df_add = neg_df_add.loc[~neg_df_add.index.isin(list(neg_df.index) + site_selected_list),
                                     :].sample(frac=1)  # remove dumplicate from add
                        neg_df = pd.concat([neg_df, neg_df_add], axis=1)

                    excel_index_keep = neg_df.index.values[:count_list[i][1]]
                    print(excel_index_keep)
                    site_selected_list.extend(excel_index_keep)
                    site_selected_list_4period.extend(excel_index_keep)
                    print()

                selected_list.extend(site_selected_list)

            site_select_df_4period = cohort_df.loc[site_selected_list_4period, :]

            age_distribution.append(
                cohort_df.loc[cohort_df['site'] == site, ["age"]].describe().rename(
                    columns={"index age": site + ' all'}))
            age_distribution.append(site_select_df_4period[["age"]].describe().rename(columns={"age": site + ' sample'}))
            print(pd.concat(age_distribution, axis=1))
            print()

        print("add_info", add_info)
        print('len(selected_list):', len(selected_list), 'len(set(selected_list)):', len(set(selected_list)))
        cohort_df['race_eth_combo'] = cohort_df.apply(write_race_eth_combo, axis=1)
        selected_df = cohort_df.loc[selected_list, :]

        print('In total: selected_df.shape', selected_df.shape)
        selected_df.to_excel(
            r'../data/recover/output/results/anyPASC2DX/incidence_sample/anyPASC-{}_incidence_sampled-seed{}.xlsx'.format(
                anypasc_type, seed))
        age_distribution_df = pd.concat(age_distribution, axis=1)
        age_distribution_df.to_excel(
            r'../data/recover/output/results/anyPASC2DX/incidence_sample/anyPASC-{}_incidence_sampled-seed{}-agedist.xlsx'.format(
                anypasc_type, seed))

        print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    #print('Caution: pitt has no race information, thus only consider gender, NO_RACE_SITE', NO_RACE_SITE)
