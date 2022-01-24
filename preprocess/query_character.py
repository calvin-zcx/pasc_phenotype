import sys

# for linux env.
sys.path.insert(0, '..')
import time
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from misc import utils
from eligibility_setting import _is_in_baseline, _is_in_followup, _is_in_acute
import functools
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess demographics')
    parser.add_argument('--dataset', choices=['COL', 'MSHS', 'MONTE', 'NYU', 'WCM', 'ALL'], default='COL',
                        help='site dataset')
    args = parser.parse_args()

    # args.input_file = r'../data/V15_COVID19/output/{}/data_pcr_cohorts_{}.pkl'.format(args.dataset, args.dataset)
    args.output_file = r'../data/V15_COVID19/output/character/pcr_cohorts_character_raw_{}.csv'.format(args.dataset)

    print('args:', args)
    return args


def cohorts_characterization_build_data(args):
    start_time = time.time()
    print('In cohorts_characterization_build_data...')
    if args.dataset == 'ALL':
        sites = ['COL', 'MSHS', 'MONTE', 'NYU', 'WCM']
    else:
        sites = [args.dataset, ]
    print('Loading: ', sites)
    # step 1: load 5 cohorts pickle data, store necessary info for analysis
    df_records_aux = []
    for site in tqdm(sites):
        input_file = r'../data/V15_COVID19/output/{}/data_pcr_cohorts_{}.pkl'.format(site, site)
        print('Loading site:', site, input_file)
        id_data = utils.load(input_file)
        print('Load covid patients pickle data done! len(id_data):', len(id_data))

        n = len(id_data)
        pid_list = []
        site_list = []
        covid_list = []

        n_no_index_enc_type = 0
        for i, (pid, item) in tqdm(enumerate(id_data.items()), total=len(id_data)):
            pid_list.append(pid)
            index_info, demo, dx, med, covid_lab, enc = item
            flag, index_date, covid_loinc, flag_name, index_age_year, index_enc_id = index_info
            birth_date, gender, race, hispanic, zipcode, state, city, nation_adi, state_adi = demo
            site_list.append(args.dataset)
            covid_list.append(flag)

            # matching index encounter type:
            index_enc_type = np.nan
            enc_type_flag = False

            if pd.notna(index_enc_id):
                # If exits encounter id
                # get index enc type by enc id! enc_item: (date, enc_type, enc_id)
                for enc_item in enc:
                    if enc_item[2] == index_enc_id:
                        index_enc_type = enc_item[1]
                        enc_type_flag = True
                        break
            else:
                # Notice: NYU covid lab data has no encounter id, need to use time to match encounter table
                # One the same day, one patient may have multiple encounter type, e.g. from outpatient --> inpatient
                # how to summarize encounter/ hospital utilization of cohorts?
                # get index enc type by enc date! enc_item: (date, enc_type, enc_id)
                # if multiple date match, count all.
                _enc_type_list = []
                for enc_item in enc:
                    if enc_item[0].date() == index_date.date():
                        # index_enc_type = enc_itelm[1]
                        _enc_type_list.append(enc_item[1])
                        enc_type_flag = True
                if enc_type_flag:
                    index_enc_type = ';'.join(_enc_type_list)

            if not enc_type_flag:
                print('not found covid index encounter type', pid, site)
                n_no_index_enc_type += 1
            # store raw information for debugging
            # add dx, med, enc in acute, and follow-up
            # currently focus on baseline information
            records_aux = [pid, site]
            records_aux.extend(index_info + [index_enc_type, ] + demo)

            lab_str = ';'.join([x[2] for x in covid_lab])  # all lab tests
            dx_str_baseline = ';'.join([x[1].replace('.', '') for x in dx if _is_in_baseline(x[0], index_date)])
            med_str_baseline = ';'.join([x[1] for x in med if _is_in_baseline(x[0], index_date)])
            enc_str_baseline = ';'.join([x[1] for x in enc if _is_in_baseline(x[0], index_date)])

            dx_str_acute = ';'.join([x[1].replace('.', '') for x in dx if _is_in_acute(x[0], index_date)])
            med_str_acute = ';'.join([x[1] for x in med if _is_in_acute(x[0], index_date)])
            enc_str_acute = ';'.join([x[1] for x in enc if _is_in_acute(x[0], index_date)])

            dx_str_followup = ';'.join([x[1].replace('.', '') for x in dx if _is_in_followup(x[0], index_date)])
            med_str_followup = ';'.join([x[1] for x in med if _is_in_followup(x[0], index_date)])
            enc_str_followup = ';'.join([x[1] for x in enc if _is_in_followup(x[0], index_date)])

            records_aux.extend([lab_str,
                                dx_str_baseline, med_str_baseline, enc_str_baseline,
                                dx_str_acute, med_str_acute, enc_str_acute,
                                dx_str_followup, med_str_followup, enc_str_followup])
            df_records_aux.append(records_aux)

        print(site, 'Done!',
              'len(id_data):', len(id_data),
              'len(df_records_aux)', len(df_records_aux),
              'n_no_index_enc_type:', n_no_index_enc_type)

    print('Integrating done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    #   step 2: build pandas, column, and dump
    df_records_aux = pd.DataFrame(df_records_aux,
                                  columns=['patid', "site", "covid", "index_date", "covid_loinc", "flag_name",
                                           "index_age_year", "index_enc_id", "index_enc_type",
                                           "birth_date", "gender", "race", "hispanic", "zipcode", "state", "city",
                                           "nation_adi", "state_adi",
                                           "lab_str",
                                           "dx_str_baseline", "med_str_baseline", "enc_str_baseline",
                                           "dx_str_acute", "med_str_acute", "enc_str_acute",
                                           "dx_str_followup", "med_str_followup", "enc_str_followup"])
    print('df_records_aux.shape:', df_records_aux.shape)

    utils.check_and_mkdir(args.output_file)
    df_records_aux.to_csv(args.output_file)
    print('dump done to {}'.format(args.output_file))
    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return df_records_aux


def cohorts_characterization_analyse(args):
    df = pd.read_csv(args.output_file, dtype={'patid': str}, parse_dates=['index_date', 'birth_date'])  #
    df_pos = df.loc[df["covid"], :]
    df_neg = df.loc[~df["covid"], :]

    # age
    pos_age_iqr = df_pos['index_age_year'].quantile([0.25, 0.5, 0.75]).to_list()
    neg_age_iqr = df_neg['index_age_year'].quantile([0.25, 0.5, 0.75]).to_list()
    print('pos:  {} ({} -- {})'.format(pos_age_iqr[1], pos_age_iqr[0], pos_age_iqr[2]))
    print('neg:  {} ({} -- {})'.format(neg_age_iqr[1], neg_age_iqr[0], pos_age_iqr[2]))

    pos_cnt, pos_interval = np.histogram(df_pos['index_age_year'], bins=[20, 40, 55, 65, 75, 85, np.inf])
    neg_cnt, neg_interval = np.histogram(df_neg['index_age_year'], bins=[20, 40, 55, 65, 75, 85, np.inf])

    def print_age_group(cnt):
        tot = np.sum(cnt)
        for c in cnt:
            print('{} ({:.1f})'.format(c, c / tot * 100))

    print('pos age group:')
    print_age_group(pos_cnt)

    print('neg age group:')
    print_age_group(neg_cnt)

    def print_series_group(cnt, vocab_dic={}):
        tot = cnt.sum()
        for index, value in cnt.items():
            if not vocab_dic:
                print('{}\t{} ({:.1f})'.format(index, value, value / tot * 100))
            else:
                print('{}\t{} ({:.1f})'.format(vocab_dic[index], value, value / tot * 100))

    pos_female = df_pos['gender'].value_counts(dropna=False)
    neg_female = df_neg['gender'].value_counts(dropna=False)
    print('pos gender group:')
    print_series_group(pos_female)
    print('neg gender group:')
    print_series_group(neg_female)

    race_dict = {"01": "American Indian or Alaska Native",
                 "02": "Asian",
                 "03": "Black or African American",
                 "04": "Native Hawaiian or Other Pacific Islander",
                 "05": "White",
                 "06": "Multiple race",
                 "07": "Refuse to answer",
                 "NI": "No information",
                 "UN": "Unknown",
                 "OT": "Other"
                 }
    pos_race = df_pos['race'].value_counts(dropna=False)
    neg_race = df_neg['race'].value_counts(dropna=False)
    print('pos race group:')
    print_series_group(pos_race, race_dict)
    print('neg race group:')
    print_series_group(neg_race, race_dict)

    hispanic_dict = {"Y": "Yes",
                     "N": "No",
                     "R": "Refuse to answer",
                     "NI": "No information",
                     "UN": "Unknown",
                     "OT": "Other"}

    pos_hisp = df_pos['hispanic'].value_counts(dropna=False)
    neg_hisp = df_neg['hispanic'].value_counts(dropna=False)
    print('pos hispanic group:')
    print_series_group(pos_hisp, hispanic_dict)
    print('neg hispanic group:')
    print_series_group(neg_hisp, hispanic_dict)

    # ADI
    pos_adi_iqr = df_pos['nation_adi'].quantile([0.25, 0.5, 0.75]).to_list()
    neg_adi_iqr = df_neg['nation_adi'].quantile([0.25, 0.5, 0.75]).to_list()
    print('adi pos:  {} ({} -- {})'.format(pos_adi_iqr[1], pos_adi_iqr[0], pos_adi_iqr[2]))
    print('adi neg:  {} ({} -- {})'.format(neg_adi_iqr[1], neg_adi_iqr[0], neg_adi_iqr[2]))

    # df_pos['index_date'] = df_pos['index_date'].astype("datetime64")
    df_pos['index_date'].groupby(df["index_date"].dt.month).count().plot(kind="bar")

    fig, ax = plt.subplots(figsize=(28, 18))
    # Add x-axis and y-axis
    hist = df_pos['index_date'].hist(bins=pd.date_range(start='1/1/2020', end='12/1/2021', freq='M'))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel('Date', fontsize=28)
    plt.ylabel('Cases', fontsize=28)
    plt.title("Monthly COVID-19 PCR positive cases, INSIGHT Data Warehouse, 2020/3 - 2021/11",
              fontdict={'fontsize':28})
    # Rotate tick marks on x-axis
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.show()

    pos_month = pd.Series(index=df_pos['index_date'], data=1, name='positive cases').resample('1M').count()
    neg_month = pd.Series(index=df_neg['index_date'], data=1, name='negative cases').resample('1M').count()
    month_result = pos_month.to_frame().join(neg_month.to_frame(), how='outer')
    month_result.to_excel('positive_and_negative_monthly_counts.xlsx')

    def count_time_period(date_series):
        bins = [datetime.datetime(2020, 1, 1, 0, 0),
                datetime.datetime(2020, 7, 1, 0, 0),
                datetime.datetime(2020, 11, 1, 0, 0),
                datetime.datetime(2021, 3, 1, 0, 0),
                datetime.datetime(2021, 7, 1, 0, 0),
                datetime.datetime(2021, 12, 30, 0, 0)]
        results = []
        for i in range(len(bins)-1):
            tot = len(date_series)
            cnt = ((bins[i] <= date_series) & (date_series < bins[i+1])).sum()
            results.append((cnt, cnt/tot))
        df = pd.DataFrame(results, columns=['count', 'fraction'])
        for x in results:
            print('{} ({:.1f})'.format(x[0], x[1]*100))
        return df
    print('positive_4monthly_counts:')
    pos_time_period = count_time_period(df_pos['index_date'])
    print('negative_4monthly_counts:')
    neg_time_period = count_time_period(df_neg['index_date'])
    pos_time_period.to_excel('../data/V15_COVID19/output/character/positive_4monthly_counts.xlsx')
    neg_time_period.to_excel('../data/V15_COVID19/output/character/negative_4monthly_counts.xlsx')

    return df


if __name__ == '__main__':
    # python query_character.py --dataset ALL 2>&1 | tee  log/query_character_ALL_data_building.txt

    start_time = time.time()
    args = parse_args()
    df = cohorts_characterization_build_data(args)
    # df = cohorts_characterization_analyse(args)
    # df_pos = df.loc[df["covid"], :]
    # df_neg = df.loc[~df["covid"], :]
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
