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
import eligibility_setting as ecs
import functools

print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess demographics')
    parser.add_argument('--cohorts', choices=['pasc_incidence', 'pasc_prevalence', 'covid'],
                        default='covid', help='cohorts')
    parser.add_argument('--dataset', choices=['COL', 'MSHS', 'MONTE', 'NYU', 'WCM', 'ALL'],
                        default='COL', help='site dataset')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    args.output_file_query12 = r'../data/V15_COVID19/output/character/matrix_cohorts_{}_query12_encoding_{}.csv'.format(
        args.cohorts,
        args.dataset)

    print('args:', args)
    return args


def _load_mapping():
    print('... loading encoding mapping dictionaries:')

    with open(r'../data/mapping/icd_pasc_mapping.pkl', 'rb') as f:
        icd_pasc = pickle.load(f)
        print('Load ICD-10 to PASC mapping done! len(icd_pasc):', len(icd_pasc))
        record_example = next(iter(icd_pasc.items()))
        print('e.g.:', record_example)

    with open(r'../data/mapping/pasc_index_mapping.pkl', 'rb') as f:
        pasc_encoding = pickle.load(f)
        print('Load PASC to encoding mapping done! len(pasc_encoding):', len(pasc_encoding))
        record_example = next(iter(pasc_encoding.items()))
        print('e.g.:', record_example)

    with open(r'../data/mapping/icd_cmr_mapping.pkl', 'rb') as f:
        icd_cmr = pickle.load(f)
        print('Load ICD-10 to CMR mapping done! len(icd_cmr):', len(icd_cmr))
        record_example = next(iter(icd_cmr.items()))
        print('e.g.:', record_example)

    with open(r'../data/mapping/cmr_index_mapping.pkl', 'rb') as f:
        cmr_encoding = pickle.load(f)
        print('Load CMR to encoding mapping done! len(cmr_encoding):', len(cmr_encoding))
        record_example = next(iter(cmr_encoding.items()))
        print('e.g.:', record_example)

    with open(r'../data/mapping/icd_ccsr_mapping.pkl', 'rb') as f:
        icd_ccsr = pickle.load(f)
        print('Load ICD-10 to CCSR mapping done! len(icd_ccsr):', len(icd_ccsr))
        record_example = next(iter(icd_ccsr.items()))
        print('e.g.:', record_example)

    with open(r'../data/mapping/ccsr_index_mapping.pkl', 'rb') as f:
        ccsr_encoding = pickle.load(f)
        print('Load CCSR to encoding mapping done! len(ccsr_encoding):', len(ccsr_encoding))
        record_example = next(iter(ccsr_encoding.items()))
        print('e.g.:', record_example)

    with open(r'../data/mapping/rxnorm_ingredient_mapping_combined.pkl', 'rb') as f:
        rxnorm_ing = pickle.load(f)
        print('Load rxRNOM_CUI to ingredient mapping done! len(rxnorm_atc):', len(rxnorm_ing))
        record_example = next(iter(rxnorm_ing.items()))
        print('e.g.:', record_example)

    with open(r'../data/mapping/rxnorm_atc_mapping.pkl', 'rb') as f:
        rxnorm_atc = pickle.load(f)
        print('Load rxRNOM_CUI to ATC mapping done! len(rxnorm_atc):', len(rxnorm_atc))
        record_example = next(iter(rxnorm_atc.items()))
        print('e.g.:', record_example)

    with open(r'../data/mapping/atcL2_index_mapping.pkl', 'rb') as f:
        atcl2_encoding = pickle.load(f)
        print('Load to ATC-Level-2 to encoding mapping done! len(atcl2_encoding):', len(atcl2_encoding))
        record_example = next(iter(atcl2_encoding.items()))
        print('e.g.:', record_example)

    with open(r'../data/mapping/atcL3_index_mapping.pkl', 'rb') as f:
        atcl3_encoding = pickle.load(f)
        print('Load to ATC-Level-3 to encoding mapping done! len(atcl3_encoding):', len(atcl3_encoding))
        record_example = next(iter(atcl3_encoding.items()))
        print('e.g.:', record_example)

    # with open(r'../data/mapping/atc_rxnorm_mapping.pkl', 'rb') as f:
    #     atc_rxnorm = pickle.load(f)
    #     print('Load ATC to rxRNOM_CUI mapping done! len(atc_rxnorm):', len(atc_rxnorm))
    #     record_example = next(iter(atc_rxnorm.items()))
    #     print('e.g.:', record_example)

    return icd_pasc, pasc_encoding, icd_cmr, cmr_encoding, icd_ccsr, ccsr_encoding, \
           rxnorm_ing, rxnorm_atc, atcl2_encoding, atcl3_encoding


def _encoding_age(age):
    # 'age20-39',  'age40-54', 'age55-64', 'age65-74', 'age75-84', 'age>=85'
    encoding = np.zeros((1, 6), dtype='int')
    if age <= 39:
        encoding[0, 0] = 1
    elif age <= 54:
        encoding[0, 1] = 1
    elif age <= 64:
        encoding[0, 2] = 1
    elif age <= 74:
        encoding[0, 3] = 1
    elif age <= 84:
        encoding[0, 4] = 1
    else:
        encoding[0, 5] = 1
    return encoding


def _encoding_gender(gender):
    encoding = np.zeros((1, 3), dtype='int')
    # female, Male, other/missing
    gender = gender.upper()
    if gender == 'F':
        encoding[0, 0] = 1
    elif gender == 'M':
        encoding[0, 1] = 1
    else:
        encoding[0, 2] = 1

    return encoding


def _encoding_race(race):
    # race_column_names = ['Asian', 'Black or African American', 'White', 'Other', 'Missing']
    # Other (American Indian or Alaska Native, Native Hawaiian or Other Pacific Islander, Multiple Race, Other)5
    # Missing (No Information, Refuse to Answer, Unknown, Missing)4
    encoding = np.zeros((1, 5), dtype='int')
    if race == '02':
        encoding[0, 0] = 1
    elif race == '03':
        encoding[0, 1] = 1
    elif race == '05':
        encoding[0, 2] = 1
    elif race == '01' or race == '04' or race == '06' or race == 'OT':
        encoding[0, 3] = 1
    else:
        encoding[0, 4] = 1
    return encoding


def _encoding_hispanic(hispanic):
    # ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']
    encoding = np.zeros((1, 3), dtype='int')
    if hispanic == 'Y':
        encoding[0, 0] = 1
    elif hispanic == 'N':
        encoding[0, 1] = 1
    else:
        encoding[0, 2] = 1

    return encoding


def _encoding_yearmonth(index_date):
    """
    ["March 2020", "April 2020", "May 2020", "June 2020", "July 2020",
    "August 2020", "September 2020", "October 2020", "November 2020", "December 2020",
    "January 2021", "February 2021", "March 2021", "April 2021", "May 2021",
    "June 2021", "July 2021", "August 2021", "September 2021", "October 2021",
    "November 2021", "December 2021", "January 2022",]
    :param index_date:
    :return:
    """
    encoding = np.zeros((1, 23), dtype='int')
    year = index_date.year
    month = index_date.month
    pos = (month-3) + (year - 2020)*12
    encoding[0, pos] = 1

    return encoding


def _encoding_inpatient(dx_list, index_date):
    flag = False
    for records in dx_list:
        dx_date, icd, dx_type, enc_type = records
        if ecs._is_in_inpatient_period(dx_date, index_date):
            if (enc_type == 'EI') or (enc_type == 'IP') or (enc_type == 'OS'):
                flag = True
                break

    return flag


def _encoding_ventilation(pro_list, obsgen_list, index_date, vent_codes):
    flag = False
    for records in pro_list:
        px_date, px, px_type, enc_type, enc_id = records
        px = px.replace('.', '').upper()
        if ecs._is_in_ventilation_period(px_date, index_date):
            if px in vent_codes:
                flag = True
                break

    for records in obsgen_list:
        px_date, px, px_type, result_text, source, enc_id = records
        if ecs._is_in_ventilation_period(px_date, index_date):
            if (px_type == 'PC_COVID') and (px == '3000') and (source == 'DR') and (result_text == 'Y'):
                flag = True
                break

    return flag


def _encoding_dx(dx_list, icd_cmr, cmr_encoding, index_date, verbos=0):
    # encoding 39 cmr comorbidity codes in the baseline
    encoding = np.zeros((1, len(cmr_encoding)), dtype='float')
    for records in dx_list:
        dx_date, icd = records[:2]
        if _is_in_baseline(dx_date, index_date):
            icd = icd.replace('.', '').upper()
            if icd in icd_cmr:
                cmr_list = icd_cmr[icd]  # one icd can mapping 1, 2, or 3 cmd categories
                for cmr in cmr_list:
                    rec = cmr_encoding[cmr]
                    pos = rec[0]
                    encoding[0, pos] += 1
            else:
                if verbos > 0:
                    print('ERROR:', icd, 'not in icd to CMR dictionary!')
    return encoding


def _encoding_med(med_list, rxnorm_ing, rxnorm_atc, atcl_encoding, index_date, atc_level=2, verbose=0):
    # encoding 2 atc level 2 diagnoses codes H02: CORTICOSTEROIDS FOR SYSTEMIC USE   L04:IMMUNOSUPPRESSANTS in the baseline
    # mapping rxnorm_cui to its ingredient(s)
    # for each ingredient, mapping to atc and thus atc[:3] is level three
    # med_array = np.zeros((n, 2), dtype='int')  # atc level 3 category
    # med_column_names =   #
    atclevel_chars = {1: 1, 2: 3, 3: 4, 4: 5, 5: 7}
    atc_n_chars = atclevel_chars.get(atc_level, 3)  # default level 2, using first 3 chars
    encoding = np.zeros((1, 2), dtype='float')
    _no_mapping_rxrnom = set([])
    for records in med_list:
        med_date, rxnorm, supply_days = records
        if _is_in_baseline(med_date, index_date):
            if rxnorm in rxnorm_atc:
                atcl_set = set([x[0][:atc_n_chars] for x in rxnorm_atc[rxnorm]])
                pos_list = [atcl_encoding[x][0] for x in atcl_set if x in atcl_encoding]
                for pos in pos_list:
                    encoding[0, pos] += 1
            elif rxnorm in rxnorm_ing:
                ing_list = rxnorm_ing[rxnorm]
                atcl_set = set([])
                for ing in ing_list:
                    if ing in rxnorm_atc:
                        atcl_set.update([x[0][:atc_n_chars] for x in rxnorm_atc[ing]])

                pos_list = [atcl_encoding[x][0] for x in atcl_set if x in atcl_encoding]
                for pos in pos_list:
                    encoding[0, pos] += 1
            else:
                _no_mapping_rxrnom.add(rxnorm)
                if verbose:
                    print('ERROR:', rxnorm, 'not in rxnorm to atc dictionary or rxnorm-to-ing-to-atc!')
    return encoding, _no_mapping_rxrnom


def build_query_1and2_matrix(args):
    start_time = time.time()
    print('In build_query_1and2_matrix...')
    # step 1: load encoding dictionary
    # icd_pasc, pasc_encoding, icd_cmr, cmr_encoding, \
    # icd_ccsr, ccsr_encoding, rxnorm_ing, rxnorm_atc, atcl2_encoding, atcl3_encoding = _load_mapping()

    ventilation_codes = utils.load(r'../data/mapping/ventilation_codes.pkl')

    # step 2: load cohorts pickle data
    print('In cohorts_characterization_build_data...')
    if args.dataset == 'ALL':
        sites = ['COL', 'MSHS', 'MONTE', 'NYU', 'WCM']
    else:
        sites = [args.dataset, ]

    data_all_sites = []
    # outcome_all_sites = []
    # outcome_med_all_sites = []
    # df_records_aux = []  # for double check, and get basic information
    # _no_mapping_rxrnom_all = set([])

    print('Try to load: ', sites)
    for site in tqdm(sites):
        print('Loading: ', site)
        input_file = r'../data/V15_COVID19/output/{}/cohorts_{}_{}.pkl'.format(site, args.cohorts, site)
        print('Load cohorts pickle data file:', input_file)
        id_data = utils.load(input_file)

        # step 3: encoding cohorts baseline covariates into matrix
        n = len(id_data)
        pid_list = []
        site_list = []
        covid_list = []
        hospitalized_list = []
        ventilation_list = []

        age_array = np.zeros((n, 6), dtype='int')
        age_column_names = ['20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75-<85 years', '85+ years']

        gender_array = np.zeros((n, 3), dtype='int')
        gender_column_names = ['Female', 'Male', 'Other/Missing']

        race_array = np.zeros((n, 5), dtype='int')
        race_column_names = ['Asian', 'Black or African American', 'White', 'Other', 'Missing']
        # Other (American Indian or Alaska Native, Native Hawaiian or Other Pacific Islander, Multiple Race, Other)5
        # Missing (No Information, Refuse to Answer, Unknown, Missing)4

        hispanic_array = np.zeros((n, 3), dtype='int')
        hispanic_column_names = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']

        yearmonth_array = np.zeros((n, 23), dtype='int')
        yearmonth_column_names = ["March 2020", "April 2020", "May 2020", "June 2020", "July 2020", "August 2020",
                                  "September 2020", "October 2020", "November 2020", "December 2020", "January 2021",
                                  "February 2021", "March 2021", "April 2021", "May 2021", "June 2021", "July 2021",
                                  "August 2021", "September 2021", "October 2021", "November 2021", "December 2021",
                                  "January 2022",]

        # dx_array = np.zeros((n, 39), dtype='int')  # 38 cmr , encoding both CBVD_POA CBVD_SQLA, thus 39
        # dx_column_names = list(cmr_encoding.keys())
        #
        # med_array = np.zeros((n, 2), dtype='int')  # atc level 3 category
        # med_column_names = ['H02', 'L04']  # H02: CORTICOSTEROIDS FOR SYSTEMIC USE   L04:IMMUNOSUPPRESSANTS

        column_names = ['patid', 'site', 'covid', 'hospitalized', 'ventilation',] + age_column_names + \
                       gender_column_names + race_column_names + hispanic_column_names + yearmonth_column_names

        print('len(column_names):', len(column_names), '\n', column_names)

        for i, (pid, item) in tqdm(enumerate(id_data.items()), total=len(id_data)):
            index_info, demo, dx, med, covid_lab, enc, procedure, obsgen, immun = item
            flag, index_date, covid_loinc, flag_name, index_age_year, index_enc_id = index_info
            birth_date, gender, race, hispanic, zipcode, state, city, nation_adi, state_adi = demo

            pid_list.append(pid)
            site_list.append(site)
            covid_list.append(flag)

            if pid in ['1375116']:
                print(pid)

            inpatient_flag = _encoding_inpatient(dx, index_date)
            vent_flag = _encoding_ventilation(procedure, obsgen, index_date, ventilation_codes)
            hospitalized_list.append(inpatient_flag)
            ventilation_list.append(vent_flag)

            # encoding query 1 information
            age_array[i, :] = _encoding_age(index_age_year)
            gender_array[i] = _encoding_gender(gender)
            race_array[i, :] = _encoding_race(race)
            hispanic_array[i, :] = _encoding_hispanic(hispanic)
            yearmonth_array[i, :] = _encoding_yearmonth(index_date)

            # dx_array[i, :] = _encoding_dx(dx, icd_cmr, cmr_encoding, index_date)
            # med_array[i, :], _no_mapping_rxrnom = _encoding_med(med, rxnorm_ing, rxnorm_atc,
            #                                                     {'H02': (0, "CORTICOSTEROIDS FOR SYSTEMIC USE"),
            #                                                      'L04': (1, "IMMUNOSUPPRESSANTS")},
            #                                                     index_date)

        print('Encoding done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

        #   step 4: build pandas, column, and dump
        data_array = np.hstack((np.asarray(pid_list).reshape(-1, 1),
                                np.asarray(site_list).reshape(-1, 1),
                                np.array(covid_list).reshape(-1, 1),
                                np.asarray(hospitalized_list).reshape(-1, 1),
                                np.array(ventilation_list).reshape(-1, 1),
                                age_array,
                                gender_array,
                                race_array,
                                hispanic_array,
                                yearmonth_array))

        df_data = pd.DataFrame(data_array, columns=column_names)
        data_all_sites.append(df_data)

        print('df_data.shape:', df_data.shape)
        print('Done site:', site)
        # end iterate sites

    df_data_all_sites = pd.concat(data_all_sites)
    print('df_data_all_sites.shape:', df_data_all_sites.shape)
    utils.check_and_mkdir(args.output_file_query12)
    df_data_all_sites.to_csv(args.output_file_query12)
    print('Done! Dump data matrix for query12 to {}'.format(args.output_file_query12))

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return df_data_all_sites


if __name__ == '__main__':
    # python query_12_cdc.py --dataset COL 2>&1 | tee  log/query_12_cdc_COL.txt
    # python query_12_cdc.py --dataset ALL 2>&1 | tee  log/query_12_cdc_ALL.txt

    start_time = time.time()
    args = parse_args()
    df_data = build_query_1and2_matrix(args)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
