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
from collections import  Counter
import datetime
from misc import utils
from eligibility_setting_elixhauser import _is_in_baseline, _is_in_followup, _is_in_acute
import functools

print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess demographics')
    parser.add_argument('--dataset', choices=['COL', 'MSHS', 'MONTE', 'NYU', 'WCM', 'ALL'], default='COL',
                        help='site dataset')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    args.output_file_covariates = r'../data/V15_COVID19/output/character/pcr_cohorts_covariate_elixh_encoding_{}.csv'.format(
        args.dataset)
    args.output_file_outcome = r'../data/V15_COVID19/output/character/pcr_cohorts_outcome_pasc_encoding_{}.csv'.format(
        args.dataset)
    args.output_file_outcome_med = r'../data/V15_COVID19/output/character/pcr_cohorts_outcome_atcl3_encoding_{}.csv'.format(
        args.dataset)

    args.output_file_raw = r'../data/V15_COVID19/output/character/pcr_cohorts_raw_elixh_df_{}.csv'.format(args.dataset)

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
    encoding = np.zeros((1, 6), dtype='float')
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
    gender = gender.upper()
    if gender == 'F':
        return 1
    else:
        return 0


def _encoding_race(race):
    # ['white', 'black', 'asian', 'other', 'unknown']
    encoding = np.zeros((1, 5), dtype='float')
    if race == '05':
        encoding[0, 0] = 1
    elif race == '03':
        encoding[0, 1] = 1
    elif race == '02':
        encoding[0, 2] = 1
    elif race == 'NI' or race == '07' or race == 'UN':
        # NI=No information, 07=Refuse to answer, UN=Unknown
        encoding[0, 3] = 1
    else:
        encoding[0, 4] = 1
    return encoding


def _encoding_hispanic(hispanic):
    # ['not hispanic', 'hispanic', 'unknown']
    encoding = np.zeros((1, 3), dtype='float')
    if hispanic == 'N':
        encoding[0, 0] = 1
    elif hispanic == 'Y':
        encoding[0, 1] = 1
    else:
        encoding[0, 2] = 1
    return encoding


def _encoding_social(nation_adi, impute_value):
    # ['ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49',
    #  'ADI50-59', 'ADI60-69', 'ADI70-79', 'ADI80-89', 'ADI90-100']
    encoding = np.zeros((1, 10), dtype='float')
    if pd.isna(nation_adi):
        nation_adi = impute_value
    if nation_adi >= 100:
        nation_adi = 99
    if nation_adi < 1:
        nation_adi = 1
    pos = int(nation_adi) // 10
    encoding[0, pos] = 1
    return encoding


def _encoding_utilization(enc_list, index_date):
    # encoding uitlization in the baseline
    # ['inpatient visits', 'outpatient visits', 'emergency visits', 'other visits']
    encoding = np.zeros((1, 4), dtype='float')
    for records in enc_list:
        enc_date, type, enc_id = records
        if _is_in_baseline(enc_date, index_date):
            if type == 'EI' or type == 'IP':
                encoding[0, 0] += 1
            elif type == 'AV' or type == 'OA' or type == 'TH':
                encoding[0, 1] += 1
            elif type == 'ED':
                encoding[0, 2] += 1
            else:
                encoding[0, 3] += 1
    # ['inpatient visits 0', 'inpatient visits 1-2', 'inpatient visits 3-4', 'inpatient visits >=5',
    #  'outpatient visits 0', 'outpatient visits 1-2', 'outpatient visits 3-4', 'outpatient visits >=5',
    #  'emergency visits 0', 'emergency visits 1-2', 'emergency visits 3-4', 'emergency visits >=5']
    #
    encoding_update = np.zeros((1, 12), dtype='float')
    for i in [0, 1, 2]:
        if encoding[0, i] <= 0:
            encoding_update[0, 0 + i*4] = 1
        elif encoding[0, i] <= 2:
            encoding_update[0, 1 + i*4] = 1
        elif encoding[0, i] <= 4:
            encoding_update[0, 2 + i*4] = 1
        else:
            encoding_update[0, 3 + i*4] = 1

    return encoding_update


def _encoding_index_period(index_date):
    # ['03/20-06/20', '07/20-10/20', '11/20-02/21', '03/21-06/21', '07/21-11/21']
    encoding = np.zeros((1, 5), dtype='float')
    # datetime.datetime(2020, 1, 1, 0, 0),
    # datetime.datetime(2020, 7, 1, 0, 0),
    # datetime.datetime(2020, 11, 1, 0, 0),
    # datetime.datetime(2021, 3, 1, 0, 0),
    # datetime.datetime(2021, 7, 1, 0, 0),
    # datetime.datetime(2021, 12, 30, 0, 0)
    if index_date < datetime.datetime(2020, 7, 1, 0, 0):
        encoding[0, 0] = 1
    elif index_date < datetime.datetime(2020, 11, 1, 0, 0):
        encoding[0, 1] = 1
    elif index_date < datetime.datetime(2021, 3, 1, 0, 0):
        encoding[0, 2] = 1
    elif index_date < datetime.datetime(2021, 7, 1, 0, 0):
        encoding[0, 3] = 1
    else:
        encoding[0, 4] = 1

    return encoding


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


def _rxnorm_to_atc(rxnorm, rxnorm_ing, rxnorm_atc, atc_level):
    assert atc_level in [1,2,3,4,5]
    atclevel_chars = {1: 1, 2: 3, 3: 4, 4: 5, 5: 7}
    atc_n_chars = atclevel_chars[atc_level]  # default level 2, using first 3 chars
    if rxnorm in rxnorm_atc:
        atcl_set = set([x[0][:atc_n_chars] for x in rxnorm_atc[rxnorm]])
    elif rxnorm in rxnorm_ing:
        ing_list = rxnorm_ing[rxnorm]
        atcl_set = set([])
        for ing in ing_list:
            if ing in rxnorm_atc:
                atcl_set.update([x[0][:atc_n_chars] for x in rxnorm_atc[ing]])
    else:
        atcl_set = set()
    return atcl_set


def _encoding_outcome_med(med_list, rxnorm_ing, rxnorm_atc, atcl_encoding, index_date, atc_level=3, verbose=0):
    # mapping rxnorm_cui to its ingredient(s)
    # for each ingredient, mapping to atc and thus atc[:3] is level three
    # med_array = np.zeros((n, 2), dtype='int')  # atc level 3 category
    # atc l3, 269 codes
    outcome_t2e = np.zeros((1, len(atcl_encoding)), dtype='float')
    outcome_flag = np.zeros((1, len(atcl_encoding)), dtype='int')
    outcome_baseline = np.zeros((1, len(atcl_encoding)), dtype='int')

    _no_mapping_rxrnom = set([])
    for records in med_list:
        med_date, rxnorm, supply_days = records
        atcl_set = _rxnorm_to_atc(rxnorm, rxnorm_ing, rxnorm_atc, atc_level)

        if len(atcl_set) > 0:
            pos_list = [atcl_encoding[x][0] for x in atcl_set if x in atcl_encoding]
        else:
            _no_mapping_rxrnom.add(rxnorm)
            if verbose:
                print('ERROR:', rxnorm, 'not in rxnorm to atc dictionary or rxnorm-to-ing-to-atc!')
            continue

        # build baseline
        if _is_in_baseline(med_date, index_date):
            for pos in pos_list:
                outcome_baseline[0, pos] += 1

        # build outcome
        if _is_in_followup(med_date, index_date):
            days = (med_date - index_date).days
            for pos in pos_list:
                if outcome_flag[0, pos] == 0:
                    # only records the first event and time
                    outcome_t2e[0, pos] = days
                    outcome_flag[0, pos] = 1

    return outcome_flag, outcome_t2e, outcome_baseline


def _encoding_outcome_dx(dx_list, icd_pasc, pasc_encoding, index_date):
    # encoding 137 outcomes from our PASC list
    #         outcome_t2e = np.zeros((n, 137), dtype='int')
    #         outcome_flag = np.zeros((n, 137), dtype='int')
    #         outcome_baseline = np.zeros((n, 137), dtype='int')

    outcome_t2e = np.zeros((1, len(pasc_encoding)), dtype='float')
    outcome_flag = np.zeros((1, len(pasc_encoding)), dtype='int')
    outcome_baseline = np.zeros((1, len(pasc_encoding)), dtype='int')

    for records in dx_list:
        dx_date, icd = records[:2]
        icd = icd.replace('.', '').upper()
        # build baseline
        if _is_in_baseline(dx_date, index_date):
            if icd in icd_pasc:
                pasc_info = icd_pasc[icd]
                pasc = pasc_info[0]
                rec = pasc_encoding[pasc]
                pos = rec[0]
                outcome_baseline[0, pos] += 1
        # build outcome
        if _is_in_followup(dx_date, index_date):
            days = (dx_date - index_date).days
            if icd in icd_pasc:
                pasc_info = icd_pasc[icd]
                pasc = pasc_info[0]
                rec = pasc_encoding[pasc]
                pos = rec[0]
                if outcome_flag[0, pos] == 0:
                    # only records the first event and time
                    outcome_t2e[0, pos] = days
                    outcome_flag[0, pos] = 1

    return outcome_flag, outcome_t2e, outcome_baseline


def build_baseline_covariates_and_outcome(args):
    start_time = time.time()
    print('In build_baseline_covariates...')
    # step 1: load encoding dictionary
    icd_pasc, pasc_encoding, icd_cmr, cmr_encoding, \
    icd_ccsr, ccsr_encoding, rxnorm_ing, rxnorm_atc, atcl2_encoding, atcl3_encoding = _load_mapping()

    # step 2: load cohorts pickle data
    print('In cohorts_characterization_build_data...')
    if args.dataset == 'ALL':
        sites = ['COL', 'MSHS', 'MONTE', 'NYU', 'WCM']
    else:
        sites = [args.dataset, ]

    data_all_sites = []
    outcome_all_sites = []
    outcome_med_all_sites = []

    df_records_aux = []  # for double check, and get basic information
    _no_mapping_rxrnom_all = set([])
    print('Try to load: ', sites)
    for site in tqdm(sites):
        print('Loading: ', site)
        input_file = r'../data/V15_COVID19/output/{}/data_pcr_cohorts_{}.pkl'.format(site, site)
        print('Load cohorts pickle data file:', input_file)
        try:
            with open(input_file, 'rb') as f:
                id_data = pickle.load(f)
                print('Load covid patients pickle data done! len(id_data):', len(id_data))
        except:
            print('Error occur, try loading again!')
            with open(input_file, 'rb') as f:
                id_data = pickle.load(f)
                print('AGAIN Load covid patients pickle data done! len(id_data):', len(id_data))

        # step 3: encoding cohorts baseline covariates into matrix
        n = len(id_data)
        pid_list = []
        site_list = []
        covid_list = []
        age_array = np.zeros((n, 6), dtype='int')
        age_column_names = ['age20-39', 'age40-54', 'age55-64', 'age65-74', 'age75-84', 'age>=85']
        gender_array = np.zeros((n, 1), dtype='int')
        gender_column_names = ['gender-female', ]
        race_array = np.zeros((n, 5), dtype='int')
        race_column_names = ['white', 'black', 'asian', 'other', 'unknown']
        hispanic_array = np.zeros((n, 3), dtype='int')
        hispanic_column_names = ['not hispanic', 'hispanic', 'unknown']
        social_array = np.zeros((n, 10), dtype='int')
        social_column_names = ['ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49',
                               'ADI50-59', 'ADI60-69', 'ADI70-79', 'ADI80-89', 'ADI90-100']
        utilization_array = np.zeros((n, 12), dtype='int')
        utilization_column_names = ['inpatient visits 0', 'inpatient visits 1-2', 'inpatient visits 3-4',
                                    'inpatient visits >=5',
                                    'outpatient visits 0', 'outpatient visits 1-2', 'outpatient visits 3-4',
                                    'outpatient visits >=5',
                                    'emergency visits 0', 'emergency visits 1-2', 'emergency visits 3-4',
                                    'emergency visits >=5']

        index_period_array = np.zeros((n, 5), dtype='int')
        index_period_names = ['03/20-06/20', '07/20-10/20', '11/20-02/21', '03/21-06/21', '07/21-11/21']

        dx_array = np.zeros((n, 39), dtype='int')  # 38 cmr , encoding both CBVD_POA CBVD_SQLA, thus 39
        dx_column_names = list(cmr_encoding.keys())

        med_array = np.zeros((n, 2), dtype='int')  # atc level 3 category
        med_column_names = ['H02', 'L04']  # H02: CORTICOSTEROIDS FOR SYSTEMIC USE   L04:IMMUNOSUPPRESSANTS

        column_names = ['patid', 'site', 'covid', ] + \
                       age_column_names + gender_column_names + race_column_names + hispanic_column_names + \
                       social_column_names + utilization_column_names + index_period_names + \
                       dx_column_names + med_column_names
        print('len(column_names):', len(column_names), '\n', column_names)

        # step 4: build outcome t2e and flag in follow-up,
        #         and outcome flag in baseline for dynamic cohort selection
        # in total, there are 137 PASC categories in our lists.
        outcome_t2e = np.zeros((n, 137), dtype='int16')
        outcome_flag = np.zeros((n, 137), dtype='int16')
        outcome_baseline = np.zeros((n, 137), dtype='int16')
        # ['patid', 'site', 'covid', ] + \
        outcome_column_names = ['flag@' + x for x in pasc_encoding.keys()] + \
                               ['t2e@' + x for x in pasc_encoding.keys()] + \
                               ['baseline@' + x for x in pasc_encoding.keys()]
        ICD_cnts_pos = Counter()
        ICD_cnts_neg = Counter()
        # build ATC-L3 medication outcomes
        # encoding 269 atc level 3 diagnoses codes in the baseline
        outcome_med_t2e = np.zeros((n, 269), dtype='int16')
        outcome_med_flag = np.zeros((n, 269), dtype='int16')
        outcome_med_baseline = np.zeros((n, 269), dtype='int16')
        outcome_med_column_names = ['flag@' + x for x in atcl3_encoding.keys()] + \
                                   ['t2e@' + x for x in atcl3_encoding.keys()] + \
                                   ['baseline@' + x for x in atcl3_encoding.keys()]
        atc_cnts_pos = Counter()
        atc_cnts_neg = Counter()

        # impute adi value by median of all:
        adi_value_list = [v[1][7] for key, v in id_data.items()]
        adi_value_default = np.nanmedian(adi_value_list)
        # pd.DataFrame(adi_value_list).describe()
        # pd.DataFrame(adi_value_list).hist(bins=20)

        for i, (pid, item) in tqdm(enumerate(id_data.items()), total=len(id_data)):
            pid_list.append(pid)
            index_info, demo, dx, med, covid_lab, enc = item
            flag, index_date, covid_loinc, flag_name, index_age_year, index_enc_id = index_info
            birth_date, gender, race, hispanic, zipcode, state, city, nation_adi, state_adi = demo
            site_list.append(site)
            covid_list.append(flag)

            if args.debug:
                # store raw information for debugging
                # add dx, med, enc in acute, and follow-up
                # currently focus on baseline information
                records_aux = [pid, site]
                records_aux.extend(index_info + demo)
                lab_str = ';'.join([x[2] for x in covid_lab])  # all lab tests
                dx_str_baseline = ';'.join([x[1].replace('.', '').upper() for x in dx if _is_in_baseline(x[0], index_date)])
                med_str_baseline = ';'.join([x[1] for x in med if _is_in_baseline(x[0], index_date)])
                enc_str_baseline = ';'.join([x[1] for x in enc if _is_in_baseline(x[0], index_date)])

                dx_str_acute = ';'.join([x[1].replace('.', '').upper() for x in dx if _is_in_acute(x[0], index_date)])
                med_str_acute = ';'.join([x[1] for x in med if _is_in_acute(x[0], index_date)])
                enc_str_acute = ';'.join([x[1] for x in enc if _is_in_acute(x[0], index_date)])

                dx_str_followup = ';'.join([x[1].replace('.', '').upper() for x in dx if _is_in_followup(x[0], index_date)])
                med_str_followup = ';'.join([x[1] for x in med if _is_in_followup(x[0], index_date)])
                enc_str_followup = ';'.join([x[1] for x in enc if _is_in_followup(x[0], index_date)])

                records_aux.extend([lab_str,
                                    dx_str_baseline, med_str_baseline, enc_str_baseline,
                                    dx_str_acute, med_str_acute, enc_str_acute,
                                    dx_str_followup, med_str_followup, enc_str_followup])
                df_records_aux.append(records_aux)

            # encoding data for modeling
            age_array[i, :] = _encoding_age(index_age_year)
            gender_array[i] = _encoding_gender(gender)
            race_array[i, :] = _encoding_race(race)
            hispanic_array[i, :] = _encoding_hispanic(hispanic)
            social_array[i, :] = _encoding_social(nation_adi, adi_value_default)
            # Only count following covariates in baseline
            utilization_array[i, :] = _encoding_utilization(enc, index_date)
            index_period_array[i, :] = _encoding_index_period(index_date)
            dx_array[i, :] = _encoding_dx(dx, icd_cmr, cmr_encoding, index_date)
            med_array[i, :], _no_mapping_rxrnom = _encoding_med(med, rxnorm_ing, rxnorm_atc,
                                                                {'H02': (0, "CORTICOSTEROIDS FOR SYSTEMIC USE"),
                                                                 'L04': (1, "IMMUNOSUPPRESSANTS")},
                                                                index_date)
            _no_mapping_rxrnom_all.update(_no_mapping_rxrnom)

            # encoding outcome
            outcome_flag[i, :], outcome_t2e[i, :], outcome_baseline[i, :] = _encoding_outcome_dx(dx, icd_pasc, pasc_encoding, index_date)

            outcome_med_flag[i, :], outcome_med_t2e[i, :], outcome_med_baseline[i, :] = _encoding_outcome_med(med, rxnorm_ing, rxnorm_atc, atcl3_encoding, index_date, atc_level=3)

            # count dx in pos and neg
            dx_set = set()
            for i_dx in dx:
                dx_t = i_dx[0]
                icd = i_dx[1].replace('.', '').upper()
                if _is_in_followup(dx_t, index_date):
                    dx_set.add(icd)

            for i_dx in dx_set:
                if flag:
                    ICD_cnts_pos[i_dx] += 1
                else:
                    ICD_cnts_neg[i_dx] += 1

            # count medication atc in pos and neg
            med_set = set()
            for i_med in med:
                t = i_med[0]
                rx = i_med[1]
                if _is_in_followup(t, index_date):
                    _added = _rxnorm_to_atc(rx, rxnorm_ing, rxnorm_atc, atc_level=3)
                    med_set.update(_added)

            for i_med in med_set:
                if flag:
                    atc_cnts_pos[i_med] += 1
                else:
                    atc_cnts_neg[i_med] += 1

        print('Encoding done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

        #   step 4: build pandas, column, and dump
        data_array = np.hstack((np.asarray(pid_list).reshape(-1, 1),
                                np.asarray(site_list).reshape(-1, 1),
                                np.array(covid_list).reshape(-1, 1),
                                age_array,
                                gender_array,
                                race_array,
                                hispanic_array,
                                social_array,
                                utilization_array,
                                index_period_array,
                                dx_array,
                                med_array))

        df_data = pd.DataFrame(data_array, columns=column_names)
        data_all_sites.append(df_data)

        # outcome_array = np.hstack((np.asarray(pid_list).reshape(-1, 1),
        #                            np.asarray(site_list).reshape(-1, 1),
        #                            np.array(covid_list).reshape(-1, 1),
        #                            outcome_flag,
        #                            outcome_t2e,
        #                            outcome_baseline
        #                            ))

        outcome_array = np.hstack((outcome_flag, outcome_t2e, outcome_baseline ))
        df_outcome = pd.DataFrame(outcome_array, columns=outcome_column_names)
        outcome_all_sites.append(df_outcome)

        outcome_med_array = np.hstack((outcome_med_flag, outcome_med_t2e, outcome_med_baseline))
        df_outcome_med = pd.DataFrame(outcome_med_array, columns=outcome_med_column_names)
        outcome_med_all_sites.append(df_outcome_med)

        print('df_data.shape:', df_data.shape,
              'df_outcome.shape:', df_outcome.shape,
              'df_outcome_med.shape:', df_outcome_med.shape)

        print('len(_no_mapping_rxrnom_all):', len(_no_mapping_rxrnom_all))
        print('Done site:', site)
        # end iterate sites

    print('len(_no_mapping_rxrnom_all):', len(_no_mapping_rxrnom_all))
    print(_no_mapping_rxrnom_all)

    df_data_all_sites = pd.concat(data_all_sites)
    print('df_data_all_sites.shape:', df_data_all_sites.shape)

    df_outcome_all_sites = pd.concat(outcome_all_sites)
    print('df_outcome_all_sites.shape:', df_outcome_all_sites.shape)

    df_outcome_med_all_sites = pd.concat(outcome_med_all_sites)
    print('df_outcome_med_all_sites.shape:', df_outcome_med_all_sites.shape)

    if args.debug:
        df_records_aux = pd.DataFrame(df_records_aux,
                                      columns=['patid', "site", "covid", "index_date", "covid_loinc", "flag_name",
                                               "index_age_year", "index_enc_id",
                                               "birth_date", "gender", "race", "hispanic", "zipcode", "state", "city",
                                               "nation_adi", "state_adi",
                                               "lab_str",
                                               "dx_str_baseline", "med_str_baseline", "enc_str_baseline",
                                               "dx_str_acute", "med_str_acute", "enc_str_acute",
                                               "dx_str_followup", "med_str_followup", "enc_str_followup"])
        print('df_records_aux.shape:', df_records_aux.shape)

    utils.check_and_mkdir(args.output_file_covariates)
    df_data_all_sites.to_csv(args.output_file_covariates)
    print('dump covariates done to {}'.format(args.output_file_covariates))

    utils.check_and_mkdir(args.output_file_outcome)
    df_outcome_all_sites.to_csv(args.output_file_outcome)
    print('dump outcome diagnosis done to {}'.format(args.output_file_outcome))

    utils.check_and_mkdir(args.output_file_outcome_med)
    df_outcome_med_all_sites.to_csv(args.output_file_outcome_med)
    print('dump outcome medication done to {}'.format(args.output_file_outcome_med))

    if args.debug:
        df_records_aux.to_csv(args.output_file_raw)
        print('dump debug file done to {}'.format(args.output_file_raw))

    utils.dump(_no_mapping_rxrnom_all, '../data/V15_COVID19/output/character/_no_mapping_rxrnom_all_set.pkl')

    ICD_cnts_pos = pd.DataFrame(ICD_cnts_pos.most_common(), columns=['ICD', 'No.person in pos'])
    ICD_cnts_neg = pd.DataFrame(ICD_cnts_neg.most_common(), columns=['ICD', 'No.person in neg'])
    df_combined_ICD_cnts = pd.merge(ICD_cnts_pos, ICD_cnts_neg, on='ICD', how='outer')
    df_combined_ICD_cnts['ICD name'] = df_combined_ICD_cnts['ICD'].apply(lambda x: icd_ccsr.get(x, ''))
    df_combined_ICD_cnts.to_csv('../data/V15_COVID19/output/character/pcr_cohorts_ICD_cnts_followup-{}.csv'.format(args.dataset))

    atc_cnts_pos = pd.DataFrame(atc_cnts_pos.most_common(), columns=['atc', 'No.person in pos'])
    atc_cnts_neg = pd.DataFrame(atc_cnts_neg.most_common(), columns=['atc', 'No.person in neg'])
    df_combined_atc_cnts = pd.merge(atc_cnts_pos, atc_cnts_neg, on='atc', how='outer')
    df_combined_atc_cnts['atc name'] = df_combined_atc_cnts['atc'].apply(lambda x: atcl3_encoding.get(x, ''))
    df_combined_atc_cnts.to_csv('../data/V15_COVID19/output/character/pcr_cohorts_atc_cnts_followup-{}.csv'.format(args.dataset))

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return df_data_all_sites, df_records_aux, df_outcome_all_sites, df_combined_ICD_cnts, df_combined_atc_cnts


def analyse_cohorts(args):
    df = pd.read_csv(args.output_file_covariates, dtype={'patid': str, 'covid': int})
    return df


if __name__ == '__main__':
    # python pre_data_elixh.py --dataset COL 2>&1 | tee  log/pre_data_elixh_COL.txt
    # python pre_data_elixh.py --dataset ALL 2>&1 | tee  log/pre_data_elixh_ALL.txt

    start_time = time.time()
    args = parse_args()
    # df_data = analyse_cohorts(args)
    df_data, df_records_aux, df_outcome_all_sites, \
    df_combined_ICD_cnts, df_combined_atc_cnts = build_baseline_covariates_and_outcome(args)
    # outcomes_df = build_outcomes_in_followup(args)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
