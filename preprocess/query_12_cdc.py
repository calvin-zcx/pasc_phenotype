import fnmatch
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
import fnmatch
from lifelines import KaplanMeierFitter, CoxPHFitter

print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess demographics')
    parser.add_argument('--cohorts', choices=['pasc_incidence', 'pasc_prevalence', 'covid'],
                        default='pasc_incidence', help='cohorts')
    parser.add_argument('--dataset', choices=['COL', 'MSHS', 'MONTE', 'NYU', 'WCM', 'ALL'],
                        default='COL', help='site dataset')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    args.output_file_query12 = r'../data/V15_COVID19/output/character/matrix_cohorts_{}_query12_encoding_cnt_{}.csv'.format(
        args.cohorts,
        args.dataset)
    args.output_file_query12_bool = r'../data/V15_COVID19/output/character/matrix_cohorts_{}_query12_encoding_bool_{}.csv'.format(
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
    pos = (month - 3) + (year - 2020) * 12
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
            # OBSGEN_TYPE=”PC_COVID” & OBSGEN_CODE = 3000 & OBSGEN_SOURCE=”DR” & RESULT_TEXT=”Y”
            if (px_type == 'PC_COVID') and (px == '3000') and (source == 'DR') and (result_text == 'Y'):
                flag = True
                break

    return flag


def _is_in_code_set_with_wildchar(code, code_set, code_set_wild):
    if code in code_set:
        return True
    for pattern in code_set_wild:
        if fnmatch.fnmatchcase(code, pattern):
            return True
    return False


def _encoding_dx(dx_list, dx_column_names, comorbidity_codes, index_date, pro_list):
    # encoding 32 cmr comorbidity codes in the baseline
    # deal with "DX: Hypertension and Type 1 or 2 Diabetes Diagnosis"  separately later after encoding
    # DX: End Stage Renal Disease on Dialysis   Both diagnosis and procedure codes used to define this condtion
    encoding = np.zeros((1, len(dx_column_names)), dtype='int')
    for records in dx_list:
        dx_date, icd, dx_type, enc_type = records
        if ecs._is_in_comorbidity_period(dx_date, index_date):
            icd = icd.replace('.', '').upper()
            for pos, col_name in enumerate(dx_column_names):
                if col_name in comorbidity_codes:
                    code_set, code_set_wild = comorbidity_codes[col_name]
                    if _is_in_code_set_with_wildchar(icd, code_set, code_set_wild):
                        encoding[0, pos] += 1

    # deal with DX: End Stage Renal Disease on Dialysis from procedure
    pos = dx_column_names.index(r'DX: End Stage Renal Disease on Dialysis')
    for records in pro_list:
        px_date, px, px_type, enc_type, enc_id = records
        if ecs._is_in_comorbidity_period(px_date, index_date):
            px = px.replace('.', '').upper()
            code_set, code_set_wild = comorbidity_codes[r'DX: End Stage Renal Disease on Dialysis']
            if _is_in_code_set_with_wildchar(px, code_set, code_set_wild):
                encoding[0, pos] += 1

    return encoding


def _encoding_med(med_list, med_column_names, comorbidity_codes, index_date):
    # encoding 2 medications:
    # MEDICATION: Corticosteroids
    # MEDICATION: Immunosuppressant drug
    # ps: (H02: CORTICOSTEROIDS FOR SYSTEMIC USE   L04:IMMUNOSUPPRESSANTS in the baseline)

    encoding = np.zeros((1, len(med_column_names)), dtype='int')
    for records in med_list:
        med_date, rxnorm, supply_days = records
        if ecs._is_in_comorbidity_period(med_date, index_date):
            for pos, col_name in enumerate(med_column_names):
                if col_name in comorbidity_codes:
                    code_set, code_set_wild = comorbidity_codes[col_name]
                    if _is_in_code_set_with_wildchar(rxnorm, code_set, code_set_wild):
                        encoding[0, pos] += 1

    return encoding


def _encoding_outcome_dx(dx_list, icd_pasc, pasc_encoding, index_date):
    # encoding 137 outcomes from our PASC list
    # outcome_t2e = np.zeros((n, 137), dtype='int')
    # outcome_flag = np.zeros((n, 137), dtype='int')
    # outcome_baseline = np.zeros((n, 137), dtype='int')

    outcome_t2e = np.zeros((1, len(pasc_encoding)), dtype='int')
    outcome_flag = np.zeros((1, len(pasc_encoding)), dtype='int')
    outcome_baseline = np.zeros((1, len(pasc_encoding)), dtype='int')

    for records in dx_list:
        dx_date, icd = records[:2]
        icd = icd.replace('.', '').upper()
        # build baseline
        if ecs._is_in_baseline(dx_date, index_date):
            if icd in icd_pasc:
                pasc_info = icd_pasc[icd]
                pasc = pasc_info[0]
                rec = pasc_encoding[pasc]
                pos = rec[0]
                outcome_baseline[0, pos] += 1
        # build outcome
        if ecs._is_in_followup(dx_date, index_date):
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
                else:
                    outcome_flag[0, pos] += 1

    return outcome_flag, outcome_t2e, outcome_baseline


def build_query_1and2_matrix(args):
    start_time = time.time()
    print('In build_query_1and2_matrix...')
    # step 1: load encoding dictionary
    # icd_pasc, pasc_encoding, icd_cmr, cmr_encoding, \
    # icd_ccsr, ccsr_encoding, rxnorm_ing, rxnorm_atc, atcl2_encoding, atcl3_encoding = _load_mapping()

    ventilation_codes = utils.load(r'../data/mapping/ventilation_codes.pkl')
    comorbidity_codes = utils.load(r'../data/mapping/tailor_comorbidity_codes.pkl')

    icd_pasc = utils.load(r'../data/mapping/icd_pasc_mapping.pkl')
    pasc_encoding = utils.load(r'../data/mapping/pasc_index_mapping.pkl')

    # step 2: load cohorts pickle data
    print('In cohorts_characterization_build_data...')
    if args.dataset == 'ALL':
        sites = ['NYU', 'MONTE', 'COL', 'MSHS', 'WCM']
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

        age_array = np.zeros((n, 6), dtype='int16')
        age_column_names = ['20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75-<85 years', '85+ years']

        gender_array = np.zeros((n, 3), dtype='int16')
        gender_column_names = ['Female', 'Male', 'Other/Missing']

        race_array = np.zeros((n, 5), dtype='int16')
        race_column_names = ['Asian', 'Black or African American', 'White', 'Other', 'Missing']
        # Other (American Indian or Alaska Native, Native Hawaiian or Other Pacific Islander, Multiple Race, Other)5
        # Missing (No Information, Refuse to Answer, Unknown, Missing)4

        hispanic_array = np.zeros((n, 3), dtype='int16')
        hispanic_column_names = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']

        yearmonth_array = np.zeros((n, 23), dtype='int16')
        yearmonth_column_names = ["March 2020", "April 2020", "May 2020", "June 2020", "July 2020", "August 2020",
                                  "September 2020", "October 2020", "November 2020", "December 2020", "January 2021",
                                  "February 2021", "March 2021", "April 2021", "May 2021", "June 2021", "July 2021",
                                  "August 2021", "September 2021", "October 2021", "November 2021", "December 2021",
                                  "January 2022", ]

        # cautious of "DX: Hypertension and Type 1 or 2 Diabetes Diagnosis" using logic afterwards, due to threshold >= 2 issue
        # DX: End Stage Renal Disease on Dialysis   Both diagnosis and procedure codes used to define this condtion
        dx_array = np.zeros((n, 32), dtype='int16')
        dx_column_names = ["DX: Alcohol Abuse", "DX: Anemia", "DX: Arrythmia", "DX: Asthma", "DX: Cancer",
                           "DX: Chronic Kidney Disease", "DX: Chronic Pulmonary Disorders", "DX: Cirrhosis",
                           "DX: Coagulopathy", "DX: Congestive Heart Failure",
                           "DX: COPD", "DX: Coronary Artery Disease", "DX: Dementia", "DX: Diabetes Type 1",
                           "DX: Diabetes Type 2", "DX: End Stage Renal Disease on Dialysis", "DX: Hemiplegia",
                           "DX: HIV", "DX: Hypertension", "DX: Hypertension and Type 1 or 2 Diabetes Diagnosis",
                           "DX: Inflammatory Bowel Disorder", "DX: Lupus or Systemic Lupus Erythematosus",
                           "DX: Mental Health Disorders", "DX: Multiple Sclerosis", "DX: Parkinson's Disease",
                           "DX: Peripheral vascular disorders ", "DX: Pregnant",
                           "DX: Pulmonary Circulation Disorder  (PULMCR_ELIX)",
                           "DX: Rheumatoid Arthritis", "DX: Seizure/Epilepsy",
                           "DX: Severe Obesity  (BMI>=40 kg/m2)", "DX: Weight Loss"]
        #
        med_array = np.zeros((n, 2), dtype='int16')  # atc level 3 category
        med_column_names = ["MEDICATION: Corticosteroids", "MEDICATION: Immunosuppressant drug", ]
        # H02: CORTICOSTEROIDS FOR SYSTEMIC USE   L04:IMMUNOSUPPRESSANTS

        # Build PASC outcome t2e and flag in follow-up, and outcome flag in baseline for dynamic cohort selection
        # in total, there are 137 PASC categories in our lists.
        outcome_t2e = np.zeros((n, 137), dtype='int16')
        outcome_flag = np.zeros((n, 137), dtype='int16')
        outcome_baseline = np.zeros((n, 137), dtype='int16')
        outcome_column_names = ['flag@' + x for x in pasc_encoding.keys()] + \
                               ['t2e@' + x for x in pasc_encoding.keys()] + \
                               ['baseline@' + x for x in pasc_encoding.keys()]

        column_names = ['patid', 'site', 'covid', 'hospitalized', 'ventilation', ] + age_column_names + \
                       gender_column_names + race_column_names + hispanic_column_names + yearmonth_column_names + \
                       dx_column_names + med_column_names + outcome_column_names

        print('len(column_names):', len(column_names), '\n', column_names)

        for i, (pid, item) in tqdm(enumerate(id_data.items()), total=len(id_data)):
            index_info, demo, dx, med, covid_lab, enc, procedure, obsgen, immun = item
            flag, index_date, covid_loinc, flag_name, index_age_year, index_enc_id = index_info
            birth_date, gender, race, hispanic, zipcode, state, city, nation_adi, state_adi = demo

            pid_list.append(pid)
            site_list.append(site)
            covid_list.append(flag)

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

            # encoding query 2 information
            dx_array[i, :] = _encoding_dx(dx, dx_column_names, comorbidity_codes, index_date, procedure)
            med_array[i, :] = _encoding_med(med, med_column_names, comorbidity_codes, index_date)

            # encoding pasc information in both baseline and followup
            outcome_flag[i, :], outcome_t2e[i, :], outcome_baseline[i, :] = _encoding_outcome_dx(dx, icd_pasc,
                                                                                                 pasc_encoding,
                                                                                                 index_date)

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
                                yearmonth_array,
                                dx_array,
                                med_array,
                                outcome_flag,
                                outcome_t2e,
                                outcome_baseline))

        df_data = pd.DataFrame(data_array, columns=column_names)
        data_all_sites.append(df_data)

        print('df_data.shape:', df_data.shape)
        del id_data
        print('Done site:', site)
        # end iterate sites

    df_data_all_sites = pd.concat(data_all_sites)
    print('df_data_all_sites.shape:', df_data_all_sites.shape)

    utils.check_and_mkdir(args.output_file_query12)
    df_data_all_sites.to_csv(args.output_file_query12)
    print('Done! Dump data matrix for query12 to {}'.format(args.output_file_query12))

    # transform count to bool with threshold 2, and deal with "DX: Hypertension and Type 1 or 2 Diabetes Diagnosis"
    df_bool = df_data_all_sites.copy()
    selected_cols = [x for x in df_bool.columns if (x.startswith('DX:') or x.startswith('MEDICATION:'))]
    df_bool.loc[:, selected_cols] = (df_bool.loc[:, selected_cols].astype('int') >= 2).astype('int')
    df_bool.loc[:, r"DX: Hypertension and Type 1 or 2 Diabetes Diagnosis"] = \
        (df_bool.loc[:, r'DX: Hypertension'] & (
                    df_bool.loc[:, r'DX: Diabetes Type 1'] | df_bool.loc[:, r'DX: Diabetes Type 2'])).astype('int')

    # keep the value of baseline count and outcome count in the file, filter later depends on the application
    selected_cols = [x for x in df_bool.columns if (x.startswith('flag@') or x.startswith('baseline@'))]
    df_bool.loc[:, selected_cols] = (df_bool.loc[:, selected_cols].astype('int') >= 1).astype('int')

    utils.check_and_mkdir(args.output_file_query12_bool)
    df_bool.to_csv(args.output_file_query12_bool)
    print('Done! Dump data bool matrix for query12 to {}'.format(args.output_file_query12_bool))

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return df_data_all_sites, df_bool


def cohorts_characterization_analyse(cohorts, dataset='ALL', severity=''):
    # severity in 'hospitalized', 'ventilation', None
    in_file = r'../data/V15_COVID19/output/character/matrix_cohorts_{}_query12_encoding_bool_{}.csv'.format(
        cohorts, dataset)

    print('Try to load:', in_file)
    df_template = pd.read_excel(r'../data/V15_COVID19/output/character/RECOVER_Adults_Queries 1-2_with_PASC.xlsx',
                                sheet_name=r'Table Shells - Adults')
    df_data = pd.read_csv(in_file, dtype={'patid': str})  # , parse_dates=['index_date', 'birth_date']

    if severity == '':
        print('Not considering severity')
        df_pos = df_data.loc[df_data["covid"], :]
        df_neg = df_data.loc[~df_data["covid"], :]
    elif severity == 'hospitalized':
        print('Considering severity hospitalized cohorts')
        df_pos = df_data.loc[df_data['hospitalized'] & df_data["covid"], :]
        df_neg = df_data.loc[df_data['hospitalized'] & (~df_data["covid"]), :]
    elif severity == 'not hospitalized':
        print('Considering severity hospitalized cohorts')
        df_pos = df_data.loc[(~df_data['hospitalized']) & df_data["covid"], :]
        df_neg = df_data.loc[(~df_data['hospitalized']) & (~df_data["covid"]), :]
    elif severity == 'ventilation':
        print('Considering severity hospitalized ventilation cohorts')
        df_pos = df_data.loc[df_data['hospitalized'] & df_data['ventilation'] & df_data["covid"], :]
        df_neg = df_data.loc[df_data['hospitalized'] & df_data['ventilation'] & (~df_data["covid"]), :]
    else:
        raise ValueError

    col_in = list(df_data.columns)
    row_out = list(df_template.iloc[:, 0])

    for pos in [True, False]:
        # generate both positive and negative cohorts
        df_out = pd.DataFrame(np.nan, index=row_out, columns=['N', '%'])

        if pos:
            df_in = df_pos
        else:
            df_in = df_neg

        out_file = r'../data/V15_COVID19/output/character/results_query12_{}-{}-{}-{}.csv'.format(
            cohorts,
            dataset,
            'POS' if pos else 'NEG',
            severity)

        df_out.loc['Number of Unique Patients', 'N'] = len(df_in)
        df_out.loc['Number of Unique Patients', '%'] = 1.0

        age_col = ['20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75-<85 years', '85+ years']
        df_out.loc[age_col, 'N'] = df_in[age_col].sum().to_list()
        df_out.loc[age_col, '%'] = df_in[age_col].mean().to_list()

        sex_col_in = ['Female', 'Male', 'Other/Missing']
        sex_col_out = ['   Female', '   Male', '   Other2 / Missing']
        df_out.loc[sex_col_out, 'N'] = df_in[sex_col_in].sum().to_list()
        df_out.loc[sex_col_out, '%'] = df_in[sex_col_in].mean().to_list()

        race_col_in = ['Asian', 'Black or African American', 'White', 'Other', 'Missing', ]
        race_col_out = ['Asian', 'Black or African American', 'White',
                        'Other (American Indian or Alaska Native, Native Hawaiian or Other Pacific Islander, Multiple Race, Other)5',
                        'Missing (No Information, Refuse to Answer, Unknown, Missing)4 ']
        df_out.loc[race_col_out, 'N'] = df_in[race_col_in].sum().to_list()
        df_out.loc[race_col_out, '%'] = df_in[race_col_in].mean().to_list()

        ethnicity_col_in = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']
        ethnicity_col_out = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other3/Missing4']
        df_out.loc[ethnicity_col_out, 'N'] = df_in[ethnicity_col_in].sum().to_list()
        df_out.loc[ethnicity_col_out, '%'] = df_in[ethnicity_col_in].mean().to_list()

        date_col = ['March 2020', 'April 2020', 'May 2020', 'June 2020', 'July 2020',
                    'August 2020', 'September 2020', 'October 2020', 'November 2020', 'December 2020',
                    'January 2021', 'February 2021', 'March 2021', 'April 2021', 'May 2021',
                    'June 2021', 'July 2021', 'August 2021', 'September 2021', 'October 2021',
                    'November 2021', 'December 2021', 'January 2022']
        df_out.loc[date_col, 'N'] = df_in[date_col].sum().to_list()
        df_out.loc[date_col, '%'] = df_in[date_col].mean().to_list()

        # date race
        for d in date_col:
            c_in = ['Asian', 'Black or African American', 'White', 'Other', 'Missing']
            c_out = ['Race: Asian', 'Race: Black or African American', 'Race: White', 'Race: Other5', 'Race: Missing4']
            c_out = [d + ' - ' + x for x in c_out]
            _df_select = df_in.loc[df_in[d] == 1, c_in]
            df_out.loc[c_out, 'N'] = _df_select.sum().to_list()
            df_out.loc[c_out, '%'] = _df_select.mean().to_list()

        # date ethnicity
        for d in date_col:
            c_in = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']
            c_out = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other3/Missing4']
            c_out = [d + ' - ' + x for x in c_out]
            _df_select = df_in.loc[df_in[d] == 1, c_in]
            df_out.loc[c_out, 'N'] = _df_select.sum().to_list()
            df_out.loc[c_out, '%'] = _df_select.mean().to_list()

        # date age
        for i, d in enumerate(date_col):
            c_in = ['20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75-<85 years', '85+ years']
            c_out = [u'Age: 20-<40\xa0years', 'Age: 40-<55 years', 'Age: 55-<65 years', 'Age: 65-<75 years',
                     'Age: 75-<85 years', 'Age: 85+ years']
            # c_out = [d + ' - ' + x for x in c_out]
            c_out_v2 = []
            for x in c_out:
                if (d in ['October 2020', 'November 2020', 'December 2020',
                          'January 2021', 'February 2021', 'March 2021', 'April 2021']) and (x == 'Age: 40-<55 years'):
                    c_out_v2.append(d + '- ' + x)
                else:
                    c_out_v2.append(d + ' - ' + x)

            _df_select = df_in.loc[df_in[d] == 1, c_in]
            df_out.loc[c_out_v2, 'N'] = _df_select.sum().to_list()
            df_out.loc[c_out_v2, '%'] = _df_select.mean().to_list()

        comorbidity_col = ['DX: Alcohol Abuse', 'DX: Anemia', 'DX: Arrythmia', 'DX: Asthma', 'DX: Cancer',
                           'DX: Chronic Kidney Disease', 'DX: Chronic Pulmonary Disorders', 'DX: Cirrhosis',
                           'DX: Coagulopathy', 'DX: Congestive Heart Failure', 'DX: COPD',
                           'DX: Coronary Artery Disease', 'DX: Dementia', 'DX: Diabetes Type 1', 'DX: Diabetes Type 2',
                           'DX: End Stage Renal Disease on Dialysis', 'DX: Hemiplegia', 'DX: HIV', 'DX: Hypertension',
                           'DX: Hypertension and Type 1 or 2 Diabetes Diagnosis', 'DX: Inflammatory Bowel Disorder',
                           'DX: Lupus or Systemic Lupus Erythematosus', 'DX: Mental Health Disorders',
                           'DX: Multiple Sclerosis', "DX: Parkinson's Disease", 'DX: Peripheral vascular disorders ',
                           'DX: Pregnant', 'DX: Pulmonary Circulation Disorder  (PULMCR_ELIX)',
                           'DX: Rheumatoid Arthritis', 'DX: Seizure/Epilepsy', 'DX: Severe Obesity  (BMI>=40 kg/m2)',
                           'DX: Weight Loss', 'MEDICATION: Corticosteroids', 'MEDICATION: Immunosuppressant drug']
        df_out.loc[comorbidity_col, 'N'] = df_in[comorbidity_col].sum()
        df_out.loc[comorbidity_col, '%'] = df_in[comorbidity_col].sum() / len(df_in)

        pasc_col = [x for x in df_in.columns if (x.startswith('flag@') )]
        df_out.loc[pasc_col, 'N'] = df_in[pasc_col].sum()
        df_out.loc[pasc_col, '%'] = df_in[pasc_col].sum() / len(df_in)

        df_out.to_csv(out_file)
        print('Dump done ', out_file)


def pasc_specific_cohorts_characterization_analyse(cohorts, dataset='ALL', severity='',
                                                   pasc='Respiratory signs and symptoms'):
    # severity in 'hospitalized', 'ventilation', None
    # build pasc specific cohorts from covid base cohorts!
    in_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_query12_encoding_bool_{}.csv'.format(dataset)
    print('In pasc_specific_cohorts_characterization_analyse, PASC: {}, Cohorts: {}, severity: {}'.format(
        pasc, cohorts, severity))
    print('Try to load:', in_file)
    df_template = pd.read_excel(r'../data/V15_COVID19/output/character/RECOVER_Adults_Queries 1-2_with_PASC.xlsx',
                                sheet_name=r'Table Shells - Adults')
    df_data = pd.read_csv(in_file, dtype={'patid': str})  # , parse_dates=['index_date', 'birth_date']
    print('df_data.shape:', df_data.shape)
    # # dump potentially significant PASC list for screening
    # selected_cols = [x for x in df_data.columns if x.startswith('flag@')]  # or x.startswith('baseline@')
    # df_data.loc[:, selected_cols] = (df_data.loc[:, selected_cols] >= 1).astype('int')
    # df_pos = df_data.loc[df_data["covid"], :]
    # df_neg = df_data.loc[~df_data["covid"], :]
    # df_m = pd.DataFrame({'pos_mean': df_pos.loc[:, selected_cols].mean(),
    #                      'neg_mean': df_neg.loc[:, selected_cols].mean(),
    #                      'pos-neg%': df_pos.loc[:, selected_cols].mean() - df_neg.loc[:, selected_cols].mean(),
    #                      'pos_count': df_pos.loc[:, selected_cols].sum(),
    #                      'neg_count': df_neg.loc[:, selected_cols].sum(),
    #                      })
    # df_m_sorted = df_m.sort_values(by=['pos-neg%'], ascending=False)
    # df_m_sorted.to_csv('../data/V15_COVID19/output/character/pasc_count_cohorts_covid_query12_ALL.csv')

    if cohorts == 'pasc_prevalence':
        print('Choose cohorts pasc_prevalence')
        df_data = df_data.loc[df_data['flag@' + pasc] >= 1, :]
        print('df_data.shape:', df_data.shape)
    elif cohorts == 'pasc_incidence':
        print('Choose cohorts pasc_incidence')
        df_data = df_data.loc[(df_data['flag@' + pasc] >= 1) & (df_data['baseline@' + pasc] == 0), :]
        print('df_data.shape:', df_data.shape)
    else:
        raise ValueError

    if severity == '':
        print('Not considering severity')
        df_pos = df_data.loc[df_data["covid"], :]
        df_neg = df_data.loc[~df_data["covid"], :]
    elif severity == 'hospitalized':
        print('Considering severity hospitalized cohorts')
        df_pos = df_data.loc[df_data['hospitalized'] & df_data["covid"], :]
        df_neg = df_data.loc[df_data['hospitalized'] & (~df_data["covid"]), :]
    elif severity == 'not hospitalized':
        print('Considering severity NOT hospitalized cohorts')
        df_pos = df_data.loc[(~df_data['hospitalized']) & df_data["covid"], :]
        df_neg = df_data.loc[(~df_data['hospitalized']) & (~df_data["covid"]), :]
    elif severity == 'ventilation':
        print('Considering severity hospitalized ventilation cohorts')
        df_pos = df_data.loc[df_data['hospitalized'] & df_data['ventilation'] & df_data["covid"], :]
        df_neg = df_data.loc[df_data['hospitalized'] & df_data['ventilation'] & (~df_data["covid"]), :]
    else:
        raise ValueError
    print('df_pos.shape:', df_pos.shape, 'df_neg.shape:', df_neg.shape, )
    pasc_cols = ['flag@' + pasc, 't2e@' + pasc, 'baseline@' + pasc]
    print('pos:', df_pos[pasc_cols].mean())
    print('neg:', df_neg[pasc_cols].mean())
    try:
        kmf1 = KaplanMeierFitter(label='pos').fit(df_pos['t2e@' + pasc], event_observed=df_pos['flag@' + pasc],
                                                  label="pos")
        kmf0 = KaplanMeierFitter(label='neg').fit(df_neg['t2e@' + pasc], event_observed=df_neg['flag@' + pasc],
                                                  label="neg")
        ax = plt.subplot(111)
        kmf1.plot_survival_function(ax=ax)
        kmf0.plot_survival_function(ax=ax)
        plt.title(pasc + '-' + cohorts + '-' + severity)
        fig_outfile = r'../data/V15_COVID19/output/character/pasc/{}/results_query12_PASC-{}-{}-{}-{}.png'.format(
            pasc, pasc, cohorts, dataset, severity)
        utils.check_and_mkdir(fig_outfile)
        plt.savefig(fig_outfile)
        plt.close()
    except:
        print('No KM curve plots')

    try:
        cph = CoxPHFitter()
        selected_cols = ['covid', 'flag@' + pasc, 't2e@' + pasc] + \
                        [x for x in df_data.columns if (x.startswith('DX:') or x.startswith('MEDICATION:'))]
        cph.fit(df_data.loc[:, selected_cols], 't2e@' + pasc, 'flag@' + pasc)
        HR = cph.hazard_ratios_['covid']
        # CI = np.exp(cph.confidence_intervals_.values.reshape(-1))
        CI = cph.confidence_intervals_.apply(np.exp)
    except:
        HR = CI = None

    print(pasc, cohorts, dataset, severity)
    print('Cox {}: HR: {} ({})'.format(pasc, HR, CI))

    col_in = list(df_data.columns)
    row_out = list(df_template.iloc[:, 0])

    for pos in [True, False]:
        # generate both positive and negative cohorts
        df_out = pd.DataFrame(np.nan, index=row_out, columns=['N', '%'])

        if pos:
            df_in = df_pos
        else:
            df_in = df_neg

        out_file = r'../data/V15_COVID19/output/character/pasc/{}/results_query12_PASC-{}-{}-{}-{}-{}.csv'.format(
            pasc,
            pasc,
            cohorts,
            dataset,
            'POS' if pos else 'NEG',
            severity)
        utils.check_and_mkdir(out_file)
        df_out.loc['Number of Unique Patients', 'N'] = len(df_in)
        df_out.loc['Number of Unique Patients', '%'] = 1.0

        age_col = ['20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75-<85 years', '85+ years']
        df_out.loc[age_col, 'N'] = df_in[age_col].sum().to_list()
        df_out.loc[age_col, '%'] = df_in[age_col].mean().to_list()

        sex_col_in = ['Female', 'Male', 'Other/Missing']
        sex_col_out = ['   Female', '   Male', '   Other2 / Missing']
        df_out.loc[sex_col_out, 'N'] = df_in[sex_col_in].sum().to_list()
        df_out.loc[sex_col_out, '%'] = df_in[sex_col_in].mean().to_list()

        race_col_in = ['Asian', 'Black or African American', 'White', 'Other', 'Missing', ]
        race_col_out = ['Asian', 'Black or African American', 'White',
                        'Other (American Indian or Alaska Native, Native Hawaiian or Other Pacific Islander, Multiple Race, Other)5',
                        'Missing (No Information, Refuse to Answer, Unknown, Missing)4 ']
        df_out.loc[race_col_out, 'N'] = df_in[race_col_in].sum().to_list()
        df_out.loc[race_col_out, '%'] = df_in[race_col_in].mean().to_list()

        ethnicity_col_in = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']
        ethnicity_col_out = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other3/Missing4']
        df_out.loc[ethnicity_col_out, 'N'] = df_in[ethnicity_col_in].sum().to_list()
        df_out.loc[ethnicity_col_out, '%'] = df_in[ethnicity_col_in].mean().to_list()

        date_col = ['March 2020', 'April 2020', 'May 2020', 'June 2020', 'July 2020',
                    'August 2020', 'September 2020', 'October 2020', 'November 2020', 'December 2020',
                    'January 2021', 'February 2021', 'March 2021', 'April 2021', 'May 2021',
                    'June 2021', 'July 2021', 'August 2021', 'September 2021', 'October 2021',
                    'November 2021', 'December 2021', 'January 2022']
        df_out.loc[date_col, 'N'] = df_in[date_col].sum().to_list()
        df_out.loc[date_col, '%'] = df_in[date_col].mean().to_list()

        # date race
        for d in date_col:
            c_in = ['Asian', 'Black or African American', 'White', 'Other', 'Missing']
            c_out = ['Race: Asian', 'Race: Black or African American', 'Race: White', 'Race: Other5', 'Race: Missing4']
            c_out = [d + ' - ' + x for x in c_out]
            _df_select = df_in.loc[df_in[d] == 1, c_in]
            df_out.loc[c_out, 'N'] = _df_select.sum().to_list()
            df_out.loc[c_out, '%'] = _df_select.mean().to_list()

        # date ethnicity
        for d in date_col:
            c_in = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']
            c_out = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other3/Missing4']
            c_out = [d + ' - ' + x for x in c_out]
            _df_select = df_in.loc[df_in[d] == 1, c_in]
            df_out.loc[c_out, 'N'] = _df_select.sum().to_list()
            df_out.loc[c_out, '%'] = _df_select.mean().to_list()

        # date age
        for i, d in enumerate(date_col):
            c_in = ['20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75-<85 years', '85+ years']
            c_out = [u'Age: 20-<40\xa0years', 'Age: 40-<55 years', 'Age: 55-<65 years', 'Age: 65-<75 years',
                     'Age: 75-<85 years', 'Age: 85+ years']
            # c_out = [d + ' - ' + x for x in c_out]
            c_out_v2 = []
            for x in c_out:
                if (d in ['October 2020', 'November 2020', 'December 2020',
                          'January 2021', 'February 2021', 'March 2021', 'April 2021']) and (x == 'Age: 40-<55 years'):
                    c_out_v2.append(d + '- ' + x)
                else:
                    c_out_v2.append(d + ' - ' + x)

            _df_select = df_in.loc[df_in[d] == 1, c_in]
            df_out.loc[c_out_v2, 'N'] = _df_select.sum().to_list()
            df_out.loc[c_out_v2, '%'] = _df_select.mean().to_list()

        comorbidity_col = ['DX: Alcohol Abuse', 'DX: Anemia', 'DX: Arrythmia', 'DX: Asthma', 'DX: Cancer',
                           'DX: Chronic Kidney Disease', 'DX: Chronic Pulmonary Disorders', 'DX: Cirrhosis',
                           'DX: Coagulopathy', 'DX: Congestive Heart Failure', 'DX: COPD',
                           'DX: Coronary Artery Disease', 'DX: Dementia', 'DX: Diabetes Type 1', 'DX: Diabetes Type 2',
                           'DX: End Stage Renal Disease on Dialysis', 'DX: Hemiplegia', 'DX: HIV', 'DX: Hypertension',
                           'DX: Hypertension and Type 1 or 2 Diabetes Diagnosis', 'DX: Inflammatory Bowel Disorder',
                           'DX: Lupus or Systemic Lupus Erythematosus', 'DX: Mental Health Disorders',
                           'DX: Multiple Sclerosis', "DX: Parkinson's Disease", 'DX: Peripheral vascular disorders ',
                           'DX: Pregnant', 'DX: Pulmonary Circulation Disorder  (PULMCR_ELIX)',
                           'DX: Rheumatoid Arthritis', 'DX: Seizure/Epilepsy', 'DX: Severe Obesity  (BMI>=40 kg/m2)',
                           'DX: Weight Loss', 'MEDICATION: Corticosteroids', 'MEDICATION: Immunosuppressant drug']
        df_out.loc[comorbidity_col, 'N'] = df_in[comorbidity_col].sum()
        df_out.loc[comorbidity_col, '%'] = df_in[comorbidity_col].sum() / len(df_in)

        pasc_col = [x for x in df_in.columns if (x.startswith('flag@'))]
        df_out.loc[pasc_col, 'N'] = df_in[pasc_col].sum()
        df_out.loc[pasc_col, '%'] = df_in[pasc_col].sum() / len(df_in)

        df_out.to_csv(out_file)
        print('Dump done ', out_file)


def screen_all_pasc_category():
    start_time = time.time()
    df = pd.read_csv(r'../data/V15_COVID19/output/character/pasc_count_cohorts_covid_query12_ALL.csv')
    for key, row in tqdm(df.iterrows(), total=len(df)):
        print('Screening pasc specific cohort:', key, row)
        pasc = row[0][5:]
        # pasc = 'Respiratory signs and symptoms'
        pasc_specific_cohorts_characterization_analyse(cohorts='pasc_incidence', dataset='ALL', severity='', pasc=pasc)
        pasc_specific_cohorts_characterization_analyse(cohorts='pasc_prevalence', dataset='ALL', severity='', pasc=pasc)
        pasc_specific_cohorts_characterization_analyse(cohorts='pasc_incidence', dataset='ALL', severity='hospitalized',
                                                       pasc=pasc)
        pasc_specific_cohorts_characterization_analyse(cohorts='pasc_prevalence', dataset='ALL',
                                                       severity='hospitalized', pasc=pasc)
        pasc_specific_cohorts_characterization_analyse(cohorts='pasc_incidence', dataset='ALL',
                                                       severity='not hospitalized', pasc=pasc)
        pasc_specific_cohorts_characterization_analyse(cohorts='pasc_prevalence', dataset='ALL',
                                                       severity='not hospitalized', pasc=pasc)
        pasc_specific_cohorts_characterization_analyse(cohorts='pasc_incidence', dataset='ALL', severity='ventilation',
                                                       pasc=pasc)
        pasc_specific_cohorts_characterization_analyse(cohorts='pasc_prevalence', dataset='ALL', severity='ventilation',
                                                       pasc=pasc)

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


if __name__ == '__main__':
    # python query_12_cdc.py --dataset COL 2>&1 | tee  log/query_12_cdc_COL.txt
    # python query_12_cdc.py --dataset ALL --cohorts pasc_incidence 2>&1 | tee  log/query_12_cdc_ALL_pasc_incidence.txt
    # python query_12_cdc.py --dataset ALL --cohorts pasc_prevalence 2>&1 | tee  log/query_12_cdc_ALL_pasc_prevalence.txt
    # python query_12_cdc.py --dataset ALL --cohorts covid 2>&1 | tee  log/query_12_cdc_ALL_covid.txt
    # python query_12_cdc.py --dataset ALL --cohorts covid 2>&1 | tee  log/query_12_cdc_ALL_covid_withPASCoutcome.txt
    # python query_12_cdc.py --dataset ALL --cohorts covid 2>&1 | tee  log/query_12_cdc_ALL_covid_screenAllPASC.txt

    start_time = time.time()
    args = parse_args()
    # df_data, df_data_bool = build_query_1and2_matrix(args)

    # cohorts_characterization_analyse(cohorts='pasc_incidence', dataset='ALL', severity='')
    # cohorts_characterization_analyse(cohorts='pasc_incidence', dataset='ALL', severity='hospitalized')
    # cohorts_characterization_analyse(cohorts='pasc_incidence', dataset='ALL', severity='not hospitalized')
    # cohorts_characterization_analyse(cohorts='pasc_incidence', dataset='ALL', severity='ventilation')
    # #
    # cohorts_characterization_analyse(cohorts='pasc_prevalence', dataset='ALL', severity='')
    # cohorts_characterization_analyse(cohorts='pasc_prevalence', dataset='ALL', severity='hospitalized')
    # cohorts_characterization_analyse(cohorts='pasc_prevalence', dataset='ALL', severity='not hospitalized')
    # cohorts_characterization_analyse(cohorts='pasc_prevalence', dataset='ALL', severity='ventilation')
    # #
    # cohorts_characterization_analyse(cohorts='covid', dataset='ALL', severity='')
    # cohorts_characterization_analyse(cohorts='covid', dataset='ALL', severity='hospitalized')
    # cohorts_characterization_analyse(cohorts='covid', dataset='ALL', severity='not hospitalized')
    # cohorts_characterization_analyse(cohorts='covid', dataset='ALL', severity='ventilation')

    screen_all_pasc_category()

    # df = pd.read_csv(r'../data/V15_COVID19/output/character/pasc_count_cohorts_covid_query12_ALL.csv')
    # for key, row in tqdm(df.iterrows(), total=len(df)):
    #     print(key, row)
    #     pasc = row[0][5:]
    # # pasc = 'Respiratory signs and symptoms'
    #     pasc_specific_cohorts_characterization_analyse(cohorts='pasc_incidence', dataset='ALL', severity='', pasc=pasc)
    #     pasc_specific_cohorts_characterization_analyse(cohorts='pasc_prevalence', dataset='ALL', severity='', pasc=pasc)
    #     pasc_specific_cohorts_characterization_analyse(cohorts='pasc_incidence', dataset='ALL', severity='hospitalized', pasc=pasc)
    #     pasc_specific_cohorts_characterization_analyse(cohorts='pasc_prevalence', dataset='ALL', severity='hospitalized', pasc=pasc)
    #     pasc_specific_cohorts_characterization_analyse(cohorts='pasc_incidence', dataset='ALL', severity='not hospitalized', pasc=pasc)
    #     pasc_specific_cohorts_characterization_analyse(cohorts='pasc_prevalence', dataset='ALL', severity='not hospitalized', pasc=pasc)
    #     pasc_specific_cohorts_characterization_analyse(cohorts='pasc_incidence', dataset='ALL', severity='ventilation', pasc=pasc)
    #     pasc_specific_cohorts_characterization_analyse(cohorts='pasc_prevalence', dataset='ALL', severity='ventilation', pasc=pasc)

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
