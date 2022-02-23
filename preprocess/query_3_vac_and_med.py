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

from iptw.PSModels import ml
from iptw.evaluation import *


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess demographics')
    parser.add_argument('--cohorts', choices=['pasc_incidence', 'pasc_prevalence', 'covid',
                                              'covid_4screen', 'covid_4screen_Covid+', 'covid_4manuscript'],
                        default='covid', help='cohorts')
    parser.add_argument('--dataset', choices=['COL', 'MSHS', 'MONTE', 'NYU', 'WCM', 'ALL'],
                        default='COL', help='site dataset')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--positive_only', action='store_true')

    args = parser.parse_args()

    args.output_file_query12 = r'../data/V15_COVID19/output/character/query3-covid-drug-and-vaccine-matrix_cohorts_{}_cnt_{}.csv'.format(
        args.cohorts,
        args.dataset)
    args.output_file_query12_bool = r'../data/V15_COVID19/output/character/query3-covid-drug-and-vaccine-matrix_cohorts_{}_bool_{}.csv'.format(
        args.cohorts,
        args.dataset)

    # args.output_med_info = r'../data/V15_COVID19/output/character/info_medication_cohorts_{}_{}.csv'.format(
    #     args.cohorts,
    #     args.dataset)
    #
    # args.output_dx_info = r'../data/V15_COVID19/output/character/info_dx_cohorts_{}_{}.csv'.format(
    #     args.cohorts,
    #     args.dataset)

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

    with open(r'../data/mapping/atcL4_index_mapping.pkl', 'rb') as f:
        atcl4_encoding = pickle.load(f)
        print('Load to ATC-Level-4 to encoding mapping done! len(atcl4_encoding):', len(atcl4_encoding))
        record_example = next(iter(atcl4_encoding.items()))
        print('e.g.:', record_example)

    # with open(r'../data/mapping/atc_rxnorm_mapping.pkl', 'rb') as f:
    #     atc_rxnorm = pickle.load(f)
    #     print('Load ATC to rxRNOM_CUI mapping done! len(atc_rxnorm):', len(atc_rxnorm))
    #     record_example = next(iter(atc_rxnorm.items()))
    #     print('e.g.:', record_example)

    return icd_pasc, pasc_encoding, icd_cmr, cmr_encoding, icd_ccsr, ccsr_encoding, \
           rxnorm_ing, rxnorm_atc, atcl2_encoding, atcl3_encoding, atcl4_encoding


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
    # inpatient status in acute phase
    flag = 0  # False
    for records in dx_list:
        dx_date, icd, dx_type, enc_type = records
        if ecs._is_in_inpatient_period(dx_date, index_date):
            if (enc_type == 'EI') or (enc_type == 'IP') or (enc_type == 'OS'):
                flag = 1  # True
                break

    return flag


def _encoding_ventilation(pro_list, obsgen_list, index_date, vent_codes):
    # ventilation status in acute phase
    flag = 0  # False
    for records in pro_list:
        px_date, px, px_type, enc_type, enc_id = records
        px = px.replace('.', '').upper()
        if ecs._is_in_ventilation_period(px_date, index_date):
            if px in vent_codes:
                flag = 1  # True
                break

    for records in obsgen_list:
        px_date, px, px_type, result_text, source, enc_id = records
        if ecs._is_in_ventilation_period(px_date, index_date):
            # OBSGEN_TYPE=”PC_COVID” & OBSGEN_CODE = 3000 & OBSGEN_SOURCE=”DR” & RESULT_TEXT=”Y”
            if (px_type == 'PC_COVID') and (px == '3000') and (source == 'DR') and (result_text == 'Y'):
                flag = 1 # True
                break

    return flag


def _encoding_critical_care(pro_list, index_date, cc_codes={"99291", "99292"}):
    # critical_care status in acute phase
    flag = 0  # False
    for records in pro_list:
        px_date, px, px_type, enc_type, enc_id = records
        px = px.replace('.', '').upper()
        if ecs._is_in_inpatient_period(px_date, index_date):
            if px in cc_codes:
                flag = 1  # True
                break

    return flag


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
    # encoding uitlization in the baseline period
    # ['inpatient visits', 'outpatient visits', 'emergency visits', 'other visits']
    encoding = np.zeros((1, 4), dtype='float')
    for records in enc_list:
        enc_date, type, enc_id = records
        if ecs._is_in_baseline(enc_date, index_date):
            if type == 'EI' or type == 'IP' or type == 'OS':
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
            encoding_update[0, 0 + i * 4] = 1
        elif encoding[0, i] <= 2:
            encoding_update[0, 1 + i * 4] = 1
        elif encoding[0, i] <= 4:
            encoding_update[0, 2 + i * 4] = 1
        else:
            encoding_update[0, 3 + i * 4] = 1

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


def _encoding_covidmed(med_list, pro_list, covidmed_column_names, covidmed_codes, index_date):
    encoding = np.zeros((1, len(covidmed_column_names)), dtype='int')
    for records in med_list:
        med_date, rxnorm, supply_days = records
        if ecs._is_in_covid_medication(med_date, index_date):
            for pos, col_name in enumerate(covidmed_column_names):
                if col_name in covidmed_codes:
                    if rxnorm in covidmed_codes[col_name]:
                        encoding[0, pos] += 1

    for records in pro_list:
        px_date, px, px_type, enc_type, enc_id = records
        if ecs._is_in_covid_medication(px_date, index_date):
            for pos, col_name in enumerate(covidmed_column_names):
                if col_name in covidmed_codes:
                    if px in covidmed_codes[col_name]:
                        encoding[0, pos] += 1
    return encoding


def _encoding_vaccine(pro_list, immun_list, _vaccine_column_names, vaccine_codes, index_date):
    encoding_pre = np.zeros((1, len(_vaccine_column_names)), dtype='int')
    encoding_post = np.zeros((1, len(_vaccine_column_names)), dtype='int')
    for records in pro_list:
        px_date, px, px_type, enc_type, enc_id = records
        if px_date < index_date:
            for pos, col_name in enumerate(_vaccine_column_names):
                if col_name in vaccine_codes:
                    if px in vaccine_codes[col_name]:
                        encoding_pre[0, pos] += 1
        else:
            for pos, col_name in enumerate(_vaccine_column_names):
                if col_name in vaccine_codes:
                    if px in vaccine_codes[col_name]:
                        encoding_post[0, pos] += 1

    def _is_duplicate(px_date, codes_under_same_category):
        # same date and same vaccine type, then treat as duplicate
        for records in pro_list:
            _px_date, _px, _px_type, _enc_type, _enc_id = records
            if (_px_date == px_date) and (_px in codes_under_same_category):
                return True
        return False

    for records in immun_list:
        px_date, px, px_type, enc_id = records
        if px_date < index_date:
            for pos, col_name in enumerate(_vaccine_column_names):
                if col_name in vaccine_codes:
                    if px in vaccine_codes[col_name]:  # should add de-duplicate codes
                        if not _is_duplicate(px_date, vaccine_codes[col_name]):
                            encoding_pre[0, pos] += 1
        else:
            for pos, col_name in enumerate(_vaccine_column_names):
                if col_name in vaccine_codes:
                    if px in vaccine_codes[col_name]:
                        if not _is_duplicate(px_date, vaccine_codes[col_name]):
                            encoding_post[0, pos] += 1

    return encoding_pre, encoding_post


def _encoding_maxfollowtime(index_date, enc, dx, med):
    # encode maximum followup in database
    if enc:
        maxfollowtime = (enc[-1][0] - index_date).days
    elif dx:
        maxfollowtime = (dx[-1][0] - index_date).days
    elif med:
        maxfollowtime = (med[-1][0] - index_date).days
    else:
        # by definition, should be >= followup-left
        # but may above condition give < follow-left results, thus, need to deal with this in defining t2e
        maxfollowtime = ecs.FOLLOWUP_LEFT

    return maxfollowtime


def _encoding_death(death, index_date):
    # death flag, death time
    encoding = np.zeros((1, 2), dtype='int')
    if death:
        encoding[0, 0] = 1
        ddate = death[0]
        if pd.notna(ddate):
            encoding[0, 1] = (ddate - index_date).days
        else:
            encoding[0, 1] = 9999
    else:
        encoding[0, 1] = 9999

    return encoding


def _encoding_outcome_dx(dx_list, icd_pasc, pasc_encoding, index_date, default_t2e):
    # encoding 137 outcomes from our PASC list
    # outcome_t2e = np.zeros((n, 137), dtype='int')
    # outcome_flag = np.zeros((n, 137), dtype='int')
    # outcome_baseline = np.zeros((n, 137), dtype='int')

    # 2022-02-18 initialize t2e:  last encounter, event, end of followup, whichever happens first
    outcome_t2e = np.ones((1, len(pasc_encoding)), dtype='float') * default_t2e
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
                    if days < outcome_t2e[0, pos]:
                        outcome_t2e[0, pos] = days
                    outcome_flag[0, pos] = 1
                else:
                    outcome_flag[0, pos] += 1
    # debug # a = pd.DataFrame({'1':pasc_encoding.keys(), '2':outcome_flag.squeeze(), '3':outcome_t2e.squeeze(), '4':outcome_baseline.squeeze()})
    return outcome_flag, outcome_t2e, outcome_baseline


def _rxnorm_to_atc(rxnorm, rxnorm_ing, rxnorm_atc, atc_level):
    assert atc_level in [1, 2, 3, 4, 5]
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


def _encoding_outcome_med(med_list, rxnorm_ing, rxnorm_atc, atcl_encoding, index_date, default_t2e, atc_level=3,
                          verbose=0):
    # mapping rxnorm_cui to its ingredient(s)
    # for each ingredient, mapping to atc and thus atc[:3] is level three
    # med_array = np.zeros((n, 2), dtype='int')  # atc level 3 category
    # atc l3, 269 codes
    outcome_t2e = np.ones((1, len(atcl_encoding)), dtype='float') * default_t2e
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
        if ecs._is_in_medication_baseline(med_date, index_date):  # 2022-02-17 USE 1 YEAR FOR MEDICATION
            for pos in pos_list:
                outcome_baseline[0, pos] += 1

        # build outcome
        if ecs._is_in_followup(med_date, index_date):
            days = (med_date - index_date).days
            for pos in pos_list:
                if outcome_flag[0, pos] == 0:
                    if days < outcome_t2e[0, pos]:
                        outcome_t2e[0, pos] = days
                    outcome_flag[0, pos] = 1
                else:
                    outcome_flag[0, pos] += 1
    # debug # a = pd.DataFrame({'atc':atcl_encoding.keys(), 'name':atcl_encoding.values(), 'outcome_flag':outcome_flag.squeeze(), 'outcome_t2e':outcome_t2e.squeeze(), 'outcome_baseline':outcome_baseline.squeeze()})
    return outcome_flag, outcome_t2e, outcome_baseline


def _update_counter(dict3, key, flag):
    if key in dict3:
        dict3[key][0] += 1
    else:
        # total, pos, negative,
        dict3[key] = [1, 0, 0]

    if flag:
        dict3[key][1] += 1
    else:
        dict3[key][2] += 1


def build_query_1and2_matrix(args):
    start_time = time.time()
    print('In build_query_1and2_matrix...')
    # step 1: load encoding dictionary
    # icd_pasc, pasc_encoding, icd_cmr, cmr_encoding, \
    # icd_ccsr, ccsr_encoding, rxnorm_ing, rxnorm_atc, atcl2_encoding, atcl3_encoding = _load_mapping()

    icd_pasc, pasc_encoding, icd_cmr, cmr_encoding, \
    icd_ccsr, ccsr_encoding, rxnorm_ing, rxnorm_atc, atcl2_encoding, atcl3_encoding, atcl4_encoding = _load_mapping()

    ventilation_codes = utils.load(r'../data/mapping/ventilation_codes.pkl')
    comorbidity_codes = utils.load(r'../data/mapping/tailor_comorbidity_codes.pkl')

    covidmed_codes = utils.load(r'../data/mapping/query3_medication_codes.pkl')
    vaccine_codes = utils.load(r'../data/mapping/query3_vaccine_codes.pkl')

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
        if args.positive_only:
            n = 0
            for i, (pid, item) in tqdm(enumerate(id_data.items()), total=len(id_data)):
                index_info, demo, dx, med, covid_lab, enc, procedure, obsgen, immun, death = item
                flag, index_date, covid_loinc, flag_name, index_age_year, index_enc_id = index_info
                if flag:
                    n += 1
        else:
            n = len(id_data)

        print('args.positive_only:', args.positive_only, n)

        pid_list = []
        site_list = []
        covid_list = []
        indexdate_list = []  # newly add 2022-02-20
        hospitalized_list = []
        ventilation_list = []
        criticalcare_list = []

        maxfollowtime_list = []  # newly add 2022-02-18
        death_array = np.zeros((n, 2), dtype='int16')  # newly add 2022-02-20
        death_column_names = ['death', 'death t2e']

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

        # newly added 2022-02-18
        social_array = np.zeros((n, 10), dtype='int16')
        social_column_names = ['ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49',
                               'ADI50-59', 'ADI60-69', 'ADI70-79', 'ADI80-89', 'ADI90-100']
        utilization_array = np.zeros((n, 12), dtype='int16')
        utilization_column_names = ['inpatient visits 0', 'inpatient visits 1-2', 'inpatient visits 3-4',
                                    'inpatient visits >=5',
                                    'outpatient visits 0', 'outpatient visits 1-2', 'outpatient visits 3-4',
                                    'outpatient visits >=5',
                                    'emergency visits 0', 'emergency visits 1-2', 'emergency visits 3-4',
                                    'emergency visits >=5']

        index_period_array = np.zeros((n, 5), dtype='int16')
        index_period_names = ['03/20-06/20', '07/20-10/20', '11/20-02/21', '03/21-06/21', '07/21-11/21']
        #

        yearmonth_array = np.zeros((n, 23), dtype='int16')
        yearmonth_column_names = [
            "YM: March 2020", "YM: April 2020", "YM: May 2020", "YM: June 2020", "YM: July 2020",
            "YM: August 2020", "YM: September 2020", "YM: October 2020", "YM: November 2020", "YM: December 2020",
            "YM: January 2021", "YM: February 2021", "YM: March 2021", "YM: April 2021", "YM: May 2021",
            "YM: June 2021", "YM: July 2021", "YM: August 2021", "YM: September 2021", "YM: October 2021",
            "YM: November 2021", "YM: December 2021", "YM: January 2022"
        ]
        # cautious of "DX: Hypertension and Type 1 or 2 Diabetes Diagnosis" using logic afterwards, due to threshold >= 2 issue
        # DX: End Stage Renal Disease on Dialysis   Both diagnosis and procedure codes used to define this condtion
        dx_array = np.zeros((n, 37), dtype='int16')
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
                           "DX: Severe Obesity  (BMI>=40 kg/m2)", "DX: Weight Loss",
                           "DX: Down's Syndrome", 'DX: Other Substance Abuse', 'DX: Cystic Fibrosis',
                           'DX: Autism', 'DX: Sickle Cell'
                           ]
        #
        med_array = np.zeros((n, 2), dtype='int16')
        # atc level 3 category # H02: CORTICOSTEROIDS FOR SYSTEMIC USE   L04:IMMUNOSUPPRESSANTS
        # --> detailed code list from CDC
        med_column_names = ["MEDICATION: Corticosteroids", "MEDICATION: Immunosuppressant drug", ]

        # add covid medication
        covidmed_array = np.zeros((n, 25), dtype='int16')
        covidmed_column_names = [
            'Anti-platelet Therapy', 'Aspirin', 'Baricitinib', 'Bamlanivimab Monoclonal Antibody Treatment',
            'Bamlanivimab and Etesevimab Monoclonal Antibody Treatment',  'Casirivimab and Imdevimab Monoclonal Antibody Treatment',
            'Any Monoclonal Antibody Treatment (Bamlanivimab, Bamlanivimab and Etesevimab, Casirivimab and Imdevimab, Sotrovimab, and unspecified monoclonal antibodies)',
            'Colchicine', 'Corticosteroids', 'Dexamethasone', 'Factor Xa Inhibitors', 'Fluvoxamine', 'Heparin',
            'Inhaled Steroids', 'Ivermectin', 'Low Molecular Weight Heparin', 'Molnupiravir', 'Nirmatrelvir',
            'Paxlovid', 'Remdesivir', 'Ritonavir', 'Sotrovimab Monoclonal Antibody Treatment',
            'Thrombin Inhibitors', 'Tocilizumab (Actemra)', 'PX: Convalescent Plasma']

        # add vaccine status
        _vaccine_column_names = [
            'pfizer_first', 'pfizer_second', 'pfizer_third', 'pfizer_booster',
            'moderna_first', 'moderna_second', 'moderna_booster',
            'janssen_first', 'janssen_booster',
            'px_pfizer', 'imm_pfizer',
            'px_moderna', 'imm_moderna',
            'px_janssen', 'imm_janssen',
            'vax_unspec',
            'pfizer_any', 'moderna_any', 'janssen_any', 'any_mrna']
        vaccine_preindex_array = np.zeros((n, 20), dtype='int16')
        vaccine_postindex_array = np.zeros((n, 20), dtype='int16')
        vaccine_preindex_column_names = ['pre_' + x for x in _vaccine_column_names]
        vaccine_postindex_column_names = ['post_' + x for x in _vaccine_column_names]



        # Build PASC outcome t2e and flag in follow-up, and outcome flag in baseline for dynamic cohort selection
        # In total, there are 137 PASC categories in our lists. See T2E later
        outcome_flag = np.zeros((n, 137), dtype='int16')
        # outcome_t2e = np.zeros((n, 137), dtype='int16')
        outcome_baseline = np.zeros((n, 137), dtype='int16')
        # outcome_column_names = ['dx-out@' + x for x in pasc_encoding.keys()] + \
        #                        ['dx-t2e@' + x for x in pasc_encoding.keys()] + \
        #                        ['dx-base@' + x for x in pasc_encoding.keys()]
        outcome_column_names = ['dx-out@' + x for x in pasc_encoding.keys()] + \
                               ['dx-base@' + x for x in pasc_encoding.keys()]

        # atcl2 outcome. time 2 event is not good for censoring or negative. update later
        # outcome_med_flag = np.zeros((n, 269), dtype='int16')
        # outcome_med_t2e = np.zeros((n, 269), dtype='int16')
        # outcome_med_baseline = np.zeros((n, 269), dtype='int16')
        # outcome_med_column_names = ['med-out@' + x for x in atcl3_encoding.keys()] + \
        #                            ['med-t2e@' + x for x in atcl3_encoding.keys()] + \
        #                            ['med-base@' + x for x in atcl3_encoding.keys()]

        # outcome_med_t2e = np.zeros((n, 909), dtype='int16')
        # outcome_med_flag = np.zeros((n, 909), dtype='int16')
        # outcome_med_baseline = np.zeros((n, 909), dtype='int16')
        # outcome_med_column_names = ['atc@' + x for x in atcl4_encoding.keys()] + \
        #                            ['atct2e@' + x for x in atcl4_encoding.keys()] + \
        #                            ['atcbase@' + x for x in atcl4_encoding.keys()]

        # column_names = ['patid', 'site', 'covid', 'index date', 'hospitalized',
        #                 'ventilation', 'criticalcare', 'maxfollowup'] + death_column_names + age_column_names + \
        #                gender_column_names + race_column_names + hispanic_column_names + \
        #                social_column_names + utilization_column_names + index_period_names + yearmonth_column_names + \
        #                dx_column_names + med_column_names + \
        #                covidmed_column_names + vaccine_preindex_column_names + vaccine_postindex_column_names + \
        #                outcome_column_names

        column_names = ['patid', 'site', 'covid', 'index date', 'hospitalized',
                        'ventilation', 'criticalcare'] + \
                       covidmed_column_names + vaccine_preindex_column_names + vaccine_postindex_column_names + \
                       outcome_column_names

        print('len(column_names):', len(column_names), '\n', column_names)
        # impute adi value by median of site , per site:
        adi_value_list = [v[1][7] for key, v in id_data.items()]
        adi_value_default = np.nanmedian(adi_value_list)

        med_count = {}
        dx_count = {}

        i = -1
        for pid, item in tqdm(id_data.items(), total=len(
                id_data), mininterval=10):  # for i, (pid, item) in tqdm(enumerate(id_data.items()), total=len(id_data)):
            index_info, demo, dx, med, covid_lab, enc, procedure, obsgen, immun = item
            flag, index_date, covid_loinc, flag_name, index_age_year, index_enc_id = index_info
            birth_date, gender, race, hispanic, zipcode, state, city, nation_adi, state_adi = demo

            if args.positive_only:
                if not flag:
                    continue
            i += 1

            # maxfollowtime
            # gaurantee at least one encounter in baseline or followup. thus can be 0 if no followup
            # later EC should be at lease one in follow-up

            pid_list.append(pid)
            site_list.append(site)
            covid_list.append(flag)
            indexdate_list.append(index_date)

            inpatient_flag = _encoding_inpatient(dx, index_date)
            hospitalized_list.append(inpatient_flag)

            vent_flag = _encoding_ventilation(procedure, obsgen, index_date, ventilation_codes)
            ventilation_list.append(vent_flag)

            criticalcare_flag = _encoding_critical_care(procedure, index_date)
            criticalcare_list.append(criticalcare_flag)

            maxfollowtime = _encoding_maxfollowtime(index_date, enc, dx, med)
            maxfollowtime_list.append(maxfollowtime)

            # encode death
            # death_array[i, :] = _encoding_death(death, index_date)

            # # encoding query 1 information
            # age_array[i, :] = _encoding_age(index_age_year)
            # gender_array[i] = _encoding_gender(gender)
            # race_array[i, :] = _encoding_race(race)
            # hispanic_array[i, :] = _encoding_hispanic(hispanic)
            #
            # social_array[i, :] = _encoding_social(nation_adi, adi_value_default)
            # utilization_array[i, :] = _encoding_utilization(enc, index_date)
            # index_period_array[i, :] = _encoding_index_period(index_date)
            # #
            # yearmonth_array[i, :] = _encoding_yearmonth(index_date)
            #
            # # encoding query 2 information
            # dx_array[i, :] = _encoding_dx(dx, dx_column_names, comorbidity_codes, index_date, procedure)
            # med_array[i, :] = _encoding_med(med, med_column_names, comorbidity_codes, index_date)

            # encoding query 3 covid medication
            # if pid == '1043542':
            #     print(pid)
            covidmed_array[i, :] = _encoding_covidmed(med, procedure, covidmed_column_names, covidmed_codes, index_date)
            vaccine_preindex_array[i, :], vaccine_postindex_array[i, :] = _encoding_vaccine(procedure, immun, _vaccine_column_names, vaccine_codes, index_date)
            # encoding pasc information in both baseline and followup
            # time 2 event: censoring in the database (should be >= followup start time),
            # maximum follow-up, death-time, event, whichever comes first
            default_t2e = np.min([
                np.maximum(ecs.FOLLOWUP_LEFT, maxfollowtime),
                np.maximum(ecs.FOLLOWUP_LEFT, ecs.FOLLOWUP_RIGHT),
                np.maximum(ecs.FOLLOWUP_LEFT, death_array[i, 1])
            ])
            outcome_flag[i, :], _, outcome_baseline[i, :] = \
                _encoding_outcome_dx(dx, icd_pasc, pasc_encoding, index_date, default_t2e)
            # outcome_flag[i, :], outcome_t2e[i, :], outcome_baseline[i, :] = \
            #     _encoding_outcome_dx(dx, icd_pasc, pasc_encoding, index_date, default_t2e)
            # later use selected ATCl4, because too high dim
            # outcome_med_flag[i, :], outcome_med_t2e[i, :], outcome_med_baseline[i, :] =
            # _encoding_outcome_med(med, rxnorm_ing, rxnorm_atc, atcl4_encoding, index_date, default_t2e, atc_level=4)
            # outcome_med_flag[i, :], outcome_med_t2e[i, :], outcome_med_baseline[i, :] = \
            #     _encoding_outcome_med(med, rxnorm_ing, rxnorm_atc, atcl3_encoding, index_date, default_t2e, atc_level=3)

            # count additional information
            # in follow-up, each person count once
            _dx_set = set()
            for i_dx in dx:
                dx_t = i_dx[0]
                icd = i_dx[1].replace('.', '').upper()
                if ecs._is_in_followup(dx_t, index_date):
                    _dx_set.add(icd)

            for i_dx in _dx_set:
                _update_counter(dx_count, i_dx, flag)

            _med_set = set()
            for i_med in med:
                t = i_med[0]
                rx = i_med[1]
                if ecs._is_in_followup(t, index_date):
                    if rx in rxnorm_ing:
                        _added = rxnorm_ing[rx]
                        _med_set.update(_added)
                    else:
                        _med_set.add(rx)

            for i_med in _med_set:
                _update_counter(med_count, i_med, flag)

        print('Encoding done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

        #   step 4: build pandas, column, and dump
        data_array = np.hstack((np.asarray(pid_list).reshape(-1, 1),
                                np.asarray(site_list).reshape(-1, 1),
                                np.array(covid_list).reshape(-1, 1).astype(int),
                                np.asarray(indexdate_list).reshape(-1, 1),
                                np.asarray(hospitalized_list).reshape(-1, 1).astype(int),
                                np.array(ventilation_list).reshape(-1, 1).astype(int),
                                np.array(criticalcare_list).reshape(-1, 1).astype(int),
                                # np.array(maxfollowtime_list).reshape(-1, 1),
                                # death_array,
                                # age_array,
                                # gender_array,
                                # race_array,
                                # hispanic_array,
                                # social_array,
                                # utilization_array,
                                # index_period_array,
                                # yearmonth_array,
                                # dx_array,
                                # med_array,
                                covidmed_array,
                                vaccine_preindex_array,
                                vaccine_postindex_array,
                                outcome_flag,
                                outcome_baseline,
                                ))

        df_data = pd.DataFrame(data_array, columns=column_names)
        data_all_sites.append(df_data)

        print('df_data.shape:', df_data.shape)
        del id_data
        print('Done site:', site)
        # end iterate sites

    # dx_count_df = pd.DataFrame.from_dict(dx_count, orient='index',
    #                                      columns=['total', 'no. in positive group', 'no. in negative group'])
    # dx_count_df.to_csv(args.output_dx_info)
    # med_count_df = pd.DataFrame.from_dict(med_count, orient='index',
    #                                       columns=['total', 'no. in positive group', 'no. in negative group'])
    # med_count_df.to_csv(args.output_med_info)

    df_data_all_sites = pd.concat(data_all_sites)
    print('df_data_all_sites.shape:', df_data_all_sites.shape)

    utils.check_and_mkdir(args.output_file_query12)
    df_data_all_sites.to_csv(args.output_file_query12)
    print('Done! Dump data matrix for query12 to {}'.format(args.output_file_query12))

    # transform count to bool with threshold 2, and deal with "DX: Hypertension and Type 1 or 2 Diabetes Diagnosis"
    # df_bool = df_data_all_sites.copy()  # not using deep copy for the sage of time
    df_bool = df_data_all_sites
    # selected_cols = [x for x in df_bool.columns if (x.startswith('DX:') or x.startswith('MEDICATION:'))]
    # df_bool.loc[:, selected_cols] = (df_bool.loc[:, selected_cols].astype('int') >= 2).astype('int')
    # df_bool.loc[:, r"DX: Hypertension and Type 1 or 2 Diabetes Diagnosis"] = \
    #     (df_bool.loc[:, r'DX: Hypertension'] & (
    #             df_bool.loc[:, r'DX: Diabetes Type 1'] | df_bool.loc[:, r'DX: Diabetes Type 2'])).astype('int')

    # keep the value of baseline count and outcome count in the file, filter later depends on the application
    selected_cols = [x for x in df_bool.columns if
                     (x.startswith('dx-out@') or x.startswith('dx-base@') or
                      x.startswith('med-out@') or x.startswith('med-base@'))]
    df_bool.loc[:, selected_cols] = (df_bool.loc[:, selected_cols].astype('int') >= 1).astype('int')

    utils.check_and_mkdir(args.output_file_query12_bool)
    df_bool.to_csv(args.output_file_query12_bool)
    print('Done! Dump data bool matrix for query12 to {}'.format(args.output_file_query12_bool))

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return df_data_all_sites, df_bool


if __name__ == '__main__':
    # python query_3_vac_and_med.py --dataset ALL --cohorts covid 2>&1 | tee  log/query_3_vac_and_med.txt

    start_time = time.time()
    args = parse_args()
    df_data, df_data_bool = build_query_1and2_matrix(args)

    # in_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuscript_bool_ALL.csv'
    # df_data = pd.read_csv(in_file, dtype={'patid': str}, parse_dates=['index date'])

    # cohorts_table_generation(args)
    # de_novo_medication_analyse(cohorts='covid_4screen_Covid+', dataset='ALL', severity='')
    # de_novo_medication_analyse_selected_and_iptw(cohorts='covid_4screen_Covid+', dataset='ALL', severity='')

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
