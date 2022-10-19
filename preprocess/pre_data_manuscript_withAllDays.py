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
# import matplotlib.pyplot as plt
# import seaborn as sns
from collections import Counter
import datetime
from misc import utils
import eligibility_setting as ecs
import functools
import fnmatch
# from lifelines import KaplanMeierFitter, CoxPHFitter

print = functools.partial(print, flush=True)

from iptw.PSModels import ml
from iptw.evaluation import *


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess demographics')
    parser.add_argument('--cohorts', choices=['pasc_incidence', 'pasc_prevalence', 'covid',
                                              'covid_4screen', 'covid_4screen_Covid+',
                                              'covid_4manuscript', 'covid_4manuNegNoCovid', 'covid_4manuNegNoCovidV2'],
                        default='covid_4manuNegNoCovidV2', help='cohorts')
    parser.add_argument('--dataset', choices=['COL', 'MSHS', 'MONTE', 'NYU', 'WCM', 'ALL'],
                        default='COL', help='site dataset')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--positive_only', action='store_true')
    # parser.add_argument("--ndays", type=int, default=30)


    args = parser.parse_args()

    # args.output_file_query12 = r'../data/V15_COVID19/output/character/matrix_cohorts_{}_cnt_AnyPASC-withAllDays_{}.csv'.format(
    #     args.cohorts,
    #     args.dataset)
    args.output_file_query12_bool = r'../data/V15_COVID19/output/character/matrix_cohorts_{}_boolbase-nout_AnyPASC-withAllDays_{}.csv'.format(
        args.cohorts,
        args.dataset)

    # args.output_med_info = r'../data/V15_COVID19/output/character/info_medication_cohorts_{}_{}-V2_2dx{}daysAnyPASC-Cross.csv'.format(
    #     args.cohorts,
    #     args.ndays,
    #     args.dataset)
    #
    # args.output_dx_info = r'../data/V15_COVID19/output/character/info_dx_cohorts_{}_{}-V2_2dx{}daysAnyPASC-Cross.csv'.format(
    #     args.cohorts,
    #     args.ndays,
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

    # change 2022-02-28, previous ingredient first, current moiety first
    with open(r'../data/mapping/rxnorm_ingredient_mapping_combined_moietyfirst.pkl', 'rb') as f:
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

    ventilation_codes = utils.load(r'../data/mapping/ventilation_codes.pkl')
    comorbidity_codes = utils.load(r'../data/mapping/tailor_comorbidity_codes.pkl')
    icd9_icd10 = utils.load(r'../data/mapping/icd9_icd10.pkl')
    rxing_index = utils.load(r'../data/mapping/selected_rxnorm_index.pkl')
    covidmed_codes = utils.load(r'../data/mapping/query3_medication_codes.pkl')
    vaccine_codes = utils.load(r'../data/mapping/query3_vaccine_codes.pkl')

    return icd_pasc, pasc_encoding, icd_cmr, cmr_encoding, icd_ccsr, ccsr_encoding, \
           rxnorm_ing, rxnorm_atc, atcl2_encoding, atcl3_encoding, atcl4_encoding, \
           ventilation_codes, comorbidity_codes, icd9_icd10, rxing_index, covidmed_codes, vaccine_codes


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


def _encoding_bmi_and_smoking(vital_list, index_date):
    # bmi_array = np.zeros((n, 5), dtype='int16')
    # bmi_names = ['BMI: <18.5 under weight', 'BMI: 18.5-<25 normal weight', 'BMI: 25-<30 overweight ', 'BMI: >=30 obese ', 'BMI: missing']
    # smoking_array = np.zeros((n, 4), dtype='int16')
    # smoking_names = ['Smoker: never', 'Smoker: current', 'Smoker: former', 'Smoker: missing']

    ht_select = wt_select = bmi_select = smoking_select = tobacco_select = np.nan
    encoding_bmi = np.zeros((1, 5), dtype='int')
    encoding_smoke = np.zeros((1, 4), dtype='int')
    for records in vital_list:
        dx_date, ht, wt, ori_bmi, smoking, tobacco = records
        if ecs._is_in_bmi_period(dx_date, index_date):
            if pd.notna(ht):
                ht_select = ht
            if pd.notna(wt):
                wt_select = wt
            if pd.notna(ori_bmi):
                bmi_select = ori_bmi

        if ecs._is_in_smoke_period(dx_date, index_date):
            if pd.notna(smoking):
                smoking_select = smoking
            if pd.notna(tobacco):
                tobacco_select = tobacco

    if pd.notna(ht_select) and pd.notna(wt_select) and utils.isfloat(ht_select) and utils.isfloat(wt_select) \
            and (ht_select > 0) and (wt_select > 0):
        bmi = wt_select / (ht_select * ht_select) * 703.069
    elif pd.notna(bmi_select):
        bmi = bmi_select
    else:
        bmi = np.nan

    # ['BMI: <18.5 under weight', 'BMI: 18.5-<25 normal weight', 'BMI: 25-<30 overweight ',
    # 'BMI: >=30 obese ', 'BMI: missing']
    if pd.notna(bmi):
        if bmi < 18.5:
            encoding_bmi[0, 0] = 1
        elif bmi < 25:
            encoding_bmi[0, 1] = 1
        elif bmi < 30:
            encoding_bmi[0, 2] = 1
        else:
            encoding_bmi[0, 3] = 1
    else:
        encoding_bmi[0, 4] = 1

    # 'Smoker: never', 'Smoker: current', 'Smoker: former', 'Smoker: missing'
    if smoking_select == '04' or tobacco_select == '02':
        encoding_smoke[0, 0] = 1
    elif (smoking_select in ('01', '02', '07', '08', '05')) or (tobacco_select == '01'):
        encoding_smoke[0, 1] = 1
    elif smoking_select == '03' or tobacco_select == '03':
        encoding_smoke[0, 2] = 1
    else:
        encoding_smoke[0, 3] = 1

    return encoding_bmi, encoding_smoke, bmi


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
                flag = 1  # True
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

    return encoding_update, encoding


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
    # if code in code_set:
    #     return True
    # updated 2022-04-13  from exact match to prefix match
    for i in range(len(code)):
        pre = code[:(len(code) - i)]
        if pre in code_set:
            # if pre != code:
            #     print(pre, code)
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


def _encoding_covidmed(med_list, pro_list, covidmed_column_names, covidmed_codes, index_date, default_t2e):
    # in -14 -- 14 days
    encoding = np.zeros((1, len(covidmed_column_names)), dtype='int')

    # also consider these drugs as outcome setting
    outcome_t2e = np.ones((1, len(covidmed_column_names)), dtype='float') * default_t2e
    outcome_flag = np.zeros((1, len(covidmed_column_names)), dtype='int')
    outcome_baseline = np.zeros((1, len(covidmed_column_names)), dtype='int')

    for records in med_list:
        med_date, rxnorm, supply_days = records
        if ecs._is_in_covid_medication(med_date, index_date):
            for pos, col_name in enumerate(covidmed_column_names):
                if col_name in covidmed_codes:
                    if rxnorm in covidmed_codes[col_name]:
                        encoding[0, pos] += 1

        if ecs._is_in_medication_baseline(med_date, index_date):
            for pos, col_name in enumerate(covidmed_column_names):
                if col_name in covidmed_codes:
                    if rxnorm in covidmed_codes[col_name]:
                        outcome_baseline[0, pos] += 1

        if ecs._is_in_followup(med_date, index_date):
            days = (med_date - index_date).days
            for pos, col_name in enumerate(covidmed_column_names):
                if col_name in covidmed_codes:
                    if rxnorm in covidmed_codes[col_name]:
                        if outcome_flag[0, pos] == 0:
                            if days < outcome_t2e[0, pos]:
                                outcome_t2e[0, pos] = days
                            outcome_flag[0, pos] = 1
                        else:
                            outcome_flag[0, pos] += 1

    for records in pro_list:
        px_date, px, px_type, enc_type, enc_id = records
        if ecs._is_in_covid_medication(px_date, index_date):
            for pos, col_name in enumerate(covidmed_column_names):
                if col_name in covidmed_codes:
                    if px in covidmed_codes[col_name]:
                        encoding[0, pos] += 1

        if ecs._is_in_medication_baseline(px_date, index_date):
            for pos, col_name in enumerate(covidmed_column_names):
                if col_name in covidmed_codes:
                    if px in covidmed_codes[col_name]:
                        outcome_baseline[0, pos] += 1

        if ecs._is_in_followup(px_date, index_date):
            days = (px_date - index_date).days
            for pos, col_name in enumerate(covidmed_column_names):
                if col_name in covidmed_codes:
                    if px in covidmed_codes[col_name]:
                        if outcome_flag[0, pos] == 0:
                            if days < outcome_t2e[0, pos]:
                                outcome_t2e[0, pos] = days
                            outcome_flag[0, pos] = 1
                        else:
                            outcome_flag[0, pos] += 1
    return encoding, outcome_flag, outcome_t2e, outcome_baseline


def _encoding_vaccine_4risk(pro_list, immun_list, vaccine_column_names, vaccine_codes, index_date):
    # modified from _encoding_vaccineV2  for risk factors
    # 2022 May 25
    # extended fully definition:
    # fully: mrna: 2 vacs, or evidence of sencond or booster
    # check vaccine status before baseline, or
    # check vaccine until followup end
    # https://www.cdc.gov/coronavirus/2019-ncov/vaccines/stay-up-to-date.html

    # vaccineV2_column_names = ['Fully vaccinated - Pre-index',
    #                           'Fully vaccinated - Post-index',
    #                           'Partially vaccinated - Pre-index',
    #                           'Partially vaccinated - Post-index',
    #                           'No evidence - Pre-index',
    #                           'No evidence - Post-index',
    #                           ]

    # vaccine_aux_column_names = [
    #     'pfizer_first', 'pfizer_second', 'pfizer_third', 'pfizer_booster',
    #     'moderna_first', 'moderna_second', 'moderna_booster',
    #     'janssen_first', 'janssen_booster',
    #     'px_pfizer', 'imm_pfizer',
    #     'px_moderna', 'imm_moderna',
    #     'px_janssen', 'imm_janssen',
    #     'vax_unspec',
    #     'pfizer_any', 'moderna_any', 'janssen_any', 'any_mrna']

    mrna = []
    jj = []
    only_2more = []

    encoding = np.zeros((1, len(vaccine_column_names)), dtype='int')
    for records in pro_list:
        px_date, px, px_type, enc_type, enc_id = records
        if px in vaccine_codes['any_mrna']:
            mrna.append((px_date, px))
        elif px in vaccine_codes['janssen_any']:
            jj.append((px_date, px))

        if (px in vaccine_codes['pfizer_second']) or (px in vaccine_codes['pfizer_third']) or \
                (px in vaccine_codes['pfizer_booster']) or \
                (px in vaccine_codes['moderna_second']) or (px in vaccine_codes['moderna_booster']) or \
                (px in vaccine_codes['janssen_booster']):
            only_2more.append((px_date, px))

    for records in immun_list:
        px_date, px, px_type, enc_id = records
        if px in vaccine_codes['any_mrna']:
            mrna.append((px_date, px))
        elif px in vaccine_codes['janssen_any']:
            jj.append((px_date, px))

        if (px in vaccine_codes['pfizer_second']) or (px in vaccine_codes['pfizer_third']) or \
                (px in vaccine_codes['pfizer_booster']) or \
                (px in vaccine_codes['moderna_second']) or (px in vaccine_codes['moderna_booster']) or \
                (px in vaccine_codes['janssen_booster']):
            only_2more.append((px_date, px))

    mrna = sorted(set(mrna), key=lambda x: x[0])
    jj = sorted(set(jj), key=lambda x: x[0])
    only_2more = sorted(set(only_2more), key=lambda x: x[0])

    # change definition of POST. all inform before end of follow up
    mrna_pre = [x for x in mrna if (x[0] - index_date).days <= 0]
    mrna_post = [x for x in mrna if ((x[0] - index_date).days <= 180)]
    jj_pre = [x for x in jj if (x[0] - index_date).days <= 0]
    jj_post = [x for x in jj if ((x[0] - index_date).days <= 180)]
    only_2more_pre = [x for x in only_2more if (x[0] - index_date).days <= 0]
    only_2more_post = [x for x in only_2more if ((x[0] - index_date).days <= 180)]

    def _fully_vaccined_mrna(vlist, only2more):
        if (len(vlist) >= 2) and ((vlist[-1][0] - vlist[0][0]).days > 20):
            return True
        elif len(only2more) > 0:
            return True
        else:
            return False

    def _fully_vaccined_jj(vlist):
        if len(vlist) >= 1:
            return True
        else:
            return False

    encoding[0, 0] = int(
        (_fully_vaccined_mrna(mrna_pre, only_2more_pre) or _fully_vaccined_jj(
            jj_pre)))  # 'Fully vaccinated - Pre-index',
    encoding[0, 1] = int(
        (_fully_vaccined_mrna(mrna_post, only_2more_post) or _fully_vaccined_jj(
            jj_post)))  # 'Fully vaccinated - Post-index',
    encoding[0, 2] = int(
        (not _fully_vaccined_mrna(mrna_pre, only_2more_pre)) and (
                len(mrna_pre) > 0))  # 'Partially vaccinated - Pre-index'
    encoding[0, 3] = int(
        (not _fully_vaccined_mrna(mrna_post, only_2more_post)) and (
                len(mrna_post) > 0))  # 'Partially vaccinated - Post-index',
    encoding[0, 4] = int((len(mrna_pre) == 0) and (len(jj_pre) == 0))  # 'No evidence - Pre-index'
    encoding[0, 5] = int((len(mrna_post) == 0) and (len(jj_post) == 0))  # 'No evidence - Post-index'

    return encoding


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


def _prefix_in_set(icd, icd_set):
    for i in range(len(icd)):
        pre = icd[:(len(icd) - i)]
        if pre in icd_set:
            return True, pre
    return False, ''


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
            # 2022-02-27, change exact match to prefix match of PASC codes
            flag, icdprefix = _prefix_in_set(icd, icd_pasc)
            if flag:  # if icd in icd_pasc:
                # if icdprefix != icd:
                #     print(icd, icdprefix)
                pasc_info = icd_pasc[icdprefix]
                pasc = pasc_info[0]
                rec = pasc_encoding[pasc]
                pos = rec[0]
                outcome_baseline[0, pos] += 1

        # build outcome
        if ecs._is_in_followup(dx_date, index_date):
            days = (dx_date - index_date).days
            flag, icdprefix = _prefix_in_set(icd, icd_pasc)
            if flag:  # if icd in icd_pasc:
                pasc_info = icd_pasc[icdprefix]
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


def _encoding_outcome_dx_withalldays(dx_list, icd_pasc, pasc_encoding, index_date, default_t2e):
    # encoding 137 outcomes from our PASC list
    # outcome_t2e = np.zeros((n, 137), dtype='int')
    # outcome_flag = np.zeros((n, 137), dtype='int')
    # outcome_baseline = np.zeros((n, 137), dtype='int')

    # 2022-02-18 initialize t2e:  last encounter, event, end of followup, whichever happens first
    outcome_t2e = np.ones((1, len(pasc_encoding)), dtype='float') * default_t2e
    outcome_flag = np.zeros((1, len(pasc_encoding)), dtype='int')
    outcome_baseline = np.zeros((1, len(pasc_encoding)), dtype='int')

    outcome_tlast = np.zeros((1, len(pasc_encoding)), dtype='int')

    outcome_t2eall = [''] * len(pasc_encoding)

    for records in dx_list:
        dx_date, icd = records[:2]
        icd = icd.replace('.', '').upper()
        # build baseline
        if ecs._is_in_baseline(dx_date, index_date):
            # 2022-02-27, change exact match to prefix match of PASC codes
            flag, icdprefix = _prefix_in_set(icd, icd_pasc)
            if flag:  # if icd in icd_pasc:
                # if icdprefix != icd:
                #     print(icd, icdprefix)
                pasc_info = icd_pasc[icdprefix]
                pasc = pasc_info[0]
                rec = pasc_encoding[pasc]
                pos = rec[0]
                outcome_baseline[0, pos] += 1

        # build outcome
        # definition of t2e might be problem when
        if ecs._is_in_followup(dx_date, index_date):
            days = (dx_date - index_date).days
            flag, icdprefix = _prefix_in_set(icd, icd_pasc)
            if flag:  # if icd in icd_pasc:
                pasc_info = icd_pasc[icdprefix]
                pasc = pasc_info[0]
                rec = pasc_encoding[pasc]
                pos = rec[0]

                outcome_t2eall[pos] += '{};'.format(days)
                if outcome_flag[0, pos] == 0:
                    # only records the first event and time
                    if days < outcome_t2e[0, pos]:
                        outcome_t2e[0, pos] = days
                    outcome_flag[0, pos] = 1
                    outcome_tlast[0, pos] = days
                else:
                    outcome_flag[0, pos] += 1

                # elif (outcome_flag[0, pos] >= 1) and ((days - outcome_tlast[0, pos]) >= ndays):
                #     outcome_flag[0, pos] += 1
                #     outcome_tlast[0, pos] = days

                # else:
                #     outcome_flag[0, pos] += 1
    # debug # a = pd.DataFrame({'1':pasc_encoding.keys(), '2':outcome_flag.squeeze(), '3':outcome_t2e.squeeze(), '4':outcome_baseline.squeeze()})
    # outcome_t2eall = np.array([outcome_t2eall])
    return outcome_flag, outcome_t2e, outcome_baseline, outcome_t2eall


def _encoding_outcome_med_rxnorm_ingredient(med_list, rxnorm_ing, ing_encoding, index_date, default_t2e, verbose=0):
    # encoding 434 top rxnorm ingredient drugs
    # initialize t2e:  last encounter,  end of followup, death, event, whichever happens first
    outcome_t2e = np.ones((1, len(ing_encoding)), dtype='float') * default_t2e
    outcome_flag = np.zeros((1, len(ing_encoding)), dtype='int')
    outcome_baseline = np.zeros((1, len(ing_encoding)), dtype='int')

    _no_mapping_rxrnom = set([])
    for records in med_list:
        med_date, rxnorm, supply_days = records
        ing_set = set(rxnorm_ing.get(rxnorm, []))

        if len(ing_set) > 0:
            pos_list = [ing_encoding[x][0] for x in ing_set if x in ing_encoding]
        else:
            _no_mapping_rxrnom.add(rxnorm)
            if verbose:
                print('Warning:', rxnorm, 'not in rxnorm to ingredeints dictionary!')
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
    # debug # a = pd.DataFrame({'rxnorm':ing_encoding.keys(), 'name':ing_encoding.values(), 'outcome_flag':outcome_flag.squeeze(), 'outcome_t2e':outcome_t2e.squeeze(), 'outcome_baseline':outcome_baseline.squeeze()})
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


def _encoding_outcome_med_atc(med_list, rxnorm_ing, rxnorm_atc, atcl_encoding, index_date, default_t2e, atc_level=3,
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


def _dx_clean_and_translate_any_ICD9_to_ICD10(dx_list, icd9_icd10, icd_ccsr):
    dx_list_new = []
    n_icd9 = 0
    for records in dx_list:
        dx_t, icd, dx_type, enc_type = records
        icd = icd.replace('.', '').upper().strip()
        if ('9' in dx_type) or ((icd.isnumeric() or (icd[0] in ('E', 'V'))) and (icd not in icd_ccsr)):
            icd10_translation = icd9_icd10.get(icd, [])
            if len(icd10_translation) > 0:
                n_icd9 += 1
                for x in icd10_translation:
                    new_records = (dx_t, x, 'from' + icd, enc_type)
                    dx_list_new.append(new_records)
                # print(icd, icd10_translation)
            else:
                dx_list_new.append((dx_t, icd, dx_type, enc_type))
        else:
            dx_list_new.append((dx_t, icd, dx_type, enc_type))

    return dx_list_new


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


def _update_counter_v2(dict3, key, flag, is_incident=False):
    if is_incident:
        if key in dict3:
            dict3[key][3] += 1
        else:
            # total_prevalent, pos_pre, negative_pre,  total_incident, pos_incident, negative_incident
            dict3[key] = [0, 0, 0, 1, 0, 0]

        if flag:
            dict3[key][4] += 1
        else:
            dict3[key][5] += 1
    else:  # is prevalence
        if key in dict3:
            dict3[key][0] += 1
        else:
            # total_prevalent, pos_pre, negative_pre,  total_incident, pos_incident, negative_incident
            dict3[key] = [1, 0, 0, 0, 0, 0]

        if flag:
            dict3[key][1] += 1
        else:
            dict3[key][2] += 1


def build_query_1and2_matrix(args):
    start_time = time.time()
    print('In build_query_1and2_matrix...')
    # step 1: load encoding dictionary
    icd_pasc, pasc_encoding, icd_cmr, cmr_encoding, \
    icd_ccsr, ccsr_encoding, rxnorm_ing, rxnorm_atc, atcl2_encoding, atcl3_encoding, atcl4_encoding, \
    ventilation_codes, comorbidity_codes, icd9_icd10, rxing_encoding, covidmed_codes, vaccine_codes = _load_mapping()

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
    med_count = {}
    dx_count = {}

    print('Try to load: ', sites)
    header = True
    mode = "w"
    for site in tqdm(sites):
        print('Loading: ', site)
        input_file = r'../data/V15_COVID19/output/{}/cohorts_{}_{}.pkl'.format(site, args.cohorts, site)
        print('Load cohorts pickle data file:', input_file)
        id_data = utils.load(input_file, chunk=4)

        # step 3: encoding cohorts baseline covariates into matrix
        if args.positive_only:
            n = 0
            for i, (pid, item) in tqdm(enumerate(id_data.items()), total=len(id_data)):
                index_info, demo, dx, med, covid_lab, enc, procedure, obsgen, immun, death, vital = item
                flag, index_date, covid_loinc, flag_name, index_age, index_enc_id = index_info
                if flag:
                    n += 1
        else:
            n = len(id_data)

        print('args.positive_only:', args.positive_only, n)

        # print('len(column_names):', len(column_names), '\n', column_names)
        # impute adi value by median of site , per site:
        adi_value_list = [v[1][7] for key, v in id_data.items()]
        adi_value_default = np.nanmedian(adi_value_list)

        i = -1
        for pid, item in tqdm(id_data.items(), total=len(
                id_data), mininterval=10):
            # for i, (pid, item) in tqdm(enumerate(id_data.items()), total=len(id_data)):
            index_info, demo, dx, med, covid_lab, enc, procedure, obsgen, immun, death, vital = item
            flag, index_date, covid_loinc, flag_name, index_age, index_enc_id = index_info
            birth_date, gender, race, hispanic, zipcode, state, city, nation_adi, state_adi = demo

            dx = _dx_clean_and_translate_any_ICD9_to_ICD10(dx, icd9_icd10, icd_ccsr)

            # dump per line
            n = 1
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

            # newly add 2022-04-08
            zip_list = []  # newly add 2022-04-08
            age_list = []
            adi_list = []
            utilization_count_array = np.zeros((n, 4), dtype='int16')
            utilization_count_names = ['inpatient no.', 'outpatient no.', 'emergency visits no.', 'other visits no.']
            bmi_list = []
            yearmonth_array = np.zeros((n, 23), dtype='int16')
            yearmonth_column_names = [
                "YM: March 2020", "YM: April 2020", "YM: May 2020", "YM: June 2020", "YM: July 2020",
                "YM: August 2020", "YM: September 2020", "YM: October 2020", "YM: November 2020", "YM: December 2020",
                "YM: January 2021", "YM: February 2021", "YM: March 2021", "YM: April 2021", "YM: May 2021",
                "YM: June 2021", "YM: July 2021", "YM: August 2021", "YM: September 2021", "YM: October 2021",
                "YM: November 2021", "YM: December 2021", "YM: January 2022"
            ]
            #
            age_array = np.zeros((n, 6), dtype='int16')
            age_column_names = ['20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75-<85 years',
                                '85+ years']

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

            # newly add 2022-04-08
            bmi_array = np.zeros((n, 5), dtype='int16')
            bmi_names = ['BMI: <18.5 under weight', 'BMI: 18.5-<25 normal weight',
                         'BMI: 25-<30 overweight ', 'BMI: >=30 obese ', 'BMI: missing']

            smoking_array = np.zeros((n, 4), dtype='int16')
            smoking_names = ['Smoker: never', 'Smoker: current', 'Smoker: former', 'Smoker: missing']

            # cautious of "DX: Hypertension and Type 1 or 2 Diabetes Diagnosis" using logic afterwards,
            # due to threshold >= 2 issue
            # DX: End Stage Renal Disease on Dialysis, Both diagnosis and procedure codes used to define this condtion
            dx_array = np.zeros((n, 40), dtype='int16')  # from 37 --> 40, # added 2022-05-25
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
                               'DX: Autism', 'DX: Sickle Cell',
                               'DX: Obstructive sleep apnea',  # added 2022-05-25
                               'DX: Epstein-Barr and Infectious Mononucleosis (Mono)',  # added 2022-05-25
                               'DX: Herpes Zoster',  # added 2022-05-25
                               ]

            # Two selected baseline medication
            med_array = np.zeros((n, 2), dtype='int16')
            # atc level 3 category # H02: CORTICOSTEROIDS FOR SYSTEMIC USE   L04:IMMUNOSUPPRESSANTS
            # --> detailed code list from CDC
            med_column_names = ["MEDICATION: Corticosteroids", "MEDICATION: Immunosuppressant drug", ]

            # vaccine info designed for risk. definition of post, and fully are different from our previous query
            vaccine_column_names = ['Fully vaccinated - Pre-index',
                                    'Fully vaccinated - Post-index',
                                    'Partially vaccinated - Pre-index',
                                    'Partially vaccinated - Post-index',
                                    'No evidence - Pre-index',
                                    'No evidence - Post-index',
                                    ]
            vaccine_array = np.zeros((n, 6), dtype='int16')

            # add covid medication
            covidmed_array = np.zeros((n, 25), dtype='int16')
            covidmed_column_names = [
                'Anti-platelet Therapy', 'Aspirin', 'Baricitinib', 'Bamlanivimab Monoclonal Antibody Treatment',
                'Bamlanivimab and Etesevimab Monoclonal Antibody Treatment',
                'Casirivimab and Imdevimab Monoclonal Antibody Treatment',
                'Any Monoclonal Antibody Treatment (Bamlanivimab, Bamlanivimab and Etesevimab, Casirivimab and Imdevimab, Sotrovimab, and unspecified monoclonal antibodies)',
                'Colchicine', 'Corticosteroids', 'Dexamethasone', 'Factor Xa Inhibitors', 'Fluvoxamine', 'Heparin',
                'Inhaled Steroids', 'Ivermectin', 'Low Molecular Weight Heparin', 'Molnupiravir', 'Nirmatrelvir',
                'Paxlovid', 'Remdesivir', 'Ritonavir', 'Sotrovimab Monoclonal Antibody Treatment',
                'Thrombin Inhibitors', 'Tocilizumab (Actemra)', 'PX: Convalescent Plasma']

            # also these drug categories as outcomes in followup
            outcome_covidmed_flag = np.zeros((n, 25), dtype='int16')
            outcome_covidmed_t2e = np.zeros((n, 25), dtype='int16')
            outcome_covidmed_baseline = np.zeros((n, 25), dtype='int16')
            outcome_covidmed_column_names = ['covidmed-out@' + x for x in covidmed_column_names] + \
                                            ['covidmed-t2e@' + x for x in covidmed_column_names] + \
                                            ['covidmed-base@' + x for x in covidmed_column_names]

            # Build PASC outcome t2e and flag in follow-up, and outcome flag in baseline for dynamic cohort selection
            # In total, there are 137 PASC categories in our lists. See T2E later
            outcome_flag = np.zeros((n, 137), dtype='int16')
            outcome_t2e = np.zeros((n, 137), dtype='int16')
            outcome_baseline = np.zeros((n, 137), dtype='int16')

            outcome_t2eall = []

            # new add for 2 dx cross categories
            outcome_column_names = ['dx-out@' + x for x in pasc_encoding.keys()] + \
                                   ['dx-t2e@' + x for x in pasc_encoding.keys()] + \
                                   ['dx-base@' + x for x in pasc_encoding.keys()] + \
                                   ['dx-t2eall@' + x for x in pasc_encoding.keys()]

            # # rxing_encoding outcome.
            # outcome_med_flag = np.zeros((n, 434), dtype='int16')
            # outcome_med_t2e = np.zeros((n, 434), dtype='int16')
            # outcome_med_baseline = np.zeros((n, 434), dtype='int16')
            # outcome_med_column_names = ['med-out@' + x for x in rxing_encoding.keys()] + \
            #                            ['med-t2e@' + x for x in rxing_encoding.keys()] + \
            #                            ['med-base@' + x for x in rxing_encoding.keys()]

            column_names = ['patid', 'site', 'covid', 'index date', 'hospitalized',
                            'ventilation', 'criticalcare', 'maxfollowup'] + death_column_names + \
                           ['zip', 'age', 'adi'] + utilization_count_names + ['bmi'] + yearmonth_column_names + \
                           age_column_names + \
                           gender_column_names + race_column_names + hispanic_column_names + \
                           social_column_names + utilization_column_names + index_period_names + \
                           bmi_names + smoking_names + \
                           dx_column_names + med_column_names + vaccine_column_names + covidmed_column_names + \
                           outcome_covidmed_column_names + outcome_column_names  # + outcome_med_column_names

            if args.positive_only:
                if not flag:
                    continue
            # i += 1
            i = 0
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
            death_array[i, :] = _encoding_death(death, index_date)

            # newly add 2022-04-08
            zip_list.append(zipcode)
            age_list.append(index_age)
            adi_list.append(nation_adi)
            # utilization count, postponed to below
            # bmi_list, postponed to below
            yearmonth_array[i, :] = _encoding_yearmonth(index_date)

            # encoding query 1 information
            age_array[i, :] = _encoding_age(index_age)
            gender_array[i] = _encoding_gender(gender)
            race_array[i, :] = _encoding_race(race)
            hispanic_array[i, :] = _encoding_hispanic(hispanic)
            #
            social_array[i, :] = _encoding_social(nation_adi, adi_value_default)
            utilization_array[i, :], utilization_count_array[i, :] = _encoding_utilization(enc, index_date)
            index_period_array[i, :] = _encoding_index_period(index_date)
            #

            # encoding bmi and smoking
            bmi_array[i, :], smoking_array[i, :], bmi = _encoding_bmi_and_smoking(vital, index_date)
            bmi_list.append(bmi)

            # encoding query 2 information
            dx_array[i, :] = _encoding_dx(dx, dx_column_names, comorbidity_codes, index_date, procedure)
            med_array[i, :] = _encoding_med(med, med_column_names, comorbidity_codes, index_date)

            # encoding pasc information in both baseline and followup
            # time 2 event: censoring in the database (should be >= followup start time),
            # maximum follow-up, death-time, event, whichever comes first
            default_t2e = np.min([
                np.maximum(ecs.FOLLOWUP_LEFT, maxfollowtime),
                np.maximum(ecs.FOLLOWUP_LEFT, ecs.FOLLOWUP_RIGHT),
                np.maximum(ecs.FOLLOWUP_LEFT, death_array[i, 1])
            ])

            vaccine_array[i, :] = _encoding_vaccine_4risk(procedure, immun, vaccine_column_names, vaccine_codes,
                                                          index_date)

            covidmed_array[i, :], \
            outcome_covidmed_flag[i, :], outcome_covidmed_t2e[i, :], outcome_covidmed_baseline[i, :] \
                = _encoding_covidmed(med, procedure, covidmed_column_names, covidmed_codes, index_date, default_t2e)

            # outcome_flag[i, :], outcome_t2e[i, :], outcome_baseline[i, :] = \
            #     _encoding_outcome_dx(dx, icd_pasc, pasc_encoding, index_date, default_t2e)

            outcome_flag[i, :], outcome_t2e[i, :], outcome_baseline[i, :], outcome_t2eall_1row = \
                _encoding_outcome_dx_withalldays(dx, icd_pasc, pasc_encoding, index_date, default_t2e)
            outcome_t2eall.append(outcome_t2eall_1row)
            # outcome_med_flag[i, :], outcome_med_t2e[i, :], outcome_med_baseline[i, :] = \
            #     _encoding_outcome_med_rxnorm_ingredient(med, rxnorm_ing, rxing_encoding, index_date, default_t2e)

            #   step 4: build pandas, column, and dump
            data_array = np.hstack((np.asarray(pid_list).reshape(-1, 1),
                                    np.asarray(site_list).reshape(-1, 1),
                                    np.array(covid_list).reshape(-1, 1).astype(int),
                                    np.asarray(indexdate_list).reshape(-1, 1),
                                    np.asarray(hospitalized_list).reshape(-1, 1).astype(int),
                                    np.array(ventilation_list).reshape(-1, 1).astype(int),
                                    np.array(criticalcare_list).reshape(-1, 1).astype(int),
                                    np.array(maxfollowtime_list).reshape(-1, 1),
                                    death_array,
                                    np.asarray(zip_list).reshape(-1, 1),
                                    np.asarray(age_list).reshape(-1, 1),
                                    np.asarray(adi_list).reshape(-1, 1),
                                    utilization_count_array,
                                    np.asarray(bmi_list).reshape(-1, 1),
                                    yearmonth_array,
                                    age_array,
                                    gender_array,
                                    race_array,
                                    hispanic_array,
                                    social_array,
                                    utilization_array,
                                    index_period_array,
                                    bmi_array,
                                    smoking_array,
                                    dx_array,
                                    med_array,
                                    vaccine_array,
                                    covidmed_array,
                                    outcome_covidmed_flag,
                                    outcome_covidmed_t2e,
                                    outcome_covidmed_baseline,
                                    outcome_flag,
                                    outcome_t2e,
                                    outcome_baseline,
                                    np.asarray(outcome_t2eall),
                                    # outcome_med_flag,
                                    # outcome_med_t2e,
                                    # outcome_med_baseline
                                    ))

            df_data = pd.DataFrame(data_array, columns=column_names)
            # data_all_sites.append(df_data)

            # transform count to bool with threshold 2, and deal with "DX: Hypertension and Type 1 or 2 Diabetes Diagnosis"
            # df_bool = df_data_all_sites.copy()  # not using deep copy for the sage of time
            # df_bool = df_data_all_sites
            df_bool = df_data
            selected_cols = [x for x in df_bool.columns if (x.startswith('DX:') or x.startswith('MEDICATION:'))]
            df_bool.loc[:, selected_cols] = (df_bool.loc[:, selected_cols].astype('int') >= 2).astype('int')
            df_bool.loc[:, r"DX: Hypertension and Type 1 or 2 Diabetes Diagnosis"] = \
                (df_bool.loc[:, r'DX: Hypertension'] & (
                        df_bool.loc[:, r'DX: Diabetes Type 1'] | df_bool.loc[:, r'DX: Diabetes Type 2'])).astype('int')

            # Warning: the covid medication part is not boolean
            # keep the value of baseline count and outcome count in the file, filter later depends on the application
            # df_data.loc[:, covidmed_column_names] = (df_data.loc[:, covidmed_column_names].astype('int') >= 1).astype('int')
            # can be done later

            selected_cols = [x for x in df_bool.columns if
                             (x.startswith('dx-base@') or
                              x.startswith('med-out@') or x.startswith('med-base@') or
                              x.startswith('covidmed-out@') or x.startswith('covidmed-base@'))]
            df_bool.loc[:, selected_cols] = (df_bool.loc[:, selected_cols].astype('int') >= 1).astype('int')

            # selected_cols = [x for x in df_bool.columns if (x.startswith('dx-out@'))]
            # df_bool.loc[:, selected_cols] = (df_bool.loc[:, selected_cols].astype('int') >= 2).astype('int')

            # utils.check_and_mkdir(args.output_file_query12_bool)

            df_bool.to_csv(args.output_file_query12_bool, mode=mode, header=header, index=False)
            if header:
                header = False
                mode = "a"

            # count additional information
            # in follow-up, each person count once
            _dx_set = set()
            _dx_set_base = set()
            for i_dx in dx:
                dx_t, icd, dx_type, enc_type = i_dx
                icd = icd.replace('.', '').upper()
                if ecs._is_in_followup(dx_t, index_date):
                    _dx_set.add(icd)
                if ecs._is_in_baseline(dx_t, index_date):
                    _dx_set_base.add(icd)

            for i_dx in _dx_set:
                _update_counter_v2(dx_count, i_dx, flag, is_incident=False)
                if i_dx not in _dx_set_base:
                    _update_counter_v2(dx_count, i_dx, flag, is_incident=True)

            _med_set = set()
            _med_set_base = set()
            for i_med in med:
                t = i_med[0]
                rx = i_med[1]
                if ecs._is_in_followup(t, index_date):
                    if rx in rxnorm_ing:
                        _added = rxnorm_ing[rx]
                        _med_set.update(_added)
                    else:
                        _med_set.add(rx)

                if ecs._is_in_medication_baseline(t, index_date):
                    if rx in rxnorm_ing:
                        _added = rxnorm_ing[rx]
                        _med_set_base.update(_added)
                    else:
                        _med_set_base.add(rx)

            for i_med in _med_set:
                _update_counter_v2(med_count, i_med, flag, is_incident=False)
                if i_med not in _med_set_base:
                    _update_counter_v2(med_count, i_med, flag, is_incident=True)

        print('Done! Dump data bool matrix for query12 to {}'.format(args.output_file_query12_bool))
        print('Encoding done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

        print('df_data.shape:', df_data.shape)
        # del id_data
        print('Done site:', site)
        # end iterate sites

    dx_count_df = pd.DataFrame.from_dict(dx_count, orient='index',
                                         columns=['total', 'no. in positive group', 'no. in negative group',
                                                  'incident total', 'incident no. in positive group',
                                                  'incident no. in negative group'])
    # dx_count_df.to_csv(args.output_dx_info)
    med_count_df = pd.DataFrame.from_dict(med_count, orient='index',
                                          columns=['total', 'no. in positive group', 'no. in negative group',
                                                   'incident total', 'incident no. in positive group',
                                                   'incident no. in negative group'])
    # med_count_df.to_csv(args.output_med_info)

    # df_data_all_sites = pd.concat(data_all_sites)
    # print('df_data_all_sites.shape:', df_data_all_sites.shape)

    # utils.check_and_mkdir(args.output_file_query12)
    # df_data_all_sites.to_csv(args.output_file_query12)
    # print('Done! Dump data matrix for query12 to {}'.format(args.output_file_query12))



    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return df_bool


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

        pasc_col = [x for x in df_in.columns if (x.startswith('flag@'))]
        df_out.loc[pasc_col, 'N'] = df_in[pasc_col].sum()
        df_out.loc[pasc_col, '%'] = df_in[pasc_col].sum() / len(df_in)

        df_out.to_csv(out_file)
        print('Dump done ', out_file)


def de_novo_medication_analyse(cohorts, dataset='ALL', severity=''):
    # severity in 'hospitalized', 'ventilation', None
    # build pasc specific cohorts from covid base cohorts!
    in_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4screen_Covid+_queryATCL4_encoding_bool_{}.csv'.format(
        dataset)
    print('In de_novo_medication_analyse,  Cohorts: {}, severity: {}'.format(cohorts, severity))
    print('Try to load:', in_file)

    df_data = pd.read_csv(in_file, dtype={'patid': str})  # , parse_dates=['index_date', 'birth_date']
    print('df_data.shape:', df_data.shape)

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

    selected_cols = [x for x in df_data.columns if x.startswith('flag@')]  # or x.startswith('baseline@')
    df_data['any_pasc'] = df_data.loc[:, selected_cols].sum(axis=1)
    df_data = df_data.loc[df_data["covid"], :]
    print('df_data.shape:', df_data.shape)
    df_pos = df_data.loc[df_data["any_pasc"] > 0, :]
    df_neg = df_data.loc[df_data["any_pasc"] == 0, :]
    print('df_pos.shape:', df_pos.shape)
    print('df_neg.shape:', df_neg.shape)

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
    records = []
    for atc in tqdm(atcl4_encoding.keys(), total=len(atcl4_encoding)):
        atc_cohort_exposed = df_data.loc[(df_data['atc@' + atc] >= 1) & (df_data['atcbase@' + atc] == 0), :]
        atc_cohort_not_exposed = df_data.loc[(df_data['atc@' + atc] == 0), :]

        atc_cohort_exposed_pasc = atc_cohort_exposed.loc[atc_cohort_exposed["any_pasc"] > 0, :]
        atc_cohort_exposed_nopasc = atc_cohort_exposed.loc[atc_cohort_exposed["any_pasc"] == 0, :]

        atc_cohort_not_exposed_pasc = atc_cohort_not_exposed.loc[atc_cohort_not_exposed["any_pasc"] > 0, :]
        atc_cohort_not_exposed_nopasc = atc_cohort_not_exposed.loc[atc_cohort_not_exposed["any_pasc"] == 0, :]

        records.append((atc, atcl4_encoding[atc][1], atcl4_encoding[atc][2], atcl3_encoding[atc[:4]][2],
                        len(atc_cohort_exposed_pasc), len(atc_cohort_exposed_nopasc),
                        len(atc_cohort_not_exposed_pasc), len(atc_cohort_not_exposed_nopasc),
                        ))

    df = pd.DataFrame(records, columns=['atcl4', 'rxnorm', 'name', 'category',
                                        'atc_exposed-pasc_case (a)', 'atc_exposed-nopasc_control (b)',
                                        'atc_unexposed-pasc_case (c)', 'atc_unexposed-nopasc_control (d)'])

    df['Odds case was exposed (a/c)'] = df['atc_exposed-pasc_case (a)'] / df['atc_unexposed-pasc_case (c)']
    df['Odds control was exposed (b/d)'] = df['atc_exposed-nopasc_control (b)'] / df['atc_unexposed-nopasc_control (d)']
    df['OR (ad/bc)'] = df['Odds case was exposed (a/c)'] / df['Odds control was exposed (b/d)']

    df.to_csv(
        r'../data/V15_COVID19/output/character/summary_covid+_screen_medication_queryATCL4_encoding_bool_{}.csv'.format(
            dataset))

    print('Dump done ')


def de_novo_medication_analyse_selected_and_iptw(cohorts, dataset='ALL', severity=''):
    # severity in 'hospitalized', 'ventilation', None
    # build pasc specific cohorts from covid base cohorts!
    in_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4screen_Covid+_queryATCL4_encoding_bool_{}.csv'.format(
        dataset)
    print('In de_novo_medication_analyse,  Cohorts: {}, severity: {}'.format(cohorts, severity))
    print('Try to load:', in_file)

    df_drug = pd.read_csv(
        r'../data/V15_COVID19/output/character/summary_covid+_screen_medication_queryATCL4_encoding_bool_ALL.csv')
    df_drug = df_drug.sort_values(by=['atc_exposed-pasc_case (a)'], ascending=False)
    drug_list = df_drug.loc[df_drug['atc_exposed-pasc_case (a)'] >= 1000, 'atcl4']

    df_data = pd.read_csv(in_file, dtype={'patid': str})  # , parse_dates=['index_date', 'birth_date']
    print('df_data.shape:', df_data.shape)

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

    selected_cols = [x for x in df_data.columns if x.startswith('flag@')]  # or x.startswith('baseline@')
    df_data['any_pasc'] = df_data.loc[:, selected_cols].sum(axis=1)
    df_data = df_data.loc[df_data["covid"], :]
    print('df_data.shape:', df_data.shape)
    df_pos = df_data.loc[df_data["any_pasc"] > 0, :]
    df_neg = df_data.loc[df_data["any_pasc"] == 0, :]
    print('df_pos.shape:', df_pos.shape)
    print('df_neg.shape:', df_neg.shape)

    records = []
    for atc in tqdm(drug_list, total=len(drug_list)):
        atc_cohort_exposed = df_data.loc[(df_data['atc@' + atc] >= 1) & (df_data['atcbase@' + atc] == 0), :].copy()
        atc_cohort_not_exposed = df_data.loc[(df_data['atc@' + atc] == 0), :].copy()

        selected_colums = list(df_data.columns[7:51]) + list(df_data.columns[74:108])
        df_covs_array = pd.concat(
            [atc_cohort_exposed.loc[:, selected_colums], atc_cohort_not_exposed.loc[:, selected_colums]])
        df_label = pd.concat([atc_cohort_exposed.loc[:, 'atc@' + atc], atc_cohort_not_exposed.loc[:, 'atc@' + atc]])
        model = ml.PropensityEstimator(learner='LR', random_seed=0).cross_validation_fit(df_covs_array, df_label)
        # , paras_grid = {
        #     'penalty': 'l2',
        #     'C': 0.03162277660168379,
        #     'max_iter': 200,
        #     'random_state': 0}
        iptw = model.predict_inverse_weight(df_covs_array, df_label, stabilized=True, clip=False)

        atc_cohort_exposed['iptw'] = iptw[:len(atc_cohort_exposed)]
        atc_cohort_not_exposed['iptw'] = iptw[len(atc_cohort_exposed):]

        atc_cohort_exposed_pasc = atc_cohort_exposed.loc[atc_cohort_exposed["any_pasc"] > 0, :]
        atc_cohort_exposed_nopasc = atc_cohort_exposed.loc[atc_cohort_exposed["any_pasc"] == 0, :]

        atc_cohort_not_exposed_pasc = atc_cohort_not_exposed.loc[atc_cohort_not_exposed["any_pasc"] > 0, :]
        atc_cohort_not_exposed_nopasc = atc_cohort_not_exposed.loc[atc_cohort_not_exposed["any_pasc"] == 0, :]

        records.append((atc, atcl4_encoding[atc][1], atcl4_encoding[atc][2], atcl3_encoding[atc[:4]][2],
                        len(atc_cohort_exposed_pasc), len(atc_cohort_exposed_nopasc),
                        len(atc_cohort_not_exposed_pasc), len(atc_cohort_not_exposed_nopasc),
                        atc_cohort_exposed_pasc['iptw'].sum(), atc_cohort_exposed_nopasc['iptw'].sum(),
                        atc_cohort_not_exposed_pasc['iptw'].sum(), atc_cohort_not_exposed_nopasc['iptw'].sum(),
                        model.best_balance, model.best_balance_k_folds_detail,
                        model.best_fit, model.best_fit_k_folds_detail,
                        ))

    df = pd.DataFrame(records, columns=['atcl4', 'rxnorm', 'name', 'category',
                                        'atc_exposed-pasc_case (a)', 'atc_exposed-nopasc_control (b)',
                                        'atc_unexposed-pasc_case (c)', 'atc_unexposed-nopasc_control (d)',
                                        'atc_exposed-pasc_case (a) iptw', 'atc_exposed-nopasc_control (b) iptw',
                                        'atc_unexposed-pasc_case (c) iptw', 'atc_unexposed-nopasc_control (d) iptw',
                                        'best_balance', 'best_balance_k_folds_detail',
                                        'best_fit', 'best_fit_k_folds_detail'
                                        ])

    df['Odds case was exposed (a/c)'] = df['atc_exposed-pasc_case (a)'] / df['atc_unexposed-pasc_case (c)']
    df['Odds control was exposed (b/d)'] = df['atc_exposed-nopasc_control (b)'] / df['atc_unexposed-nopasc_control (d)']
    df['OR (ad/bc)'] = df['Odds case was exposed (a/c)'] / df['Odds control was exposed (b/d)']

    df['Odds case was exposed (a/c) iptw'] = df['atc_exposed-pasc_case (a) iptw'] / df[
        'atc_unexposed-pasc_case (c) iptw']
    df['Odds control was exposed (b/d) iptw'] = df['atc_exposed-nopasc_control (b) iptw'] / df[
        'atc_unexposed-nopasc_control (d) iptw']
    df['OR (ad/bc) iptw'] = df['Odds case was exposed (a/c) iptw'] / df['Odds control was exposed (b/d) iptw']

    df.to_csv(
        r'../data/V15_COVID19/output/character/summary_covid+_screen_medication_queryATCL4_encoding_bool_{}-iptw.csv'.format(
            dataset))

    print('Dump done ')


def de_novo_medication_analyse_atcl3(cohorts, dataset='ALL', severity=''):
    # severity in 'hospitalized', 'ventilation', None
    # build pasc specific cohorts from covid base cohorts!
    in_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4screen_queryATCL4_encoding_bool_{}.csv'.format(
        dataset)
    print('In de_novo_medication_analyse,  Cohorts: {}, severity: {}'.format(cohorts, severity))
    print('Try to load:', in_file)

    df_data = pd.read_csv(in_file, dtype={'patid': str})  # , parse_dates=['index_date', 'birth_date']
    print('df_data.shape:', df_data.shape)

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

    selected_cols = [x for x in df_data.columns if x.startswith('flag@')]  # or x.startswith('baseline@')
    df_data['any_pasc'] = df_data.loc[:, selected_cols].sum(axis=1)
    df_data = df_data.loc[df_data["covid"], :]
    print('df_data.shape:', df_data.shape)
    df_pos = df_data.loc[df_data["any_pasc"] > 0, :]
    df_neg = df_data.loc[df_data["any_pasc"] == 0, :]
    print('df_pos.shape:', df_pos.shape)
    print('df_neg.shape:', df_neg.shape)

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
    records = []
    for atc in tqdm(atcl4_encoding.keys(), total=len(atcl4_encoding)):
        atc_cohort_exposed = df_data.loc[(df_data['atc@' + atc] >= 1) & (df_data['atcbase@' + atc] == 0), :]
        atc_cohort_not_exposed = df_data.loc[(df_data['atc@' + atc] == 0), :]

        atc_cohort_exposed_pasc = atc_cohort_exposed.loc[atc_cohort_exposed["any_pasc"] > 0, :]
        atc_cohort_exposed_nopasc = atc_cohort_exposed.loc[atc_cohort_exposed["any_pasc"] == 0, :]

        atc_cohort_not_exposed_pasc = atc_cohort_not_exposed.loc[atc_cohort_not_exposed["any_pasc"] > 0, :]
        atc_cohort_not_exposed_nopasc = atc_cohort_not_exposed.loc[atc_cohort_not_exposed["any_pasc"] == 0, :]

        records.append((atc, atcl4_encoding[atc][1], atcl4_encoding[atc][2], atcl3_encoding[atc[:4]][2],
                        len(atc_cohort_exposed_pasc), len(atc_cohort_exposed_nopasc),
                        len(atc_cohort_not_exposed_pasc), len(atc_cohort_not_exposed_nopasc),
                        ))

    df = pd.DataFrame(records, columns=['atcl4', 'rxnorm', 'name', 'category',
                                        'atc_exposed-pasc_case (a)', 'atc_exposed-nopasc_control (b)',
                                        'atc_unexposed-pasc_case (c)', 'atc_unexposed-nopasc_control (d)'])

    df['Odds case was exposed (a/c)'] = df['atc_exposed-pasc_case (a)'] / df['atc_unexposed-pasc_case (c)']
    df['Odds control was exposed (b/d)'] = df['atc_exposed-nopasc_control (b)'] / df['atc_unexposed-nopasc_control (d)']
    df['OR (ad/bc)'] = df['Odds case was exposed (a/c)'] / df['Odds control was exposed (b/d)']

    df.to_csv(
        r'../data/V15_COVID19/output/character/summary_covid_4screen_queryATCL4_encoding_bool_{}.csv'.format(dataset))

    print('Dump done ')


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


def cohorts_table_generation(args):
    df_data = pd.read_csv(
        r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4screen_Covid+_queryATCL4_encoding_bool_ALL.csv',
        dtype={'patid': str})  #
    print('df_data.shape:', df_data.shape)
    df_data = df_data.drop(columns=['patid', 'site'])

    df_data['Unnamed: 0'] = 1
    df_data.rename(columns={'Unnamed: 0': 'No.'}, inplace=True)
    selected_cols = [x for x in df_data.columns if x.startswith('flag@')]  # or x.startswith('baseline@')
    df_data['any_pasc'] = df_data.loc[:, selected_cols].sum(axis=1)
    df_data['covid'] = df_data['covid'].apply(lambda x: int(x == True))
    df_data['hospitalized'] = df_data['hospitalized'].apply(lambda x: int(x == True))
    df_data['ventilation'] = df_data['ventilation'].apply(lambda x: int(x == True))

    df_data = df_data.loc[df_data["covid"] == 1, :]
    print('df_data.shape:', df_data.shape)
    df_pos = df_data.loc[df_data["any_pasc"] > 0, :]
    df_neg = df_data.loc[df_data["any_pasc"] == 0, :]
    print('df_pos.shape:', df_pos.shape)
    print('df_neg.shape:', df_neg.shape)

    def smd(m1, m2, v1, v2):
        VAR = np.sqrt((v1 + v2) / 2)
        smd = np.divide(
            m1 - m2,
            VAR, out=np.zeros_like(m1), where=VAR != 0)
        return smd

    df = pd.DataFrame({'Overall': df_data.sum(),
                       'Overall-mean': df_data.mean(),
                       'df_pos': df_pos.sum(),
                       'df_pos-mean': df_pos.mean(),
                       'df_neg': df_neg.sum(),
                       'df_neg-mean': df_neg.mean(),
                       'smd': smd(df_pos.mean(), df_neg.mean(), df_pos.var(), df_neg.var())
                       })

    df.to_csv(
        r'../data/V15_COVID19/output/character/table_matrix_cohorts_covid_4screen_Covid+_queryATCL4_encoding_bool_ALL.csv')
    # age

    return df


def enrich_med_rwd_info():
    atclevel_chars = {1: 1, 2: 3, 3: 4, 4: 5, 5: 7}
    rx_name = utils.load(r'../data/mapping/rxnorm_name.pkl')
    df = pd.read_csv(r'../data/V15_COVID19/output/character/info_medication_cohorts_covid_4manuNegNoCovid_ALL.csv',
                     dtype={'Unnamed: 0': str}).rename(columns={'Unnamed: 0': "rxnorm"})
    df = df.sort_values(by=['no. in positive group'], ascending=False)

    df['ratio'] = df['no. in positive group'] / df['no. in negative group']
    df['name'] = df['rxnorm'].apply(lambda x: rx_name.get(x, [''])[0])
    df['atc-l3'] = ''
    df['atc-l4'] = ''

    rx_atc = utils.load(r'../data/mapping/rxnorm_atc_mapping.pkl')
    atc_name = utils.load(r'../data/mapping/atc_name.pkl')

    for index, row in df.iterrows():
        rx = row[0]
        atcset = rx_atc.get(rx, [])

        atc3_col = []
        atc4_col = []
        for _ra in atcset:
            atc, name = _ra
            atc3 = atc[:4]
            atc3name = atc_name.get(atc3, [''])[0]
            atc3_col.append(atc3 + ':' + atc3name)

            atc4 = atc[:5]
            atc4name = atc_name.get(atc4, [''])[0]
            atc4_col.append(atc4 + ':' + atc4name)

        atc3_col = '$'.join(atc3_col)
        df.loc[index, 'atc-l3'] = atc3_col
        atc4_col = '$'.join(atc4_col)
        df.loc[index, 'atc-l4'] = atc4_col

    df.to_csv(r'../data/V15_COVID19/output/character/info_medication_cohorts_covid_4manuNegNoCovid_ALL_enriched.csv',
              index=False)
    return df


def rwd_dx_and_pasc_comparison():
    df_ccsr = pd.read_csv(r'../data/mapping/DXCCSR_v2022-1/DXCCSR_v2022-1.CSV', dtype=str)
    df_ccsr["'ICD-10-CM CODE'"] = df_ccsr["'ICD-10-CM CODE'"].apply(lambda x: x.strip("'"))
    df_ccsr_sub = df_ccsr[["'ICD-10-CM CODE'",
                           "'ICD-10-CM CODE DESCRIPTION'",
                           "'CCSR CATEGORY 1'",
                           "'CCSR CATEGORY 1 DESCRIPTION'"
                           ]]
    # df_pasc = pd.read_excel(r'../data/mapping/PASC_Adult_Combined_List_20220127_v3.xlsx',
    #                         sheet_name=r'PASC Screening List',
    #                         usecols="A:N")
    df_pasc = pd.read_excel(r'../data/mapping/PASC_Adult_Combined_List_submit_withRWEV3.xlsx',
                            sheet_name=r'PASC Screening List',
                            usecols="A:O")
    df_pasc['ICD-10-CM Code'] = df_pasc['ICD-10-CM Code'].apply(lambda x: x.strip().upper().replace('.', ''))
    print('df_pasc.shape', df_pasc.shape)
    # pasc_codes = df_pasc_list['ICD-10-CM Code'].str.upper().replace('.', '', regex=False)  # .to_list()

    df_icd = pd.read_csv(r'../data/V15_COVID19/output/character/info_dx_cohorts_covid_4manuNegNoCovidV2_ALL-V2.csv',
                         dtype={'Unnamed: 0': str}).rename(columns={'Unnamed: 0': "dx code"})
    df_icd['ratio'] = df_icd['no. in positive group'] / df_icd['no. in negative group']
    df_icd['incident ratio'] = df_icd['incident no. in positive group'] / df_icd['incident no. in negative group']

    df_icd = pd.merge(df_icd, df_ccsr_sub, left_on='dx code', right_on="'ICD-10-CM CODE'", how='left')

    df = pd.merge(df_icd, df_pasc, left_on='dx code', right_on='ICD-10-CM Code', how='outer')
    df.to_csv(r'../data/V15_COVID19/output/character/info_dx_cohorts_covid_4manuNegNoCovidV2_ALL_with_PASC-V2.csv',
              index=False)

    df_pasc_withrwd = pd.merge(df_pasc, df_icd, left_on='ICD-10-CM Code', right_on='dx code', how='left')

    for index, row in df_pasc_withrwd.iterrows():
        icd = row['ICD-10-CM Code']
        if pd.isna(row['dx code']):
            _df = df_icd.loc[df_icd['dx code'].str.startswith(icd), :]
            dx_code = ';'.join(_df['dx code'])
            total = _df['total'].sum()
            npos = _df['no. in positive group'].sum()
            nneg = _df['no. in negative group'].sum()
            ratio = npos / nneg
            df_pasc_withrwd.loc[index, 'dx code'] = dx_code
            df_pasc_withrwd.loc[index, 'total'] = total
            df_pasc_withrwd.loc[index, 'no. in positive group'] = npos
            df_pasc_withrwd.loc[index, 'no. in negative group'] = nneg
            df_pasc_withrwd.loc[index, 'ratio'] = ratio

            total_inc = _df['incident total'].sum()
            npos_inc = _df['incident no. in positive group'].sum()
            nneg_inc = _df['incident no. in negative group'].sum()
            ratio_inc = npos_inc / nneg_inc
            df_pasc_withrwd.loc[index, 'incident total'] = total_inc
            df_pasc_withrwd.loc[index, 'incident no. in positive group'] = npos_inc
            df_pasc_withrwd.loc[index, 'incident no. in negative group'] = nneg_inc
            df_pasc_withrwd.loc[index, 'incident ratio'] = ratio_inc

    df_pasc_withrwd.to_csv(
        r'../data/V15_COVID19/output/character/PASC_Adult_Combined_List_with_covid_4manuNegNoCovidV2.csv',
        index=False)

    return df, df_pasc_withrwd


def enrich_med_rwd_info_4_neruo_and_pulmonary():
    atclevel_chars = {1: 1, 2: 3, 3: 4, 4: 5, 5: 7}
    rx_name = utils.load(r'../data/mapping/rxnorm_name.pkl')
    # df = pd.read_excel(r'../data/V15_COVID19/output/character/Query-Medication_cohorts_covid_4manuscript_ALL_neurologic_pulmonary_query_summary_table.xlsx',
    #                  dtype={'Unnamed: 0': str}).rename(columns={'Unnamed: 0': "rxnorm"})
    df = pd.read_excel(
        r'../data/V15_COVID19/output/character/Query-Medication_for_neurologic_pulmonary_CP_more_drugs.xlsx',
        dtype={'Unnamed: 0': str}).rename(columns={'Unnamed: 0': "rxnorm"})
    # df = df.sort_values(by=['no. in positive group'], ascending=False)

    # df['ratio'] = df['no. in positive group'] / df['no. in negative group']
    df['name'] = df['rxnorm'].apply(lambda x: rx_name.get(x, [''])[0])
    df['atc-l3'] = ''
    df['atc-l4'] = ''

    rx_atc = utils.load(r'../data/mapping/rxnorm_atc_mapping.pkl')
    atc_name = utils.load(r'../data/mapping/atc_name.pkl')

    for index, row in df.iterrows():
        rx = row[0]
        atcset = rx_atc.get(rx, [])

        atc3_col = []
        atc4_col = []
        for _ra in atcset:
            atc, name = _ra
            atc3 = atc[:4]
            atc3name = atc_name.get(atc3, [''])[0]
            atc3_col.append(atc3 + ':' + atc3name)

            atc4 = atc[:5]
            atc4name = atc_name.get(atc4, [''])[0]
            atc4_col.append(atc4 + ':' + atc4name)

        atc3_col = '$'.join(atc3_col)
        df.loc[index, 'atc-l3'] = atc3_col
        atc4_col = '$'.join(atc4_col)
        df.loc[index, 'atc-l4'] = atc4_col

    # df.to_csv(r'../data/V15_COVID19/output/character/Query-Medication_cohorts_covid_4manuscript_ALL_neurologic_pulmonary_query_summary_table_enriched.csv')
    df.to_csv(
        r'../data/V15_COVID19/output/character/Query-Medication_for_neurologic_pulmonary_CP_more_drugs_enriched.csv')

    return df


if __name__ == '__main__':
    # python pre_data_manuscript.py --dataset ALL --cohorts covid_4manuscript 2>&1 | tee  log/pre_data_manuscript.txt
    # python pre_data_manuscript.py --dataset ALL --cohorts covid_4manuNegNoCovid 2>&1 | tee  log/pre_data_manuscript_covid_4manuNegNoCovid.txt

    # python pre_data_manuscript_withAllDays.py --dataset ALL --cohorts covid_4manuNegNoCovidV2 2>&1 | tee  log/pre_data_manuscript_withAllDays_covid_4manuNegNoCovidV2.txt

    # enrich_med_rwd_info_4_neruo_and_pulmonary()
    #
    start_time = time.time()
    args = parse_args()
    df_data_bool = build_query_1and2_matrix(args)

    # in_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuscript_bool_ALL.csv'
    # df_data = pd.read_csv(in_file, dtype={'patid': str}, parse_dates=['index date'])

    # cohorts_table_generation(args)
    # de_novo_medication_analyse(cohorts='covid_4screen_Covid+', dataset='ALL', severity='')
    # de_novo_medication_analyse_selected_and_iptw(cohorts='covid_4screen_Covid+', dataset='ALL', severity='')
    # df_med = enrich_med_rwd_info()
    # df_dx, df_pasc_withrwd = rwd_dx_and_pasc_comparison()
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
