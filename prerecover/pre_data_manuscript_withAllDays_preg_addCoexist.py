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
                                              'covid_4manuscript', 'covid_4manuNegNoCovid',
                                              'covid_4manuNegNoCovidV2', 'covid_4manuNegNoCovidV2age18'],
                        default='covid_4manuNegNoCovidV2age18', help='cohorts')
    parser.add_argument('--dataset', default='wcm', help='site dataset')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--positive_only', action='store_true')
    # parser.add_argument("--ndays", type=int, default=30)

    args = parser.parse_args()

    # args.output_file_query12 = r'../data/V15_COVID19/output/character/matrix_cohorts_{}_cnt_AnyPASC-withAllDays_{}.csv'.format(
    #     args.cohorts,
    #     args.dataset)
    # args.output_file_query12_bool = r'../data/recover/output/{}/matrix_cohorts_{}_boolbase-nout-withAllDays_{}.csv'.format(
    #     args.dataset,
    #     args.cohorts,
    #     args.dataset)

    # args.output_med_info = r'../data/recover/output/{}/info_medication_cohorts_{}_{}.csv'.format(
    #     args.dataset,
    #     args.cohorts,
    #     args.dataset)
    #
    # args.output_dx_info = r'../data/recover/output/{}/info_dx_cohorts_{}_{}.csv'.format(
    #     args.dataset,
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

    # 2023-2-09
    icd_OBC = utils.load(r'../data/mapping/icd_OBComorbidity_mapping.pkl')
    OBC_encoding = utils.load(r'../data/mapping/OBComorbidity_index_mapping.pkl')
    icd_SMMpasc = utils.load(r'../data/mapping/icd_SMMpasc_mapping.pkl')
    SMMpasc_encoding = utils.load(r'../data/mapping/SMMpasc_index_mapping.pkl')

    return icd_pasc, pasc_encoding, icd_cmr, cmr_encoding, icd_ccsr, ccsr_encoding, \
        rxnorm_ing, rxnorm_atc, atcl2_encoding, atcl3_encoding, atcl4_encoding, \
        ventilation_codes, comorbidity_codes, icd9_icd10, rxing_index, covidmed_codes, vaccine_codes, \
        icd_OBC, OBC_encoding, icd_SMMpasc, SMMpasc_encoding


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


def _encoding_age_preg(age):
    # 'pregage:18-<25 years', 'pregage:25-<30 years', 'pregage:30-<35 years',
    # 'pregage:35-<40 years', 'pregage:40-<45 years', 'pregage:45-50 years'
    encoding = np.zeros((1, 6), dtype='int')
    if age < 18:
        pass
    elif age < 25:
        encoding[0, 0] = 1
    elif age < 30:
        encoding[0, 1] = 1
    elif age < 35:
        encoding[0, 2] = 1
    elif age < 40:
        encoding[0, 3] = 1
    elif age < 45:
        encoding[0, 4] = 1
    elif age <= 50:
        encoding[0, 5] = 1
    else:
        pass

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
    "November 2021", "December 2021", "January 2022", "February 2022", "March 2022",
    "April 2022", "May 2022", "June 2022", "July 2022", "August 2022",
    "September 2022", "October 2022", "November 2022", "December 2022", "January 2023",
    "February 2023",]
    :param index_date:
    :return:
    """
    encoding = np.zeros((1, 36), dtype='int')
    year = index_date.year
    month = index_date.month
    pos = (month - 3) + (year - 2020) * 12
    if pos >= encoding.shape[1]:
        print('In _encoding_yearmonth out of bounds, pos:', pos)
        pos = encoding.shape[1] - 1
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
                ht_select = utils.tofloat(ht)
            if pd.notna(wt):
                wt_select = utils.tofloat(wt)
            if pd.notna(ori_bmi):
                bmi_select = utils.tofloat(ori_bmi)

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
    # impute_value can also be nan, because of no zip code there? how?
    encoding = np.zeros((1, 10), dtype='float')
    if pd.isna(nation_adi):
        nation_adi = impute_value
    if nation_adi >= 100:
        nation_adi = 99
    if nation_adi < 1:
        nation_adi = 1

    if pd.notna(nation_adi):
        pos = int(nation_adi) // 10
        encoding[0, pos] = 1
    return encoding


def _encoding_utilization(enc_list, index_date):
    # encoding uitlization in the baseline period
    # ['inpatient visits', 'outpatient visits', 'emergency visits', 'other visits']
    encoding = np.zeros((1, 4), dtype='float')
    for records in enc_list:
        enc_date, type, enc_id = records[:3]
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
    # '03/20-06/20', '07/20-10/20', '11/20-02/21',
    # '03/21-06/21', '07/21-10/21', '11/21-02/22',
    # '03/22-06/22', '07/22-10/22'
    # encoding = np.zeros((1, 5), dtype='float')
    encoding = np.zeros((1, 8), dtype='float')

    # datetime.datetime(2020, 1, 1, 0, 0),
    # datetime.datetime(2020, 7, 1, 0, 0),
    # datetime.datetime(2020, 11, 1, 0, 0),
    # datetime.datetime(2021, 3, 1, 0, 0),
    # datetime.datetime(2021, 7, 1, 0, 0),
    # datetime.datetime(2021, 12, 30, 0, 0)
    if pd.to_datetime(index_date) < datetime.datetime(2020, 7, 1, 0, 0):
        encoding[0, 0] = 1
    elif pd.to_datetime(index_date) < datetime.datetime(2020, 11, 1, 0, 0):
        encoding[0, 1] = 1
    elif pd.to_datetime(index_date) < datetime.datetime(2021, 3, 1, 0, 0):
        encoding[0, 2] = 1
    elif pd.to_datetime(index_date) < datetime.datetime(2021, 7, 1, 0, 0):
        encoding[0, 3] = 1
    # update 2023-2-6
    elif pd.to_datetime(index_date) < datetime.datetime(2021, 11, 1, 0, 0):
        encoding[0, 4] = 1
    elif pd.to_datetime(index_date) < datetime.datetime(2022, 3, 1, 0, 0):
        encoding[0, 5] = 1
    elif pd.to_datetime(index_date) < datetime.datetime(2022, 7, 1, 0, 0):
        encoding[0, 6] = 1
    elif pd.to_datetime(index_date) < datetime.datetime(2022, 11, 1, 0, 0):
        encoding[0, 7] = 1

    # else:
    #     encoding[0, 4] = 1

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


def _encoding_dx_pregnancy(dx_list, icd_obcpasc, obcpasc_encoding, index_date):
    # encoding 26 obstetric comorbidity codes in the baseline
    encoding = np.zeros((1, len(obcpasc_encoding)), dtype='int')

    for records in dx_list:
        dx_date, icd = records[:2]
        icd = icd.replace('.', '').upper()
        # build baseline
        if ecs._is_in_pregnancy_comorbidity_period(dx_date, index_date):
            # 2022-02-27, change exact match to prefix match of PASC codes
            flag, icdprefix = _prefix_in_set(icd, icd_obcpasc)
            if flag:  # if icd in icd_pasc:
                # if icdprefix != icd:
                #     print(icd, icdprefix)
                pasc_info = icd_obcpasc[icdprefix]
                pasc = pasc_info[0]
                rec = obcpasc_encoding[pasc]
                pos = rec[0]
                encoding[0, pos] += 1

    return encoding


def _encoding_med(med_list, med_column_names, comorbidity_codes, index_date):
    # encoding 2 medications:
    # MEDICATION: Corticosteroids
    # MEDICATION: Immunosuppressant drug
    # ps: (H02: CORTICOSTEROIDS FOR SYSTEMIC USE   L04:IMMUNOSUPPRESSANTS in the baseline)

    encoding = np.zeros((1, len(med_column_names)), dtype='int')
    for records in med_list:
        med_date, rxnorm, supply_days, encid, medtable = records
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
        med_date, rxnorm, supply_days, encid, medtable = records
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


def _encoding_outcome_severe_maternal_morbidity_withalldays(dx_list, icd_smm, smm_encoding, index_date, default_t2e,
                                                            pro_list):
    # 2022-02-18 initialize t2e:  last encounter, event, end of followup, whichever happens first
    outcome_t2e = np.ones((1, len(smm_encoding)), dtype='float') * default_t2e
    outcome_flag = np.zeros((1, len(smm_encoding)), dtype='int')
    outcome_baseline = np.zeros((1, len(smm_encoding)), dtype='int')

    # outcome_tlast = np.zeros((1, len(smm_encoding)), dtype='int')

    outcome_t2eall = [''] * len(smm_encoding)

    code_list = dx_list + pro_list

    for records in code_list:
        dx_date, icd = records[:2]
        icd = icd.replace('.', '').upper()
        # build baseline
        if ecs._is_in_baseline(dx_date, index_date):
            # 2022-02-27, change exact match to prefix match of PASC codes
            flag, icdprefix = _prefix_in_set(icd, icd_smm)
            if flag:  # if icd in icd_pasc:
                pasc_info = icd_smm[icdprefix]
                pasc = pasc_info[0]
                rec = smm_encoding[pasc]
                pos = rec[0]
                outcome_baseline[0, pos] += 1

        # build outcome
        # definition of t2e might be problem when
        if ecs._is_in_followup(dx_date, index_date):
            days = (dx_date - index_date).days
            flag, icdprefix = _prefix_in_set(icd, icd_smm)
            if flag:  # if icd in icd_pasc:
                pasc_info = icd_smm[icdprefix]
                pasc = pasc_info[0]
                rec = smm_encoding[pasc]
                pos = rec[0]

                outcome_t2eall[pos] += '{};'.format(days)
                if outcome_flag[0, pos] == 0:
                    # only records the first event and time, because sorted
                    if days < outcome_t2e[0, pos]:
                        outcome_t2e[0, pos] = days
                    outcome_flag[0, pos] = 1
                    # outcome_tlast[0, pos] = days
                else:
                    outcome_flag[0, pos] += 1

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
        med_date, rxnorm, supply_days, encid, medtable = records
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
        med_date, rxnorm, supply_days, encid, medtable = records
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
        if icd == '':
            # e.g. in mshs, many dx colum is '', not icd10 encoded
            continue
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


def _encoding_outcome_preeclampsia(dx_list, index_date):

    outcome_flag = np.zeros(2, dtype='int')
    outcome_baseline = np.zeros(2, dtype='int')

    icd_pasc = {}
    icd_pasc['O141'] = (0, 'Severe preeclampsia')
    icd_pasc['O142'] = (0, 'Severe preeclampsia')
    icd_pasc['O11'] = (0, 'Severe preeclampsia')
    icd_pasc['O13'] = (1, 'Gestational hypertension')
    icd_pasc['O140'] = (1, 'Gestational hypertension')
    icd_pasc['O149'] = (1, 'Gestational hypertension')

    for records in dx_list:
        dx_date, icd = records[:2]
        icd = icd.replace('.', '').upper()
        # build baseline
        if ecs._is_in_baseline(dx_date, index_date):
            # 2022-02-27, change exact match to prefix match of PASC codes
            flag, icdprefix = _prefix_in_set(icd, icd_pasc)
            if flag:  # if icd in icd_pasc:
                pasc_info = icd_pasc[icdprefix]
                pos = pasc_info[0]
                pasc = pasc_info[1]
                outcome_baseline[pos] += 1

        # build outcome
        if ecs._is_in_followup(dx_date, index_date):
            days = (dx_date - index_date).days
            flag, icdprefix = _prefix_in_set(icd, icd_pasc)
            if flag:  # if icd in icd_pasc:
                pasc_info = icd_pasc[icdprefix]
                pos = pasc_info[0]
                pasc = pasc_info[1]
                outcome_flag[pos] += 1

    return outcome_flag, outcome_baseline


def build_query_1and2_matrix(args):
    start_time = time.time()
    print('In build_query_1and2_matrix...')
    # step 1: load encoding dictionary
    icd_pasc, pasc_encoding, icd_cmr, cmr_encoding, \
        icd_ccsr, ccsr_encoding, rxnorm_ing, rxnorm_atc, atcl2_encoding, atcl3_encoding, atcl4_encoding, \
        ventilation_codes, comorbidity_codes, icd9_icd10, rxing_encoding, covidmed_codes, vaccine_codes, \
        icd_OBC, OBC_encoding, icd_SMMpasc, SMMpasc_encoding = _load_mapping()

    # step 2: load cohorts pickle data
    print('In cohorts_characterization_build_data...')
    if args.dataset == 'all':
        sites = ['mcw', 'nebraska', 'utah', 'utsw',
                 'wcm', 'montefiore', 'mshs', 'columbia', 'nyu',
                 'ufh', 'usf', 'miami', 'nch',  # 'emory',
                 'pitt', 'psu', 'temple', 'michigan',
                 'ochsner', 'ucsf', 'lsu',
                 'vumc']
    else:
        sites = [args.dataset, ]

    # sites = [ 'wcm',]

    print('Try to load: ', sites)

    for site in tqdm(sites):
        print('Loading: ', site)
        input_file = r'../data/recover/output/{}/cohorts_{}_{}.pkl'.format(site, args.cohorts, site)

        print('Load cohorts pickle data file:', input_file)
        id_data = utils.load(input_file, chunk=4)

        # load pregnancy existing matrix
        matrix_data_file = r'../data/recover/output/pregnancy_data/pregnancy_{}.csv'.format(site)
        output_file = r'../data/recover/output/pregnancy_data/pregnancy_{}_withPreeclampsia.csv'.format(args.dataset)

        print('Load data covariates file:', matrix_data_file)
        df_matrix = pd.read_csv(matrix_data_file, dtype={'patid': str, 'site': str, 'zip': str},
                         parse_dates=['index date', 'flag_delivery_date', 'flag_pregnancy_start_date',
                                      'flag_pregnancy_end_date'])
        print('df.shape:', df_matrix.shape)

        # impute adi value by median of site , per site:
        # adi_value_list = [v[1][7] for key, v in id_data.items()]
        # adi_value_default = np.nanmedian(adi_value_list)

        df_matrix['out-Severe preeclampsia'] = 0
        df_matrix['out-Gestational hypertension'] = 0
        df_matrix['base-Severe preeclampsia'] = 0
        df_matrix['base-Gestational hypertension'] = 0

        n_not = 0
        for index, rows in tqdm(df_matrix.iterrows(), total=df_matrix.shape[0], mininterval=10):
            #for pid, item in tqdm(id_data.items(), total=len(id_data), mininterval=10):
            # for i, (pid, item) in tqdm(enumerate(id_data.items()), total=len(id_data)):
            pid = rows['patid']
            if pid not in id_data:
                print('pid', pid, 'is not in id_data')
                n_not+=1
                continue
            item = id_data[pid]
            index_info, demo, dx, med, covid_lab, encounter, procedure, obsgen, immun, death, vital = item
            flag, index_date, covid_loinc, flag_name, index_age, index_enc_id = index_info
            birth_date, gender, race, hispanic, zipcode, state, city, nation_adi, state_adi = demo

            # transform Timestamp to datetime.date when using sql and csv mixture types
            index_date = index_date.date()

            dx = _dx_clean_and_translate_any_ICD9_to_ICD10(dx, icd9_icd10, icd_ccsr)
            outcome_flag, outcome_baseline = _encoding_outcome_preeclampsia(dx, index_date)
            df_matrix.loc[index, 'out-Severe preeclampsia'] = outcome_flag[0]
            df_matrix.loc[index, 'out-Gestational hypertension'] = outcome_flag[1]
            df_matrix.loc[index, 'base-Severe preeclampsia'] = outcome_baseline[0]
            df_matrix.loc[index, 'base-Gestational hypertension'] = outcome_baseline[1]

        print(n_not, 'not found in id_data')
        df_matrix.to_csv(output_file, index=False)
        print('Done site', site, 'Encoding done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

        # end iterate sites

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return df_matrix


if __name__ == '__main__':
    # python pre_data_manuscript_withAllDays.py --dataset ALL --cohorts covid_4manuNegNoCovidV2 2>&1 | tee  log/pre_data_manuscript_withAllDays_covid_4manuNegNoCovidV2.txt
    # python pre_data_manuscript_withAllDays_preg_addCoexist.py --cohorts covid_4manuNegNoCovidV2age18 2>&1 | tee  log/pre_data_manuscript_withAllDays_preg_addCoexist.txt

    start_time = time.time()
    args = parse_args()
    df_matrix = build_query_1and2_matrix(args)

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
