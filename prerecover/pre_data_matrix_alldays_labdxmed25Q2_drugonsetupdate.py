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
import random

import datetime
from misc import utils
import eligibility_setting_test as ecs

print('Using eligibility_setting_test! accelerated datetime')
import functools
import fnmatch

# from lifelines import KaplanMeierFitter, CoxPHFitter

print = functools.partial(print, flush=True)

from iptw.PSModels import ml
from iptw.evaluation import *


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess demographics')
    parser.add_argument('--cohorts', choices=['covid_posneg18base', 'covid_posOnly18base'],
                        default='covid_posOnly18base', help='cohorts')
    parser.add_argument('--dataset', default='wcm_pcornet_all', help='site dataset')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--positive_only', action='store_true')
    # parser.add_argument("--ndays", type=int, default=30)
    parser.add_argument("--random_seed", type=int, default=0)

    args = parser.parse_args()
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

    # 2023-11-9
    zip_ruca = utils.load(r'../data/mapping/zip_ruca_mapping.pkl')
    icd_cci = utils.load(r'../data/mapping/icd_cci_mapping.pkl')
    cci_encoding = utils.load(r'../data/mapping/cci_index_mapping.pkl')
    covid_med_update = utils.load(r'../data/mapping/covid_drugs_updated.pkl')

    icd_addedPASC = utils.load(r'../data/mapping/icd_addedPASC_mapping.pkl')
    addedPASC_encoding = utils.load(r'../data/mapping/addedPASC_index_mapping.pkl')
    icd_brainfog = utils.load(r'../data/mapping/icd_brainfog_mapping.pkl')
    brainfog_encoding = utils.load(r'../data/mapping/brainfog_index_mapping.pkl')

    pax_contra = utils.load(r'../data/mapping/n3c_pax_contraindication.pkl')
    pax_risk = utils.load(r'../data/mapping/n3c_pax_indication.pkl')

    fips_ziplist = utils.load(r'../data/mapping/fips_to_ziplist_2020.pkl')

    icd_cognitive_fatigue_respiratory = utils.load(r'../data/mapping/icd_cognitive-fatigue-respiratory_mapping.pkl')
    cognitive_fatigue_respiratory_encoding = utils.load(
        r'../data/mapping/cognitive-fatigue-respiratory_index_mapping.pkl')

    # 2024-2-13 added pax risk covs
    icd_addedPaxRisk = utils.load(r'../data/mapping/icd_addedPaxRisk_mapping.pkl')
    addedPaxRisk_encoding = utils.load(r'../data/mapping/addedPaxRisk_index_mapping.pkl')

    # add negative control outcomes
    icd_negctrlpasc = utils.load(r'../data/mapping/icd_negative-outcome-control_mapping.pkl')
    negctrlpasc_encoding = utils.load(r'../data/mapping/negative-outcome-control_index_mapping.pkl')

    # add ssri drugs
    ssri_med = utils.load(r'../data/mapping/ssri_snri_drugs_mapping.pkl')

    # add mental covs 2024-09-05
    icd_mental = utils.load(r'../data/mapping/icd_mental_mapping.pkl')
    mental_encoding = utils.load(r'../data/mapping/mental_index_mapping.pkl')

    # add mecfs covs 2024-12-14
    icd_mecfs = utils.load(r'../data/mapping/icd_mecfs_mapping.pkl')
    mecfs_encoding = utils.load(r'../data/mapping/mecfs_index_mapping.pkl')

    # add cvd death covs 2024-12-14
    icd_cvddeath = utils.load(r'../data/mapping/icd_cvddeath_mapping.pkl')
    cvddeath_encoding = utils.load(r'../data/mapping/cvddeath_index_mapping.pkl')

    # add CNS LDN drugs 2025-4-8
    cnsldn_med = utils.load(r'../data/mapping/cns_ldn_drugs_mapping.pkl')

    # add CNS LDN covs 2025-4-8
    icd_covCNSLDN = utils.load(r'../data/mapping/icd_covCNS-LDN_mapping.pkl')
    covCNSLDN_encoding = utils.load(r'../data/mapping/covCNS-LDN_index_mapping.pkl')

    return (icd_pasc, pasc_encoding, icd_cmr, cmr_encoding, icd_ccsr, ccsr_encoding,
            rxnorm_ing, rxnorm_atc, atcl2_encoding, atcl3_encoding, atcl4_encoding,
            ventilation_codes, comorbidity_codes, icd9_icd10, rxing_index, covidmed_codes, vaccine_codes,
            icd_OBC, OBC_encoding, icd_SMMpasc, SMMpasc_encoding,
            zip_ruca, icd_cci, cci_encoding, covid_med_update,
            icd_addedPASC, addedPASC_encoding, icd_brainfog, brainfog_encoding, pax_contra, pax_risk, fips_ziplist,
            icd_cognitive_fatigue_respiratory, cognitive_fatigue_respiratory_encoding,
            icd_addedPaxRisk, addedPaxRisk_encoding, icd_negctrlpasc, negctrlpasc_encoding, ssri_med,
            icd_mental, mental_encoding,
            icd_mecfs, mecfs_encoding, icd_cvddeath, cvddeath_encoding, cnsldn_med, icd_covCNSLDN, covCNSLDN_encoding
            )


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
    if isinstance(gender, str):
        gender = gender.upper()
        if gender == 'F':
            encoding[0, 0] = 1
        elif gender == 'M':
            encoding[0, 1] = 1
        else:
            encoding[0, 2] = 1
    else:
        encoding[0, 2] = 1

    return encoding


def _encoding_race(race):
    # race_column_names = ['Asian', 'Black or African American', 'White', 'Other', 'Missing']
    # Other (American Indian or Alaska Native, Native Hawaiian or Other Pacific Islander, Multiple Race, Other)5
    # Missing (No Information, Refuse to Answer, Unknown, Missing)4
    encoding = np.zeros((1, 5), dtype='int')
    if race == '02':  # Asian
        encoding[0, 0] = 1
    elif race == '03':  # 'Black or African American'
        encoding[0, 1] = 1
    elif race == '05':  # 'White'
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
    "February 2023", --> 2023-12 --> 2024-12]
    :param index_date:
    :return:
    """
    # encoding = np.zeros((1, 46), dtype='int')
    encoding = np.zeros((1, 58), dtype='int')

    year = index_date.year
    month = index_date.month
    pos = int((month - 3) + (year - 2020) * 12)
    # if pos >= encoding.shape[1]:
    #     print('In _encoding_yearmonth out of bounds, pos:', pos)
    #     pos = encoding.shape[1] - 1
    if 0 <= pos < encoding.shape[1]:
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


def _encoding_ruca(zipcode, zip_ruca, fips_ziplist):
    # 'RUCA1@1', 'RUCA1@2', 'RUCA1@3', 'RUCA1@4', 'RUCA1@5',
    # 'RUCA1@6', 'RUCA1@7', 'RUCA1@8', 'RUCA1@9', 'RUCA1@10',
    # 'RUCA1@99', 'ZIPMiss'
    encoding = np.zeros((1, 12), dtype='int')
    zipcode5 = np.nan

    if isinstance(zipcode, str):
        if (len(zipcode) >= 5) and (len(zipcode) <= 9):
            zipcode5 = zipcode[:5]
        elif len(zipcode) >= 10:
            # no zip code, just fips code, then map to its first mapped zip
            if zipcode in fips_ziplist:
                ziplist = fips_ziplist[zipcode]
                if len(ziplist) > 0 and (len(ziplist[0]) >= 5):
                    zipcode5 = ziplist[0][:5]  # just use first

    if pd.notna(zipcode5) and (zipcode5 in zip_ruca):
        rec = zip_ruca[zipcode5]
        ruca = int(rec[0])
        if ruca in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            encoding[0, ruca - 1] = 1
        elif ruca == 99:
            encoding[0, 10] = 1
        else:
            encoding[0, 11] = 1
    else:
        encoding[0, 11] = 1
        ruca = np.nan

    return ruca, encoding


# def _encoding_social(nation_adi, impute_value):
#     # ['ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49',
#     #  'ADI50-59', 'ADI60-69', 'ADI70-79', 'ADI80-89', 'ADI90-100']
#     # impute_value can also be nan, because of no zip code there? how?
#     encoding = np.zeros((1, 10), dtype='float')
#     if pd.isna(nation_adi):
#         nation_adi = impute_value
#     if nation_adi >= 100:
#         nation_adi = 99
#     if nation_adi < 1:
#         nation_adi = 1
#
#     if pd.notna(nation_adi):
#         pos = int(nation_adi) // 10
#         encoding[0, pos] = 1
#     return encoding

def _encoding_social(nation_adi, impute_value):
    # 2023-11-16 add missing, not using imputation
    # ['ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49',
    #  'ADI50-59', 'ADI60-69', 'ADI70-79', 'ADI80-89', 'ADI90-100', 'ADIMissing']
    # impute_value can also be nan, because of no zip code there? how?
    encoding = np.zeros((1, 11), dtype='float')
    if pd.isna(nation_adi):
        pos = 10
    else:
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
        enc_date, type, enc_id = records[:3]
        if ecs._is_in_baseline1year(enc_date, index_date):
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
    # '03/20-06/20', '07/20-10/20', '11/20-02/21',
    # '03/21-06/21', '07/21-10/21', '11/21-02/22',
    # '03/22-06/22', '07/22-10/22', '11/22-02/23',
    # '03/23-06/23', '07/23-10/23', '11/23-02/24',
    # '03/24-06/24', '07/24-10/24', '11/24-02/25',
    # encoding = np.zeros((1, 5), dtype='float')
    # encoding = np.zeros((1, 8), dtype='float')
    # encoding = np.zeros((1, 12), dtype='float')
    encoding = np.zeros((1, 15), dtype='float')

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
    # update 2023-11-10
    elif pd.to_datetime(index_date) < datetime.datetime(2023, 3, 1, 0, 0):
        encoding[0, 8] = 1
    elif pd.to_datetime(index_date) < datetime.datetime(2023, 7, 1, 0, 0):
        encoding[0, 9] = 1
    elif pd.to_datetime(index_date) < datetime.datetime(2023, 11, 1, 0, 0):
        encoding[0, 10] = 1
    elif pd.to_datetime(index_date) < datetime.datetime(2024, 3, 1, 0, 0):
        encoding[0, 11] = 1
    # update 2025-4-21
    elif pd.to_datetime(index_date) < datetime.datetime(2024, 7, 1, 0, 0):
        encoding[0, 12] = 1
    elif pd.to_datetime(index_date) < datetime.datetime(2024, 11, 1, 0, 0):
        encoding[0, 13] = 1
    elif pd.to_datetime(index_date) < datetime.datetime(2025, 3, 1, 0, 0):
        encoding[0, 14] = 1
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


def _encoding_cci_and_score(dx_list, icd_cci, cci_encoding, index_date):
    # encoding 17 cci comorbidity codes in the baseline and two summarized scores: one charlson, one quan. Ref: Quan,
    # H., Li, B., Couris, C.M., Fushimi, K., Graham, P., Hider, P., Januel, J.M. and Sundararajan, V., 2011. Updating
    # and validating the Charlson comorbidity index and score for risk adjustment in hospital discharge abstracts
    # using data from 6 countries. American journal of epidemiology, 173(6), pp.676-682.
    # 'CCI:Myocardial Infarction', 'CCI:Congestive Heart Failure', 'CCI:Periphral Vascular Disease',
    # 'CCI:Cerebrovascular Disease', 'CCI:Dementia', 'CCI:Chronic Pulmonary Disease',
    # 'CCI:Connective Tissue Disease-Rheumatic Disease', 'CCI:Peptic Ulcer Disease',
    # 'CCI:Mild Liver Disease', 'CCI:Diabetes without complications', 'CCI:Diabetes with complications',
    # 'CCI:Paraplegia and Hemiplegia', 'CCI:Renal Disease', 'CCI:Cancer',
    # 'CCI:Moderate or Severe Liver Disease', 'CCI:Metastatic Carcinoma', 'CCI:AIDS/HIV',
    # 'score_cci_charlson' (max 29), 'score_cci_quan' (max 24)
    encoding = np.zeros((1, 17 + 2), dtype='int')
    # last two zeros are for padding, dim consistency
    charlson_weight = np.array([1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1,
                                2, 2, 2, 2, 3,
                                6, 6, 0, 0, ])
    quan_weight = np.array([0, 2, 0, 0, 2,
                            1, 1, 0, 2, 0,
                            1, 2, 1, 2, 4,
                            6, 4, 0, 0])
    for records in dx_list:
        dx_date, icd, dx_type, enc_type = records
        icd = icd.replace('.', '').upper()
        # build baseline
        if ecs._is_in_comorbidity_period(dx_date, index_date):
            flag, icdprefix = _prefix_in_set(icd, icd_cci)
            if flag:
                cci_info = icd_cci[icdprefix]
                cci_category = cci_info[0]
                rec = cci_encoding[cci_category]
                pos = rec[0]
                encoding[0, pos] += 1

    encoding_bool = (encoding >= 1).astype('int').squeeze()
    score_cci_charlson = np.dot(charlson_weight, encoding_bool)
    score_cci_quan = np.dot(quan_weight, encoding_bool)

    encoding[0, -2] = int(score_cci_charlson)
    encoding[0, -1] = int(score_cci_quan)

    return encoding


def _encoding_addPaxRisk(dx_list, icd_cci, cci_encoding, index_date):
    # 'addPaxRisk:Drug Abuse', 'addPaxRisk:Obesity', 'addPaxRisk:tuberculosis',
    encoding = np.zeros((1, 3), dtype='int')

    for records in dx_list:
        dx_date, icd, dx_type, enc_type = records
        icd = icd.replace('.', '').upper()
        # build baseline
        if ecs._is_in_comorbidity_period(dx_date, index_date):
            flag, icdprefix = _prefix_in_set(icd, icd_cci)
            if flag:
                cci_info = icd_cci[icdprefix]
                cci_category = cci_info[0]
                rec = cci_encoding[cci_category]
                pos = rec[0]
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


def _encoding_pax_n3ccov(pax_contra, pax_risk, dx_list, med_list, pro_list, index_date):
    # 'pax_contra' (-14-+14), 'pax_risk' (-3yrs - 0)
    pax_risk_med, pax_risk_pro, pax_risk_dx, pax_risk_other = pax_risk
    encoding = np.zeros((1, 2), dtype='int')
    # deal with dx
    for records in dx_list:
        dx_date, icd, dx_type, enc_type = records
        icd = icd.replace('.', '').upper()
        # build baseline
        if ecs._is_in_comorbidity_period(dx_date, index_date):
            if icd in pax_risk_dx:
                encoding[0, 1] += 1

    for records in med_list:
        med_date, rxnorm, supply_days, encid, medtable = records
        if ecs._is_in_comorbidity_period(med_date, index_date):
            if rxnorm in pax_risk_med:
                encoding[0, 1] += 1
        if ecs._is_in_covid_medication(med_date, index_date):
            if rxnorm in pax_contra:
                encoding[0, 0] += 1

    for records in pro_list:
        px_date, px, px_type, enc_type, enc_id = records
        if ecs._is_in_comorbidity_period(px_date, index_date):
            if px in pax_risk_pro:
                encoding[0, 1] += 1

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


def _encoding_covidtreat(med_list, covidtreat_names, covidtreat_codes, index_date, default_t2e):
    # 2023-11-13
    # capture all records of paxlovid and remdesivir
    #
    encoding = np.zeros((1, len(covidtreat_names)), dtype='int')
    # because this is not for outcome or censoring, just use these covarites to identify drug initiation
    # thus, not modeling censoring or loss of followup here, use 9999 to denote not prescribing
    outcome_t2e = np.ones((1, len(covidtreat_names)), dtype='float') * 9999  # default_t2e
    outcome_t2eall = [''] * len(covidtreat_names)

    for records in med_list:
        med_date, rxnorm, supply_days, encid, medtable = records
        days = (med_date - index_date).days
        for pos, col_name in enumerate(covidtreat_names):
            if rxnorm in covidtreat_codes[col_name]:
                outcome_t2eall[pos] += '{};'.format(days)
                encoding[0, pos] += 1
                if days < outcome_t2e[0, pos]:
                    outcome_t2e[0, pos] = days

    return encoding, outcome_t2e, outcome_t2eall


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


def _encoding_followup_any_dx(index_date, dx):
    if dx:
        for r in dx:
            if ecs._is_in_followup(r[0], index_date):
                return True
    return False


def _encoding_death(death, index_date):
    # death flag, death time
    encoding = np.zeros((1, 2), dtype='int')
    encoding[0, 1] = 9999  # no death date
    if death:
        ddate = death[0]
        if pd.notna(ddate):
            try:
                # if have interpretable death date and records
                encoding[0, 1] = (ddate - index_date).days
                encoding[0, 0] = 1
            except Exception as e:
                # death but no date, regarding no death records.
                encoding[0, 1] = 9999
                encoding[0, 0] = 0

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
        # dx_date, icd = records[:2]
        dx_date, icd, dx_type, enc_type = records
        icd = icd.replace('.', '').upper()
        # build baseline
        if ecs._is_in_baseline(dx_date, index_date):
            # 2022-02-27, change exact match to prefix match of PASC codes
            flag, icdprefix = _prefix_in_set(icd, icd_pasc)
            if flag:
                pasc_info = icd_pasc[icdprefix]
                pasc = pasc_info[0]
                rec = pasc_encoding[pasc]
                pos = rec[0]
                outcome_baseline[0, pos] += 1

        # build outcome
        if ecs._is_in_followup(dx_date, index_date):
            days = (dx_date - index_date).days
            flag, icdprefix = _prefix_in_set(icd, icd_pasc)
            if flag:
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

    # debug # a = pd.DataFrame({'1':pasc_encoding.keys(), '2':outcome_flag.squeeze(), '3':outcome_t2e.squeeze(), '4':outcome_baseline.squeeze()})
    # outcome_t2eall = np.array([outcome_t2eall])
    return outcome_flag, outcome_t2e, outcome_baseline, outcome_t2eall


def _encoding_outcome_dx_withalldaysoveralltime(dx_list, icd_pasc, pasc_encoding, index_date, default_t2e):
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
        # dx_date, icd = records[:2]
        dx_date, icd, dx_type, enc_type = records
        icd = icd.replace('.', '').upper()

        flag, icdprefix = _prefix_in_set(icd, icd_pasc)

        if flag:
            days = (dx_date - index_date).days
            pasc_info = icd_pasc[icdprefix]
            pasc = pasc_info[0]
            rec = pasc_encoding[pasc]
            pos = rec[0]

            outcome_t2eall[pos] += '{};'.format(days)

            if ecs._is_in_baseline(dx_date, index_date):
                outcome_baseline[0, pos] += 1

            if ecs._is_in_followup(dx_date, index_date):
                # days = (dx_date - index_date).days
                # flag, icdprefix = _prefix_in_set(icd, icd_pasc)
                if outcome_flag[0, pos] == 0:
                    # only records the first event and time
                    if days < outcome_t2e[0, pos]:
                        outcome_t2e[0, pos] = days
                    outcome_flag[0, pos] = 1
                    outcome_tlast[0, pos] = days
                else:
                    outcome_flag[0, pos] += 1

    # debug # a = pd.DataFrame({'1':pasc_encoding.keys(), '2':outcome_flag.squeeze(), '3':outcome_t2e.squeeze(), '4':outcome_baseline.squeeze()})
    # outcome_t2eall = np.array([outcome_t2eall])
    return outcome_flag, outcome_t2e, outcome_baseline, outcome_t2eall


def _encoding_outcome_U099_withalldays(dx_list, index_date, default_t2e):
    # outcome_U099_flag = np.zeros((n, 2), dtype='int16')
    # outcome_U099_t2e = np.zeros((n, 2), dtype='int16')
    # outcome_U099_baseline = np.zeros((n, 1), dtype='int16')
    # outcome_U099_t2eall = []  # all time, baseline, acute, post acute, can be negative
    # outcome_U099_column_names = ['U099-acute-flag', 'U099-postacute-flag',
    #                              'U099-acute-t2e', 'U099-postacute-t2e',
    #                              'U099-base', 'U099-t2eall']

    icd_pasc = {'U099', 'B948'}
    # initialize t2e:  last encounter, event, end of followup, whichever happens first
    # 0: acute, 1: post acute
    outcome_t2e = np.ones((1, 2), dtype='float') * default_t2e
    outcome_flag = np.zeros((1, 2), dtype='int')
    outcome_baseline = np.zeros((1, 1), dtype='int')
    # outcome_tlast = np.zeros((1, 2), dtype='int')
    outcome_t2eall = [''] * 1
    outcome_t2ecodeall = [''] * 1

    for records in dx_list:
        # dx_date, icd = records[:2]
        dx_date, icd, dx_type, enc_type = records
        icd = icd.replace('.', '').upper()

        flag, icdprefix = _prefix_in_set(icd, icd_pasc)
        if flag:
            days = (dx_date - index_date).days
            # save t2e all time, -, 0, +
            outcome_t2eall[0] += '{};'.format(days)
            outcome_t2ecodeall[0] += '{}_{};'.format(icd, days)
            # build baseline
            if ecs._is_in_baseline(dx_date, index_date):
                outcome_baseline[0, 0] += 1

            if ecs._is_in_acute(dx_date, index_date):
                if outcome_flag[0, 0] == 0:
                    # only records the first event and time
                    if days < outcome_t2e[0, 0]:
                        outcome_t2e[0, 0] = days
                    outcome_flag[0, 0] = 1
                    # outcome_tlast[0, pos] = days
                else:
                    outcome_flag[0, 0] += 1

            if ecs._is_in_followup(dx_date, index_date):
                if outcome_flag[0, 1] == 0:
                    # only records the first event and time
                    if days < outcome_t2e[0, 1]:
                        outcome_t2e[0, 1] = days
                    outcome_flag[0, 1] = 1
                else:
                    outcome_flag[0, 1] += 1

    # debug # a = pd.DataFrame({'1':pasc_encoding.keys(), '2':outcome_flag.squeeze(), '3':outcome_t2e.squeeze(), '4':outcome_baseline.squeeze()})
    # outcome_t2eall = np.array([outcome_t2eall])
    return outcome_flag, outcome_t2e, outcome_baseline, outcome_t2eall, outcome_t2ecodeall


def _encoding_outcome_hospitalization_withalldays(enc_list, index_date, default_t2e):
    # outcome_hospitalization_flag = np.zeros((n, 2), dtype='int16')
    # outcome_hospitalization_t2e = np.zeros((n, 2), dtype='int16')
    # outcome_hospitalization_baseline = np.zeros((n, 1), dtype='int16')
    # outcome_hospitalization_t2eall = []  # all time, baseline, acute, post acute, can be negative
    # outcome_hospitalization_column_names = ['hospitalization-acute-flag', 'hospitalization-postacute-flag',
    #                                         'hospitalization-acute-t2e', 'hospitalization-postacute-t2e',
    #                                         'hospitalization-base', 'hospitalization-t2eall']

    enc_type_list = {'EI', 'IP', 'OS'}
    # initialize t2e:  last encounter, event, end of followup, whichever happens first
    # 0: acute, 1: post acute
    outcome_t2e = np.ones((1, 2), dtype='float') * default_t2e
    outcome_flag = np.zeros((1, 2), dtype='int')
    outcome_baseline = np.zeros((1, 1), dtype='int')
    # outcome_tlast = np.zeros((1, 2), dtype='int')
    outcome_t2eall = [''] * 1

    for records in enc_list:
        # dx_date, icd = records[:2]
        # dx_date, icd, dx_type, enc_type = records
        enc_date, etype, enc_id = records[:3]
        if isinstance(etype, str):
            etype = etype.strip().upper()

        if etype in enc_type_list:
            days = (enc_date - index_date).days
            # save t2e all time, -, 0, +
            outcome_t2eall[0] += '{};'.format(days)

            # build baseline
            if ecs._is_in_baseline(enc_date, index_date):
                outcome_baseline[0, 0] += 1

            if ecs._is_in_acute(enc_date, index_date):
                if outcome_flag[0, 0] == 0:
                    # only records the first event and time
                    if days < outcome_t2e[0, 0]:
                        outcome_t2e[0, 0] = days
                    outcome_flag[0, 0] = 1
                    # outcome_tlast[0, pos] = days
                else:
                    outcome_flag[0, 0] += 1

            if ecs._is_in_followup(enc_date, index_date):
                if outcome_flag[0, 1] == 0:
                    # only records the first event and time
                    if days < outcome_t2e[0, 1]:
                        outcome_t2e[0, 1] = days
                    outcome_flag[0, 1] = 1
                else:
                    outcome_flag[0, 1] += 1

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


def build_feature_matrix(args):
    start_time = time.time()

    # 1. Load target file
    infile_name = '../iptw/cns_output/Matrix-cns-adhd-CNS-ADHD-acuteIncident-0-30-25Q2-v3.csv'
    df_cohort = pd.read_csv(infile_name,
                            dtype={'patid': str, 'site': str, 'zip': str},
                            parse_dates=['index date', 'dob', 'index_date_drug'])
    print('df.shape:', df_cohort.shape)

    days_since_covid_match_pool = df_cohort.loc[
        (df_cohort['treated'] == 1) & (df_cohort['ADHD_before_drug_onset'] == 1),
        'index_date_drug_days']
    # use return: days_since_covid_match_pool.sample().iloc[0]

    output_file_query12_bool = infile_name.replace('.csv', '-drugonsetupdate.csv')

    output_med_info = infile_name.replace('.csv', '-info_medication_cohorts-drugonsetupdate.csv')
    # r'../data/recover/output/{}/info_medication_cohorts_{}_{}.csv'.format(args.dataset, args.cohorts, args.dataset)

    output_dx_info = infile_name.replace('.csv', '-info_dx_cohorts-drugonsetupdate.csv')
    # r'../data/recover/output/{}/info_dx_cohorts_{}_{}.csv'.format(args.dataset, args.cohorts, args.dataset)

    #
    print('In build_feature_matrix...')
    # step 1: load encoding dictionary
    (icd_pasc, pasc_encoding, icd_cmr, cmr_encoding,
     icd_ccsr, ccsr_encoding, rxnorm_ing, rxnorm_atc, atcl2_encoding, atcl3_encoding, atcl4_encoding,
     ventilation_codes, comorbidity_codes, icd9_icd10, rxing_encoding, covidmed_codes, vaccine_codes,
     icd_OBC, OBC_encoding, icd_SMMpasc, SMMpasc_encoding,
     zip_ruca, icd_cci, cci_encoding, covid_med_update,
     icd_addedPASC, addedPASC_encoding, icd_brainfog, brainfog_encoding, pax_contra, pax_risk,
     fips_ziplist, icd_CFR, CFR_encoding, icd_addedPaxRisk, addedPaxRisk_encoding, icd_negctrlpasc,
     negctrlpasc_encoding, ssri_med,
     icd_mental, mental_encoding, icd_mecfs, mecfs_encoding, icd_cvddeath, cvddeath_encoding,
     cnsldn_med, icd_covCNSLDN, covCNSLDN_encoding) = _load_mapping()

    # step 2: load cohorts pickle data
    print('In cohorts_characterization_build_data...')
    # if args.dataset == 'ALL':
    #     sites = ['NYU', 'MONTE', 'COL', 'MSHS', 'WCM']
    # else:
    #     sites = [args.dataset, ]
    # sites = [args.dataset, ]
    med_count = {}
    dx_count = {}

    # print('Try to load: ', sites)
    header = True
    mode = "w"
    previous_site = None
    i = -1
    for index, row in tqdm(df_cohort.iterrows(), total=len(df_cohort)):
        # 'index date', 'flag_delivery_date', 'flag_pregnancy_start_date', 'flag_pregnancy_end_date'
        index_date_covid = row['index date']  # pd.to_datetime(row['index date'])
        index_date_drug = row['index_date_drug']
        index_date_drug_days = row['index_date_drug_days']
        site = row['site']
        pid = row['patid']
        treated = row['treated']
        if treated == 0:
            index_date_drug_days = days_since_covid_match_pool.sample().iloc[0]
            index_date_drug = index_date_covid + datetime.timedelta(days=index_date_drug_days)

        if site != previous_site:
            print('previous_site:', previous_site, 'site:', site)
            print('Loading: ', site)
            previous_site = site
            input_file = r'../data/recover/output/{}/cohorts_{}_{}.pkl.gz'.format(site, args.cohorts, site)
            #
            print('Load cohorts pickle data file:', input_file)
            id_data = utils.load(input_file, num_cpu=8)  # chunk=4,

            # impute adi value by median of site , per site:
            adi_value_list = [v[1][7] for key, v in id_data.items() if len(v[1]) > 0]
            adi_value_default = np.nanmedian(adi_value_list)

        item = id_data.get(pid, [])
        # for pid, item in tqdm(id_data.items(), total=len(id_data), mininterval=10):
        if item:
            # for i, (pid, item) in tqdm(enumerate(id_data.items()), total=len(id_data)):
            index_info, demo, dx_raw, med, covid_lab, encounter, procedure, obsgen, immun, death, vital, id_lab_select = item
            flag, index_date, covid_loinc, flag_name, index_age, index_enc_id, index_enc_type, index_source = index_info
            if len(demo) == 0:
                print(pid, ':Warning! no demo info, skip this patient')
                continue
            birth_date, gender, race, hispanic, zipcode, state, city, nation_adi, state_adi = demo

            # transform Timestamp to datetime.date when using sql and csv mixture types
            index_date = index_date.date()
            index_date_drug = index_date_drug.date()

            dx = _dx_clean_and_translate_any_ICD9_to_ICD10(dx_raw, icd9_icd10, icd_ccsr)

            # stream dump per line
            n = 1
            pid_list = []
            site_list = []
            covid_list = []
            indexdate_list = []  # newly add 2022-02-20
            hospitalized_list = []
            ventilation_list = []
            criticalcare_list = []

            maxfollowtime_list = []  # newly add 2022-02-18
            followupanydx_list = []  # 2023-11-10, because we didn't require this in the cohort generation part
            deathdate_list = []  # 2023-11-10
            death_array = np.zeros((n, 2), dtype='int16')  # newly add 2022-02-20, change 2023-11-10, as a outcome
            death_column_names = ['death date', 'death', 'death t2e']

            # newly add 2022-04-08
            zip_list = []  # newly add 2022-04-08
            dob_list = []  # newly add 2023-2-9
            age_list = []
            adi_list = []
            ruca_list = []
            utilization_count_array = np.zeros((n, 4), dtype='int16')
            utilization_count_names = ['inpatient no.', 'outpatient no.', 'emergency visits no.', 'other visits no.']
            bmi_list = []
            yearmonth_array = np.zeros((n, 58), dtype='int16')  # 46
            yearmonth_column_names = [
                "YM: March 2020", "YM: April 2020", "YM: May 2020", "YM: June 2020", "YM: July 2020",
                "YM: August 2020", "YM: September 2020", "YM: October 2020", "YM: November 2020", "YM: December 2020",
                "YM: January 2021", "YM: February 2021", "YM: March 2021", "YM: April 2021", "YM: May 2021",
                "YM: June 2021", "YM: July 2021", "YM: August 2021", "YM: September 2021", "YM: October 2021",
                "YM: November 2021", "YM: December 2021",
                "YM: January 2022", "YM: February 2022", "YM: March 2022", "YM: April 2022", "YM: May 2022",
                "YM: June 2022", "YM: July 2022", "YM: August 2022", "YM: September 2022",
                "YM: October 2022", "YM: November 2022", "YM: December 2022",
                "YM: January 2023", "YM: February 2023", "YM: March 2023", "YM: April 2023", "YM: May 2023",
                "YM: June 2023", "YM: July 2023", "YM: August 2023", "YM: September 2023", "YM: October 2023",
                "YM: November 2023", "YM: December 2023",

                "YM: January 2024", "YM: February 2024", "YM: March 2024", "YM: April 2024", "YM: May 2024",
                "YM: June 2024", "YM: July 2024", "YM: August 2024", "YM: September 2024", "YM: October 2024",
                "YM: November 2024", "YM: December 2024",
            ]
            #
            age_array = np.zeros((n, 6), dtype='int16')
            age_column_names = ['20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75-<85 years',
                                '85+ years']

            age_preg_array = np.zeros((n, 6), dtype='int16')
            age_preg_column_names = ['pregage:18-<25 years', 'pregage:25-<30 years', 'pregage:30-<35 years',
                                     'pregage:35-<40 years', 'pregage:40-<45 years', 'pregage:45-50 years']

            gender_array = np.zeros((n, 3), dtype='int16')
            gender_column_names = ['Female', 'Male', 'Other/Missing']

            race_array = np.zeros((n, 5), dtype='int16')
            race_column_names = ['Asian', 'Black or African American', 'White', 'Other', 'Missing']
            # Other (American Indian or Alaska Native, Native Hawaiian or Other Pacific Islander, Multiple Race, Other)5
            # Missing (No Information, Refuse to Answer, Unknown, Missing)4

            hispanic_array = np.zeros((n, 3), dtype='int16')
            hispanic_column_names = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']

            # newly added 2022-02-18
            social_array = np.zeros((n, 11), dtype='int16')
            social_column_names = ['ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49',
                                   'ADI50-59', 'ADI60-69', 'ADI70-79', 'ADI80-89', 'ADI90-100',
                                   'ADIMissing']
            # 2023-11-13 RUCA codes
            ruca_array = np.zeros((n, 12), dtype='int16')
            ruca_column_names = ['RUCA1@1', 'RUCA1@2', 'RUCA1@3', 'RUCA1@4', 'RUCA1@5',
                                 'RUCA1@6', 'RUCA1@7', 'RUCA1@8', 'RUCA1@9', 'RUCA1@10',
                                 'RUCA1@99', 'ZIPMissing']

            utilization_array = np.zeros((n, 12), dtype='int16')
            utilization_column_names = ['inpatient visits 0', 'inpatient visits 1-2', 'inpatient visits 3-4',
                                        'inpatient visits >=5',
                                        'outpatient visits 0', 'outpatient visits 1-2', 'outpatient visits 3-4',
                                        'outpatient visits >=5',
                                        'emergency visits 0', 'emergency visits 1-2', 'emergency visits 3-4',
                                        'emergency visits >=5']

            # index_period_array = np.zeros((n, 5), dtype='int16')
            # index_period_names = ['03/20-06/20', '07/20-10/20', '11/20-02/21', '03/21-06/21', '07/21-11/21']
            index_period_array = np.zeros((n, 15), dtype='int16')
            index_period_names = ['03/20-06/20', '07/20-10/20', '11/20-02/21',
                                  '03/21-06/21', '07/21-10/21', '11/21-02/22',
                                  '03/22-06/22', '07/22-10/22', '11/22-02/23',
                                  '03/23-06/23', '07/23-10/23', '11/23-02/24',
                                  '03/24-06/24', '07/24-10/24', '11/24-02/25', ]
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

            # 2023-11-12 CCI comorbidity and summarized score
            cci_array = np.zeros((n, 17 + 2), dtype='int16')  #
            cci_column_names = [
                'CCI:Myocardial Infarction', 'CCI:Congestive Heart Failure', 'CCI:Periphral Vascular Disease',
                'CCI:Cerebrovascular Disease', 'CCI:Dementia', 'CCI:Chronic Pulmonary Disease',
                'CCI:Connective Tissue Disease-Rheumatic Disease', 'CCI:Peptic Ulcer Disease', 'CCI:Mild Liver Disease',
                'CCI:Diabetes without complications', 'CCI:Diabetes with complications',
                'CCI:Paraplegia and Hemiplegia',
                'CCI:Renal Disease', 'CCI:Cancer', 'CCI:Moderate or Severe Liver Disease', 'CCI:Metastatic Carcinoma',
                'CCI:AIDS/HIV',
                'score_cci_charlson', 'score_cci_quan']

            # 2023-2-10 add Obstetric Comorbidity
            obcdx_array = np.zeros((n, len(OBC_encoding)), dtype='int16')  # 26-dim
            obcdx_column_names = list(OBC_encoding.keys())  # obc:

            # 2023-11014
            # baseline covs for potential pax EC
            pax_n3ccov_names = ['pax_contra', 'pax_risk']
            pax_n3ccov_array = np.zeros((n, 2), dtype='int16')

            # 2023-11-13 paxlovid and remdesivir updated codes
            covidtreat_names = ['paxlovid', 'remdesivir']
            covidtreat_flag = np.zeros((n, 2), dtype='int16')
            covidtreat_t2e = np.zeros((n, 2), dtype='int16')  # date of earliest prescriptions
            covidtreat_t2eall = []
            covidtreat_column_names = (
                    ['treat-flag@' + x for x in covidtreat_names] +
                    ['treat-t2e@' + x for x in covidtreat_names] +
                    ['treat-t2eall@' + x for x in covidtreat_names])

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

            ## 2023-11-13 add some new pasc categories, e.g., smell/taste, brain fog, etc.
            outcome_addedPASC_flag = np.zeros((n, 11), dtype='int16')
            outcome_addedPASC_t2e = np.zeros((n, 11), dtype='int16')
            outcome_addedPASC_baseline = np.zeros((n, 11), dtype='int16')
            outcome_addedPASC_t2eall = []
            outcome_addedPASC_column_names = ['dxadd-out@' + x for x in addedPASC_encoding.keys()] + \
                                             ['dxadd-t2e@' + x for x in addedPASC_encoding.keys()] + \
                                             ['dxadd-base@' + x for x in addedPASC_encoding.keys()] + \
                                             ['dxadd-t2eall@' + x for x in addedPASC_encoding.keys()]

            # brain fog
            outcome_brainfog_flag = np.zeros((n, 7), dtype='int16')
            outcome_brainfog_t2e = np.zeros((n, 7), dtype='int16')
            outcome_brainfog_baseline = np.zeros((n, 7), dtype='int16')
            outcome_brainfog_t2eall = []
            outcome_brainfog_column_names = ['dxbrainfog-out@' + x for x in brainfog_encoding.keys()] + \
                                            ['dxbrainfog-t2e@' + x for x in brainfog_encoding.keys()] + \
                                            ['dxbrainfog-base@' + x for x in brainfog_encoding.keys()] + \
                                            ['dxbrainfog-t2eall@' + x for x in brainfog_encoding.keys()]

            # cognitive_fatigue_respiratory
            outcome_CFR_flag = np.zeros((n, 3), dtype='int16')
            outcome_CFR_t2e = np.zeros((n, 3), dtype='int16')
            outcome_CFR_baseline = np.zeros((n, 3), dtype='int16')
            outcome_CFR_t2eall = []
            outcome_CFR_column_names = (
                    ['dxCFR-out@' + x for x in CFR_encoding.keys()] + \
                    ['dxCFR-t2e@' + x for x in CFR_encoding.keys()] + \
                    ['dxCFR-base@' + x for x in CFR_encoding.keys()] + \
                    ['dxCFR-t2eall@' + x for x in CFR_encoding.keys()])

            addPaxRisk_array = np.zeros((n, 3), dtype='int16')  #
            addPaxRisk_column_names = [
                'addPaxRisk:Drug Abuse', 'addPaxRisk:Obesity', 'addPaxRisk:tuberculosis',
            ]

            # rxing_encoding outcome. Commented out 2023-11-13, used later when necessary
            # outcome_med_flag = np.zeros((n, 434), dtype='int16')
            # outcome_med_t2e = np.zeros((n, 434), dtype='int16')
            # outcome_med_baseline = np.zeros((n, 434), dtype='int16')
            # outcome_med_column_names = ['med-out@' + x for x in rxing_encoding.keys()] + \
            #                            ['med-t2e@' + x for x in rxing_encoding.keys()] + \
            #                            ['med-base@' + x for x in rxing_encoding.keys()]

            # acute and postacute U099
            # U099 acute, post acute
            outcome_U099_flag = np.zeros((n, 2), dtype='int16')
            outcome_U099_t2e = np.zeros((n, 2), dtype='int16')
            outcome_U099_baseline = np.zeros((n, 1), dtype='int16')
            outcome_U099_t2eall = []  # all time, baseline, acute, post acute, can be negative
            outcome_U099_t2ecodeall = []
            outcome_U099_column_names = ['U099-acute-flag', 'U099-postacute-flag',
                                         'U099-acute-t2e', 'U099-postacute-t2e',
                                         'U099-base', 'U099-t2eall', 'U099-t2ecodeall']

            outcome_hospitalization_flag = np.zeros((n, 2), dtype='int16')
            outcome_hospitalization_t2e = np.zeros((n, 2), dtype='int16')
            outcome_hospitalization_baseline = np.zeros((n, 1), dtype='int16')
            outcome_hospitalization_t2eall = []  # all time, baseline, acute, post acute, can be negative
            outcome_hospitalization_column_names = ['hospitalization-acute-flag', 'hospitalization-postacute-flag',
                                                    'hospitalization-acute-t2e', 'hospitalization-postacute-t2e',
                                                    'hospitalization-base', 'hospitalization-t2eall']

            #
            # 2023-2-10 severe maternal morbidity
            outcome_smm_flag = np.zeros((n, len(SMMpasc_encoding)), dtype='int16')  # 21
            outcome_smm_t2e = np.zeros((n, len(SMMpasc_encoding)), dtype='int16')
            outcome_smm_baseline = np.zeros((n, len(SMMpasc_encoding)), dtype='int16')
            outcome_smm_t2eall = []

            outcome_smm_column_names = ['smm-out@' + x for x in SMMpasc_encoding.keys()] + \
                                       ['smm-t2e@' + x for x in SMMpasc_encoding.keys()] + \
                                       ['smm-base@' + x for x in SMMpasc_encoding.keys()] + \
                                       ['smm-t2eall@' + x for x in SMMpasc_encoding.keys()]

            # 2025-4-21 add from SSRI, CNS, etc previous add columns work into primary table
            # 2024-4-3 ssri, snri drug updated codes
            # 2024-7-28 add more
            # 2024-09-06 'wellbutrin' --> change name to --> bupropion
            ssritreat_names = ['fluvoxamine', 'fluoxetine', 'escitalopram', 'citalopram', 'sertraline',
                               'paroxetine', 'vilazodone',  # ssri
                               'desvenlafaxine', 'duloxetine', 'levomilnacipran', 'milnacipran', 'venlafaxine',  # snri
                               'bupropion']  # others # 'wellbutrin'
            ssritreat_flag = np.zeros((n, 13), dtype='int16')
            ssritreat_t2e = np.zeros((n, 13), dtype='int16')  # date of earliest prescriptions
            ssritreat_t2eall = []
            ssritreat_column_names = (
                    ['treat-flag@' + x for x in ssritreat_names] +
                    ['treat-t2e@' + x for x in ssritreat_names] +
                    ['treat-t2eall@' + x for x in ssritreat_names])

            # 2025-04-08 add CNS and LDN drug and guanfacine
            cnsldn_names = [
                'naltrexone', 'LDN_name', 'adderall_combo', 'lisdexamfetamine', 'methylphenidate',
                'amphetamine', 'amphetamine_nocombo', 'dextroamphetamine', 'dextroamphetamine_nocombo', 'modafinil',
                'pitolisant', 'solriamfetol', 'armodafinil', 'atomoxetine', 'benzphetamine',
                'azstarys_combo', 'dexmethylphenidate', 'dexmethylphenidate_nocombo', 'diethylpropion',
                'methamphetamine',
                'phendimetrazine', 'phentermine', 'caffeine', 'fenfluramine_delet', 'oxybate_delet',
                'doxapram_delet', 'guanfacine']
            cnsldntreat_flag = np.zeros((n, 27), dtype='int16')
            cnsldntreat_t2e = np.zeros((n, 27), dtype='int16')  # date of earliest prescriptions
            cnsldntreat_t2eall = []
            cnsldntreat_column_names = (
                    ['cnsldn-flag@' + x for x in cnsldn_names] +
                    ['cnsldn-t2e@' + x for x in cnsldn_names] +
                    ['cnsldn-t2eall@' + x for x in cnsldn_names])

            # 2025-4-8 add CNS and LDN related COVs, reuse outcome functions
            # currently there are 5 dim covs, will add later
            outcome_covCNSLDN_flag = np.zeros((n, 5), dtype='int16')
            outcome_covCNSLDN_t2e = np.zeros((n, 5), dtype='int16')
            outcome_covCNSLDN_baseline = np.zeros((n, 5), dtype='int16')
            outcome_covCNSLDN_t2eall = []
            outcome_covCNSLDN_column_names = (
                    ['dxcovCNSLDN-out@' + x for x in covCNSLDN_encoding.keys()] + \
                    ['dxcovCNSLDN-t2e@' + x for x in covCNSLDN_encoding.keys()] + \
                    ['dxcovCNSLDN-base@' + x for x in covCNSLDN_encoding.keys()] + \
                    ['dxcovCNSLDN-t2eall@' + x for x in covCNSLDN_encoding.keys()])

            # 2024-09-26 add more mental covs breakdown, majorly for baseline covs
            # print('len(mental_encoding), should be 13', len(mental_encoding))
            outcome_mental_flag = np.zeros((n, 13), dtype='int16')  # use 13 to warn if changed len(mental_encoding)
            outcome_mental_t2e = np.zeros((n, 13), dtype='int16')
            outcome_mental_baseline = np.zeros((n, 13), dtype='int16')
            outcome_mental_t2eall = []
            outcome_mental_column_names = ['mental-out@' + x for x in mental_encoding.keys()] + \
                                          ['mental-t2e@' + x for x in mental_encoding.keys()] + \
                                          ['mental-base@' + x for x in mental_encoding.keys()] + \
                                          ['mental-t2eall@' + x for x in mental_encoding.keys()]

            # 2024-12-14 add ME/CFS 1-dim outcome
            outcome_mecfs_flag = np.zeros((n, 1), dtype='int16')
            outcome_mecfs_t2e = np.zeros((n, 1), dtype='int16')
            outcome_mecfs_baseline = np.zeros((n, 1), dtype='int16')
            outcome_mecfs_t2eall = []
            outcome_mecfs_column_names = (
                    ['dxMECFS-out@' + x for x in mecfs_encoding.keys()] + \
                    ['dxMECFS-t2e@' + x for x in mecfs_encoding.keys()] + \
                    ['dxMECFS-base@' + x for x in mecfs_encoding.keys()] + \
                    ['dxMECFS-t2eall@' + x for x in mecfs_encoding.keys()])

            # 2024-12-14 cvd for death attribution
            outcome_cvddeath_flag = np.zeros((n, 1), dtype='int16')
            outcome_cvddeath_t2e = np.zeros((n, 1), dtype='int16')
            outcome_cvddeath_baseline = np.zeros((n, 1), dtype='int16')
            outcome_cvddeath_t2eall = []
            outcome_cvddeath_column_names = (
                    ['dxCVDdeath-out@' + x for x in cvddeath_encoding.keys()] + \
                    ['dxCVDdeath-t2e@' + x for x in cvddeath_encoding.keys()] + \
                    ['dxCVDdeath-base@' + x for x in cvddeath_encoding.keys()] + \
                    ['dxCVDdeath-t2eall@' + x for x in cvddeath_encoding.keys()])

            #
            column_names = ['patid', 'site', 'covid', 'index date', 'index_date_drug', 'index_date_drug_days', 'treated',
                            'hospitalized',
                            'ventilation', 'criticalcare', 'maxfollowup', 'followupanydx'] + death_column_names + \
                           ['zip', 'dob', 'age', 'adi', 'RUCA1'] + utilization_count_names + [
                               'bmi'] + yearmonth_column_names + \
                           age_column_names + age_preg_column_names + \
                           gender_column_names + race_column_names + hispanic_column_names + \
                           social_column_names + ruca_column_names + utilization_column_names + index_period_names + \
                           bmi_names + smoking_names + \
                           dx_column_names + med_column_names + vaccine_column_names + cci_column_names + \
                           obcdx_column_names + pax_n3ccov_names + \
                           covidtreat_column_names + covidmed_column_names + outcome_covidmed_column_names + \
                           outcome_column_names + outcome_addedPASC_column_names + outcome_brainfog_column_names + \
                           outcome_CFR_column_names + addPaxRisk_column_names + outcome_U099_column_names + \
                           outcome_hospitalization_column_names + outcome_smm_column_names + \
                           ssritreat_column_names + cnsldntreat_column_names + outcome_covCNSLDN_column_names + \
                           outcome_mental_column_names + outcome_mecfs_column_names + outcome_cvddeath_column_names  # + outcome_med_column_names

            # if args.positive_only:
            #     if not flag:
            #         continue
            # i += 1
            # update only 1 row here
            i = 0
            # maxfollowtime
            # gaurantee at least one encounter in baseline or followup. thus can be 0 if no followup
            # later EC should be at lease one in follow-up

            pid_list.append(pid)
            site_list.append(site)

            covid_list.append(flag)
            indexdate_list.append(index_date)

            indexdate_list.extend([index_date, index_date_drug, index_date_drug_days, treated])

            inpatient_flag = _encoding_inpatient(dx_raw, index_date)
            hospitalized_list.append(inpatient_flag)

            vent_flag = _encoding_ventilation(procedure, obsgen, index_date, ventilation_codes)
            ventilation_list.append(vent_flag)

            criticalcare_flag = _encoding_critical_care(procedure, index_date)
            criticalcare_list.append(criticalcare_flag)

            maxfollowtime = _encoding_maxfollowtime(index_date, encounter, dx_raw, med)
            maxfollowtime_list.append(maxfollowtime)

            flag_followup_any_dx = _encoding_followup_any_dx(index_date, dx_raw)
            followupanydx_list.append(flag_followup_any_dx)

            # encode death
            if death:
                death_date = death[0]
                deathdate_list.append(death_date)
            else:
                deathdate_list.append(np.nan)

            death_array[i, :] = _encoding_death(death, index_date)

            # newly add 2022-04-08
            zip_list.append(zipcode)
            dob_list.append(birth_date)
            age_list.append(index_age)
            adi_list.append(nation_adi)

            # 2023-11-13, ruca_array computed here but added after adi array later
            ruca1, ruca_array[i, :] = _encoding_ruca(zipcode, zip_ruca, fips_ziplist)
            ruca_list.append(ruca1)
            # utilization count, postponed to below
            # bmi_list, postponed to below
            yearmonth_array[i, :] = _encoding_yearmonth(index_date)

            # encoding query 1 information
            age_array[i, :] = _encoding_age(index_age)
            age_preg_array[i, :] = _encoding_age_preg(index_age)  # 18-50, bin every 5 years

            gender_array[i] = _encoding_gender(gender)
            race_array[i, :] = _encoding_race(race)
            hispanic_array[i, :] = _encoding_hispanic(hispanic)
            #
            social_array[i, :] = _encoding_social(nation_adi, adi_value_default)
            utilization_array[i, :], utilization_count_array[i, :] = _encoding_utilization(encounter, index_date_drug) #index_date)
            index_period_array[i, :] = _encoding_index_period(index_date)
            #

            # encoding bmi and smoking
            bmi_array[i, :], smoking_array[i, :], bmi = _encoding_bmi_and_smoking(vital, index_date_drug) #index_date)
            bmi_list.append(bmi)

            # encoding query 2 information
            dx_array[i, :] = _encoding_dx(dx, dx_column_names, comorbidity_codes, index_date_drug, procedure) # index_date
            med_array[i, :] = _encoding_med(med, med_column_names, comorbidity_codes, index_date_drug) #index_date)

            # encoding pasc information in both baseline and followup
            # time 2 event: censoring in the database (should be >= followup start time),
            # maximum follow-up, death-time, event, whichever comes first
            default_t2e = np.min([
                np.maximum(ecs.FOLLOWUP_LEFT, maxfollowtime),
                np.maximum(ecs.FOLLOWUP_LEFT, ecs.FOLLOWUP_RIGHT),
                np.maximum(ecs.FOLLOWUP_LEFT, death_array[i, 1])
            ])

            vaccine_array[i, :] = _encoding_vaccine_4risk(procedure, immun, vaccine_column_names, vaccine_codes,
                                                          index_date_drug) #index_date)

            cci_array[i, :] = _encoding_cci_and_score(dx_raw, icd_cci, cci_encoding, index_date_drug) #index_date)

            obcdx_array[i, :] = _encoding_dx_pregnancy(dx, icd_OBC, OBC_encoding, index_date_drug) #index_date)

            # 2023-11-14 n3c messy 2 covs: 'pax_contra', 'pax_risk'
            pax_n3ccov_array[i, :] = _encoding_pax_n3ccov(pax_contra, pax_risk, dx_raw, med, procedure, index_date_drug) #index_date)

            # 2023-11-13 updated capture of paxlovid and remdesivir
            covidtreat_flag[i, :], covidtreat_t2e[i, :], covidtreat_t2eall_1row = \
                _encoding_covidtreat(med, covidtreat_names, covid_med_update, index_date_drug, default_t2e) # index_date
            covidtreat_t2eall.append(covidtreat_t2eall_1row)
            #

            covidmed_array[i, :], \
            outcome_covidmed_flag[i, :], outcome_covidmed_t2e[i, :], outcome_covidmed_baseline[i, :] \
                = _encoding_covidmed(med, procedure, covidmed_column_names, covidmed_codes, index_date_drug, default_t2e) # index_date

            # outcome_flag[i, :], outcome_t2e[i, :], outcome_baseline[i, :] = \
            #     _encoding_outcome_dx(dx, icd_pasc, pasc_encoding, index_date, default_t2e)

            outcome_flag[i, :], outcome_t2e[i, :], outcome_baseline[i, :], outcome_t2eall_1row = \
                _encoding_outcome_dx_withalldays(dx, icd_pasc, pasc_encoding, index_date, default_t2e) #still actertain pasc during 30-180
            outcome_t2eall.append(outcome_t2eall_1row)
            #
            # # added PASC, 2023-11-13
            outcome_addedPASC_flag[i, :], outcome_addedPASC_t2e[i, :], outcome_addedPASC_baseline[i,
                                                                       :], outcome_addedPASC_t2eall_1row = \
                _encoding_outcome_dx_withalldays(dx, icd_addedPASC, addedPASC_encoding, index_date, default_t2e)
            outcome_addedPASC_t2eall.append(outcome_addedPASC_t2eall_1row)

            # brain fog, 2023-11-13
            outcome_brainfog_flag[i, :], outcome_brainfog_t2e[i, :], outcome_brainfog_baseline[i,
                                                                     :], outcome_brainfog_t2eall_1row = \
                _encoding_outcome_dx_withalldays(dx, icd_brainfog, brainfog_encoding, index_date, default_t2e)
            outcome_brainfog_t2eall.append(outcome_brainfog_t2eall_1row)

            # med columns might be commented out for brevity. rereun when needed, 2023-11-13
            # outcome_med_flag[i, :], outcome_med_t2e[i, :], outcome_med_baseline[i, :] = \
            #     _encoding_outcome_med_rxnorm_ingredient(med, rxnorm_ing, rxing_encoding, index_date, default_t2e)

            # add Cognitive, Fatigue, Respiratory condiiotn, 2023-11-13
            outcome_CFR_flag[i, :], outcome_CFR_t2e[i, :], outcome_CFR_baseline[i, :], outcome_CFR_t2eall_1row = \
                _encoding_outcome_dx_withalldays(dx, icd_CFR, CFR_encoding, index_date, default_t2e)
            outcome_CFR_t2eall.append(outcome_CFR_t2eall_1row)

            addPaxRisk_array[i, :] = _encoding_addPaxRisk(dx_raw, icd_addedPaxRisk, addedPaxRisk_encoding, index_date)

            # acute U099, 2024-2-23
            (outcome_U099_flag[i, :], outcome_U099_t2e[i, :], outcome_U099_baseline[i, :], outcome_U099_t2eall_1row,
             outcome_U099_t2ecodeall_1row) = \
                _encoding_outcome_U099_withalldays(dx, index_date, default_t2e)
            outcome_U099_t2eall.append(outcome_U099_t2eall_1row)
            outcome_U099_t2ecodeall.append(outcome_U099_t2ecodeall_1row)

            # acute hospitalization, 2024-2-23
            outcome_hospitalization_flag[i, :], outcome_hospitalization_t2e[i, :], outcome_hospitalization_baseline[i,
                                                                                   :], outcome_hospitalization_t2eall_1row = \
                _encoding_outcome_hospitalization_withalldays(encounter, index_date, default_t2e)
            outcome_hospitalization_t2eall.append(outcome_hospitalization_t2eall_1row)

            ##
            outcome_smm_flag[i, :], outcome_smm_t2e[i, :], outcome_smm_baseline[i, :], outcome_smm_t2eall_1row = \
                _encoding_outcome_severe_maternal_morbidity_withalldays(
                    dx, icd_SMMpasc, SMMpasc_encoding, index_date, default_t2e, procedure)
            outcome_smm_t2eall.append(outcome_smm_t2eall_1row)

            ###
            # 2024-4-3  capture ssri and snri for exploration
            ssritreat_flag[i, :], ssritreat_t2e[i, :], ssritreat_t2eall_1row = \
                _encoding_covidtreat(med, ssritreat_names, ssri_med, index_date, default_t2e)
            ssritreat_t2eall.append(ssritreat_t2eall_1row)
            #

            # 2025-4-8  capture CNS and LDN for exploration
            cnsldntreat_flag[i, :], cnsldntreat_t2e[i, :], cnsldntreat_t2eall_1row = \
                _encoding_covidtreat(med, cnsldn_names, cnsldn_med, index_date, default_t2e)
            cnsldntreat_t2eall.append(cnsldntreat_t2eall_1row)

            # add CNSLDN covariates condition, 2025-4-8
            outcome_covCNSLDN_flag[i, :], outcome_covCNSLDN_t2e[i, :], outcome_covCNSLDN_baseline[i,
                                                                       :], outcome_covCNSLDN_t2eall_1row = \
                _encoding_outcome_dx_withalldaysoveralltime(dx, icd_covCNSLDN, covCNSLDN_encoding, index_date_drug, #index_date,
                                                            default_t2e)
            outcome_covCNSLDN_t2eall.append(outcome_covCNSLDN_t2eall_1row)

            # mental cov breakdown, majorly use baseline portion, 2024-9-6
            # baseline are counts can be > 1, and t2eall are only stored during post-acute phase
            # if want to store time all time, need revision
            outcome_mental_flag[i, :], outcome_mental_t2e[i, :], outcome_mental_baseline[i,
                                                                 :], outcome_mental_t2eall_1row = \
                _encoding_outcome_dx_withalldaysoveralltime(dx, icd_mental, mental_encoding, index_date_drug, default_t2e) #index_date
            outcome_mental_t2eall.append(outcome_mental_t2eall_1row)

            # add ME/CFS condition, 2024-12-14
            outcome_mecfs_flag[i, :], outcome_mecfs_t2e[i, :], outcome_mecfs_baseline[i, :], outcome_mecfs_t2eall_1row = \
                _encoding_outcome_dx_withalldaysoveralltime(dx, icd_mecfs, mecfs_encoding, index_date, default_t2e)
            outcome_mecfs_t2eall.append(outcome_mecfs_t2eall_1row)

            # add CVD death condition, 2024-12-14
            # option 1, CVD outcome post-acute (use this one for consistency, implict implies CVD diagnosis close to death)
            # only check post-acute associated dx and death.
            # not store a dimension for acute death assocaited dx, need to extract from stored all time dimension
            # option 2, CVD outcome acute + postacute
            outcome_cvddeath_flag[i, :], outcome_cvddeath_t2e[i, :], outcome_cvddeath_baseline[i,
                                                                     :], outcome_cvddeath_t2eall_1row = \
                _encoding_outcome_dx_withalldaysoveralltime(dx, icd_cvddeath, cvddeath_encoding, index_date,
                                                            default_t2e)
            outcome_cvddeath_t2eall.append(outcome_cvddeath_t2eall_1row)

            #   step 4: build pandas, column, and dump
            data_array = np.hstack((np.asarray(pid_list).reshape(-1, 1),
                                    np.asarray(site_list).reshape(-1, 1),
                                    np.array(covid_list).reshape(-1, 1).astype(int),
                                    np.asarray(indexdate_list).reshape(-1, 1),
                                    np.asarray(hospitalized_list).reshape(-1, 1).astype(int),
                                    np.array(ventilation_list).reshape(-1, 1).astype(int),
                                    np.array(criticalcare_list).reshape(-1, 1).astype(int),
                                    np.array(maxfollowtime_list).reshape(-1, 1),
                                    np.array(followupanydx_list).reshape(-1, 1).astype(int),
                                    np.array(deathdate_list).reshape(-1, 1),
                                    death_array,
                                    np.asarray(zip_list).reshape(-1, 1),
                                    np.asarray(dob_list).reshape(-1, 1),
                                    np.asarray(age_list).reshape(-1, 1),
                                    np.asarray(adi_list).reshape(-1, 1),
                                    np.asarray(ruca_list).reshape(-1, 1),
                                    utilization_count_array,
                                    np.asarray(bmi_list).reshape(-1, 1),
                                    yearmonth_array,
                                    age_array,
                                    age_preg_array,
                                    gender_array,
                                    race_array,
                                    hispanic_array,
                                    social_array,
                                    ruca_array,
                                    utilization_array,
                                    index_period_array,
                                    bmi_array,
                                    smoking_array,
                                    dx_array,
                                    med_array,
                                    vaccine_array,
                                    cci_array,
                                    obcdx_array,
                                    pax_n3ccov_array,
                                    covidtreat_flag,
                                    covidtreat_t2e,
                                    np.asarray(covidtreat_t2eall),
                                    covidmed_array,
                                    outcome_covidmed_flag,
                                    outcome_covidmed_t2e,
                                    outcome_covidmed_baseline,
                                    outcome_flag,
                                    outcome_t2e,
                                    outcome_baseline,
                                    np.asarray(outcome_t2eall),
                                    outcome_addedPASC_flag,
                                    outcome_addedPASC_t2e,
                                    outcome_addedPASC_baseline,
                                    np.asarray(outcome_addedPASC_t2eall),
                                    outcome_brainfog_flag,
                                    outcome_brainfog_t2e,
                                    outcome_brainfog_baseline,
                                    np.asarray(outcome_brainfog_t2eall),
                                    outcome_CFR_flag,
                                    outcome_CFR_t2e,
                                    outcome_CFR_baseline,
                                    np.asarray(outcome_CFR_t2eall),
                                    addPaxRisk_array,
                                    outcome_U099_flag,
                                    outcome_U099_t2e,
                                    outcome_U099_baseline,
                                    np.asarray(outcome_U099_t2eall),
                                    np.asarray(outcome_U099_t2ecodeall),
                                    outcome_hospitalization_flag,
                                    outcome_hospitalization_t2e,
                                    outcome_hospitalization_baseline,
                                    np.asarray(outcome_hospitalization_t2eall),
                                    outcome_smm_flag,
                                    outcome_smm_t2e,
                                    outcome_smm_baseline,
                                    np.asarray(outcome_smm_t2eall),
                                    ssritreat_flag,
                                    ssritreat_t2e,
                                    np.asarray(ssritreat_t2eall),
                                    cnsldntreat_flag,
                                    cnsldntreat_t2e,
                                    np.asarray(cnsldntreat_t2eall),
                                    outcome_covCNSLDN_flag,
                                    outcome_covCNSLDN_t2e,
                                    outcome_covCNSLDN_baseline,
                                    np.asarray(outcome_covCNSLDN_t2eall),
                                    outcome_mental_flag,
                                    outcome_mental_t2e,
                                    outcome_mental_baseline,
                                    np.asarray(outcome_mental_t2eall),
                                    outcome_mecfs_flag,
                                    outcome_mecfs_t2e,
                                    outcome_mecfs_baseline,
                                    np.asarray(outcome_mecfs_t2eall),
                                    outcome_cvddeath_flag,
                                    outcome_cvddeath_t2e,
                                    outcome_cvddeath_baseline,
                                    np.asarray(outcome_cvddeath_t2eall),
                                    ))

            df_data = pd.DataFrame(data_array, columns=column_names)
            # data_all_sites.append(df_data)

            # Baseline comorbidities required at lest 2 appearance
            # transform count to bool with threshold 2, and deal with "DX: Hypertension and Type 1 or 2 Diabetes Diagnosis"
            # df_bool = df_data_all_sites.copy()  # not using deep copy for the sake of time
            # df_bool = df_data_all_sites
            df_bool = df_data
            # selected_cols = [x for x in df_bool.columns if (
            #         x.startswith('DX:') or
            #         x.startswith('MEDICATION:') or
            #         x.startswith('obc:')
            # )]
            # Baseline covariates to adjust for
            # 2023-11-14 baseline covariates also count, turn to bool later >= 1 or >= 2
            # df_bool.loc[:, selected_cols] = (df_bool.loc[:, selected_cols].astype('int') >= 2).astype('int')
            # df_bool.loc[:, r"DX: Hypertension and Type 1 or 2 Diabetes Diagnosis"] = \ (df_bool.loc[:,
            # r'DX: Hypertension'] & ( df_bool.loc[:, r'DX: Diabetes Type 1'] | df_bool.loc[:, r'DX: Diabetes Type
            # 2'])).astype('int')

            df_bool.loc[:, r"DX: Hypertension and Type 1 or 2 Diabetes Diagnosis"] = \
                ((df_bool.loc[:, r'DX: Hypertension'] >= 1) & (
                        (df_bool.loc[:, r'DX: Diabetes Type 1'] >= 1) | (
                        df_bool.loc[:, r'DX: Diabetes Type 2'] >= 1))).astype('int')

            # Warning: the covid medication part is not boolean
            # keep the value of baseline count and outcome count in the file, filter later depends on the application
            # df_data.loc[:, covidmed_column_names] = (df_data.loc[:, covidmed_column_names].astype('int') >= 1).astype('int')
            # can be done later

            # Baseline flag for incident outcomes
            # 2022-11-3
            # Only binarize baseline flag --> stringent manner. The outcome keeps count
            # selected_cols = [x for x in df_bool.columns if
            #                  (x.startswith('dx-base@') or
            #                   x.startswith('med-base@') or
            #                   x.startswith('covidmed-base@') or
            #                   x.startswith('smm-base@')
            #                   )]  # x.startswith('med-out@') or x.startswith('covidmed-out@') or
            selected_cols = [x for x in df_bool.columns if
                             (x.startswith('dx-base@') or
                              x.startswith('dxadd-base@') or
                              x.startswith('dxbrainfog-base@') or
                              x.startswith('covidmed-base@') or
                              x.startswith('smm-base@') or
                              x.startswith('dxdxCFR-base@')
                              )]
            df_bool.loc[:, selected_cols] = (df_bool.loc[:, selected_cols].astype('int') >= 1).astype('int')

            # selected_cols = [x for x in df_bool.columns if (x.startswith('dx-out@'))]
            # df_bool.loc[:, selected_cols] = (df_bool.loc[:, selected_cols].astype('int') >= 2).astype('int')
            # utils.check_and_mkdir(args.output_file_query12_bool)

            df_bool.to_csv(output_file_query12_bool, mode=mode, header=header, index=False)
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

    print('Done! Dump data bool matrix for query12 to {}'.format(output_file_query12_bool))
    print('Encoding done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    # print('df_data.shape:', df_data.shape)
    # del id_data
    print('Done site:', site)
        # end iterate sites

    dx_count_df = pd.DataFrame.from_dict(dx_count, orient='index',
                                         columns=['total', 'no. in positive group', 'no. in negative group',
                                                  'incident total', 'incident no. in positive group',
                                                  'incident no. in negative group'])
    dx_count_df.to_csv(output_dx_info)

    med_count_df = pd.DataFrame.from_dict(med_count, orient='index',
                                          columns=['total', 'no. in positive group', 'no. in negative group',
                                                   'incident total', 'incident no. in positive group',
                                                   'incident no. in negative group'])
    med_count_df.to_csv(output_med_info)

    # df_data_all_sites = pd.concat(data_all_sites)
    # print('df_data_all_sites.shape:', df_data_all_sites.shape)

    # utils.check_and_mkdir(args.output_file_query12)
    # df_data_all_sites.to_csv(args.output_file_query12)
    # print('Done! Dump data matrix for query12 to {}'.format(args.output_file_query12))

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return df_bool, dx_count_df, med_count_df


if __name__ == '__main__':
    # python pre_data_manuscript.py --dataset ALL --cohorts covid_4manuscript 2>&1 | tee  log/pre_data_manuscript.txt
    # python pre_data_manuscript.py --dataset ALL --cohorts covid_4manuNegNoCovid 2>&1 | tee  log/pre_data_manuscript_covid_4manuNegNoCovid.txt
    # python pre_data_manuscript_withAllDays.py --dataset ALL --cohorts covid_4manuNegNoCovidV2 2>&1 | tee  log/pre_data_manuscript_withAllDays_covid_4manuNegNoCovidV2.txt

    # enrich_med_rwd_info_4_neruo_and_pulmonary()
    #
    start_time = time.time()
    args = parse_args()
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)

    df_data_bool, dx_count_df, med_count_df = build_feature_matrix(args)

    # in_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuscript_bool_ALL.csv'
    # df_data = pd.read_csv(in_file, dtype={'patid': str}, parse_dates=['index date'])

    # cohorts_table_generation(args)
    # de_novo_medication_analyse(cohorts='covid_4screen_Covid+', dataset='ALL', severity='')
    # de_novo_medication_analyse_selected_and_iptw(cohorts='covid_4screen_Covid+', dataset='ALL', severity='')
    # df_med = enrich_med_rwd_info()
    # df_dx, df_pasc_withrwd = rwd_dx_and_pasc_comparison()
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
