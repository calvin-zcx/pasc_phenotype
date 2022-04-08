import sys
# for linux env.
import pandas as pd

sys.path.insert(0,'..')
import time
import pickle
import argparse
from misc import utils
from eligibility_setting import _is_in_baseline, _is_in_followup, INDEX_AGE_MINIMUM
import functools
from collections import Counter
print = functools.partial(print, flush=True)


# 1. aggregate preprocessed data [covid lab, demo, diagnosis, medication] into cohorts data structure
# 2. applying EC [eligibility criteria] to include and exclude patients for each cohorts
# 3. summarize basic statistics when applying each EC criterion


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess cohorts')
    parser.add_argument('--dataset', choices=['COL', 'MSHS', 'MONTE', 'NYU', 'WCM'], default='COL', help='site dataset')
    parser.add_argument('--positive_only', action='store_true')

    args = parser.parse_args()

    args.covid_lab_file = r'../data/V15_COVID19/output/{}/patient_covid_lab_{}.pkl'.format(args.dataset, args.dataset)
    args.demo_file = r'../data/V15_COVID19/output/{}/patient_demo_{}.pkl'.format(args.dataset, args.dataset)
    args.dx_file = r'../data/V15_COVID19/output/{}/diagnosis_{}.pkl'.format(args.dataset, args.dataset)
    args.med_file = r'../data/V15_COVID19/output/{}/medication_{}.pkl'.format(args.dataset, args.dataset)
    args.enc_file = r'../data/V15_COVID19/output/{}/encounter_{}.pkl'.format(args.dataset, args.dataset)
    # added 2022-02-02
    args.pro_file = r'../data/V15_COVID19/output/{}/procedures_{}.pkl'.format(args.dataset, args.dataset)
    args.obsgen_file = r'../data/V15_COVID19/output/{}/obs_gen_{}.pkl'.format(args.dataset, args.dataset)
    args.immun_file = r'../data/V15_COVID19/output/{}/immunization_{}.pkl'.format(args.dataset, args.dataset)
    # added 2022-02-20
    args.death_file = r'../data/V15_COVID19/output/{}/death_{}.pkl'.format(args.dataset, args.dataset)
    # added 2022-04-08
    args.vital_file = r'../data/V15_COVID19/output/{}/vital_{}.pkl'.format(args.dataset, args.dataset)

    args.pasc_list_file = r'../data/mapping/PASC_Adult_Combined_List_20220127_v3.xlsx'
    args.covid_list_file = r'../data/V15_COVID19/covid_phenotype/COVID_ICD.xlsx'

    # changed 2022-04-08 V2, add vital information in V2
    # args.output_file_covid = r'../data/V15_COVID19/output/{}/cohorts_covid_4manuNegNoCovid_{}.pkl'.format(args.dataset, args.dataset)
    args.output_file_covid = r'../data/V15_COVID19/output/{}/cohorts_covid_4manuNegNoCovidV2_{}.pkl'.format(args.dataset, args.dataset)

    print('args:', args)
    return args


def read_preprocessed_data(args):
    """
    load pre-processed data, [make potential transformation]
    :param args contains all data file names
    :return: a list of imported pickle files
    :Notice:
        1.COL data: e.g:
        df.shape: (16666999, 19)

        2. WCM data: e.g.
        df.shape: (47319049, 19)

    """
    start_time = time.time()
    # Load covid list covid_list_file
    df_covid_list = pd.read_excel(args.covid_list_file, sheet_name=r'Sheet1')
    print('df_covid_list.shape', df_covid_list.shape)
    covid_codes_set = set([x.upper().strip().replace('.', '') for x in df_covid_list['Code']])  # .to_list()
    print('Load COVID phenotyping ICD codes list done from {}\nlen(pasc_codes)'.format(args.covid_list_file),
          len(df_covid_list['Code']), 'len(covid_codes_set):', len(covid_codes_set))

    # Load pasc list
    df_pasc_list = pd.read_excel(args.pasc_list_file, sheet_name=r'PASC Screening List', usecols="A:N")
    print('df_pasc_list.shape', df_pasc_list.shape)
    # pasc_codes = df_pasc_list['ICD-10-CM Code'].str.upper().replace('.', '', regex=False)  # .to_list()
    pasc_codes = [x.upper().strip().replace('.', '') for x in df_pasc_list['ICD-10-CM Code']]  # .to_list()
    pasc_codes_set = set(pasc_codes)
    print('Load compiled pasc list done from {}\nlen(pasc_codes)'.format(args.pasc_list_file),
          len(pasc_codes), 'len(pasc_codes_set):', len(pasc_codes_set))

    # load pcr/antigen covid patients, and their corresponding infos
    id_lab = utils.load(args.covid_lab_file)
    id_demo = utils.load(args.demo_file)
    id_dx = utils.load(args.dx_file)
    id_med = utils.load(args.med_file)
    id_enc = utils.load(args.enc_file)
    id_pro = utils.load(args.pro_file)
    id_obsgen = utils.load(args.obsgen_file)
    id_immun = utils.load(args.immun_file)
    id_death = utils.load(args.death_file)
    id_vital = utils.load(args.vital_file)

    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return covid_codes_set, df_pasc_list, pasc_codes_set, id_lab, id_demo, id_dx, id_med, id_enc, id_pro, id_obsgen,\
           id_immun, id_death, id_vital


def _eligibility_age(id_indexrecord, age_minimum_criterion):
    print("Step: applying _eligibility_age, exclude index age < ", age_minimum_criterion,
          'input cohorts size:', len(id_indexrecord))
    N = len(id_indexrecord)
    n_pos_before = n_neg_before = 0
    n_pos_after = n_neg_after = 0
    n_pos_exclude = n_neg_exclude = 0
    exclude_list = []
    for pid, row in id_indexrecord.items():
        # (True/False, lab_date, lab_code, result_label, age)
        covid_flag = row[0]
        age = row[4]
        if covid_flag:
            n_pos_before += 1
        else:
            n_neg_before += 1

        if age < age_minimum_criterion:
            # exclude, DEFAULT 20
            exclude_list.append(pid)
            if covid_flag:
                n_pos_exclude += 1
            else:
                n_neg_exclude += 1
        else:
            # include
            if covid_flag:
                n_pos_after += 1
            else:
                n_neg_after += 1
    # Applying excluding:
    # print('exclude_list', exclude_list)
    [id_indexrecord.pop(pid, None) for pid in exclude_list]
    # Summary:
    print('...Before EC, total: {}\tpos: {}\tneg: {}'.format(N, n_pos_before, n_neg_before))
    print('...Excluding, total: {}\tpos: {}\tneg: {}'.format(len(exclude_list), n_pos_exclude, n_neg_exclude))
    print('...After  EC, total: {}\tpos: {}\tneg: {}'.format(len(id_indexrecord), n_pos_after, n_neg_after))

    return id_indexrecord


def _eligibility_baseline_any_dx(id_indexrecord, id_dx, func_is_in_baseline):
    print("Step: applying _eligibility_baseline_any_dx",
          'input cohorts size:', len(id_indexrecord))
    N = len(id_indexrecord)
    n_pos_before = n_neg_before = 0
    n_pos_after = n_neg_after = 0
    n_pos_exclude = n_neg_exclude = 0
    exclude_list = []
    for pid, row in id_indexrecord.items():
        # (True/False, lab_date, lab_code, result_label, age)
        covid_flag = row[0]
        index_date = row[1]
        v_dx = id_dx.get(pid, [])
        if covid_flag:
            n_pos_before += 1
        else:
            n_neg_before += 1

        if not v_dx:
            exclude_list.append(pid)
            if covid_flag:
                n_pos_exclude += 1
            else:
                n_neg_exclude += 1
        else:
            flag_baseline = False
            for r in v_dx:
                dx_date = r[0]
                if func_is_in_baseline(dx_date, index_date):
                    flag_baseline = True
                    break

            if flag_baseline:
                if covid_flag:
                    n_pos_after += 1
                else:
                    n_neg_after += 1
            else:
                exclude_list.append(pid)
                if covid_flag:
                    n_pos_exclude += 1
                else:
                    n_neg_exclude += 1

    # Applying excluding:
    # print('exclude_list', exclude_list)
    [id_indexrecord.pop(pid, None) for pid in exclude_list]
    # Summary:
    print('...Before EC, total: {}\tpos: {}\tneg: {}'.format(N, n_pos_before, n_neg_before))
    print('...Excluding, total: {}\tpos: {}\tneg: {}'.format(len(exclude_list), n_pos_exclude, n_neg_exclude))
    print('...After  EC, total: {}\tpos: {}\tneg: {}'.format(len(id_indexrecord), n_pos_after, n_neg_after))

    return id_indexrecord


def _eligibility_followup_any_dx(id_indexrecord, id_dx, func_is_in_followup):
    print("Step: applying _eligibility_followup_any_dx",
          'input cohorts size:', len(id_indexrecord))
    N = len(id_indexrecord)
    n_pos_before = n_neg_before = 0
    n_pos_after = n_neg_after = 0
    n_pos_exclude = n_neg_exclude = 0
    exclude_list = []
    for pid, row in id_indexrecord.items():
        # (True/False, lab_date, lab_code, result_label, age)
        covid_flag = row[0]
        index_date = row[1]
        v_dx = id_dx.get(pid, [])
        if covid_flag:
            n_pos_before += 1
        else:
            n_neg_before += 1

        if not v_dx:
            exclude_list.append(pid)
            if covid_flag:
                n_pos_exclude += 1
            else:
                n_neg_exclude += 1
        else:
            flag_followup = False
            for r in v_dx:
                dx_date = r[0]
                dx = r[1].replace('.', '').upper().strip()
                dx_type = r[2]
                # if int(dx_type) == 9:
                #     print('icd code 9:', dx)
                if func_is_in_followup(dx_date, index_date):
                    flag_followup = True
                    break

            if flag_followup:
                if covid_flag:
                    n_pos_after += 1
                else:
                    n_neg_after += 1
            else:
                exclude_list.append(pid)
                if covid_flag:
                    n_pos_exclude += 1
                else:
                    n_neg_exclude += 1

    # Applying excluding:
    # print('exclude_list', exclude_list)
    [id_indexrecord.pop(pid, None) for pid in exclude_list]
    # Summary:
    print('...Before EC, total: {}\tpos: {}\tneg: {}'.format(N, n_pos_before, n_neg_before))
    print('...Excluding, total: {}\tpos: {}\tneg: {}'.format(len(exclude_list), n_pos_exclude, n_neg_exclude))
    print('...After  EC, total: {}\tpos: {}\tneg: {}'.format(len(id_indexrecord), n_pos_after, n_neg_after))

    return id_indexrecord


def _eligibility_baseline_or_followup_any_dx(id_indexrecord, id_dx, func_is_in_baseline, func_is_in_followup):
    print("Step: applying _eligibility_baseline_or_followup_any_dx",
          'input cohorts size:', len(id_indexrecord))
    N = len(id_indexrecord)
    n_pos_before = n_neg_before = 0
    n_pos_after = n_neg_after = 0
    n_pos_exclude = n_neg_exclude = 0
    exclude_list = []
    for pid, row in id_indexrecord.items():
        # (True/False, lab_date, lab_code, result_label, age)
        covid_flag = row[0]
        index_date = row[1]
        v_dx = id_dx.get(pid, [])
        if covid_flag:
            n_pos_before += 1
        else:
            n_neg_before += 1

        if not v_dx:
            exclude_list.append(pid)
            if covid_flag:
                n_pos_exclude += 1
            else:
                n_neg_exclude += 1
        else:
            flag_baseline_or_followup = False
            for r in v_dx:
                dx_date = r[0]
                if func_is_in_baseline(dx_date, index_date) or func_is_in_followup(dx_date, index_date):
                    flag_baseline_or_followup = True
                    break

            if flag_baseline_or_followup:
                if covid_flag:
                    n_pos_after += 1
                else:
                    n_neg_after += 1
            else:
                exclude_list.append(pid)
                if covid_flag:
                    n_pos_exclude += 1
                else:
                    n_neg_exclude += 1

    # Applying excluding:
    # print('exclude_list', exclude_list)
    [id_indexrecord.pop(pid, None) for pid in exclude_list]
    # Summary:
    print('...Before EC, total: {}\tpos: {}\tneg: {}'.format(N, n_pos_before, n_neg_before))
    print('...Excluding, total: {}\tpos: {}\tneg: {}'.format(len(exclude_list), n_pos_exclude, n_neg_exclude))
    print('...After  EC, total: {}\tpos: {}\tneg: {}'.format(len(id_indexrecord), n_pos_after, n_neg_after))

    return id_indexrecord


def _eligibility_followup_any_pasc(id_indexrecord, id_dx, pasc_codes_set, func_is_in_followup):
    print("Step: applying _eligibility_followup_any_pasc",
          'input cohorts size:', len(id_indexrecord))
    N = len(id_indexrecord)
    n_pos_before = n_neg_before = 0
    n_pos_after = n_neg_after = 0
    n_pos_exclude = n_neg_exclude = 0
    exclude_list = []
    for pid, row in id_indexrecord.items():
        # (True/False, lab_date, lab_code, result_label, age)
        covid_flag = row[0]
        index_date = row[1]
        v_dx = id_dx.get(pid, [])
        if covid_flag:
            n_pos_before += 1
        else:
            n_neg_before += 1

        if not v_dx:
            exclude_list.append(pid)
            if covid_flag:
                n_pos_exclude += 1
            else:
                n_neg_exclude += 1
        else:
            flag_followup = False
            for r in v_dx:
                dx_date = r[0]
                dx = r[1].replace('.', '').upper()
                dx_type = r[2]
                # if int(dx_type) == 9:
                #     print('icd code 9:', dx)

                if func_is_in_followup(dx_date, index_date) and (dx in pasc_codes_set):
                    flag_followup = True
                    break

            if flag_followup:
                if covid_flag:
                    n_pos_after += 1
                else:
                    n_neg_after += 1
            else:
                exclude_list.append(pid)
                if covid_flag:
                    n_pos_exclude += 1
                else:
                    n_neg_exclude += 1

    # Applying excluding:
    # print('exclude_list', exclude_list)
    [id_indexrecord.pop(pid, None) for pid in exclude_list]
    # Summary:
    print('...Before EC, total: {}\tpos: {}\tneg: {}'.format(N, n_pos_before, n_neg_before))
    print('...Excluding, total: {}\tpos: {}\tneg: {}'.format(len(exclude_list), n_pos_exclude, n_neg_exclude))
    print('...After  EC, total: {}\tpos: {}\tneg: {}'.format(len(id_indexrecord), n_pos_after, n_neg_after))

    return id_indexrecord


def _eligibility_baseline_no_pasc(id_indexrecord, id_dx, pasc_codes_set, func_is_in_baseline):
    print("Step: applying _eligibility_baseline_no_pasc",
          'input cohorts size:', len(id_indexrecord))
    N = len(id_indexrecord)
    n_pos_before = n_neg_before = 0
    n_pos_after = n_neg_after = 0
    n_pos_exclude = n_neg_exclude = 0
    exclude_list = []
    for pid, row in id_indexrecord.items():
        # (True/False, lab_date, lab_code, result_label, age)
        covid_flag = row[0]
        index_date = row[1]
        v_dx = id_dx.get(pid, [])
        if covid_flag:
            n_pos_before += 1
        else:
            n_neg_before += 1

        if not v_dx:
            # include, because no pasc
            if covid_flag:
                n_pos_after += 1
            else:
                n_neg_after += 1
        else:
            flag_has_baseline_pasc = False
            for r in v_dx:
                dx_date = r[0]
                dx = r[1].replace('.', '').upper()
                dx_type = r[2]
                # if int(dx_type) == 9:
                #     print('icd code 9:', dx)
                if func_is_in_baseline(dx_date, index_date) and (dx in pasc_codes_set):
                    flag_has_baseline_pasc = True
                    break

            if flag_has_baseline_pasc:
                exclude_list.append(pid)
                if covid_flag:
                    n_pos_exclude += 1
                else:
                    n_neg_exclude += 1
            else:
                if covid_flag:
                    n_pos_after += 1
                else:
                    n_neg_after += 1

    # Applying excluding:
    # print('exclude_list', exclude_list)
    [id_indexrecord.pop(pid, None) for pid in exclude_list]
    # Summary:
    print('...Before EC, total: {}\tpos: {}\tneg: {}'.format(N, n_pos_before, n_neg_before))
    print('...Excluding, total: {}\tpos: {}\tneg: {}'.format(len(exclude_list), n_pos_exclude, n_neg_exclude))
    print('...After  EC, total: {}\tpos: {}\tneg: {}'.format(len(id_indexrecord), n_pos_after, n_neg_after))

    return id_indexrecord


def _eligibility_negative_followup_no_covid_dx(id_indexrecord, id_dx, covid_codes_set, func_is_in_followup):
    print("Step: applying _eligibility_negative_followup_no_covid",
          'input cohorts size:', len(id_indexrecord))
    N = len(id_indexrecord)
    n_pos_before = n_neg_before = 0
    n_pos_after = n_neg_after = 0
    n_pos_exclude = n_neg_exclude = 0
    exclude_list = []
    for pid, row in id_indexrecord.items():
        # (True/False, lab_date, lab_code, result_label, age)
        covid_flag = row[0]
        index_date = row[1]
        v_dx = id_dx.get(pid, [])
        if covid_flag:
            n_pos_before += 1
        else:
            n_neg_before += 1

        if not v_dx:
            # include, because no dx.covid for negative
            if covid_flag:
                n_pos_after += 1
            else:
                n_neg_after += 1
        else:
            flag_negative_has_followup_covid = False
            for r in v_dx:
                dx_date = r[0]
                dx = r[1].replace('.', '').upper().strip()
                dx_type = r[2]
                # if int(dx_type) == 9:
                #     print('icd code 9:', dx)
                if func_is_in_followup(dx_date, index_date) and (dx in covid_codes_set) and (not covid_flag):
                    flag_negative_has_followup_covid = True
                    break

            if flag_negative_has_followup_covid:
                exclude_list.append(pid)
                if covid_flag:
                    n_pos_exclude += 1
                else:
                    n_neg_exclude += 1
            else:
                if covid_flag:
                    n_pos_after += 1
                else:
                    n_neg_after += 1

    # Applying excluding:
    # print('exclude_list', exclude_list)
    [id_indexrecord.pop(pid, None) for pid in exclude_list]
    # Summary:
    print('...Before EC, total: {}\tpos: {}\tneg: {}'.format(N, n_pos_before, n_neg_before))
    print('...Excluding, total: {}\tpos: {}\tneg: {}'.format(len(exclude_list), n_pos_exclude, n_neg_exclude))
    print('...After  EC, total: {}\tpos: {}\tneg: {}'.format(len(id_indexrecord), n_pos_after, n_neg_after))

    return id_indexrecord


def _eligibility_negative_no_covid_dx(id_indexrecord, id_dx, covid_codes_set):
    print("Step: applying _eligibility_negative_no_covid_dx",
          'input cohorts size:', len(id_indexrecord))
    N = len(id_indexrecord)
    n_pos_before = n_neg_before = 0
    n_pos_after = n_neg_after = 0
    n_pos_exclude = n_neg_exclude = 0
    exclude_list = []
    for pid, row in id_indexrecord.items():
        # (True/False, lab_date, lab_code, result_label, age)
        covid_flag = row[0]
        index_date = row[1]
        v_dx = id_dx.get(pid, [])
        if covid_flag:
            n_pos_before += 1
        else:
            n_neg_before += 1

        if not v_dx:
            # include, because no dx.covid for negative
            if covid_flag:
                n_pos_after += 1
            else:
                n_neg_after += 1
        else:
            flag_negative_has_covid = False
            for r in v_dx:
                dx_date = r[0]
                dx = r[1].replace('.', '').upper().strip()
                dx_type = r[2]
                # if int(dx_type) == 9:
                #     print('icd code 9:', dx)
                if (dx in covid_codes_set) and (not covid_flag):
                    flag_negative_has_covid = True
                    break

            if flag_negative_has_covid:
                exclude_list.append(pid)
                if covid_flag:
                    n_pos_exclude += 1
                else:
                    n_neg_exclude += 1
            else:
                if covid_flag:
                    n_pos_after += 1
                else:
                    n_neg_after += 1

    # Applying excluding:
    # print('exclude_list', exclude_list)
    [id_indexrecord.pop(pid, None) for pid in exclude_list]
    # Summary:
    print('...Before EC, total: {}\tpos: {}\tneg: {}'.format(N, n_pos_before, n_neg_before))
    print('...Excluding, total: {}\tpos: {}\tneg: {}'.format(len(exclude_list), n_pos_exclude, n_neg_exclude))
    print('...After  EC, total: {}\tpos: {}\tneg: {}'.format(len(id_indexrecord), n_pos_after, n_neg_after))

    return id_indexrecord


def integrate_data_and_apply_eligibility(args):
    start_time = time.time()
    print('In integrate_data_and_apply_eligibility, site:', args.dataset)
    covid_codes_set, df_pasc_list, pasc_codes_set, id_lab, id_demo, id_dx, id_med, id_enc, id_pro, id_obsgen, \
    id_immun, id_death, id_vital = read_preprocessed_data(args)

    # Step 1. Load included patients build id --> index records
    #    lab-confirmed positive:  first positive record
    #    lab-confirmed negative: first negative record
    id_indexrecord = {}
    n_pos = n_neg = 0
    n_with_ni = 0
    for i, (pid, row) in enumerate(id_lab.items()):
        v_lables = [x[2].upper() for x in row]
        if 'POSITIVE' in v_lables:
            n_pos += 1
            position = v_lables.index('POSITIVE')
            indexrecord = row[position]
            id_indexrecord[pid] = [True, ] + list(indexrecord)
        # else:
        elif v_lables.count('NEGATIVE') == len(v_lables):
            n_neg += 1
            position = 0
            indexrecord = row[position]
            if not args.positive_only:
                id_indexrecord[pid] = [False, ] + list(indexrecord)
        else:
            # ignore NI, because may leak presume positive to negative
            n_with_ni += 1

    print('Step1: Initial Included cohorts from\nlen(id_lab):', len(id_lab))
    print('Not using NI due to potential positive leaking, thus Total included:\n'
          'len(id_indexrecord):', len(id_indexrecord), 'n_pos:', n_pos, 'n_neg:', n_neg, 'n_with_ni', n_with_ni)
    # Can calculate more statistics

    def _local_build_data(_id_indexrecord):
        data = {}
        for pid, row in _id_indexrecord.items():
            # (True/False, lab_date, lab_code, result_label, age, enc-id)
            lab = id_lab[pid]
            demo = id_demo.get(pid, [])
            dx = id_dx[pid]
            med = id_med[pid]
            enc = id_enc[pid]
            procedure = id_pro[pid]
            obsgen = id_obsgen[pid]
            immun = id_immun[pid]
            death = id_death.get(pid, [])
            vital = id_vital.get(pid, [])
            data[pid] = [row, demo, dx, med, lab, enc, procedure, obsgen, immun, death, vital]
        print('building data done, len(id_indexrecord):', len(id_indexrecord), 'len(data) ', len(data))
        return data

    # Step 2: Applying EC. exclude index age < INDEX_AGE_MINIMUM
    # goal: adult population is our targeted population
    id_indexrecord = _eligibility_age(id_indexrecord, age_minimum_criterion=INDEX_AGE_MINIMUM)

    # Step 3: Applying EC. Any diagnosis in the baseline period
    # goal: baseline information, and access to healthcare
    id_indexrecord = _eligibility_baseline_any_dx(id_indexrecord, id_dx, _is_in_baseline)
    # data = _local_build_data(id_indexrecord)

    # Step 4: Applying EC. Any diagnosis in the follow-up period
    # goal: alive beyond acute phase, and have information for screening PASC,
    # and capture baseline PASC-like diagnosis for  Non-Covid patients
    # Our cohort 1 for screening PASC
    id_indexrecord = _eligibility_followup_any_dx(id_indexrecord, id_dx, _is_in_followup)

    # Step 5: Applying EC. No COVID diagnosis in the Negative Cohorts, any records in any time, excluding Covid Leaking
    id_indexrecord = _eligibility_negative_no_covid_dx(id_indexrecord, id_dx, covid_codes_set)

    data = _local_build_data(id_indexrecord)
    utils.dump(data, args.output_file_covid, chunk=4)

    # # Notes for other potential cohorts:
    # #  Sensitivity 1: Applying EC. No (initial, or screened) PASC diagnoses in the baseline  --> healthy population
    # # Goal: screen PASC without baseline PASC
    # # Move to matrix part to build cohorts dynamically with a better flexibility
    # # print('Adult PASC incidence cohorts:')
    # # id_indexrecord = _eligibility_baseline_no_pasc(id_indexrecord, id_dx, pasc_codes_set, _is_in_baseline)
    # # data = _local_build_data(id_indexrecord)
    # # utils.dump(data, args.output_file_pasc_incidence)

    # # Sensitivity 2: for separating recruiting window and observation window,
    # # also move to matrix part for a better flexibility

    # # Sensitivity 3: for excluding covid diagnosis in follow-up for control
    # # also move to matrix part for a better flexibility

    # # Sensitivity 4:
    # If using Dx to define Covid, we need to change initial inclusion list and rerun all the codes

    # # Sensitivity 5:
    # If using Flu or other viral codes to define control, we need to change initial inclusion list
    # and rerun all the codes

    # __step : save data structure for later encoding. save last cohort
    _last_cohort_raw_data = [] # No need to return this
    # [id_indexrecord, id_lab, id_demo, id_dx, id_med, id_enc, id_pro, id_obsgen, id_immun, id_death, id_vital]

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return data, _last_cohort_raw_data


if __name__ == '__main__':
    # python pre_cohort_manuscript.py --dataset COL 2>&1 | tee  log/pre_cohort_manuscript_COL.txt
    # python pre_cohort_manuscript.py --dataset WCM 2>&1 | tee  log/pre_cohort_manuscript_WCM.txt
    # python pre_cohort_manuscript.py --dataset NYU 2>&1 | tee  log/pre_cohort_manuscript_NYU.txt
    # python pre_cohort_manuscript.py --dataset MONTE 2>&1 | tee  log/pre_cohort_manuscript_MONTE.txt
    # python pre_cohort_manuscript.py --dataset MSHS 2>&1 | tee  log/pre_cohort_manuscript_MSHS.txt

    start_time = time.time()
    args = parse_args()
    data, raw_data = integrate_data_and_apply_eligibility(args)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

