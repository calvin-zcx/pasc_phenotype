import sys
# for linux env.
import pandas as pd

sys.path.insert(0, '..')
import time
import pickle
import numpy as np
import argparse
from misc import utils
from eligibility_setting import _is_in_baseline, _is_in_followup, INDEX_AGE_MINIMUM, INDEX_AGE_MINIMUM_18
import functools
from collections import Counter

print = functools.partial(print, flush=True)


# 1. aggregate preprocessed data [covid lab, demo, diagnosis, medication] into cohorts data structure
# 2. applying EC [eligibility criteria] to include and exclude patients for each cohorts
# 3. summarize basic statistics when applying each EC criterion

# updated 2023-11-8
# 1. using lab + dx + med
# 2. more information
# 3. try to mimimize selection bias, not using any EC after t0


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess cohorts')
    parser.add_argument('--dataset', default='wcm_pcornet_all', help='site dataset')
    parser.add_argument('--positive_only', action='store_true')
    parser.add_argument('--cohorttype', choices=['lab-dx-med-preg', 'lab-dx-med', 'lab-dx', 'lab'],
                        default='lab-dx-med-preg')

    args = parser.parse_args()

    args.preg_delivery_file = r'../data/recover/output/{}/patient_pregnant_{}.pkl'.format(args.dataset,
                                                                                          args.dataset)
    # load covid identified by lab + dx + med
    args.covid_lab_file = r'../data/recover/output/{}/patient_covid_{}_{}.pkl'.format(args.dataset,
                                                                                      args.cohorttype,
                                                                                      args.dataset)
    args.demo_file = r'../data/recover/output/{}/patient_demo_{}.pkl'.format(args.dataset, args.dataset)
    args.dx_file = r'../data/recover/output/{}/diagnosis_{}.pkl'.format(args.dataset, args.dataset)
    args.med_file = r'../data/recover/output/{}/medication_{}.pkl'.format(args.dataset, args.dataset)
    args.enc_file = r'../data/recover/output/{}/encounter_{}.pkl'.format(args.dataset, args.dataset)
    # added 2022-02-02
    args.pro_file = r'../data/recover/output/{}/procedures_{}.pkl'.format(args.dataset, args.dataset)
    args.obsgen_file = r'../data/recover/output/{}/obs_gen_{}.pkl'.format(args.dataset, args.dataset)
    args.immun_file = r'../data/recover/output/{}/immunization_{}.pkl'.format(args.dataset, args.dataset)
    # added 2022-02-20
    args.death_file = r'../data/recover/output/{}/death_{}.pkl'.format(args.dataset, args.dataset)
    # added 2022-04-08
    args.vital_file = r'../data/recover/output/{}/vital_{}.pkl'.format(args.dataset, args.dataset)
    # added 2023-11-8
    args.select_lab_file = r'../data/recover/output/{}/lab_result_select_{}.pkl'.format(args.dataset, args.dataset)

    args.pasc_list_file = r'../data/mapping/PASC_Adult_Combined_List_20220127_v3.xlsx'
    args.covid_list_file = r'../data/V15_COVID19/covid_phenotype/COVID_ICD.xlsx'

    # changed 2022-04-08 V2, add vital information in V2
    # args.output_file_covid = r'../data/V15_COVID19/output/{}/cohorts_covid_4manuNegNoCovid_{}.pkl'.format(
    # args.dataset, args.dataset)
    args.output_file_covid = r'../data/recover/output/{}/cohorts_preg_roughcal_{}{}.pkl.gz'.format(
        args.dataset,
        args.dataset,
        '' if args.cohorttype == 'lab-dx-med-preg' else '_' + args.cohorttype)  #
    # args.output_file_covid2 = r'../data/recover/output/{}/cohorts_covid_posOnly18base_{}{}.pkl'.format(
    #     args.dataset,
    #     args.dataset,
    #     '' if args.cohorttype == 'lab-dx-med-preg' else '_' + args.cohorttype)
    args.output_file_cohortinfo = r'../data/recover/output/{}/cohorts_preg_roughcal_{}{}_info.csv'.format(
        args.dataset,
        args.dataset,
        '' if args.cohorttype == 'lab-dx-med-preg' else '_' + args.cohorttype)

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
    id_lab_select = utils.load(args.select_lab_file)

    # for index pregnant delivery event, replace id_lab
    id_pregnant = utils.load(args.preg_delivery_file)

    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return covid_codes_set, df_pasc_list, pasc_codes_set, id_lab, id_demo, id_dx, id_med, id_enc, id_pro, id_obsgen, \
           id_immun, id_death, id_vital, id_lab_select, id_pregnant


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
        # 2024-7-22 (True/False/None, pregnant, lab_date, lab_code, result_label, age)
        covid_flag = row[0]
        age = row[5]  # row[4]
        if covid_flag:
            n_pos_before += 1
        else:
            n_neg_before += 1

        if age < age_minimum_criterion:
            # exclude, DEFAULT 20  --> change to 18
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

    info = {'before N': N, 'before N Pos': n_pos_before, 'before N Neg': n_neg_before,
            'exclude N': len(exclude_list), 'exclude N Pos': n_pos_exclude, 'exclude N Neg': n_neg_exclude,
            'after N': len(id_indexrecord), 'after N Pos': n_pos_after, 'after N Neg': n_neg_after,
            'ec': '_eligibility_age'}

    return id_indexrecord, info


def _eligibility_baseline_any_dx(id_indexrecord, id_dx, func_is_in_baseline):
    print("Step: applying _eligibility_baseline_any_dx",
          'input cohorts size:', len(id_indexrecord))
    N = len(id_indexrecord)
    n_pos_before = n_neg_before = 0
    n_pos_after = n_neg_after = 0
    n_pos_exclude = n_neg_exclude = 0
    exclude_list = []
    for pid, row in id_indexrecord.items():
        # (True/False/None, pregnant,  lab_date, lab_code, result_label, age, ***)
        covid_flag = row[0]
        index_date = row[2]  # row[1]
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

    info = {'before N': N, 'before N Pos': n_pos_before, 'before N Neg': n_neg_before,
            'exclude N': len(exclude_list), 'exclude N Pos': n_pos_exclude, 'exclude N Neg': n_neg_exclude,
            'after N': len(id_indexrecord), 'after N Pos': n_pos_after, 'after N Neg': n_neg_after,
            'ec': "_eligibility_baseline_any_dx"}

    return id_indexrecord, info


def _eligibility_followup_any_dx(id_indexrecord, id_dx, func_is_in_followup):
    print("Step: applying _eligibility_followup_any_dx",
          'input cohorts size:', len(id_indexrecord))
    N = len(id_indexrecord)
    n_pos_before = n_neg_before = 0
    n_pos_after = n_neg_after = 0
    n_pos_exclude = n_neg_exclude = 0
    exclude_list = []
    for pid, row in id_indexrecord.items():
        # (True/False/None, preg, lab_date, lab_code, result_label, age, ***)
        covid_flag = row[0]
        index_date = row[2]  # row[1]
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

    info = {'before N': N, 'before N Pos': n_pos_before, 'before N Neg': n_neg_before,
            'exclude N': len(exclude_list), 'exclude N Pos': n_pos_exclude, 'exclude N Neg': n_neg_exclude,
            'after N': len(id_indexrecord), 'after N Pos': n_pos_after, 'after N Neg': n_neg_after,
            'ec': "_eligibility_followup_any_dx"}

    return id_indexrecord, info


def _eligibility_baseline_or_followup_any_dx(id_indexrecord, id_dx, func_is_in_baseline, func_is_in_followup):
    print("Step: applying _eligibility_baseline_or_followup_any_dx",
          'input cohorts size:', len(id_indexrecord))
    N = len(id_indexrecord)
    n_pos_before = n_neg_before = 0
    n_pos_after = n_neg_after = 0
    n_pos_exclude = n_neg_exclude = 0
    exclude_list = []
    for pid, row in id_indexrecord.items():
        # (True/False/None, preg, lab_date, lab_code, result_label, age, ***)
        covid_flag = row[0]
        index_date = row[2]  # row[1]
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

    info = {'before N': N, 'before N Pos': n_pos_before, 'before N Neg': n_neg_before,
            'exclude N': len(exclude_list), 'exclude N Pos': n_pos_exclude, 'exclude N Neg': n_neg_exclude,
            'after N': len(id_indexrecord), 'after N Pos': n_pos_after, 'after N Neg': n_neg_after,
            'ec': "_eligibility_baseline_or_followup_any_dx"}

    return id_indexrecord, info


def _eligibility_followup_any_pasc(id_indexrecord, id_dx, pasc_codes_set, func_is_in_followup):
    print("Step: applying _eligibility_followup_any_pasc",
          'input cohorts size:', len(id_indexrecord))
    N = len(id_indexrecord)
    n_pos_before = n_neg_before = 0
    n_pos_after = n_neg_after = 0
    n_pos_exclude = n_neg_exclude = 0
    exclude_list = []
    for pid, row in id_indexrecord.items():
        # (True/False/None, preg,  lab_date, lab_code, result_label, age, ***)
        covid_flag = row[0]
        index_date = row[2]  # row[1]
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

    info = {'before N': N, 'before N Pos': n_pos_before, 'before N Neg': n_neg_before,
            'exclude N': len(exclude_list), 'exclude N Pos': n_pos_exclude, 'exclude N Neg': n_neg_exclude,
            'after N': len(id_indexrecord), 'after N Pos': n_pos_after, 'after N Neg': n_neg_after,
            'ec': "_eligibility_followup_any_pasc"}

    return id_indexrecord, info


def _eligibility_baseline_no_pasc(id_indexrecord, id_dx, pasc_codes_set, func_is_in_baseline):
    print("Step: applying _eligibility_baseline_no_pasc",
          'input cohorts size:', len(id_indexrecord))
    N = len(id_indexrecord)
    n_pos_before = n_neg_before = 0
    n_pos_after = n_neg_after = 0
    n_pos_exclude = n_neg_exclude = 0
    exclude_list = []
    for pid, row in id_indexrecord.items():
        # (True/False/None, preg, lab_date, lab_code, result_label, age)
        covid_flag = row[0]
        index_date = row[2]  # row[1]
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

    info = {'before N': N, 'before N Pos': n_pos_before, 'before N Neg': n_neg_before,
            'exclude N': len(exclude_list), 'exclude N Pos': n_pos_exclude, 'exclude N Neg': n_neg_exclude,
            'after N': len(id_indexrecord), 'after N Pos': n_pos_after, 'after N Neg': n_neg_after,
            'ec': "_eligibility_baseline_no_pasc"}

    return id_indexrecord, info


def _eligibility_negative_followup_no_covid_dx(id_indexrecord, id_dx, covid_codes_set, func_is_in_followup):
    print("Step: applying _eligibility_negative_followup_no_covid",
          'input cohorts size:', len(id_indexrecord))
    N = len(id_indexrecord)
    n_pos_before = n_neg_before = 0
    n_pos_after = n_neg_after = 0
    n_pos_exclude = n_neg_exclude = 0
    exclude_list = []
    for pid, row in id_indexrecord.items():
        # (True/False/None, preg, lab_date, lab_code, result_label, age, ***)
        covid_flag = row[0]
        index_date = row[2]  # row[1]
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

    info = {'before N': N, 'before N Pos': n_pos_before, 'before N Neg': n_neg_before,
            'exclude N': len(exclude_list), 'exclude N Pos': n_pos_exclude, 'exclude N Neg': n_neg_exclude,
            'after N': len(id_indexrecord), 'after N Pos': n_pos_after, 'after N Neg': n_neg_after,
            'ec': "_eligibility_negative_followup_no_covid_dx"}

    return id_indexrecord, info


def _eligibility_negative_no_covid_dx(id_indexrecord, id_dx, covid_codes_set):
    print("Step: applying _eligibility_negative_no_covid_dx",
          'input cohorts size:', len(id_indexrecord))
    N = len(id_indexrecord)
    n_pos_before = n_neg_before = 0
    n_pos_after = n_neg_after = 0
    n_pos_exclude = n_neg_exclude = 0
    exclude_list = []
    for pid, row in id_indexrecord.items():
        # (True/False/None, Preg, lab_date, lab_code, result_label, age, ***)
        covid_flag = row[0]
        index_date = row[2]  # row[1]
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

    info = {'before N': N, 'before N Pos': n_pos_before, 'before N Neg': n_neg_before,
            'exclude N': len(exclude_list), 'exclude N Pos': n_pos_exclude, 'exclude N Neg': n_neg_exclude,
            'after N': len(id_indexrecord), 'after N Pos': n_pos_after, 'after N Neg': n_neg_after,
            'ec': "_eligibility_negative_no_covid_dx"}

    return id_indexrecord, info


def _eligibility_covid_positive_only(id_indexrecord):
    print("Step: applying _eligibility_covid_positive_only",
          'input cohorts size:', len(id_indexrecord))
    N = len(id_indexrecord)
    n_pos_before = n_neg_before = 0
    n_pos_after = n_neg_after = 0
    n_pos_exclude = n_neg_exclude = 0
    exclude_list = []
    for pid, row in id_indexrecord.items():
        # (True/False/None, preg, lab_date, lab_code, result_label, age, ***)
        covid_flag = row[0]
        index_date = row[2]  # row[1]
        if covid_flag:
            n_pos_before += 1
            n_pos_after += 1
        else:
            n_neg_before += 1
            exclude_list.append(pid)
            n_neg_exclude += 1

    # Applying excluding:
    # print('exclude_list', exclude_list)
    [id_indexrecord.pop(pid, None) for pid in exclude_list]
    # Summary:
    print('...Before EC, total: {}\tpos: {}\tneg: {}'.format(N, n_pos_before, n_neg_before))
    print('...Excluding, total: {}\tpos: {}\tneg: {}'.format(len(exclude_list), n_pos_exclude, n_neg_exclude))
    print('...After  EC, total: {}\tpos: {}\tneg: {}'.format(len(id_indexrecord), n_pos_after, n_neg_after))

    info = {'before N': N, 'before N Pos': n_pos_before, 'before N Neg': n_neg_before,
            'exclude N': len(exclude_list), 'exclude N Pos': n_pos_exclude, 'exclude N Neg': n_neg_exclude,
            'after N': len(id_indexrecord), 'after N Pos': n_pos_after, 'after N Neg': n_neg_after,
            'ec': "_eligibility_covid_positive_only"}

    return id_indexrecord, info


def _clean_covid_pcr_label(x):
    if isinstance(x, str):
        x = x.strip().upper()
        if x.startswith(('NOT DETECTED', 'NEG', 'NOT', 'NEGATIVE', 'UNDETECTED')):
            x = 'NEGATIVE'
        elif x.startswith(('DETECTED', 'POSITIVE', 'POS')):
            x = 'POSITIVE'
        elif x.startswith(('Pregnant', 'PREGNANT')):
            x = 'PREGNANT'
        else:
            x = 'NI'
    else:
        x = 'NI'
    return x


def integrate_data_and_apply_eligibility(args):
    start_time = time.time()
    print('In integrate_data_and_apply_eligibility, site:', args.dataset)
    covid_codes_set, df_pasc_list, pasc_codes_set, id_lab, id_demo, id_dx, id_med, id_enc, id_pro, id_obsgen, \
    id_immun, id_death, id_vital, id_lab_select, id_pregnant = read_preprocessed_data(args)

    # Step 1. Load included patients build id --> index records
    #    lab-confirmed positive:  first positive record
    #    lab-confirmed negative: first negative record
    id_indexrecord = {}
    cohort_info = []
    n_pos = n_neg = 0
    n_preg_wo_covidrecs = 0
    n_preg_in_NI = 0
    n_preg = 0
    n_with_ni_nopreg = 0

    # rows in id_lab can be a list of the following elements:
    """
    id_lab[patid].append((lab_date,   lab_code, result_label, age, enc_id, '', 'lab'))
    id_dx[patid].append ((dx_date,    dx,       'Positive', age, enc_id, enc_type, 'dx'))
    id_med[patid].append((start_date, rxnorm,   'Positive', age, encid, '', 'med-pr'))
    id_med[patid].append((start_date, rxnorm,   'Positive', age, encid, '', 'med-medadmin'))
    id_med[patid].append((start_date, rxnorm,   'Positive', age, encid, dispense_source, 'med-dispense'))

    id_dx[patid].append ((dx_date,    dx,       'Pregnant', age, enc_id, enc_type, 'preg-dx'))
    id_dx[patid].append ((dx_date,    dx,       'Pregnant', age, enc_id, enc_type, 'preg-px'))
    id_dx[patid].append ((dx_date,    dx,       'Pregnant', age, enc_id, enc_type, 'preg-encdrg'))
    """
    # for i, (pid, row) in enumerate(id_lab.items()):
    for i, (pid, row) in enumerate(id_pregnant.items()):
        # v_lables = [x[2].upper() for x in row]
        # how to deal with pregnant information?
        # group-1: positive,
        # group-2: negative, and
        # group-3: additional added from pregnant-related information w/o covid-related information
        v_lables = [_clean_covid_pcr_label(x[2]) for x in row]
        # v_labels_only_covid = [_clean_covid_pcr_label(x[2]) for x in v_lables]
        # 2024-7-22, add the second dimension pregnant
        if 'PREGNANT' in v_lables:
            n_preg += 1
            position = 0  # sorted before, choose the first one
            indexrecord = row[position]
            # 'PREGNANT' (dx_date, dx/pro/enc, 'Pregnant', age, enc_id, enc_type, 'preg-dx')
            id_indexrecord[pid] = [None, 'PREGNANT'] + list(indexrecord)

        # if v_lables.count('PREGNANT') == len(v_lables):
        #     # only have pregnant related records, no covid related records
        #     n_preg_wo_covidrecs += 1
        #     # position = 0  # use first pregnant related event as index
        #     # indexrecord = row[position]
        #     # id_indexrecord[pid] = [None, True] + list(indexrecord)
        # else:
        #     # have any covid related records
        #     if 'POSITIVE' in v_lables:
        #         # if has any covid positive record, then positive
        #         n_pos += 1
        #         position = v_lables.index('POSITIVE')  # first positive
        #         indexrecord = row[position]
        #         # id_indexrecord[pid] = [True, 'PREGNANT' in v_lables] + list(indexrecord)
        #
        #         # # 2024-7-23 add positive patients with negative portions --> discard here, use another file
        #         if 'NEGATIVE' in v_lables:
        #             neg_start = v_lables.index('NEGATIVE')  # first negative
        #             if neg_start < position:
        #                 neg_indexrecord = row[neg_start]
        #                 id_indexrecord[pid] = [False, indexrecord] + list(neg_indexrecord)
        #
        #     else:  # ( no pos, only negative/NI/preg)
        #         if 'NI' not in v_lables:  # ni not in <---> all negative or pregnant (cannot be all pregnant)
        #             # and ((v_lables.count('NEGATIVE') + v_lables.count('PREGNANT')) == len(v_lables)):  # can be empty [], then true for this. potential bug
        #             # if all covid records are negative
        #             n_neg += 1
        #             # ### position = 0  # if all negative, selected first
        #             # position = v_lables.index('NEGATIVE')  # first negative among negative/pregnant
        #             # indexrecord = row[position]
        #             # # if not args.positive_only:
        #             # id_indexrecord[pid] = [False, 'PREGNANT' in v_lables] + list(indexrecord)
        #
        #         elif ('NI' in v_lables) and ('PREGNANT' in v_lables):  # have NI or negative   and preg
        #             n_preg_in_NI += 1
        #             # position = v_lables.index('NI')  # first negative
        #             # if 'NEGATIVE' in v_lables:
        #             #     pos2 = v_lables.index('NEGATIVE')
        #             #     if pos2 <= position:
        #             #         position = pos2
        #             # indexrecord = row[position]  # earliest NI or negative event
        #             # id_indexrecord[pid] = [None, True] + list(indexrecord)
        #
        #         else:  # ('NI' in v_lables) and ('PREGNANT' not in v_lables) # have NI or negative and no preg, discard
        #             # no pos, no pregnant, only NI or with Negative
        #             # ignore NI, because may leak presume positive to negative
        #             # check number first
        #             n_with_ni_nopreg += 1
        #             # position = v_lables.index('NI')  # first negative
        #             # if 'NEGATIVE' in v_lables:
        #             #     pos2 = v_lables.index('NEGATIVE')
        #             #     if pos2 <= position:
        #             #         position = pos2
        #             # indexrecord = row[position]  # earliest NI or negative event
        #             # id_indexrecord[pid] = [np.nan, False] + list(indexrecord)

            # Notes:
            # 1. For all preg w/o covid records, or preg with NI, label as None, bool(None): False, pd.isna(None):True
            #      will not influence covid+, only add to negative cohorts
            # 2. negative may include NI? check NI number first
            # 3. negative may include those positive later, but negative and for being negative for a followup period
            #    which means, positive patients can contribute to their negative period

    print('Step1: Initial Included cohorts from\nlen(id_lab):', len(id_lab))
    print('Not using NI due to potential positive leaking, thus Total included:\n',
          'len(id_pregnant)', len(id_pregnant),
          'len(id_lab):', len(id_lab),
          'len(id_indexrecord):', len(id_indexrecord),
          'n_preg:', n_preg,
          'n_preg_wo_covidrecs:', n_preg_wo_covidrecs,
          'n_pos:', n_pos,
          'n_neg:', n_neg,
          'n_preg_in_NI:', n_preg_in_NI,
          'n_with_ni_nopreg', n_with_ni_nopreg)
    # Can calculate more statistics

    info = {'before N': len(id_pregnant), 'before N Pos': n_pos, 'before N Neg': n_neg, 'exclude N': n_with_ni_nopreg,
            'exclude N Pos': np.nan, 'exclude N Neg': np.nan, 'after N': len(id_indexrecord), 'after N Pos': n_pos,
            'after N Neg': n_neg, 'ec': 'exclude NI PCR'}
    cohort_info.append(info)

    def _local_build_data(_id_indexrecord):
        data = {}
        for pid, row in _id_indexrecord.items():
            # (True/False, lab_date, lab_code, result_label, age, enc-id)
            pregnant = id_pregnant[pid]
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
            lab_select = id_lab_select.get(pid, [])
            data[pid] = [row, demo, dx, med, lab, enc, procedure, obsgen, immun, death, vital, lab_select, pregnant] # add pregnant record at the end
        print('building data done, len(id_indexrecord):', len(id_indexrecord), 'len(data) ', len(data))
        return data

    print('OF NOTE: the 2nd element of index record is a tuple!!!\n'
          ' id_indexrecord[pid] = [False, indexrecord] + list(neg_indexrecord)')


    # id_indexrecord, info = _eligibility_age(id_indexrecord, age_minimum_criterion=INDEX_AGE_MINIMUM_18)
    id_indexrecord, info = _eligibility_age(id_indexrecord, age_minimum_criterion=14)
    cohort_info.append(info)
    data = _local_build_data(id_indexrecord)

    # dump cohort selection process
    cohort_info = pd.DataFrame(cohort_info)
    cohort_info.to_csv(args.output_file_cohortinfo)
    print(cohort_info)

    # utils.dump(data, args.output_file_covid, chunk=4)
    # # utils.dump(data2, args.output_file_covid2, chunk=4)

    utils.dump_compressed(data, args.output_file_covid)
    # utils.dump(data, args.output_file_covid)

    # __step : save data structure for later encoding. save last cohort
    _last_cohort_raw_data = []  # No need to return this
    # [id_indexrecord, id_lab, id_demo, id_dx, id_med, id_enc, id_pro, id_obsgen, id_immun, id_death, id_vital]

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return data, cohort_info, _last_cohort_raw_data


if __name__ == '__main__':
    # python pre_cohort_manuscript.py --dataset COL 2>&1 | tee  log/pre_cohort_manuscript_COL.txt
    # python pre_cohort_manuscript.py --dataset WCM 2>&1 | tee  log/pre_cohort_manuscript_WCM.txt
    # python pre_cohort_manuscript.py --dataset NYU 2>&1 | tee  log/pre_cohort_manuscript_NYU.txt
    # python pre_cohort_manuscript.py --dataset MONTE 2>&1 | tee  log/pre_cohort_manuscript_MONTE.txt
    # python pre_cohort_manuscript.py --dataset MSHS 2>&1 | tee  log/pre_cohort_manuscript_MSHS.txt

    start_time = time.time()
    args = parse_args()
    # id_pregnant = utils.load(args.preg_delivery_file)
    # zz
    data, cohort_info, raw_data = integrate_data_and_apply_eligibility(args)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
