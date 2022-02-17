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

    args.pasc_list_file = r'../data/mapping/PASC_Adult_Combined_List_20220127_v3.xlsx'

    args.output_file_covid = r'../data/V15_COVID19/output/{}/cohorts_covid_4screen_Covid+_{}.pkl'.format(args.dataset, args.dataset)
    # args.output_file_pasc_incidence = r'../data/V15_COVID19/output/{}/cohorts_pasc_incidence_{}.pkl'.format(args.dataset, args.dataset)
    # args.output_file_pasc_prevalence = r'../data/V15_COVID19/output/{}/cohorts_pasc_prevalence_{}.pkl'.format(args.dataset, args.dataset)

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
    # Load pasc list
    df_pasc_list = pd.read_excel(args.pasc_list_file, sheet_name=r'PASC Screening List', usecols="A:N")
    print('df_pasc_list.shape', df_pasc_list.shape)
    pasc_codes = df_pasc_list['ICD-10-CM Code'].str.upper().replace('.', '', regex=False)  # .to_list()
    pasc_codes_set = set(pasc_codes)
    print('0-Load compiled pasc list done from {}\nlen(pasc_codes)'.format(args.pasc_list_file),
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

    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return df_pasc_list, pasc_codes_set, id_lab, id_demo, id_dx, id_med, id_enc, id_pro, id_obsgen, id_immun


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
                dx = r[1].replace('.', '').upper()
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


def integrate_data_and_apply_eligibility(args):
    start_time = time.time()
    print('In integrate_data_and_apply_eligibility, site:', args.dataset)
    df_pasc_list, pasc_codes_set, id_lab, id_demo, id_dx, id_med, id_enc, id_pro, id_obsgen, id_immun = read_preprocessed_data(args)

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
            demo = id_demo[pid]
            dx = id_dx[pid]
            med = id_med[pid]
            enc = id_enc[pid]
            procedure = id_pro.get(pid, [])
            obsgen = id_obsgen.get(pid, [])
            immun = id_immun.get(pid, [])
            data[pid] = [row, demo, dx, med, lab, enc, procedure, obsgen, immun]
        print('building data done, len(id_indexrecord):', len(id_indexrecord), 'len(data) ', len(data))
        return data

    # Step 2: Applying EC. exclude index age < INDEX_AGE_MINIMUM
    id_indexrecord = _eligibility_age(id_indexrecord, age_minimum_criterion=INDEX_AGE_MINIMUM)

    # Step 3: Applying EC. Any diagnosis in the baseline period
    # print('Adult COVID cohorts:')
    id_indexrecord = _eligibility_baseline_any_dx(id_indexrecord, id_dx, _is_in_baseline)
    data = _local_build_data(id_indexrecord)

    # Step 4: Applying EC. Any diagnosis in the follow-up period
    # print('Adult COVID any dx in baseline and follow-up cohorts for pasc screening:')
    # id_indexrecord = _eligibility_followup_any_dx(id_indexrecord, id_dx, _is_in_followup)
    # data = _local_build_data(id_indexrecord)
    utils.dump(data, args.output_file_covid)

    # __step : save data structure for later encoding. save last cohort
    _last_cohort_raw_data = [id_indexrecord, id_lab, id_demo, id_dx, id_med, id_enc, id_pro, id_obsgen, id_immun]

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return data, _last_cohort_raw_data


if __name__ == '__main__':
    # python pre_cohort_for_screen.py --dataset COL 2>&1 | tee  log/pre_cohort_for_screen_COL.txt
    # python pre_cohort_for_screen.py --dataset WCM 2>&1 | tee  log/pre_cohort_for_screen_WCM.txt
    # python pre_cohort_for_screen.py --dataset NYU 2>&1 | tee  log/pre_cohort_for_screen_NYU.txt
    # python pre_cohort_for_screen.py --dataset MONTE 2>&1 | tee  log/pre_cohort_for_screen_MONTE.txt
    # python pre_cohort_for_screen.py --dataset MSHS 2>&1 | tee  log/pre_cohort_for_screen_MSHS.txt

    start_time = time.time()
    args = parse_args()
    data, raw_data = integrate_data_and_apply_eligibility(args)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

