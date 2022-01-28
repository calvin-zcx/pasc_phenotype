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

    args = parser.parse_args()

    args.covid_lab_file = r'../data/V15_COVID19/output/{}/patient_covid_lab_{}.pkl'.format(args.dataset, args.dataset)
    args.demo_file = r'../data/V15_COVID19/output/{}/patient_demo_{}.pkl'.format(args.dataset, args.dataset)
    args.dx_file = r'../data/V15_COVID19/output/{}/diagnosis_{}.pkl'.format(args.dataset, args.dataset)
    args.med_file = r'../data/V15_COVID19/output/{}/medication_{}.pkl'.format(args.dataset, args.dataset)
    args.enc_file = r'../data/V15_COVID19/output/{}/encounter_{}.pkl'.format(args.dataset, args.dataset)
    args.pasc_list_file = r'../data/mapping/PASC_Adult_Combined_List_20220127_v3.xlsx'

    # args.output_file = r'../data/V15_COVID19/output/{}/data_pcr_cohorts_{}.pkl'.format(args.dataset, args.dataset)
    # args.output_file = r'../data/V15_COVID19/output/{}/data_pcr_incidence_cohorts_{}.pkl'.format(args.dataset, args.dataset)
    args.output_file = r'../data/V15_COVID19/output/{}/data_pcr_prevalence_cohorts_{}.pkl'.format(args.dataset, args.dataset)

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
    # 0. Load pasc list
    df_pasc_list = pd.read_excel(args.pasc_list_file, sheet_name=r'PASC Screening List', usecols="A:N")
    print('df_pasc_list.shape', df_pasc_list.shape)
    pasc_codes = df_pasc_list['ICD-10-CM Code'].str.upper().replace('.', '', regex=False)  # .to_list()
    pasc_codes_set = set(pasc_codes)
    print('0-Load compiled pasc list done from {}\nlen(pasc_codes)'.format(args.pasc_list_file),
          len(pasc_codes), 'len(pasc_codes_set):', len(pasc_codes_set))

    # 1. load covid patients lab list
    with open(args.covid_lab_file, 'rb') as f:
        id_lab = pickle.load(f)
        print('1-Load covid patients lab list done from {}!\nlen(id_lab):'.format(args.covid_lab_file), len(id_lab))

    # 2. load demographics file
    with open(args.demo_file, 'rb') as f:
        id_demo = pickle.load(f)
        print('2-load demographics file done from {}!\nlen(id_demo):'.format(args.demo_file), len(id_demo))

    # 3. load diagnosis file.
    # NYU may use joblib file and load method.
    # with open(args.dx_file, 'rb') as f:
    #     id_dx = pickle.load(f)
    #     print('3-load diagnosis file done! len(id_dx):', len(id_dx))
    id_dx = utils.load(args.dx_file)
    print('3-load diagnosis file done from {}!\nlen(id_dx):'.format(args.dx_file), len(id_dx))

    # 4. load medication file
    with open(args.med_file, 'rb') as f:
        id_med = pickle.load(f)
        print('4-load medication file done from {}!\nlen(id_med):'.format(args.med_file), len(id_med))

    # 5. load encounter file
    with open(args.enc_file, 'rb') as f:
        id_enc = pickle.load(f)
        print('5-load encounter file done from {}!\nlen(id_med):'.format(args.enc_file), len(id_enc))

    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return df_pasc_list, pasc_codes_set, id_lab, id_demo, id_dx, id_med, id_enc


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
    df_pasc_list, pasc_codes_set, id_lab, id_demo, id_dx, id_med, id_enc = read_preprocessed_data(args)

    # Step 1. Load included patients build id --> index records
    #    lab-confirmed positive:  first positive record
    #    lab-confirmed negative: first negative record
    id_indexrecord = {}
    n_pos = n_neg = 0
    for pid, row in id_lab.items():
        v_lables= [x[2].upper() for x in row]
        if 'POSITIVE' in v_lables:
            n_pos += 1
            position = v_lables.index('POSITIVE')
            indexrecord = row[position]
            id_indexrecord[pid] = [True, ] + list(indexrecord)
        else:
            n_neg += 1
            position = 0
            indexrecord = row[position]
            id_indexrecord[pid] = [False, ] + list(indexrecord)
    print('Step1: Initial Included cohorts:')
    print('Total len(id_indexrecord):', len(id_indexrecord), 'n_pos:', n_pos, 'n_neg:', n_neg)
    # Can calculate more statistics

    # Step 2: Applying EC. exclude index age < INDEX_AGE_MINIMUM
    id_indexrecord = _eligibility_age(id_indexrecord, age_minimum_criterion=INDEX_AGE_MINIMUM)

    # Step 3: Applying EC. Any diagnosis in the baseline period
    id_indexrecord = _eligibility_baseline_any_dx(id_indexrecord, id_dx, _is_in_baseline)

    # Step 4: Applying EC. Any PASC diagnoses in the follow-up
    id_indexrecord = _eligibility_followup_any_pasc(id_indexrecord, id_dx, pasc_codes_set, _is_in_followup)

    # Step 5: Applying EC. No PASC diagnoses in the baseline
    # id_indexrecord = _eligibility_baseline_no_pasc(id_indexrecord, id_dx, pasc_codes_set, _is_in_baseline)

    print('Final selected cohorts total len(id_indexrecord):', len(id_indexrecord))

    # step 4: build data structure for later encoding.
    raw_data = [id_indexrecord, id_lab, id_demo, id_dx, id_med, id_enc]
    data = {}
    for pid, row in id_indexrecord.items():
        # (True/False, lab_date, lab_code, result_label, age, enc-id)
        lab = id_lab[pid]
        demo = id_demo[pid]
        dx = id_dx[pid]
        med = id_med[pid]
        enc = id_enc[pid]
        data[pid] = [row, demo, dx, med, lab, enc]
    print('Final data: len(data):', len(data))

    utils.check_and_mkdir(args.output_file)
    # pickle.dump(data, open(args.output_file, 'wb'))
    utils.dump(data, args.output_file)

    print('dump done to {}'.format(args.output_file))
    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return data, raw_data


if __name__ == '__main__':
    # python pre_cohort.py --dataset COL 2>&1 | tee  log/pre_cohort_COL.txt
    # python pre_cohort.py --dataset WCM 2>&1 | tee  log/pre_cohort_WCM.txt
    # python pre_cohort.py --dataset NYU 2>&1 | tee  log/pre_cohort_NYU.txt
    # python pre_cohort.py --dataset MONTE 2>&1 | tee  log/pre_cohort_MONTE.txt
    # python pre_cohort.py --dataset MSHS 2>&1 | tee  log/pre_cohort_MSHS.txt

    start_time = time.time()
    args = parse_args()
    data, raw_data = integrate_data_and_apply_eligibility(args)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    print('Get some statistics of final selected cohorts')
    dx_encounter_type = []
    for key, row in data.items():
        dx = row[2]
        for b in dx:
            dx_encounter_type.append(b[3])
    c = Counter(dx_encounter_type)
    print('Encounter type of diagnosis:', c.most_common())

    gender_type = []
    for key, row in data.items():
        gender = row[1][1]
        gender_type.append(gender)
    c = Counter(gender_type)
    print('Gender type:', c.most_common())

    race_type = []
    for key, row in data.items():
        race = row[1][2]
        race_type.append(race)
    c = Counter(race_type)
    print('Race type:', c.most_common())

    state_type = []
    for key, row in data.items():
        race = row[1][4]
        state_type.append(race)
    c = Counter(state_type)
    print('State type:', c.most_common(60))