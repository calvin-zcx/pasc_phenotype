import sys
# for linux env.
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

    args.output_file = r'../data/V15_COVID19/output/{}/data_pcr_cohorts_{}.pkl'.format(args.dataset, args.dataset)

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
    # 1. load covid patients lab list
    with open(args.covid_lab_file, 'rb') as f:
        id_lab = pickle.load(f)
        print('1-Load covid patients lab list done! len(id_lab):', len(id_lab))

    # 2. load demographics file
    with open(args.demo_file, 'rb') as f:
        id_demo = pickle.load(f)
        print('2-load demographics file done! len(id_demo):', len(id_demo))

    # 3. load diagnosis file.
    # NYU may use joblib file and load method.
    # with open(args.dx_file, 'rb') as f:
    #     id_dx = pickle.load(f)
    #     print('3-load diagnosis file done! len(id_dx):', len(id_dx))
    id_dx = utils.load(args.dx_file)
    print('3-load diagnosis file done! len(id_dx):', len(id_dx))

    # 4. load medication file
    with open(args.med_file, 'rb') as f:
        id_med = pickle.load(f)
        print('4-load medication file done! len(id_med):', len(id_med))

    # 5. load encounter file
    with open(args.enc_file, 'rb') as f:
        id_enc = pickle.load(f)
        print('5-load encounter file done! len(id_med):', len(id_enc))

    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return id_lab, id_demo, id_dx, id_med, id_enc


def integrate_data_and_apply_eligibility(args):
    start_time = time.time()
    print('In integrate_data_and_apply_eligibility, site:', args.dataset)
    id_lab, id_demo, id_dx, id_med, id_enc = read_preprocessed_data(args)

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
    n_pos = n_neg = 0
    exclude_list = []
    for pid, row in id_indexrecord.items():
        # (True/False, lab_date, lab_code, result_label, age)
        covid_flag = row[0]
        age = row[4]
        if age < INDEX_AGE_MINIMUM:  # DEFAULT 20
            # excluded
            exclude_list.append(pid)
        else:
            # included
            if covid_flag:
                n_pos += 1
            else:
                n_neg += 1
    [id_indexrecord.pop(pid, None) for pid in exclude_list]
    print('Step2: exclude index age < {}, len(exclude_list):'.format(INDEX_AGE_MINIMUM), len(exclude_list),)
    print('Total len(id_indexrecord): ', len(id_indexrecord), 'n_pos:', n_pos, 'n_neg:', n_neg)
    # print('exclude_list', exclude_list)

    # Step 3. Applying EC exclude no diagnosis in baseline period [-18th month, -1st month] or
    #         no dx in follow-up [1st month, 6th month]
    n_pos = n_neg = 0
    exclude_list = []
    n_exclude_due_to_baseline = n_exclude_due_to_followup = n_exclude_due_to_both = 0
    for pid, row in id_indexrecord.items():
        # (True/False, lab_date, lab_code, result_label, age)
        covid_flag = row[0]
        index_date = row[1]
        v_dx = id_dx.get(pid, [])
        if not v_dx:
            exclude_list.append(pid)
        else:
            flag_follow = False
            flag_baseline = False
            for r in v_dx:
                dx_date = r[0]
                if _is_in_followup(dx_date, index_date):  # 30 <= (dx_date - index_date).days <= 180:
                    flag_follow = True
                    break

            for r in v_dx:
                dx_date = r[0]
                if _is_in_baseline(dx_date, index_date):  # -540 <= (dx_date - index_date).days <= -30:
                    flag_baseline = True
                    break

            if not flag_follow:
                n_exclude_due_to_followup += 1
            if not flag_baseline:
                n_exclude_due_to_baseline += 1
            if (not flag_baseline) and (not flag_follow):
                n_exclude_due_to_both += 1

            if flag_follow and flag_baseline:
                if covid_flag:
                    n_pos += 1
                else:
                    n_neg += 1
            else:
                exclude_list.append(pid)

    [id_indexrecord.pop(pid, None) for pid in exclude_list]
    print('Step3: exclude no diagnosis in baseline period [-18th month, -1st month] or '
          'no dx in follow-up [1st month, 6th month], len(exclude_list):', len(exclude_list))
    print('n_exclude_due_to_followup:', n_exclude_due_to_followup,
          'n_exclude_due_to_baseline:', n_exclude_due_to_baseline,
          'n_exclude_due_to_both:', n_exclude_due_to_both)

    print('Total len(id_indexrecord):', len(id_indexrecord), 'n_pos:', n_pos, 'n_neg:', n_neg)
    # print('exclude_list', exclude_list)

    print('Finally selected cohorts:')
    print('Total len(id_indexrecord):', len(id_indexrecord), 'n_pos:', n_pos, 'n_neg:', n_neg)

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