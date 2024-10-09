import sys
# for linux env.
import pandas as pd

sys.path.insert(0, '..')
import time
import pickle
import numpy as np
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
    parser.add_argument('--dataset', default='wcm', help='site dataset')
    parser.add_argument('--positive_only', action='store_true')

    args = parser.parse_args()

    # args.covid_lab_file = r'../data/recover/output/{}/patient_covid_lab_{}.pkl'.format(args.dataset, args.dataset)
    args.patient_list_file = r'../data/recover/output_hf/{}/patient_hf_list_{}.pkl'.format(args.dataset, args.dataset)

    args.demo_file = r'../data/recover/output_hf/{}/patient_demo_{}.pkl'.format(args.dataset, args.dataset)
    args.dx_file = r'../data/recover/output_hf/{}/diagnosis_{}.pkl'.format(args.dataset, args.dataset)
    args.med_file = r'../data/recover/output_hf/{}/medication_{}.pkl'.format(args.dataset, args.dataset)
    args.enc_file = r'../data/recover/output_hf/{}/encounter_{}.pkl'.format(args.dataset, args.dataset)
    # added 2022-02-02
    args.pro_file = r'../data/recover/output_hf/{}/procedures_{}.pkl'.format(args.dataset, args.dataset)
    args.obsgen_file = r'../data/recover/output_hf/{}/obs_gen_{}.pkl'.format(args.dataset, args.dataset)
    args.immun_file = r'../data/recover/output_hf/{}/immunization_{}.pkl'.format(args.dataset, args.dataset)
    # added 2022-02-20
    args.death_file = r'../data/recover/output_hf/{}/death_{}.pkl'.format(args.dataset, args.dataset)
    # added 2022-04-08
    args.vital_file = r'../data/recover/output_hf/{}/vital_{}.pkl'.format(args.dataset, args.dataset)

    args.nts_file = r'../data/recover/output_hf/{}/nts_{}.csv'.format(args.dataset, args.dataset)

    # args.pasc_list_file = r'../data/mapping/PASC_Adult_Combined_List_20220127_v3.xlsx'
    # args.covid_list_file = r'../data/V15_COVID19/covid_phenotype/COVID_ICD.xlsx'

    # changed 2022-04-08 V2, add vital information in V2
    # args.output_file_covid = r'../data/V15_COVID19/output/{}/cohorts_covid_4manuNegNoCovid_{}.pkl'.format(args.dataset, args.dataset)
    args.output_file_hf = r'../data/recover/output_hf/{}/cohorts_hf_{}.pkl'.format(args.dataset, args.dataset)
    args.output_file_cohortinfo = r'../data/recover/output_hf/{}/cohorts_hf_{}_info.csv'.format(args.dataset, args.dataset)

    print('args:', args)
    return args


if __name__ == '__main__':

    start_time = time.time()
    args = parse_args()

    df_nts = pd.read_csv(args.nts_file)
    a = df_nts['note_type_source_value'].value_counts()
    print(a)