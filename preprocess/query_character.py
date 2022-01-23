import sys

# for linux env.
sys.path.insert(0, '..')
import time
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import  tqdm
from misc import utils
from eligibility_setting import _is_in_baseline, _is_in_followup, _is_in_acute
import functools
print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess demographics')
    parser.add_argument('--dataset', choices=['COL', 'MSHS', 'MONTE', 'NYU', 'WCM', 'ALL'], default='COL', help='site dataset')
    args = parser.parse_args()

    # args.input_file = r'../data/V15_COVID19/output/{}/data_pcr_cohorts_{}.pkl'.format(args.dataset, args.dataset)
    args.output_file = r'../data/V15_COVID19/output/character/pcr_cohorts_character_raw_{}.csv'.format(args.dataset)

    print('args:', args)
    return args


def cohorts_characterization_build_data(args):
    start_time = time.time()
    print('In cohorts_characterization_build_data...')
    if args.dataset == 'ALL':
        sites = ['COL', 'MSHS', 'MONTE', 'NYU', 'WCM']
    else:
        sites = [args.dataset,]
    print('Loading: ', sites)
    # step 1: load 5 cohorts pickle data, store necessary info for analysis
    df_records_aux = []
    for site in tqdm(sites):
        input_file = r'../data/V15_COVID19/output/{}/data_pcr_cohorts_{}.pkl'.format(site, site)
        print('Loading site:', site, input_file)
        id_data = utils.load(input_file)
        print('Load covid patients pickle data done! len(id_data):', len(id_data))

        n = len(id_data)
        pid_list = []
        site_list = []
        covid_list = []

        n_no_index_enc_type = 0
        for i, (pid, item) in tqdm(enumerate(id_data.items()), total=len(id_data)):
            pid_list.append(pid)
            index_info, demo, dx, med, covid_lab, enc = item
            flag, index_date, covid_loinc, flag_name, index_age_year, index_enc_id = index_info
            birth_date, gender, race, hispanic, zipcode, state, city, nation_adi, state_adi = demo
            site_list.append(args.dataset)
            covid_list.append(flag)

            index_enc_type = np.nan
            enc_type_flag = False
            for enc_item in enc:
                if enc_item[2] == index_enc_id:
                    index_enc_type = enc_item[1]
                    enc_type_flag = True
                    break

            if not enc_type_flag:
                print('not found covid index encounter type', pid, site)
                n_no_index_enc_type += 1
            # store raw information for debugging
            # add dx, med, enc in acute, and follow-up
            # currently focus on baseline information
            records_aux = [pid, args.dataset]
            records_aux.extend(index_info + [index_enc_type, ] + demo)

            lab_str = ';'.join([x[2] for x in covid_lab])  # all lab tests
            dx_str_baseline = ';'.join([x[1].replace('.', '') for x in dx if _is_in_baseline(x[0], index_date)])
            med_str_baseline = ';'.join([x[1] for x in med if _is_in_baseline(x[0], index_date)])
            enc_str_baseline = ';'.join([x[1] for x in enc if _is_in_baseline(x[0], index_date)])

            dx_str_acute = ';'.join([x[1].replace('.', '') for x in dx if _is_in_acute(x[0], index_date)])
            med_str_acute = ';'.join([x[1] for x in med if _is_in_acute(x[0], index_date)])
            enc_str_acute = ';'.join([x[1] for x in enc if _is_in_acute(x[0], index_date)])

            dx_str_followup = ';'.join([x[1].replace('.', '') for x in dx if _is_in_followup(x[0], index_date)])
            med_str_followup = ';'.join([x[1] for x in med if _is_in_followup(x[0], index_date)])
            enc_str_followup = ';'.join([x[1] for x in enc if _is_in_followup(x[0], index_date)])

            records_aux.extend([lab_str,
                                dx_str_baseline, med_str_baseline, enc_str_baseline,
                                dx_str_acute, med_str_acute, enc_str_acute,
                                dx_str_followup, med_str_followup, enc_str_followup])
            df_records_aux.append(records_aux)

        print(site, 'Done!',
              'len(id_data):', len(id_data),
              'len(df_records_aux)', len(df_records_aux),
              'n_no_index_enc_type:', n_no_index_enc_type)

    print('Integrating done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    #   step 2: build pandas, column, and dump
    df_records_aux = pd.DataFrame(df_records_aux,
                                  columns=['patid', "site", "covid", "index_date", "covid_loinc", "flag_name",
                                           "index_age_year", "index_enc_id", "index_enc_type",
                                           "birth_date", "gender", "race", "hispanic", "zipcode", "state", "city",
                                           "nation_adi", "state_adi",
                                           "lab_str",
                                           "dx_str_baseline", "med_str_baseline", "enc_str_baseline",
                                           "dx_str_acute", "med_str_acute", "enc_str_acute",
                                           "dx_str_followup", "med_str_followup", "enc_str_followup"])
    print('df_records_aux.shape:', df_records_aux.shape)

    utils.check_and_mkdir(args.output_file)
    df_records_aux.to_csv(args.output_file)
    print('dump done to {}'.format(args.output_file))
    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return df_records_aux


def cohorts_characterization_analyse(args):
    df = pd.read_csv(args.output_file_covariates, dtype={'patid': str, 'covid': int})
    return df


if __name__ == '__main__':
    # python query_character.py --dataset ALL 2>&1 | tee  log/query_character_ALL_data_building.txt

    start_time = time.time()
    args = parse_args()
    df = cohorts_characterization_build_data(args)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
