import sys

# for linux env.
sys.path.insert(0, '..')
import os
import pickle
import numpy as np
from collections import defaultdict, OrderedDict
import pandas as pd
import requests
import functools
from misc import utils
import re
from tqdm import tqdm
import psycopg2
import urllib
import time
from sqlalchemy import create_engine
import json
from datetime import datetime

print = functools.partial(print, flush=True)


def get_table_size(connect_string, table_name):
    engine = create_engine(connect_string)
    query = "SELECT  pg_size_pretty (pg_total_relation_size('{}') );".format(table_name)
    df = pd.read_sql_query(query, engine)
    size = df.iloc[0, 0]
    return size


def get_table_rows(connect_string, table_name):
    engine = create_engine(connect_string)
    query = "SELECT COUNT(*) FROM {};".format(table_name)
    df = pd.read_sql_query(query, engine)
    rows = df.iloc[0, 0]
    return rows


def get_table_columns(connect_string, table_name):
    engine = create_engine(connect_string)
    query = "SELECT * FROM {} LIMIT 1;".format(table_name)
    df = pd.read_sql_query(query, engine)
    cols = df.columns
    return cols


if __name__ == '__main__':
    # python pre_codemapping.py 2>&1 | tee  log/pre_codemapping_zip_adi.txt
    start_time = time.time()
    pd.set_option('display.max_colwidth', None)

    # df_site = pd.read_excel('RECOVER Adult Site schemas_edit.xlsx')
    df_site = pd.read_excel('RECOVER Adult Site schemas_edit2025.xlsx')

    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']
    # ['duke', 'intermountain', 'missouri', 'iowa', 'northwestern', 'ochin', 'osu', 'wakeforest',  'musc']
    # site_list = site_list.to_list() + ['northwestern', 'wakeforest',
    #                                    'chop', 'nemours', 'nationwide', 'seattle', 'colorado', 'lurie',
    #                                    'cchmc', 'national', 'indiana', 'stanford', ]  # these two sites with label 0
    site_list = site_list.to_list() + [
                                       'chop', 'nemours', 'nationwide', 'seattle', 'colorado', 'lurie',
                                       'cchmc', 'national', 'indiana',
                                        'usf', 'miami', 'musc'  # not for S11, S12, add northwestern, wakeforest, stanford to selected

    ]  # these two sites with label 0
    # Intermountain does not have covid data? From dmi
    #  Indiana is the site that is not present.
    print('len(site_list):', len(site_list), site_list)
    with open('../misc/pg_credential.json') as _ff_:
        cred_dict = json.load(_ff_)
        pg_username = cred_dict['pg_username']
        pg_password = cred_dict['pg_password']
        pg_server = cred_dict['pg_server']
        pg_port = cred_dict['pg_port']
        pg_database = cred_dict['pg_database']
        connect_string = f"postgresql+psycopg2://{pg_username}:{urllib.parse.quote_plus(pg_password)}@{pg_server}:{pg_port}/{pg_database}"
        print('pg_database', pg_database)
    try:
        # conn = psycopg2.connect(user=pg_username,
        #                         password=pg_password,
        #                         host=pg_server,
        #                         port=pg_port,
        #                         database=pg_database)
        engine = create_engine(connect_string)

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)

    # select target table?
    table_list = ['condition', 'covid_elements', 'death', 'death_cause', 'demographic', 'diagnosis', 'dispensing',
                  'encounter', 'enrollment', 'harvest', 'hash_token', 'immunization', 'lab_history', 'lab_result_cm',
                  'lds_address_history', 'med_admin', 'obs_clin', 'obs_gen', 'pcornet_trial', 'prescribing', 'pro_cm',
                  'procedures', 'provider', 'vital', 'geocoded_2010', 'geocoded', 'geocoded_2020', ]
    # no date colum: death_cause, 'harvest', 'hash_token', 'lab_history','provider',
    table_dict = {'condition': 'report_date', 'covid_elements': 'admit_date', 'death': 'death_date',
                  'demographic': 'birth_date',
                  'diagnosis': 'admit_date', 'dispensing': 'dispense_date',
                  'encounter': 'admit_date', 'enrollment': 'enr_start_date',
                  'immunization': 'vx_record_date', 'lab_result_cm': 'result_date',  # 'lab_order_date',
                  'lds_address_history': 'address_period_start',
                  'med_admin': 'medadmin_start_date', 'obs_clin': 'obsclin_start_date', 'obs_gen': 'obsgen_start_date',
                  'pcornet_trial': 'trial_enroll_date', 'prescribing': 'rx_order_date', 'pro_cm': 'pro_time',
                  'procedures': 'admit_date', 'vital': 'measure_date',
                  'geocoded_2010': 'patid', 'geocoded': 'patid', 'geocoded_2020': 'patid', 'provider': 'providerid'}
    results = []
    error_msg = []
    # site_list = ['temple', 'usf']
    for site in tqdm(site_list):
        # for table in table_list:
        for table, col in tqdm(table_dict.items()):
            print('In', site, table)
            # should use according column, build a mapping later
            query = "select MIN({}) AS MinDate,MAX({}) AS MaxDate from {}_pcornet_all.{};".format(col, col, site, table)
            try:
                df = pd.read_sql(query, engine)
                df['site'] = site
                df['table'] = table
                df['date_col'] = col

                table_name = '{}_pcornet_all.{}'.format(site, table)
                table_size = get_table_size(connect_string, table_name)
                table_rows = get_table_rows(connect_string, table_name)
                df['table_size'] = table_size
                df['table_rows'] = table_rows
                df['query'] = query
                col_names = get_table_columns(connect_string, table_name)
                df['col_names'] = str(len(col_names)) + ':' + ','.join(col_names)
                # df.rename(index={0: site + '-' + table + '-' + col}, inplace=True)
                results.append(df)
                # print(df)
                # print(df.iloc[0, :])
            except Exception as e:
                print(site, e)
                error_msg.append(site + '-' + table + '-' + col + str(e))
        print('Done', site)
        # break

    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d")  # now.strftime("%m/%d/%Y, %H:%M:%S")

    pd_results = pd.concat(results, ignore_index=True)
    df_combined = pd.merge(pd_results, df_site, left_on='site', right_on='Schema name',
                           how='left')  # df_site.loc[df_site['selected'] == 1]
    # df_combined.to_csv('output/db_info/dev_s11_pcornet_all_table_date-{}.csv'.format(date_time))
    df_combined.to_csv('output/db_info/dev_s12_pcornet_all_table_date-{}.csv'.format(date_time))

    pd_error = pd.DataFrame(error_msg)
    # pd_error.to_csv('output/db_info/dev_s11_pcornet_all_table_date_ErrorMsg-{}.csv'.format(date_time))
    pd_error.to_csv('output/db_info/dev_s12_pcornet_all_table_date_ErrorMsg-{}.csv'.format(date_time))

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
