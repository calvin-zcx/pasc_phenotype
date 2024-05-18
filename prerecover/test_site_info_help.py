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

    df_site = pd.read_excel('RECOVER Adult Site schemas_edit.xlsx')

    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']
    # ['duke', 'intermountain', 'missouri', 'iowa', 'northwestern', 'ochin', 'osu', 'wakeforest',  'musc']
    site_list = site_list.to_list() + ['northwestern', 'wakeforest',
                                       'chop', 'nemours', 'nationwide', 'seattle', 'colorado', 'lurie',
                                       'cchmc', 'national', 'indiana', 'stanford', ]  # these two sites with label 0
    # Intermountain does not have covid data? From dmi
    #  Indiana is the site that is not present.
    print('len(site_list):', len(site_list), site_list)


    finished_site = [
        'columbia_pcornet_all', 'duke_pcornet_all', 'emory_pcornet_all', 'intermountain_pcornet_all', 'iowa_pcornet_all',
        'lsu_pcornet_all', 'mcw_pcornet_all', 'michigan_pcornet_all', 'missouri_pcornet_all', 'montefiore_pcornet_all',
        'mshs_pcornet_all', 'wcm_pcornet_all', 'nch_pcornet_all', 'nebraska_pcornet_all', 'northwestern_pcornet_all',
        'nyu_pcornet_all', 'ochsner_pcornet_all', 'osu_pcornet_all', 'pitt_pcornet_all', 'psu_pcornet_all',
        'temple_pcornet_all', 'ufh_pcornet_all', 'utah_pcornet_all', 'utsw_pcornet_all',
        'vumc_pcornet_all', 'wakeforest_pcornet_all',]
    print('len(finished_site)', len(finished_site))

    sa = set(site_list)
    sb = set([x.split('_')[0] for x in finished_site])

    print('len(sa-sb)', len(sa  -sb), sa - sb)
    print('len(sb-sa)', len(sb - sa), sb - sa)


    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
