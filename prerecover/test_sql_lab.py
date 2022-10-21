import sys

# for linux env.
sys.path.insert(0, '..')
import psycopg2
import pandas as pd
import time
import connectorx as cx
import urllib
from tqdm import tqdm
import functools
from misc import utils
import json
print = functools.partial(print, flush=True)


if __name__ == '__main__':
    start_time = time.time()

    with open('pg_credential.json') as _ff_:
        cred_dict = json.load(_ff_)
        pg_username = cred_dict['pg_username']
        pg_password = cred_dict['pg_password']
        pg_server = cred_dict['pg_server']
        pg_port = cred_dict['pg_port']
        pg_database = cred_dict['pg_database']
        connect_string = f"postgresql+psycopg2://{pg_username}:{urllib.parse.quote_plus(pg_password)}@{pg_server}:{pg_port}/{pg_database}"

    try:
        conn = psycopg2.connect(user=pg_username,
                                password=pg_password,
                                host=pg_server,
                                port=pg_port,
                                database=pg_database)
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)

    connect_string = f"postgres://{pg_username}:{urllib.parse.quote_plus(pg_password)}@{pg_server}:{pg_port}/{pg_database}"

    query = """
    SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA
    order by SCHEMA_NAME;
    """
    query = """
     SELECT nspname
    FROM pg_catalog.pg_namespace
    order by nspname;
    """
    df_name = pd.read_sql(query, conn)
    df_name.to_csv('site_name_all.csv')


    # list the name of sites
    # the names should corrospond to the schema name in the database
    site_names = [  # 'mshs', 'montefiore', 'nyu', 'wcm', 'columbia',
        # 'emory', 'nch', 'psu', #'pitt',
        # 'ufh', 'vumc', 'lsu', 'mcw', 'ochsner',
        # 'temple_v1', 'ucsf', 'utah', 'utsw',
        'wcm_v1'
    ]


    for site in tqdm(site_names):
        query = """
        SELECT Count(*) as CODE_COUNT, LAB_LOINC, RAW_RESULT, RESULT_NUM, RESULT_QUAL 
        FROM {}.LAB_RESULT_CM
        WHERE LAB_LOINC in ('29953-7',
        '33253-6',
        '40655-3',
        '42254-3',
        '47383-5',
        '5047-6',
        '5048-4',
        '59069-5',
        '8061-4',
        '9423-5')
        GROUP BY LAB_LOINC, RAW_RESULT, RESULT_NUM, RESULT_QUAL
        ORDER BY LAB_LOINC, CODE_COUNT, RAW_RESULT
        """.format(site)
        print(site, query)

        start_time = time.time()
        df = pd.read_sql(query, conn)
        # df = cx.read_sql(connect_string, query)
        # save query result
        df.to_csv(r'output\ana_lab_{}.csv'.format(site))
        print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
