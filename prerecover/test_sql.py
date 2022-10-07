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

print = functools.partial(print, flush=True)


if __name__ == '__main__':
    start_time = time.time()

    try:
        conn = psycopg2.connect(user="wcm_analyst",
                                password="zy6ryZbz9yfuAL7X",
                                host="aurora-stack-auroradbcluster-1rbqp9aty8v4q.cluster-cbs1thv2ku8o.us-east-1.rds.amazonaws.com",
                                port="5432",
                                database="recover")
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)

    pg_username = "wcm_analyst"
    pg_password = "zy6ryZbz9yfuAL7X"
    pg_server = "aurora-stack-auroradbcluster-1rbqp9aty8v4q.cluster-cbs1thv2ku8o.us-east-1.rds.amazonaws.com"
    pg_port = '5432'
    pg_database = 'recover'
    connect_string = f"postgres://{pg_username}:{urllib.parse.quote_plus(pg_password)}@{pg_server}:{pg_port}/{pg_database}"

    # list the name of sites
    # the names should corrospond to the schema name in the database
    site_names = ['mshs', 'montefiore', 'nyu', 'wcm' 'ufh', 'vumc', 'lsu', 'mcw', 'ochsner', 'temple', 'ucsf', 'utah', 'utsw']
    # , 'wcm_pcornet', 'nyu_pcornet', 'mshs_pcornet', 'montefiore_pcornet'
    # columbia,'mshs','montefiore','nyu','wcm'
    # mcw, ufh, pitt, ochsner, vumc

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

        start_time = time.time()
        df = pd.read_sql(query, conn)
        # df = cx.read_sql(connect_string, query)
        # save query result
        df.to_csv(r'output\ana_lab_{}.csv'.format(site))
        print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
