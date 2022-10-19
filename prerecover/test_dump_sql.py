# dumping data from Recover database
import sys
# for linux env.
sys.path.insert(0, '..')
import psycopg2
import pandas as pd
import time
import urllib
import functools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
# from memory_profiler import memory_usage
from sqlalchemy import create_engine
print = functools.partial(print, flush=True)


def get_table_size(connect_string, table_name, unit='gb'):
    engine = create_engine(connect_string)
    query = "SELECT ( pg_total_relation_size('{}') );".format(table_name)
    df = pd.read_sql_query(query, engine)
    size = df.iloc[0, 0]
    if unit == 'kb':
        size = size / 1024
    elif unit == 'mb':
        size = size / 1024 / 1024
    elif unit == 'gb':
        size = size / 1024 / 1024 / 1024
    else:
        size = size  # others for byte
    return size


def get_table_rows(connect_string, table_name):
    engine = create_engine(connect_string)
    query = "SELECT COUNT(*) FROM {};".format(table_name)
    df = pd.read_sql_query(query, engine)
    rows = df.iloc[0, 0]
    return rows


def export_csv(connect_string, sql_query, csv_file_path, chunksize=None, n_chunk=1, compression='infer'):
    start_time = time.time()
    engine = create_engine(connect_string)
    connection = engine.connect().execution_options(
        stream_results=True, max_row_buffer=chunksize
    )
    header = True
    mode = "w"
    n = 0
    for df in tqdm(pd.read_sql(sql_query, connection, chunksize=chunksize), total=n_chunk):
        df.to_csv(csv_file_path, mode=mode, header=header, index=False, compression=compression)
        n += 1
        if header:
            header = False
            mode = "a"
    print('Finish:', sql_query,
          'no. of chunk:', n,
          'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    connection.close()
    return df


def extract_all(site_names, folder_path, start_date='2021-03-01', end_date='2022-04-01'):
    '''This function pulls all the encounters between a time interval
    This function saves a combined dataset of all sites data
    input 1 - site_names should be a list of strings mirroring the exact database name for each site
    input 2 - folder_path is a string indicating where the resutls will be saved
    input 3 - start_date: is a string 'YYYY-MM-DD' indicating the start of interval. Default value
    input 4 - end_date: is a string 'YYYY-MM-DD' indicating the end of interval
    returns the combined data set as pandas dataframe
    '''

    # encounter = pd.DataFrame()
    s_0 = [ 'lab_result_cm',]
    table_list = ['lab_result_cm']
    counter = 1
    for i in site_names:
        print(f"""querying {counter}. {i} data started""")
        for j in table_list:
            print('j',j)
            query = f"""
            select *
            from {i}.{j}
            """
            chunksize = 100000
            csv_file_path = folder_path + i + '\\' + j + '.csv'
            mode = "w"
            header = True
            k = 0
            for df in pd.read_sql(query, conn, chunksize=chunksize):
                if k % 10 == 0:
                    print('K = ', k)
                df.to_csv(csv_file_path, mode=mode, header=header, index=False)
                if header:
                    header = False
                    mode = "a"
                k = k + 1

            del df
            print("*" * 20)
        print(f"""{i} data query has finished""")
        print("*" * 50)
        counter += 1

    return counter


if __name__ == '__main__':
    start_time = time.time()

    # 1. connect
    # 2. get file size
    # 3. determine file compressed or not
    # 4. read and dump file by chunk
    # cred_dict = {
    #     "pg_username" : "wcm_analyst",
    #     "pg_password" : "zy6ryZbz9yfuAL7X",
    #     "pg_server" : "aurora-stack-auroradbcluster-1rbqp9aty8v4q.cluster-cbs1thv2ku8o.us-east-1.rds.amazonaws.com",
    #     "pg_port" : '5432',
    #     "pg_database" : 'recover',
    # }
    with open('pg_credential.json') as _ff_:
        cred_dict = json.load(_ff_)
        pg_username = cred_dict['pg_username']
        pg_password = cred_dict['pg_password']
        pg_server = cred_dict['pg_server']
        pg_port = cred_dict['pg_port']
        pg_database = cred_dict['pg_database']
        connect_string = f"postgresql+psycopg2://{pg_username}:{urllib.parse.quote_plus(pg_password)}@{pg_server}:{pg_port}/{pg_database}"


    try:
        table_name = 'columbia.lab_result_cm'
        table_size = get_table_size(connect_string, table_name, unit='gb')
        table_rows = get_table_rows(connect_string, table_name)
        print('Read sql table:', table_name, '| Table size: {:.2f} GB'.format(table_size), '| No. of rows:', table_rows)
        chunksize = 100000
        n_chunk = int(np.ceil(table_rows / chunksize))

        sql_query = """select *
                    from {}.{}
                """.format('columbia', 'lab_result_cm')
        csv_file_path = 'output/' + table_name + '.csv'
        if table_size > 5:
            csv_file_path += '.gz'
        print('Dump to:', csv_file_path)

        df_last = export_csv(connect_string, sql_query, csv_file_path, chunksize, n_chunk, compression='infer')

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)

    # # list the name of sites
    # # the names should corrospond to the schema name in the database
    # site_names = ['wcm'] #'ufh', 'vumc', 'lsu', 'mcw', 'ochsner', 'temple', 'ucsf', 'utah', 'utsw'
    #     # , 'wcm_pcornet', 'nyu_pcornet', 'mshs_pcornet', 'montefiore_pcornet'
    # # columbia,'mshs','montefiore','nyu','wcm'
    # # mcw, ufh, pitt, ochsner, vumc
    #
    # # the folder path where you want the results to be saved
    # folder_path = "D:\ZhenxingXu\data\COVID19\\"
    #
    # # encounter = extract_encounter(site_names, folder_path, start_date='2001-03-01', end_date='2022-09-30')
    # all_data = extract_all(site_names, folder_path, start_date='2001-03-01', end_date='2022-09-30')

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


