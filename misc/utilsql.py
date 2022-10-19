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
from sqlalchemy import create_engine
print = functools.partial(print, flush=True)


def load_sql_credential():
    with open('../misc/pg_credential.json') as _ff_:
        cred_dict = json.load(_ff_)
        pg_username = cred_dict['pg_username']
        pg_password = cred_dict['pg_password']
        pg_server = cred_dict['pg_server']
        pg_port = cred_dict['pg_port']
        pg_database = cred_dict['pg_database']
        connect_string = f"postgresql+psycopg2://{pg_username}:{urllib.parse.quote_plus(pg_password)}@{pg_server}:{pg_port}/{pg_database}"

    return connect_string, cred_dict


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