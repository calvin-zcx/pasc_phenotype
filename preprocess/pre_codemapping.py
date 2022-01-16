import sys
# for linux env.
sys.path.insert(0,'..')
import os
import shutil
import zipfile
import urllib.parse
import urllib.request
import tqdm
import pickle
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import Counter, defaultdict
import pandas as pd
import json
import requests
import functools
import utils
print = functools.partial(print, flush=True)
import time


def rxnorm_atc_from_NIH_UMLS():
    # To get code mapping from rxnorm_cui to ATC.
    # Data source: RxNorm_full_01032022.zip,
    # MD5 checksum: 8c8c0267fb2e09232fb852f7500c5b18, Release Notes 01/03/2022
    # https://www.nlm.nih.gov/research/umls/rxnorm/index.html,
    # https://www.nlm.nih.gov/research/umls/rxnorm/docs/techdoc.html#conso
    # https://mor.nlm.nih.gov/RxNav/search?searchBy=RXCUI&searchTerm=6883
    start_time = time.time()
    rx_df = pd.read_csv(r'../data/mapping/RXNCONSO.RRF', sep='|', header=None, dtype=str)
    print('rx_df.shape:', rx_df.shape)
    atc_df = rx_df.loc[rx_df[11] == 'ATC']
    print('atc_df.shape:', atc_df.shape)
    ra = set()
    rxrnom_atcset = defaultdict(set)
    for index, row in atc_df.iterrows():
        rx = row[0].strip()
        atc = row[13].strip()
        name = row[14]
        ra.add((rx, atc, name))
        rxrnom_atcset[rx].add(atc)

    print('unique rxrnom-atc-name records: len(ra):', len(ra))
    print('len(rxrnom_atcset):', len(rxrnom_atcset))

    df = pd.DataFrame(ra, columns=['rxnorm', 'atc', 'name']).sort_values(by='rxnorm', key=lambda x: x.astype(int))
    df.to_csv(r'../data/mapping/rxnorm_atc_mapping_from_NIH_UMLS_full_01032022.csv')

    output_file = r'../data/mapping/rxnorm_atc_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(rxrnom_atcset, open(output_file, 'wb'))

    print('dump done to {}'.format(output_file))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return rxrnom_atcset, df


if __name__ == '__main__':
    rxrnom_atcset, df_rxrnom_atc = rxnorm_atc_from_NIH_UMLS()
    print('Done!')
