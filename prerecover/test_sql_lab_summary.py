import sys

# for linux env.
sys.path.insert(0, '..')
import psycopg2
import pandas as pd
import time
# import connectorx as cx
import urllib
from tqdm import tqdm
import functools
from misc import utils
import json
print = functools.partial(print, flush=True)

if __name__ == '__main__':
    start_time = time.time()

    sites = ['mcw', 'nebraska', 'utah', 'utsw',
             'wcm', 'montefiore', 'mshs', 'columbia', 'nyu',
             'ufh', 'usf', 'nch', 'miami',  'emory',
             'pitt', 'psu', 'temple', 'michigan',
             'ochsner', 'ucsf', 'lsu',
             'vumc']

    results = []
    for site in sites:
        fname = r'output\ana_loinc\ana_lab_{}.csv'.format(site)
        df = pd.read_csv(fname)
        df['site'] = site
        print(site, df.shape[0])
        results.append(df)


    results = pd.concat(results)
    results.to_csv(r'output\ana_loinc\ana_lab_22sites.csv')

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
