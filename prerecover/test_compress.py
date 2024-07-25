import sys
# for linux env.
import pandas as pd

sys.path.insert(0, '..')
import time
import pickle
import numpy as np
import argparse
from misc import utils
from eligibility_setting import _is_in_baseline, _is_in_followup, INDEX_AGE_MINIMUM, INDEX_AGE_MINIMUM_18
import functools
from collections import Counter
import blosc
import pgzip

print = functools.partial(print, flush=True)

if __name__ == '__main__':
    start_time = time.time()

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    dataset = 'emory_pcornet_all' #'duke_pcornet_all'
    input_file = r'../data/recover/output/{}/cohorts_covid_posnegpreg_{}.pkl'.format(
            dataset,
            dataset)


    print('Load cohorts pickle data file:', input_file)
    id_data = utils.load(input_file, chunk=4)
    print('Load time! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    # num_cpu = 8  # use all available CPUs
    # with pgzip.open("test-{}.pkl.gz".format(dataset), "wb", thread=num_cpu, blocksize=2 * 10 ** 8) as fw:
    #     pickle.dump(id_data, fw)
    # print('Dump! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    #
    # start_time = time.time()
    # with pgzip.open("test-{}.pkl.gz".format(dataset), "rb", thread=num_cpu) as fr:
    #     data = pickle.load(fr)

    utils.dump_compressed(id_data, dataset + '.pkl.test.gz')
    data = utils.load(dataset + '.pkl.test.gz')

    print('len(data):', len(data))
    print('read! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
