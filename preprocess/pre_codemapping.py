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
from collections import Counter, defaultdict, OrderedDict
import pandas as pd
import json
import requests
import functools
import utils
print = functools.partial(print, flush=True)
import time


def rxnorm_ingredient_from_NIH_UMLS():
    # To get code mapping from rxnorm_cui to ingredients.
    # for (Single Active) Ingredients
    # rx_ing[rx] = [ing1, ]
    # for Multiple Ingredients
    # rx_ing[rx] = [ing1, ing2, ...]
    # Data source: RxNorm_full_01032022.zip, RxNorm_full_01032022\rrf\RXNREL.RRF
    # MD5 checksum: 8c8c0267fb2e09232fb852f7500c5b18, Release Notes 01/03/2022
    # https://www.nlm.nih.gov/research/umls/rxnorm/index.html,
    # https://www.nlm.nih.gov/research/umls/rxnorm/docs/techdoc.html#conso
    # https://mor.nlm.nih.gov/RxNav/search?searchBy=RXCUI&searchTerm=6883
    # type name for IN, MIN, PIN: https://www.nlm.nih.gov/research/umls/rxnorm/docs/appendix5.html
    # relation: https://www.nlm.nih.gov/research/umls/rxnorm/docs/appendix1.html#PIN
    # step 1: find all 13569 IN, 3767 MIN, 3221 PIN from RXNCONSO.RRF, col11=='RXNORM'  col12 == 'IN' | 'MIN' | PIN'
    # step 2: select RXNREL.RRF where col2 == 'CUI' and col6 == 'CUI
    # step 3: if rxcui2 col4 is IN:  rx_ing[col0] .append( col4 )
    # step 4: if rxcui2 col4 is MIN and rxcui1 col0 is SCD and relation is ingredients_of, rx_ing[col0] = rx_ing[col4]

    start_time = time.time()
    link_df = pd.read_csv(r'../data/mapping/RXNREL.RRF', sep='|', header=None, dtype=str)
    node_df = pd.read_csv(r'../data/mapping/RXNCONSO.RRF', sep='|', header=None, dtype=str)

    print('link_df.shape:', link_df.shape)  # (7204009, 17)
    print('node_df.shape:', node_df.shape)  # node_df.shape: (1101174, 19)
    link_cui_df = link_df.loc[(link_df[2] == 'CUI') & (link_df[6] == 'CUI'), :]
    print('all cui-cui relations, link_cui_df.shape:', link_cui_df.shape)  # (7204009, 17)

    rx_name_df = node_df.loc[(node_df[11]=='RXNORM'), [0, 14]]
    rx_name = {row[0]:row[14] for index, row in rx_name_df.iterrows()}

    IN_and_name = node_df.loc[(node_df[11]=='RXNORM') & (node_df[12]=='IN'), [0, 14, 12]]
    MIN_and_name = node_df.loc[(node_df[11] == 'RXNORM') & (node_df[12] == 'MIN'), [0, 14, 12]]
    PIN_and_name = node_df.loc[(node_df[11] == 'RXNORM') & (node_df[12] == 'PIN'), [0, 14, 12]]
    print('IN_and_name.shape:', IN_and_name.shape)   # (13569, 3)
    print('MIN_and_name.shape:', MIN_and_name.shape)  # (3767, 3)
    print('PIN_and_name.shape:', PIN_and_name.shape)  # (3221, 3)

    IN_set = set(IN_and_name[0])
    MIN_set = set(MIN_and_name[0])
    PIN_set = set(PIN_and_name[0])

    print('len(IN_set):', len(IN_set))  # (13569, 3)
    print('len(MIN_set):', len(MIN_set))  # (3767, 3)
    print('len(PIN_set):', len(PIN_set))  # (3221, 3)

    rx_ing = defaultdict(set)
    i = 0
    for index, row in link_cui_df.iterrows():
        rx1 = row[0]
        rx2 = row[4]
        relation_str = row[7]
        if rx2 in IN_set:
            rx_ing[rx1].add(rx2)
            i += 1
    print('add all records from IN, len(rx_ing):', len(rx_ing), 'add records:', i)

    i = 0
    for index, row in link_cui_df.iterrows():
        rx1 = row[0]
        rx2 = row[4]
        relation_str = row[7]
        if (rx2 in MIN_set) and (relation_str == 'ingredients_of'):
            rx_ing[rx1].update(rx_ing[rx2])  # one scd may have multiple MIN
            i += 1
    print('add (MIN	ingredients_of	SCD) from MIN, len(rx_ing):', len(rx_ing), 'add records:', i)

    records = []
    for key, val in rx_ing.items():
        rx_ing[key] = sorted(val)
        name = rx_name[key]
        records.append((key, name, len(val), ';'.join(val)))

    df_rx_ing = pd.DataFrame(records, columns=['rxnorm_cui', 'name', 'num of ingredient(s)', 'ingredient(s)'])
    print('df_rx_ing.shape', df_rx_ing.shape)
    df_rx_ing.to_csv(r'../data/mapping/rxnorm_ingredient_mapping.csv')

    print('rxnorm to active ingredient(s): len(rx_ing):', len(rx_ing))
    output_file = r'../data/mapping/rxnorm_ingredient_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(rx_ing, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return rx_ing, df_rx_ing


# Data source 5: nih rxclass api https://rxnav.nlm.nih.gov/api-RxClass.getClassByRxNormDrugId.html
def _parse_from_nih_rxnorm_api(rxcui):
    r = requests.get('https://rxnav.nlm.nih.gov/REST/rxcui/{}/property.json?propName=Active_moiety_RxCUI'.format(rxcui))
    # Active_ingredient_RxCUI or use Active_moiety_RxCUI  ?  e.g. 1114700 different
    # moiety seems more low level
    # moiety: {"propConceptGroup":{"propConcept":
    # [{"propCategory":"ATTRIBUTES","propName":"Active_moiety_RxCUI","propValue":"161"},
    # {"propCategory":"ATTRIBUTES","propName":"Active_moiety_RxCUI","propValue":"33408"}]}}
    # ingredient: {"propConceptGroup":{"propConcept":
    # [{"propCategory":"ATTRIBUTES","propName":"Active_ingredient_RxCUI","propValue":"161"},
    # {"propCategory":"ATTRIBUTES","propName":"Active_ingredient_RxCUI","propValue":"221141"}]}}
    data = r.json()
    ing_set = set()
    if ('propConceptGroup' in data) and ('propConcept' in data['propConceptGroup']):
        for x in data['propConceptGroup']['propConcept']:
            ing = x['propValue']
            ing_set.add(ing)
    return ing_set


def add_rxnorm_ingredient_by_umls_api():
    # 199903 is still not in our dictionary
    # 199903 has trade name 211759
    # 211759 has ingredient 5487
    # 1. find all rxnorm in umls file
    # 2. api search
    # 3. compare, contrast, and update existing dictionary

    start_time = time.time()
    with open(r'../data/mapping/rxnorm_ingredient_mapping.pkl', 'rb') as f:
        rxnorm_ing = pickle.load(f)
        print('Load rxRNOM_CUI to ingredient mapping done! len(rxnorm_atc):', len(rxnorm_ing))
        record_example = next(iter(rxnorm_ing.items()))
        print('e.g.:', record_example)

    node_df = pd.read_csv(r'../data/mapping/RXNCONSO.RRF', sep='|', header=None, dtype=str)
    print('node_df.shape:', node_df.shape)  # node_df.shape: (1101174, 19)

    node_cui_df = node_df.loc[(node_df[11]=='RXNORM'), [0, 14]].drop_duplicates()
    rxnorm_name = {row[0]: row[14] for index, row in node_cui_df.iterrows()}
    rxnorm_set = set(node_df.loc[(node_df[11]=='RXNORM'), 0])
    print('unique rxnorm codes number: ', len(rxnorm_set))

    rx_ing_api = defaultdict(set)
    n_no_return = 0
    n_has_return = 0
    i = 0
    for rx in rxnorm_set:
        i+=1
        ings = _parse_from_nih_rxnorm_api(rx)
        if ings:
            rx_ing_api[rx].update(ings)
            n_has_return+=1
        else:
            n_no_return += 1

        if rx in rxnorm_ing:
            print(i, rx, ':already found:', ';'.join(rxnorm_ing[rx]), 'new found', ';'.join(sorted(ings)))

        if i % 10000 == 0:
            print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
            print('Total search:', len(rxnorm_set), 'search:', i, 'n_has_return:', n_has_return, 'n_no_return:', n_no_return)

    print('Total search:', len(rxnorm_set), 'n_has_return:', n_has_return, 'n_no_return:', n_no_return)

    records = []
    for key, val in rx_ing_api.items():
        rx_ing_api[key] = sorted(val)
        name = rxnorm_name[key]
        records.append((key, name, len(val), ';'.join(val)))

    df_rx_ing = pd.DataFrame(records, columns=['rxnorm_cui', 'name', 'num of ingredient(s)', 'ingredient(s)'])
    print('df_rx_ing.shape', df_rx_ing.shape)
    df_rx_ing.to_csv(r'../data/mapping/rxnorm_ingredient_mapping_from_api.csv')

    print('rxnorm to active ingredient(s): len(rx_ing_api):', len(rx_ing_api))
    output_file = r'../data/mapping/rxnorm_ingredient_mapping_from_api.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(rx_ing_api, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return rx_ing_api, df_rx_ing


def rxnorm_atc_from_NIH_UMLS():
    # To get code mapping from rxnorm_cui to ATC.
    # Warning: ATC only have rxnorm ingredient (single) correspondence
    #          thus, we need rxrnom to its active ingredient mapping
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
    rxnorm_atcset = defaultdict(set)
    atc_rxnormset = defaultdict(set)
    for index, row in atc_df.iterrows():
        rx = row[0].strip()
        atc = row[13].strip()
        name = row[14]
        ra.add((rx, atc, name))
        rxnorm_atcset[rx].add((atc, name))
        atc_rxnormset[atc].add((rx, name))

    print('unique rxrnom-atc-name records: len(ra):', len(ra))
    print('len(rxnorm_atcset):', len(rxnorm_atcset))
    print('len(atc_rxnormset):', len(atc_rxnormset))

    # select atc-l3 len(atc) == 4 to index:
    atc3_info = defaultdict(list)
    for x in ra:
        rx, atc, name = x
        if len(atc) == 4:
            atc3_info[atc].extend([rx, name])
    atc3_info_sorted = OrderedDict(sorted(atc3_info.items()))
    atc3_index = {}
    for i, (key, val) in enumerate(atc3_info_sorted.items()):
        atc3_index[key] = [i, ] + val
    print('Unique atc level 3 codes: len(atc3_index):', len(atc3_index))
    output_file = r'../data/mapping/atcL3_index_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(atc3_index, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    # dump for debug
    df = pd.DataFrame(ra, columns=['rxnorm', 'atc', 'name']).sort_values(by='rxnorm', key=lambda x: x.astype(int))
    df.to_csv(r'../data/mapping/rxnorm_atc_mapping_from_NIH_UMLS_full_01032022.csv')

    # dump
    output_file = r'../data/mapping/rxnorm_atc_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(rxnorm_atcset, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    output_file = r'../data/mapping/atc_rxnorm_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(atc_rxnormset, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return rxnorm_atcset, atc_rxnormset, atc3_index, df


def zip_aid_mapping():
    # To get code mapping from rxnorm_cui to ATC.
    # Data source: https://www.neighborhoodatlas.medicine.wisc.edu/download
    # 52 states files
    # details in 2019 ADI_9 Digit Zip Code_v3.1_ReadMe
    start_time = time.time()
    readme_df = pd.read_csv(r'../data/mapping/ADI/2019_ADI_9_V3.1_readme_statistics_check.csv')
    print('readme_df.shape:', readme_df.shape)
    ## df2 = pd.read_csv(r'../data/mapping/ADI/wcm_zip_state.csv')
    ## readme_df['State_abr'] = readme_df['State_abr'].apply(str.upper)
    ## df_combined = pd.merge(readme_df, df2, left_on='State_abr', right_on='address_state', how='left')
    ## df_combined.to_csv(r'../data/mapping/ADI/2019_ADI_9_V3.1_readme_statistics_check_v2.csv')
    zip_adi = {}
    zip5_df = []
    for index, row in readme_df.iterrows():
        state = row[1]
        name = row[2]
        n_records_adi = row[3]
        n_records_wcm = row[4]
        input_file = r'../data/mapping/ADI/{}_2019_ADI_9 Digit Zip Code_v3.1.txt'.format(state)
        # specified_dtype = {'Unnamed: 0': int, 'X': int, 'TYPE': str, 'ZIPID': str, 'FIPS.x': str,
        #               'GISJOIN': str, 'FIPS.y': str, 'ADI_NATRANK': int, 'ADI_STATERNK': int}
        if os.path.exists(input_file):
            df = pd.read_csv(input_file, dtype=str)
            print(index, input_file, 'df.shape:', df.shape, 'n_records_adi:', n_records_adi, 'n_records_wcm:', n_records_wcm)
            if df.shape[0] != n_records_adi:
                print('ERROR in ', input_file, 'df.shape[0] != n_records_adi')
            df['nation_rank'] = pd.to_numeric(df['ADI_NATRANK'], errors='coerce')
            df['state_rank'] = pd.to_numeric(df['ADI_STATERNK'], errors='coerce')
            df['zip5'] = df["ZIPID"].apply(lambda x : x[:6] if pd.notna(x) else np.nan)
            zip5_scores = df.groupby(["zip5"])[['nation_rank', "state_rank"]].median().reset_index()
            print('......zip5_scores.shape:', zip5_scores.shape)

            zip9_list = df[['ZIPID', 'nation_rank', 'state_rank']].values.tolist()
            zip5_list = zip5_scores[['zip5', 'nation_rank', 'state_rank']].values.tolist()
            # save zip5 for debugging
            zip5_df.append(zip5_scores[['zip5', 'nation_rank', 'state_rank']])
            # n_null_zip9 = n_null_zip5 = 0
            zip_adi.update({x[0][1:]: x[1:] for x in zip9_list if pd.notna(x[0]) })
            print('......len(zip_adi) after adding zip9:', len(zip_adi))
            zip_adi.update({x[0][1:]: x[1:] for x in zip5_list if pd.notna(x[0])})
            print('......len(zip_adi) after adding zip5:', len(zip_adi))
            print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
        else:
            print(index, input_file, 'NOT EXIST!')

    output_file = r'../data/mapping/zip9or5_adi_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(zip_adi, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    # double check zip5
    zip5_df = pd.concat(zip5_df)
    print('zip5_df.shape', zip5_df.shape)
    zip5_df.to_csv(r'../data/mapping/zip5_for_debug.csv')
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return zip_adi, zip5_df


def ICD10_to_CCSR():
    # To get code mapping from icd10 to ccsr.
    # Data source: https://www.hcup-us.ahrq.gov/toolssoftware/ccsr/dxccsr.jsp
    # CCSR v2022.1: Fiscal Year 2022, Released October 2021 - valid for ICD 10-CM diagnosis codes through September 2022
    # CCSR for ICD-10-CM Diagnoses Tool, v2022.1 (ZIP file, 4.8 MB) released 10/28/21
    start_time = time.time()
    df = pd.read_csv(r'../data/mapping/DXCCSR_v2022-1/DXCCSR_v2022-1.CSV')
    print('df.shape:', df.shape)

    df_dimension = pd.read_excel(r'../data/mapping/DXCCSR_v2022-1/DXCCSR-Reference-File-v2022-1.xlsx',
                                 sheet_name='CCSR_Categories', skiprows=[0])
    df_dimension = df_dimension.reset_index()

    ccsr_index = {}
    for index, row in df_dimension.iterrows():
        ccsr = row[1]
        name = row[2]
        ccsr_index[ccsr] = (index, name)

    print('len(ccsr_index):', len(ccsr_index))
    output_file = r'../data/mapping/ccsr_index_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(ccsr_index, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    icd_ccsr = {}
    for index, row in df.iterrows():
        icd = row[0].strip(r"'").strip(r'"').strip()
        icd_name = row[1]
        ccsr = row[6].strip(r"'").strip(r'"').strip()
        ccsr_name = row[7]
        icd_ccsr[icd] = [ccsr, ccsr_name, icd, icd_name]

    print('len(icd_ccsr):', len(icd_ccsr))
    output_file = r'../data/mapping/icd_ccsr_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(icd_ccsr, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return icd_ccsr, ccsr_index,  df


if __name__ == '__main__':
    # python pre_codemapping.py 2>&1 | tee  log/pre_codemapping_zip_adi.txt
    start_time = time.time()
    # rxnorm_atcset, atc_rxnormset, atc3_index, df_rxrnom_atc = rxnorm_atc_from_NIH_UMLS()
    # rx_ing, df_rx_ing = rxnorm_ingredient_from_NIH_UMLS()
    rx_ing_api, df_rx_ing_api = add_rxnorm_ingredient_by_umls_api()
    # zip_adi, zip5_df = zip_aid_mapping()
    # icd_ccsr, ccsr_index, ccsr_df = ICD10_to_CCSR()
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
