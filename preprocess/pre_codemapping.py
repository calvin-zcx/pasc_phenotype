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

print = functools.partial(print, flush=True)
import time


def build_rxnorm_or_atc_to_name():
    print('In build_rxnorm_or_atc_to_name()...')
    start_time = time.time()
    node_df = pd.read_csv(r'../data/mapping/RXNCONSO.RRF', sep='|', header=None, dtype=str)

    print('node_df.shape:', node_df.shape)  # node_df.shape: (1101174, 19)

    rx_name_df = node_df.loc[(node_df[11] == 'RXNORM'), [0, 14]]
    rx_name = defaultdict(list)
    for index, row in rx_name_df.iterrows():
        rx_name[row[0]].append(row[14])

    print('rxnorm to name: len(rx_name):', len(rx_name))
    output_file = r'../data/mapping/rxnorm_name.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(rx_name, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    atc_df = node_df.loc[node_df[11] == 'ATC', [13, 14]]
    atc_name = defaultdict(list)
    for index, row in atc_df.iterrows():
        atc_name[row[13]].append(row[14])

    print('atc to name: len(rx_name):', len(rx_name))
    output_file = r'../data/mapping/atc_name.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(atc_name, open(output_file, 'wb'))

    print('dump done to {}'.format(output_file))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return rx_name, atc_name


def build_NDC_to_rxnorm():
    # Data source: RxNorm_full_01032022.zip, RxNorm_full_01032022\rrf\RXNSAT.RRF
    # MD5 checksum: 8c8c0267fb2e09232fb852f7500c5b18, Release Notes 01/03/2022
    print('In build_NDC_to_rxnorm()...')
    start_time = time.time()
    df = pd.read_csv(r'../data/mapping/RXNSAT.RRF', sep='|', header=None, dtype=str)
    df_ndc = df.loc[df[8] == 'NDC', :]
    print('df.shape:', df.shape)
    print('df_ndc.shape:', df_ndc.shape)

    ndc_rx = defaultdict(set)
    for index, row in tqdm(df_ndc.iterrows(), total=len(df_ndc)):
        rxnorm = row[0]
        ndc = row[10]
        ndc_normalized = utils.ndc_normalization(ndc)
        ndc_rx[ndc].add(rxnorm)
        if ndc_normalized:
            ndc_rx[ndc_normalized].add(rxnorm)

    print('ndc to rxnorm mapping: len(ndc_rx):', len(ndc_rx))
    output_file = r'../data/mapping/ndc_rxnorm_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(ndc_rx, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return ndc_rx


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
    print('In rxnorm_ingredient_from_NIH_UMLS()...')
    start_time = time.time()
    link_df = pd.read_csv(r'../data/mapping/RXNREL.RRF', sep='|', header=None, dtype=str)
    node_df = pd.read_csv(r'../data/mapping/RXNCONSO.RRF', sep='|', header=None, dtype=str)

    print('link_df.shape:', link_df.shape)  # (7204009, 17)
    print('node_df.shape:', node_df.shape)  # node_df.shape: (1101174, 19)
    link_cui_df = link_df.loc[(link_df[2] == 'CUI') & (link_df[6] == 'CUI'), :]
    print('all cui-cui relations, link_cui_df.shape:', link_cui_df.shape)  # (7204009, 17)

    rx_name_df = node_df.loc[(node_df[11] == 'RXNORM'), [0, 14]]
    rx_name = {row[0]: row[14] for index, row in rx_name_df.iterrows()}

    IN_and_name = node_df.loc[(node_df[11] == 'RXNORM') & (node_df[12] == 'IN'), [0, 14, 12]]
    MIN_and_name = node_df.loc[(node_df[11] == 'RXNORM') & (node_df[12] == 'MIN'), [0, 14, 12]]
    PIN_and_name = node_df.loc[(node_df[11] == 'RXNORM') & (node_df[12] == 'PIN'), [0, 14, 12]]
    print('IN_and_name.shape:', IN_and_name.shape)  # (13569, 3)
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
    # Notice: Here we use Active_ingredient_RxCUI
    # r = requests.get('https://rxnav.nlm.nih.gov/REST/rxcui/{}/property.json?propName=Active_ingredient_RxCUI'.format(rxcui))
    # change at 2022-02-28
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
    # if api exists, use api, then use my derived dictionary
    print('In add_rxnorm_ingredient_by_umls_api()...')
    start_time = time.time()
    with open(r'../data/mapping/rxnorm_ingredient_mapping.pkl', 'rb') as f:
        rxnorm_ing = pickle.load(f)
        print('Load rxRNOM_CUI to ingredient mapping done! len(rxnorm_ing):', len(rxnorm_ing))
        record_example = next(iter(rxnorm_ing.items()))
        print('e.g.:', record_example)

    node_df = pd.read_csv(r'../data/mapping/RXNCONSO.RRF', sep='|', header=None, dtype=str)
    print('node_df.shape:', node_df.shape)  # node_df.shape: (1101174, 19)

    node_cui_df = node_df.loc[(node_df[11] == 'RXNORM'), [0, 14]].drop_duplicates()
    rxnorm_name = {row[0]: row[14] for index, row in node_cui_df.iterrows()}
    rxnorm_set = set(node_df.loc[(node_df[11] == 'RXNORM'), 0])
    print('unique rxnorm codes number: ', len(rxnorm_set))

    rx_ing_api = defaultdict(set)
    n_no_return = 0
    n_has_return = 0
    i = 0
    for rx in rxnorm_set:
        i += 1
        ings = _parse_from_nih_rxnorm_api(rx)
        if ings:
            rx_ing_api[rx].update(ings)
            n_has_return += 1
        else:
            n_no_return += 1

        if rx in rxnorm_ing:
            print(i, rx, ':already found:', ';'.join(rxnorm_ing[rx]), 'vs new found:', ';'.join(sorted(ings)))

        if i % 10000 == 0:
            print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
            print('Total search:', len(rxnorm_set), 'search:', i, 'n_has_return:', n_has_return, 'n_no_return:',
                  n_no_return)

    print('Total search:', len(rxnorm_set), 'n_has_return:', n_has_return, 'n_no_return:', n_no_return)

    records = []
    for key, val in rx_ing_api.items():
        val = sorted(val)
        rx_ing_api[key] = val
        name = rxnorm_name[key]
        records.append((key, name, len(val), ';'.join(val)))

    records = sorted(records, key=lambda x: int(x[0]))
    df_rx_ing = pd.DataFrame(records, columns=['rxnorm_cui', 'name', 'num of ingredient(s)', 'ingredient(s)'])
    print('df_rx_ing.shape', df_rx_ing.shape)
    # df_rx_ing.to_csv(r'../data/mapping/rxnorm_ingredient_mapping_from_api.csv')
    df_rx_ing.to_csv(r'../data/mapping/rxnorm_ingredient_mapping_from_api_moiety.csv')

    print('rxnorm to active ingredient(s): len(rx_ing_api):', len(rx_ing_api))
    # output_file = r'../data/mapping/rxnorm_ingredient_mapping_from_api.pkl'
    output_file = r'../data/mapping/rxnorm_ingredient_mapping_from_api_moiety.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(rx_ing_api, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return rx_ing_api, df_rx_ing


def combine_rxnorm_ingredients_dicts():
    # compare, contrast, and update existing dictionary
    # if api exists, use api, then use my derived dictionary
    # sometimes, our files give better results than api

    print('In combine_rxnorm_ingredients_dicts()...')
    start_time = time.time()
    with open(r'../data/mapping/rxnorm_ingredient_mapping.pkl', 'rb') as f:
        rx_ing = pickle.load(f)
        print('Load rxRNOM_CUI to ingredient mapping generated from umls files done! len(rx_ing):', len(rx_ing))
        record_example = next(iter(rx_ing.items()))
        print('e.g.:', record_example)

    # rxnorm_ingredient_mapping_from_api_moiety
    with open(r'../data/mapping/rxnorm_ingredient_mapping_from_api_moiety.pkl',
              'rb') as f:  # with open(r'../data/mapping/rxnorm_ingredient_mapping_from_api.pkl', 'rb') as f:
        rx_ing_api = pickle.load(f)
        print('Load rxRNOM_CUI to ingredient mapping generated from API done! len(rx_ing_api):', len(rx_ing_api))
        record_example = next(iter(rx_ing_api.items()))
        print('e.g.:', record_example)

    with open(r'../data/mapping/rxnorm_atc_mapping.pkl', 'rb') as f:
        rxnorm_atc = pickle.load(f)
        print('Load rxRNOM_CUI to ATC mapping done! len(rxnorm_atc):', len(rxnorm_atc))
        record_example = next(iter(rxnorm_atc.items()))
        print('e.g.:', record_example)

    n_no_change = n_new_add = 0
    n_exist_but_different = 0
    i = 0
    # using UMLS file as default, api as aux, gave best coverage of atc
    # change order 20022-02-28
    default_rx_ing = rx_ing_api
    second_rx_ing = rx_ing

    n_default = len(default_rx_ing)
    for key, val in second_rx_ing.items():
        i += 1
        if key in default_rx_ing:  # then use default_rx_ing records
            n_no_change += 1
            val_default = set(default_rx_ing[key])
            if set(val) != val_default:
                n_exist_but_different += 1
                print(n_exist_but_different, key, 'api:', rx_ing_api[key], 'file:', rx_ing[key])
        else:  # add new records from our generated file
            n_new_add += 1
            default_rx_ing[key] = val

    print('Combine {} + {} into:'.format(n_default, len(second_rx_ing)), len(default_rx_ing),
          "n_no_change:", n_no_change, "n_new_add:", n_new_add, "n_exist_but_different:", n_exist_but_different)

    node_df = pd.read_csv(r'../data/mapping/RXNCONSO.RRF', sep='|', header=None, dtype=str)
    print('node_df.shape:', node_df.shape)  # node_df.shape: (1101174, 19)
    node_cui_df = node_df.loc[(node_df[11] == 'RXNORM'), [0, 14]].drop_duplicates()
    rxnorm_name = {row[0]: row[14] for index, row in node_cui_df.iterrows()}

    # sort records, and check atc coverage
    records = []
    n_ing = n_ing_has_atc = 0
    for key, val in default_rx_ing.items():
        val = sorted(val)
        default_rx_ing[key] = val
        name = rxnorm_name[key]
        records.append((key, name, len(val), ';'.join(val)))

        n_ing += len(val)
        for x in val:
            if x in rxnorm_atc:
                n_ing_has_atc += 1

    print('len(default_rx_ing):', len(default_rx_ing), 'n_ing:', n_ing, 'n_ing_has_atc:', n_ing_has_atc)

    records = sorted(records, key=lambda x: int(x[0]))
    df_records = pd.DataFrame(records, columns=['rxnorm_cui', 'name', 'num of ingredient(s)', 'ingredient(s)'])
    print('df_records.shape', df_records.shape)
    df_records.to_csv(r'../data/mapping/rxnorm_ingredient_mapping_combined_moietyfirst.csv')

    print('rxnorm to ingredients before add by hands: ', len(default_rx_ing))
    default_rx_ing = _add_rxnorm_ing_by_hands(default_rx_ing)
    print('rxnorm to ingredients after add by hands: ', len(default_rx_ing))

    output_file = r'../data/mapping/rxnorm_ingredient_mapping_combined_moietyfirst.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(default_rx_ing, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return default_rx_ing, df_records


def _add_rxnorm_ing_by_hands(rx_ing):
    rx_ing['1360201'] = ['435']  # albuterol 0.09 MG/ACTUAT Metered Dose Inhaler
    rx_ing['1086966'] = ['612', '6581']
    rx_ing['845494'] = ['8640']
    rx_ing['845492'] = ['8640']
    rx_ing['763179'] = ['8640']
    rx_ing['763181'] = ['8640']
    rx_ing['763185'] = ['8640']
    rx_ing['763183'] = ['8640']
    rx_ing['1659930'] = ['7242']
    rx_ing['1359855'] = ['274783']
    rx_ing['1658648'] = ['5224']
    rx_ing['1362058'] = ['5224']
    rx_ing['1362049'] = ['5224']
    rx_ing['1657985'] = ['5224']
    rx_ing['1362026'] = ['5224']
    rx_ing['848335'] = ['5224']

    rx_ing['1150984'] = ['18631']
    rx_ing['749783'] = ['18631']
    rx_ing['749780'] = ['18631']
    rx_ing['1656667'] = ['6585']
    rx_ing['834023'] = ['6902']
    rx_ing['762675'] = ['6902']
    rx_ing['1435176'] = ['161']

    rx_ing['1360330'] = ['8163']
    rx_ing['1242903'] = ['8163']

    rx_ing['1490492'] = ['5224']
    rx_ing['1001690'] = ['36721', '6585']

    rx_ing['1360509'] = ['19831', '25255', '389132']
    rx_ing['1360402'] = ['19831']

    rx_ing['1654007'] = ['4177']
    rx_ing['1360463'] = ['4177']

    rx_ing['748961'] = ['7514']
    rx_ing['1539955'] = ['7514']

    rx_ing['1665039'] = ['6754']
    rx_ing['1359948'] = ['6754']

    rx_ing['1359934'] = ['139825']

    rx_ing['1652647'] = ['86009']
    rx_ing['1652648'] = ['86009']
    rx_ing['314684'] = ['86009']

    rx_ing['1719670'] = ['5093']

    rx_ing['1359894'] = ['1223']

    rx_ing['1551307'] = ['1551291']
    rx_ing['1551301'] = ['1551291']

    rx_ing['1807460'] = ['3322']

    rx_ing['1098138'] = ['2473']
    rx_ing['1098123'] = ['2473']

    rx_ing['1360105'] = ['475968']

    rx_ing['1360180'] = ['7052']
    rx_ing['2003715'] = ['7052']

    rx_ing['1657069'] = ['253337']

    rx_ing['670078'] = ['197']

    rx_ing['1148107'] = ['18631']

    rx_ing['1242906'] = ['8163']

    rx_ing['866824'] = ['7512']

    rx_ing['1540004'] = ['3289', '5032']

    rx_ing['1145689'] = ['9796']

    rx_ing['1115906'] = ['3966']
    rx_ing['1115908'] = ['3966']

    rx_ing['312191'] = ['7824']

    rx_ing['211870'] = ['1151']

    rx_ing['308056'] = ['8410']

    rx_ing['577455'] = ['4337']
    rx_ing['283458'] = ['4337']

    rx_ing['1242905'] = ['8163']

    rx_ing['1245693'] = ['3992']

    rx_ing['1921466'] = ['327361']
    rx_ing['825169'] = ['327361']
    rx_ing['1656703'] = ['327361']
    rx_ing['1872983'] = ['327361']
    rx_ing['1855526'] = ['327361']
    rx_ing['1921467'] = ['327361']
    rx_ing['1873086'] = ['327361']
    rx_ing['1921468'] = ['327361']
    rx_ing['1855527'] = ['327361']
    rx_ing['1921244'] = ['327361']
    rx_ing['1921245'] = ['327361']
    rx_ing['1594358'] = ['327361']

    rx_ing['1488053'] = ['87636']

    rx_ing['1012661'] = ['1815']
    rx_ing['1724785'] = ['1815']

    rx_ing['1429987'] = ['5640']

    rx_ing['2119388'] = ['6130']
    rx_ing['1292462'] = ['6130']

    rx_ing['1360002'] = ['3423']
    rx_ing['1359758'] = ['3423']

    rx_ing['208561'] = ['10600']
    rx_ing['1923430'] = ['10600']
    rx_ing['1923431'] = ['10600']

    return rx_ing


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
            print(index, input_file, 'df.shape:', df.shape, 'n_records_adi:', n_records_adi, 'n_records_wcm:',
                  n_records_wcm)
            if df.shape[0] != n_records_adi:
                print('ERROR in ', input_file, 'df.shape[0] != n_records_adi')
            df['nation_rank'] = pd.to_numeric(df['ADI_NATRANK'], errors='coerce')
            df['state_rank'] = pd.to_numeric(df['ADI_STATERNK'], errors='coerce')
            df['zip5'] = df["ZIPID"].apply(lambda x: x[:6] if pd.notna(x) else np.nan)
            zip5_scores = df.groupby(["zip5"])[['nation_rank', "state_rank"]].median().reset_index()
            print('......zip5_scores.shape:', zip5_scores.shape)

            zip9_list = df[['ZIPID', 'nation_rank', 'state_rank']].values.tolist()
            zip5_list = zip5_scores[['zip5', 'nation_rank', 'state_rank']].values.tolist()
            # save zip5 for debugging
            zip5_df.append(zip5_scores[['zip5', 'nation_rank', 'state_rank']])
            # n_null_zip9 = n_null_zip5 = 0
            zip_adi.update({x[0][1:]: x[1:] for x in zip9_list if pd.notna(x[0])})
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


def zip_adi_mapping_2020():
    # To get code mapping from rxnorm_cui to ATC.
    # Data source: https://www.neighborhoodatlas.medicine.wisc.edu/download
    # 52 states files
    # details in 2019 ADI_9 Digit Zip Code_v3.1_ReadMe
    start_time = time.time()
    # readme_df = pd.read_csv(r'../data/mapping/ADI/2019_ADI_9_V3.1_readme_statistics_check.csv')
    readme_df = pd.read_csv(r'../data/mapping/ADI/2020/2020 ADI_9 Digit Zip Code_v3.2_ReadMe.csv')

    print('readme_df.shape:', readme_df.shape)
    ## df2 = pd.read_csv(r'../data/mapping/ADI/wcm_zip_state.csv')
    ## readme_df['State_abr'] = readme_df['State_abr'].apply(str.upper)
    ## df_combined = pd.merge(readme_df, df2, left_on='State_abr', right_on='address_state', how='left')
    ## df_combined.to_csv(r'../data/mapping/ADI/2019_ADI_9_V3.1_readme_statistics_check_v2.csv')
    zip_adi = {}
    zip5_df = []
    for index, row in tqdm(readme_df.iterrows(), total=len(readme_df)):
        state = row['State_abr']  # row[1]
        name = row['State_full']  # row[2]
        n_records_adi = row['#_Records']  # row[3]
        # n_records_wcm = row[4]
        # input_file = r'../data/mapping/ADI/{}_2019_ADI_9 Digit Zip Code_v3.1.txt'.format(state)
        input_file = r'../data/mapping/ADI/2020/{}_2020_ADI_9 Digit Zip Code_v3.2.csv'.format(state.upper())

        # specified_dtype = {'Unnamed: 0': int, 'X': int, 'TYPE': str, 'ZIPID': str, 'FIPS.x': str,
        #               'GISJOIN': str, 'FIPS.y': str, 'ADI_NATRANK': int, 'ADI_STATERNK': int}
        if os.path.exists(input_file):
            df = pd.read_csv(input_file, dtype=str)
            print(index, input_file, 'df.shape:', df.shape, 'n_records_adi:',
                  n_records_adi, )  # 'n_records_wcm:', n_records_wcm)
            if df.shape[0] != n_records_adi:
                print('ERROR in ', input_file, 'df.shape[0] {} != n_records_adi {}'.format(df.shape[0], n_records_adi))
            df['nation_rank'] = pd.to_numeric(df['ADI_NATRANK'], errors='coerce')
            df['state_rank'] = pd.to_numeric(df['ADI_STATERANK'], errors='coerce')
            df['zip5'] = df["ZIP_4"].apply(lambda x: x[:5] if pd.notna(
                x) else np.nan)  # ZIPID --> ZIP_4  :6 -->:5 because preivous version starts with G
            zip5_scores = df.groupby(["zip5"])[['nation_rank', "state_rank"]].median().reset_index()
            print('......zip5_scores.shape:', zip5_scores.shape)

            zip9_list = df[['ZIP_4', 'nation_rank', 'state_rank']].values.tolist()  # ZIPID --> ZIP_4
            zip5_list = zip5_scores[['zip5', 'nation_rank', 'state_rank']].values.tolist()
            # save zip5 for debugging
            zip5_df.append(zip5_scores[['zip5', 'nation_rank', 'state_rank']])
            # n_null_zip9 = n_null_zip5 = 0
            # zip_adi.update({x[0][1:]: x[1:] for x in zip9_list if pd.notna(x[0])})
            # print('......len(zip_adi) after adding zip9:', len(zip_adi))
            # zip_adi.update({x[0][1:]: x[1:] for x in zip5_list if pd.notna(x[0])})
            zip_adi.update({x[0]: x[1:] for x in zip9_list if pd.notna(x[0])})
            print('......len(zip_adi) after adding zip9:', len(zip_adi))
            zip_adi.update({x[0]: x[1:] for x in zip5_list if pd.notna(x[0])})
            print('......len(zip_adi) after adding zip5:', len(zip_adi))
            print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
        else:
            print(index, input_file, 'NOT EXIST!')

    output_file = r'../data/mapping/zip9or5_adi_mapping_2020.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(zip_adi, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    # double check zip5
    zip5_df = pd.concat(zip5_df)
    print('zip5_df.shape', zip5_df.shape)
    zip5_df.to_csv(r'../data/mapping/zip5_for_debug_2020.csv')
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return zip_adi, zip5_df


def zip_fips_adi_mapping_2020():
    # To get code mapping from rxnorm_cui to ATC.
    # Data source: https://www.neighborhoodatlas.medicine.wisc.edu/download
    # 52 states files
    # details in 2019 ADI_9 Digit Zip Code_v3.1_ReadMe
    start_time = time.time()
    # readme_df = pd.read_csv(r'../data/mapping/ADI/2019_ADI_9_V3.1_readme_statistics_check.csv')
    readme_df = pd.read_csv(r'../data/mapping/ADI/2020/2020 ADI_9 Digit Zip Code_v3.2_ReadMe.csv')

    print('readme_df.shape:', readme_df.shape)
    ## df2 = pd.read_csv(r'../data/mapping/ADI/wcm_zip_state.csv')
    ## readme_df['State_abr'] = readme_df['State_abr'].apply(str.upper)
    ## df_combined = pd.merge(readme_df, df2, left_on='State_abr', right_on='address_state', how='left')
    ## df_combined.to_csv(r'../data/mapping/ADI/2019_ADI_9_V3.1_readme_statistics_check_v2.csv')
    zip_adi = {}
    zip5_df = []
    fips_adi = {}
    fips_zip = defaultdict(list)
    for index, row in tqdm(readme_df.iterrows(), total=len(readme_df)):
        state = row['State_abr']  # row[1]
        name = row['State_full']  # row[2]
        n_records_adi = row['#_Records']  # row[3]
        # n_records_wcm = row[4]
        # input_file = r'../data/mapping/ADI/{}_2019_ADI_9 Digit Zip Code_v3.1.txt'.format(state)
        input_file = r'../data/mapping/ADI/2020/{}_2020_ADI_9 Digit Zip Code_v3.2.csv'.format(state.upper())

        # specified_dtype = {'Unnamed: 0': int, 'X': int, 'TYPE': str, 'ZIPID': str, 'FIPS.x': str,
        #               'GISJOIN': str, 'FIPS.y': str, 'ADI_NATRANK': int, 'ADI_STATERNK': int}
        if os.path.exists(input_file):
            df = pd.read_csv(input_file, dtype=str)
            print(index, input_file, 'df.shape:', df.shape, 'n_records_adi:',
                  n_records_adi, )  # 'n_records_wcm:', n_records_wcm)
            if df.shape[0] != n_records_adi:
                print('ERROR in ', input_file, 'df.shape[0] {} != n_records_adi {}'.format(df.shape[0], n_records_adi))
            df['nation_rank'] = pd.to_numeric(df['ADI_NATRANK'], errors='coerce')
            df['state_rank'] = pd.to_numeric(df['ADI_STATERANK'], errors='coerce')
            df['zip5'] = df["ZIP_4"].apply(lambda x: x[:5] if pd.notna(
                x) else np.nan)  # ZIPID --> ZIP_4  :6 -->:5 because preivous version starts with G
            zip5_scores = df.groupby(["zip5"])[['nation_rank', "state_rank"]].median().reset_index()
            print('......zip5_scores.shape:', zip5_scores.shape)

            zip9_list = df[['ZIP_4', 'nation_rank', 'state_rank']].values.tolist()  # ZIPID --> ZIP_4
            zip5_list = zip5_scores[['zip5', 'nation_rank', 'state_rank']].values.tolist()
            # save zip5 for debugging
            zip5_df.append(zip5_scores[['zip5', 'nation_rank', 'state_rank']])

            fips_list = df[['FIPS', 'nation_rank', 'state_rank']].values.tolist()
            # n_null_zip9 = n_null_zip5 = 0
            # zip_adi.update({x[0][1:]: x[1:] for x in zip9_list if pd.notna(x[0])})
            # print('......len(zip_adi) after adding zip9:', len(zip_adi))
            # zip_adi.update({x[0][1:]: x[1:] for x in zip5_list if pd.notna(x[0])})
            zip_adi.update({x[0]: x[1:] for x in zip9_list if pd.notna(x[0])})
            print('......len(zip_adi) after adding zip9:', len(zip_adi))
            zip_adi.update({x[0]: x[1:] for x in zip5_list if pd.notna(x[0])})
            print('......len(zip_adi) after adding zip5:', len(zip_adi))
            fips_adi.update({x[0]: x[1:] for x in fips_list if pd.notna(x[0])})
            print('......len(fips_adi)  fips:', len(fips_adi))

            zip_fips_list = df[['ZIP_4', 'FIPS']].values.tolist()
            for rec in zip_fips_list:
                zip9 = rec[0]
                fips = rec[1]
                if pd.notna(fips) and pd.notna(zip9):
                    fips_zip[fips].append(zip9)

            print('......len(fips_zip) :', len(fips_zip))
            print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

        else:
            print(index, input_file, 'NOT EXIST!')

    output_file = r'../data/mapping/zip9or5_adi_mapping_2020.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(zip_adi, open(output_file, 'wb'))
    print('dump zip_adi done to {}'.format(output_file))

    output_file2 = r'../data/mapping/fips_adi_mapping_2020.pkl'
    utils.check_and_mkdir(output_file2)
    pickle.dump(fips_adi, open(output_file2, 'wb'))
    print('dump fips_adi done to {}'.format(output_file2))

    output_file3 = r'../data/mapping/fips_to_ziplist_2020.pkl'
    utils.check_and_mkdir(output_file3)
    pickle.dump(fips_zip, open(output_file3, 'wb'))
    print('dump fips_adi done to {}'.format(output_file2))

    # double check zip5
    zip5_df = pd.concat(zip5_df)
    print('zip5_df.shape', zip5_df.shape)
    zip5_df.to_csv(r'../data/mapping/zip5_for_debug_2020.csv')
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return zip_adi, fips_adi, fips_zip, zip5_df


def zip_adi_mapping_2021():
    # To get code mapping from rxnorm_cui to ATC.
    # Data source: https://www.neighborhoodatlas.medicine.wisc.edu/download
    # 52 states files
    # details in 2021 ADI_9 Digit Zip Code_v4.0_ReadMe.txt
    start_time = time.time()
    # readme_df = pd.read_csv(r'../data/mapping/ADI/2019_ADI_9_V3.1_readme_statistics_check.csv')
    # readme_df = pd.read_csv(r'../data/mapping/ADI/2020/2020 ADI_9 Digit Zip Code_v3.2_ReadMe.csv')
    readme_df = pd.read_csv(r'../data/mapping/ADI/2021/2021 ADI_9 Digit Zip Code_v4.0_ReadMe_edited.csv')

    print('readme_df.shape:', readme_df.shape)
    ## df2 = pd.read_csv(r'../data/mapping/ADI/wcm_zip_state.csv')
    ## readme_df['State_abr'] = readme_df['State_abr'].apply(str.upper)
    ## df_combined = pd.merge(readme_df, df2, left_on='State_abr', right_on='address_state', how='left')
    ## df_combined.to_csv(r'../data/mapping/ADI/2019_ADI_9_V3.1_readme_statistics_check_v2.csv')
    zip_adi = {}
    zip5_df = []
    for index, row in tqdm(readme_df.iterrows(), total=len(readme_df)):
        state = row['Abbr']
        name = row['Full']
        n_records_adi = int(row['Freq'].replace(',', ''))
        # n_records_wcm = row[4]
        # input_file = r'../data/mapping/ADI/{}_2019_ADI_9 Digit Zip Code_v3.1.txt'.format(state)
        input_file = r'../data/mapping/ADI/2021/{}_2021_ADI_9 Digit Zip Code_v4.csv'.format(state.upper())

        # specified_dtype = {'Unnamed: 0': int, 'X': int, 'TYPE': str, 'ZIPID': str, 'FIPS.x': str,
        #               'GISJOIN': str, 'FIPS.y': str, 'ADI_NATRANK': int, 'ADI_STATERNK': int}
        if os.path.exists(input_file):
            df = pd.read_csv(input_file, dtype=str)
            print(index, input_file, 'df.shape:', df.shape, 'n_records_adi:',
                  n_records_adi, )  # 'n_records_wcm:', n_records_wcm)
            if df.shape[0] != n_records_adi:
                print('ERROR in ', input_file, 'df.shape[0] {} != n_records_adi {}'.format(df.shape[0], n_records_adi))
            df['nation_rank'] = pd.to_numeric(df['ADI_NATRANK'], errors='coerce')
            df['state_rank'] = pd.to_numeric(df['ADI_STATERNK'], errors='coerce')
            df['zip5'] = df["BENE_ZIP_CD"].apply(lambda x: x[:5] if pd.notna(
                x) else np.nan)  # ZIPID --> ZIP_4  :6 -->:5 because preivous version starts with G
            zip5_scores = df.groupby(["zip5"])[['nation_rank', "state_rank"]].median().reset_index()
            print('......zip5_scores.shape:', zip5_scores.shape)

            zip9_list = df[['BENE_ZIP_CD', 'nation_rank', 'state_rank']].values.tolist()  # ZIPID --> ZIP_4
            zip5_list = zip5_scores[['zip5', 'nation_rank', 'state_rank']].values.tolist()
            # save zip5 for debugging
            zip5_df.append(zip5_scores[['zip5', 'nation_rank', 'state_rank']])
            # n_null_zip9 = n_null_zip5 = 0
            # zip_adi.update({x[0][1:]: x[1:] for x in zip9_list if pd.notna(x[0])})
            # print('......len(zip_adi) after adding zip9:', len(zip_adi))
            # zip_adi.update({x[0][1:]: x[1:] for x in zip5_list if pd.notna(x[0])})
            zip_adi.update({x[0]: x[1:] for x in zip9_list if pd.notna(x[0])})
            print('......len(zip_adi) after adding zip9:', len(zip_adi))
            zip_adi.update({x[0]: x[1:] for x in zip5_list if pd.notna(x[0])})
            print('......len(zip_adi) after adding zip5:', len(zip_adi))
            print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
        else:
            print(index, input_file, 'NOT EXIST!')

    output_file = r'../data/mapping/zip9or5_adi_mapping_2021.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(zip_adi, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    # double check zip5
    zip5_df = pd.concat(zip5_df)
    print('zip5_df.shape', zip5_df.shape)
    zip5_df.to_csv(r'../data/mapping/zip5_for_debug_2021.csv')
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return zip_adi, zip5_df


def selected_rxnorm_ingredient_to_index():
    start_time = time.time()
    rx_df = pd.read_csv(r'../data/mapping/info_medication_cohorts_covid_4manuNegNoCovid_ALL_enriched.csv',
                        dtype={'rxnorm': str})
    # ['rxnorm', 'total', 'no. in positive group', 'no. in negative group',
    #        'ratio', 'name', 'atc-l3', 'atc-l4']
    rx_df = rx_df.sort_values(by='ratio', ascending=False)
    rx_df = rx_df.loc[rx_df.loc[:, 'no. in positive group'] >= 100, :]
    print('rx_df.shape:', rx_df.shape)

    rxnorm_index = {}
    for i, (index, row) in enumerate(rx_df.iterrows()):
        rx = row[0].strip()
        rxnorm_index[rx] = [i, ] + row.tolist()

    print('len(rxnorm_index):', len(rxnorm_index))
    utils.dump(rxnorm_index, r'../data/mapping/selected_rxnorm_index.pkl')
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return rxnorm_index


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
    return icd_ccsr, ccsr_index, df


def ICD_to_elixhauser_comorbidity():
    # To get code mapping from icd10 to elixhauser_comorbidity.
    # Data source: https://www.hcup-us.ahrq.gov/toolssoftware/comorbidityicd10/comorbidity_icd10.jsp#down
    # Elixhauser Comorbidity Software Refined Tool, v2022.1 (ZIP file, 1.5 MB) released 10/29/21

    start_time = time.time()
    # df_dimension = pd.read_excel(r'../data/mapping/CMR_v2022-1/CMR-Reference-File-v2022-1.xlsx',
    #                              sheet_name='Comorbidity_Measures', skiprows=[0])
    df_dimension = pd.read_csv(r'../data/mapping/CMR_v2022-1/_my_comorbidity_index.csv', dtype=str)
    print('df_dimension.shape:', df_dimension.shape)
    df = pd.read_excel(r'../data/mapping/CMR_v2022-1/CMR-Reference-File-v2022-1.xlsx',
                       sheet_name='DX_to_Comorb_Mapping', skiprows=[0])
    df = df.drop(['Unnamed: 42', 'Unnamed: 43'], axis=1)
    print('df.shape:', df.shape)
    df = df.set_index('ICD-10-CM Diagnosis')
    df_array = df.iloc[:, 2:]

    rows, cols = np.where(df_array > 0)
    rows_name = df_array.index[rows]
    cols_name = df_array.columns[cols]
    icd_cmr = defaultdict(list)
    for i, j in zip(rows_name, cols_name):
        icd_cmr[i].append(j)

    df_dimension = df_dimension.reset_index()
    cmr_index = {}
    for index, row in df_dimension.iterrows():
        cmr = row['col_name']
        name = row['Comorbidity Description']
        cmr_index[cmr] = (index, name)

    print('len(icd_cmr):', len(icd_cmr))
    output_file = r'../data/mapping/icd_cmr_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(icd_cmr, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('len(cmr_index):', len(cmr_index))
    output_file = r'../data/mapping/cmr_index_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(cmr_index, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return icd_cmr, cmr_index, df


def ICD_to_negative_control_pasc():
    # To get code mapping from icd10 to PASC our compiled list.
    # Data source: ../data/mapping/PASC_Adult_Combined_List_20220127_v3.xlsx May be updated later

    start_time = time.time()
    pasc_list_file = r'../data/mapping/PASC_Adult_Combined_List_submit.xlsx'
    df_pasc_list = pd.read_excel(pasc_list_file, sheet_name=r'code_list', usecols="A:N")
    print('df_pasc_list.shape', df_pasc_list.shape)
    df_pasc_list['ICD-10-CM Code'] = df_pasc_list['ICD-10-CM Code'].apply(lambda x: x.strip().upper().replace('.', ''))
    pasc_codes = df_pasc_list['ICD-10-CM Code']  # .str.upper().replace('.', '', regex=False)  # .to_list()
    pasc_codes_set = set(pasc_codes)
    print('Load compiled pasc list done from {}\nlen(pasc_codes)'.format(pasc_list_file),
          len(pasc_codes), 'len(pasc_codes_set):', len(pasc_codes_set))

    icd_pasc = {}
    pasc_index = {}

    def select_negative_control_ccsr_category(x):
        if pd.notna(x) and ((x in ['FAC003', 'FAC006', 'FAC008']) or x.startswith('NEO') or x.startswith('EXT')):
            return True
        else:
            return False

    df_pasc_list_neg = df_pasc_list.loc[
                       df_pasc_list['CCSR CATEGORY 1'].apply(lambda x: select_negative_control_ccsr_category(x)), :]

    for index, row in df_pasc_list_neg.iterrows():
        hd_domain = row['HD Domain (Defined by Nature paper)']
        ccsr_code = row['CCSR CATEGORY 1']
        ccsr_category = row['CCSR CATEGORY 1 DESCRIPTION']
        icd = row['ICD-10-CM Code']
        icd_name = row['ICD-10-CM Code Description']
        icd_pasc[icd] = [ccsr_code + '=' + ccsr_category, ccsr_code, hd_domain, icd_name]

    df_dim = df_pasc_list_neg[['CCSR CATEGORY 1 DESCRIPTION', 'CCSR CATEGORY 1']].value_counts().reset_index()
    df_dim = df_dim.sort_values(by='CCSR CATEGORY 1').reset_index()  # , ascending=False)
    for index, row in df_dim.iterrows():
        ccsr_category = row['CCSR CATEGORY 1 DESCRIPTION']
        cnt = row[0]
        codes = row[
            'CCSR CATEGORY 1']  # set(df_pasc_list_neg.loc[df_pasc_list_neg['CCSR CATEGORY 1 DESCRIPTION']==ccsr_category, 'CCSR CATEGORY 1'])
        pasc_index[codes + '=' + ccsr_category] = [index, cnt, codes]

    print('len(icd_pasc):', len(icd_pasc))
    output_file = r'../data/mapping/icd_negative-outcome-control_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(icd_pasc, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('len(pasc_index):', len(pasc_index))
    output_file = r'../data/mapping/negative-outcome-control_index_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(pasc_index, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return icd_pasc, pasc_index, df_pasc_list


def ICD_to_PASC():
    # To get code mapping from icd10 to PASC our compiled list.
    # Data source: ../data/mapping/PASC_Adult_Combined_List_20220127_v3.xlsx May be updated later

    start_time = time.time()
    pasc_list_file = r'../data/mapping/PASC_Adult_Combined_List_20220127_v3.xlsx'
    df_pasc_list = pd.read_excel(pasc_list_file, sheet_name=r'PASC Screening List', usecols="A:N")
    print('df_pasc_list.shape', df_pasc_list.shape)
    df_pasc_list['ICD-10-CM Code'] = df_pasc_list['ICD-10-CM Code'].apply(lambda x: x.strip().upper().replace('.', ''))
    pasc_codes = df_pasc_list['ICD-10-CM Code']  # .str.upper().replace('.', '', regex=False)  # .to_list()
    pasc_codes_set = set(pasc_codes)
    print('Load compiled pasc list done from {}\nlen(pasc_codes)'.format(pasc_list_file),
          len(pasc_codes), 'len(pasc_codes_set):', len(pasc_codes_set))

    icd_pasc = {}
    pasc_index = {}

    for index, row in df_pasc_list.iterrows():
        hd_domain = row['HD Domain (Defined by Nature paper)']
        ccsr_code = row['CCSR CATEGORY 1']
        ccsr_category = row['CCSR CATEGORY 1 DESCRIPTION']
        icd = row['ICD-10-CM Code']
        icd_name = row['ICD-10-CM Code Description']
        icd_pasc[icd] = [ccsr_category, ccsr_code, hd_domain, icd_name]

    df_dim = df_pasc_list['CCSR CATEGORY 1 DESCRIPTION'].value_counts().reset_index()
    for index, row in df_dim.iterrows():
        ccsr_category = row[0]
        cnt = row[1]
        pasc_index[ccsr_category] = [index, cnt]

    print('len(pasc_index):', len(pasc_index))
    output_file = r'../data/mapping/icd_pasc_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(icd_pasc, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('len(pasc_index):', len(pasc_index))
    output_file = r'../data/mapping/pasc_index_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(pasc_index, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return icd_pasc, pasc_index, df_pasc_list


def ICD_to_PASC_added_extension():
    # 2023-11-9
    # allow self-defined categories added
    # To get code mapping from icd10 to PASC our compiled list.
    # Data source: ../data/mapping/PASC_Adult_Combined_List_20220127_v3.xlsx May be updated later

    start_time = time.time()
    pasc_list_file = r'../data/mapping/PASC_Adult_Combined_List_added_extension.xlsx'
    df_pasc_list = pd.read_excel(pasc_list_file, sheet_name=r'Sheet1')  # , usecols="A:N")
    print('df_pasc_list.shape', df_pasc_list.shape)
    df_pasc_list['ICD-10-CM Code'] = df_pasc_list['ICD-10-CM Code'].apply(lambda x: x.strip().upper().replace('.', ''))
    pasc_codes = df_pasc_list['ICD-10-CM Code']  # .str.upper().replace('.', '', regex=False)  # .to_list()
    pasc_codes_set = set(pasc_codes)
    print('Load compiled pasc list done from {}\nlen(pasc_codes)'.format(pasc_list_file),
          len(pasc_codes), 'len(pasc_codes_set):', len(pasc_codes_set))

    icd_pasc = {}
    pasc_index = {}

    for index, row in df_pasc_list.iterrows():
        hd_domain = row['HD Domain (Defined by Nature paper)']
        ccsr_code = row['CCSR CATEGORY 1']
        ccsr_category = row['self selected category']  # row['CCSR CATEGORY 1 DESCRIPTION']
        icd = row['ICD-10-CM Code']
        icd_name = row['ICD-10-CM Code Description']
        icd_pasc[icd] = [ccsr_category, ccsr_code, hd_domain, icd_name]

    # df_dim = df_pasc_list['CCSR CATEGORY 1 DESCRIPTION'].value_counts().reset_index()
    df_dim = df_pasc_list['self selected category'].value_counts(sort=False).reset_index()
    for index, row in df_dim.iterrows():
        ccsr_category = row[0]
        cnt = row[1]
        pasc_index[ccsr_category] = [index, cnt]

    print('len(pasc_index):', len(pasc_index))
    output_file = r'../data/mapping/icd_addedPASC_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(icd_pasc, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('len(pasc_index):', len(pasc_index))
    output_file = r'../data/mapping/addedPASC_index_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(pasc_index, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return icd_pasc, pasc_index, df_pasc_list


def ICD_to_PASC_ME_CFS():
    # 2023-12-14 add ME_CFS
    # allow self-defined categories added
    # To get code mapping from icd10 to PASC our compiled list.

    start_time = time.time()
    pasc_list_file = r'../data/mapping/PASC_extension_ME_CFS.xlsx'
    df_pasc_list_all = pd.read_excel(pasc_list_file, sheet_name=r'Sheet1')  # , usecols="A:N")
    print('df_pasc_list_all.shape', df_pasc_list_all.shape)
    df_pasc_list = df_pasc_list_all.loc[df_pasc_list_all['include']=='yes', :]
    print('after selet include==yes, df_pasc_list.shape', df_pasc_list.shape)

    df_pasc_list['ICD-10-CM Code'] = df_pasc_list['ICD-10-CM Code'].apply(lambda x: x.strip().upper().replace('.', ''))
    pasc_codes = df_pasc_list['ICD-10-CM Code']  # .str.upper().replace('.', '', regex=False)  # .to_list()
    pasc_codes_set = set(pasc_codes)
    print('Load compiled pasc list done from {}\nlen(pasc_codes)'.format(pasc_list_file),
          len(pasc_codes), 'len(pasc_codes_set):', len(pasc_codes_set))

    icd_pasc = {}
    pasc_index = {}

    for index, row in df_pasc_list.iterrows():
        hd_domain = row['HD Domain']
        ccsr_code = row['CCSR CATEGORY 1']
        ccsr_category = row['self selected category']  # row['CCSR CATEGORY 1 DESCRIPTION']
        icd = row['ICD-10-CM Code']
        icd_name = row['ICD-10-CM Code Description']
        icd_pasc[icd] = [ccsr_category, ccsr_code, hd_domain, icd_name]

    # df_dim = df_pasc_list['CCSR CATEGORY 1 DESCRIPTION'].value_counts().reset_index()
    df_dim = df_pasc_list['self selected category'].value_counts(sort=False).reset_index()
    for index, row in df_dim.iterrows():
        ccsr_category = row[0]
        cnt = row[1]
        pasc_index[ccsr_category] = [index, cnt]

    print('len(pasc_index):', len(pasc_index))
    output_file = r'../data/mapping/icd_mecfs_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(icd_pasc, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('len(pasc_index):', len(pasc_index))
    output_file = r'../data/mapping/mecfs_index_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(pasc_index, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return icd_pasc, pasc_index, df_pasc_list


def ICD_to_PASC_brainfog():
    start_time = time.time()
    dict_df_cci = pd.read_excel(r'../data/mapping/RECOVER Brain Fog Code Lists 11.04.2022-category_namerevised.xlsx',
                                dtype=str,
                                sheet_name=None)
    print('len(dict_df_cci)', len(dict_df_cci))

    icd_cci = {}
    cci_index = {}

    for ith, (key, df_cci) in enumerate(dict_df_cci.items()):
        category = df_cci['Category'].apply(lambda x: x.strip()).unique()
        print(len(category), category)
        assert len(category) == 1
        category = category[0]
        cci_index[category] = [ith, len(df_cci)]
        print('category:', category, 'index:', ith, '#code:', len(df_cci), )
        for index, row in df_cci.iterrows():
            icd = row['DX CODE'].strip().upper().replace('.', '')
            name = row['DX Description']
            type = 'icd10'  # all icd10 in this spreadsheet #row['code type']
            # category = row['Category'].strip()
            icd_cci[icd] = [category, type, name]

    print('len(icd_cci):', len(icd_cci))
    output_file = r'../data/mapping/icd_brainfog_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(icd_cci, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('len(cci_index):', len(cci_index))
    output_file = r'../data/mapping/brainfog_index_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(cci_index, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return icd_cci, cci_index, dict_df_cci


def ICD_to_PASC_cognitive_fatigue_respiratory():
    start_time = time.time()
    pasc_list_file = r'../data/mapping/global_burden_Cognitive_Fatigue_Respiratory.xlsx'
    df_pasc_list = pd.read_excel(pasc_list_file,
                                 dtype=str,
                                 sheet_name='Sheet1')
    print('df_pasc_list.shape', df_pasc_list.shape)
    df_pasc_list['ICD-10-CM CODE'] = df_pasc_list['ICD-10-CM CODE'].apply(
        lambda x: x.strip().strip(r'\'').upper().replace('.', ''))
    pasc_codes = df_pasc_list['ICD-10-CM CODE']  # .str.upper().replace('.', '', regex=False)  # .to_list()
    pasc_codes_set = set(pasc_codes)
    print('Load compiled pasc list done from {}\nlen(pasc_codes)'.format(pasc_list_file),
          len(pasc_codes), 'len(pasc_codes_set):', len(pasc_codes_set))

    icd_pasc = {}
    pasc_index = {}

    for index, row in df_pasc_list.iterrows():
        category = row['Symptom cluster']
        icd = row['ICD-10-CM CODE']
        icd_name = row['ICD-10-CM CODE DESCRIPTION']
        icd_pasc[icd] = [category, icd, icd_name]

    df_dim = df_pasc_list['Symptom cluster'].value_counts(sort=False).reset_index()
    for index, row in df_dim.iterrows():
        ccsr_category = row[0]
        cnt = row[1]
        pasc_index[ccsr_category] = [index, cnt]

    print('len(pasc_index):', len(pasc_index))
    output_file = r'../data/mapping/icd_cognitive-fatigue-respiratory_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(icd_pasc, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('len(pasc_index):', len(pasc_index))
    output_file = r'../data/mapping/cognitive-fatigue-respiratory_index_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(pasc_index, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return icd_pasc, pasc_index, df_pasc_list

def ICD_to_CVD_death():
    start_time = time.time()
    pasc_list_file = r'../data/mapping/death_cardiovascular.xlsx'
    df_pasc_list = pd.read_excel(pasc_list_file,
                                 dtype=str,
                                 sheet_name='Sheet1')
    print('df_pasc_list.shape', df_pasc_list.shape)
    df_pasc_list['icd10cm'] = df_pasc_list['icd10cm'].apply(
        lambda x: x.strip().strip(r'\'').upper().replace('.', ''))
    pasc_codes = df_pasc_list['icd10cm']  # .str.upper().replace('.', '', regex=False)  # .to_list()
    pasc_codes_set = set(pasc_codes)
    print('Load compiled pasc list done from {}\nlen(pasc_codes)'.format(pasc_list_file),
          len(pasc_codes), 'len(pasc_codes_set):', len(pasc_codes_set))

    icd_pasc = {}
    pasc_index = {}

    for index, row in df_pasc_list.iterrows():
        category = row['category']
        subcategory = row['cvd death subgroup']
        icd = row['icd10cm']
        icd_name = row['long description']
        icd_pasc[icd] = [category, icd, icd_name, subcategory]

    df_dim = df_pasc_list['category'].value_counts(sort=False).reset_index()
    for index, row in df_dim.iterrows():
        ccsr_category = row[0]
        cnt = row[1]
        pasc_index[ccsr_category] = [index, cnt]

    print('len(pasc_index):', len(pasc_index))
    output_file = r'../data/mapping/icd_cvddeath_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(icd_pasc, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('len(pasc_index):', len(pasc_index))
    output_file = r'../data/mapping/cvddeath_index_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(pasc_index, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return icd_pasc, pasc_index, df_pasc_list


def load_cdc_mapping():
    # input_file = r'../data/mapping/CDC_COVIDv22_CodeList_v1.xlsx'
    # change at 2022-5-24, Add DX: Obstructive sleep apnea <----> OSA
    input_file = r'../data/mapping/CDC_COVIDv22_CodeList_v2.xlsx'

    mapping_file = r'../data/mapping/ventilation&comorbidity_sheetname.csv'
    df_map = pd.read_csv(mapping_file, dtype=str)
    df_all = pd.read_excel(input_file, sheet_name=None, dtype=str)  # read all sheets
    print('len(df_all):', len(df_all))
    print('len(df_map):', len(df_map))

    table_name = sorted(df_all.keys())
    tailor_comorbidity = {}
    for i, (key, row) in enumerate(df_map.iterrows()):
        query_name = row[0]
        sheet_name = row[1]
        notes = row[2]
        if (query_name.startswith('DX:') or query_name.startswith('MEDICATION:')) and \
                (query_name != r'DX: Hypertension and Type 1 or 2 Diabetes Diagnosis'):
            df = df_all[sheet_name]
            code_set = set()
            code_set_wildchar = set()
            for c in df.loc[:, 'code1']:
                if bool(re.search(r"\s", c)):
                    print(c)
                c = c.replace('.', '').upper().strip()
                if '*' in c:
                    code_set_wildchar.add(c)
                else:
                    code_set.add(c)

            tailor_comorbidity[query_name] = [code_set, code_set_wildchar]
            print('Done:', i, query_name, sheet_name)

    print('tailor_commorbidity  len(tailor_comorbidity):', len(tailor_comorbidity))
    utils.dump(tailor_comorbidity, r'../data/mapping/tailor_comorbidity_codes.pkl')

    vent = df_all['MECHANICAL_VENT']
    vent_dict = {}
    for key, row in vent.iterrows():
        if bool(re.search(r"\s", row['code1'])):
            print(row['code1'])
        code_raw = row['code1'].strip()
        code_type = row['codetype1']
        des = row['descrip']
        code = code_raw.replace('.', '').upper()
        vent_dict[code] = [code_raw, code_type, des]
    print('MECHANICAL_VENT  len(vent_dict):', len(vent_dict))
    utils.dump(vent_dict, r'../data/mapping/ventilation_codes.pkl')

    return df_all, tailor_comorbidity, vent_dict


def load_query3_vaccine_and_drug_mapping():
    df_map_vac = pd.read_excel(r'../data/mapping/query3-vaccine_sheet_mapping.xlsx', sheet_name='Sheet1', dtype=str)
    df_map_med = pd.read_excel(r'../data/mapping/query3-medication_sheet_mapping.xlsx', sheet_name='Sheet1', dtype=str)
    df_all = pd.read_excel(r'../data/mapping/RECOVER Query 3 Code List_2.28.22.xlsx', sheet_name=None,
                           dtype=str)  # read all sheets
    print('len(df_all):', len(df_all))
    print('len(df_map_vac):', len(df_map_vac))
    print('len(df_map_med):', len(df_map_med))

    # table_name = sorted(df_all.keys())

    med_code = {}
    for i, (key, row) in enumerate(df_map_med.iterrows()):
        query_name = row[0]
        sheet_name = row[1]
        notes = row[2]
        if sheet_name in df_all:
            df = df_all[sheet_name]
            code_dict = {}
            for ikey, irow in df.iterrows():
                if bool(re.search(r"\s", irow['code1'])):
                    print(irow['code1'])
                code_raw = irow['code1'].strip()
                code_type = irow['codetype1']
                des = irow['descrip']
                if '*' in code_raw:
                    print(code_raw, sheet_name)
                # code = code_raw.replace('.', '').upper()
                code_dict[code_raw] = [code_raw, code_type, des]

            med_code[query_name] = code_dict
        else:
            print('Not found sheet:', sheet_name)

        print('Done:', i, query_name, sheet_name, len(df))

    print('med_code done,  len(med_code):', len(med_code))
    utils.dump(med_code, r'../data/mapping/query3_medication_codes.pkl')

    vac_code = {}
    for i, (key, row) in enumerate(df_map_vac.iterrows()):
        query_name = row[0]
        sheet_name = row[1]
        notes = row[2]
        if sheet_name in df_all:
            df = df_all[sheet_name]
            code_dict = {}
            for ikey, irow in df.iterrows():
                if bool(re.search(r"\s", irow['code1'])):
                    print(irow['code1'])
                code_raw = irow['code1'].strip()
                code_type = irow['codetype1']
                des = irow['descrip']
                if '*' in code_raw:
                    print(code_raw, sheet_name)
                # code = code_raw.replace('.', '').upper()
                code_dict[code_raw] = [code_raw, code_type, des]

            vac_code[query_name] = code_dict

        else:
            print('Not found sheet:', sheet_name)

        print('Done:', i, query_name, sheet_name, len(df))

    print('vac_code done,  len(vac_code):', len(vac_code))
    utils.dump(vac_code, r'../data/mapping/query3_vaccine_codes.pkl')

    return df_all, med_code, vac_code


def build_icd9_to_icd10():
    df = pd.read_csv(r'../data/mapping/icd9toicd10cmgem.csv', dtype=str)
    print('df.shape:', df.shape)

    icd9_icd10 = defaultdict(list)
    for i, (key, row) in enumerate(df.iterrows()):
        icd9 = row[0].strip()
        icd10 = row[1].strip()
        icd9_icd10[icd9].append(icd10)

    print('len(icd9_icd10)', len(icd9_icd10))

    for key, records in icd9_icd10.items():
        # add a set operation to reduce duplicates
        records_sorted_list = list(set(records))
        icd9_icd10[key] = records_sorted_list

    print('len(icd9_icd10)', len(icd9_icd10))
    utils.dump(icd9_icd10, r'../data/mapping/icd9_icd10.pkl')

    return icd9_icd10


#
def _pre_smm_blood_pcs():
    dfselectpcs = pd.read_csv(r'../data/mapping/ICD-10-PCS Order File 2023/my_temp_code.csv')
    dfpcs = pd.read_excel(r'../data/mapping/ICD-10-PCS Order File 2023/icd10pcs_order_2023_edit_simple_copy.xlsx',
                          )
    selectpcslist = dfselectpcs['pcs'].to_list()
    selpcs = dfpcs.loc[dfpcs['icd10pcs'].isin(selectpcslist), :]

    dfcom = df_combined = pd.merge(dfselectpcs, dfpcs, left_on='pcs', right_on='icd10pcs', how='left')
    dfcom.to_csv(r'../data/mapping/ICD-10-PCS Order File 2023/my_temp_code_addinfo.csv')


def ICD_to_PASC_Severe_Maternal_Morbidity():
    # 2023-2-9
    # To get code mapping from icd10 to Maternal PASC list using SMM
    # https://www.cdc.gov/reproductivehealth/maternalinfanthealth/smm/severe-morbidity-ICD.htm.
    # Data source: ../data/mapping/Severe Maternal Morbidity List-2023-02-08.xlsx

    start_time = time.time()
    pasc_list_file = r'../data/mapping/Severe Maternal Morbidity List-2023-02-08.xlsx'
    df_pasc_list = pd.read_excel(pasc_list_file, sheet_name=r'Sheet1')
    print('df_pasc_list.shape', df_pasc_list.shape)
    df_pasc_list['Code'] = df_pasc_list['Code'].apply(lambda x: x.strip().upper().replace('.', ''))
    df_pasc_list['Comorbidity'] = df_pasc_list['Comorbidity'].apply(lambda x: 'smm:' + x.strip())

    pasc_codes = df_pasc_list['Code']  # .str.upper().replace('.', '', regex=False)  # .to_list()
    pasc_codes_set = set(pasc_codes)
    print('Load compiled pasc list done from {}\nlen(pasc_codes)'.format(pasc_list_file),
          len(pasc_codes), 'len(pasc_codes_set):', len(pasc_codes_set))

    icd_pasc = {}
    pasc_index = {}

    for index, row in df_pasc_list.iterrows():
        comorbidity = row['Comorbidity']
        codetype = row['CodeType']
        icd = row['Code']
        icd_name = row['Description']
        icd_pasc[icd] = [comorbidity, codetype, icd_name]

    df_dim = df_pasc_list['Comorbidity'].value_counts()[df_pasc_list['Comorbidity'].unique()].reset_index()
    for index, row in df_dim.iterrows():
        ccsr_category = row[0]
        cnt = row[1]
        pasc_index[ccsr_category] = [index, cnt]

    print('len(pasc_index):', len(pasc_index))
    output_file = r'../data/mapping/icd_SMMpasc_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(icd_pasc, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('len(pasc_index):', len(pasc_index))
    output_file = r'../data/mapping/SMMpasc_index_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(pasc_index, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return icd_pasc, pasc_index, df_pasc_list


def ICD_to_Obstetric_Comorbidity():
    # 2023-2-9
    start_time = time.time()
    pasc_list_file = r'../data/mapping/Obstetric Comorbidity List-2023-02-08.xlsx'
    df_pasc_list = pd.read_excel(pasc_list_file, sheet_name=r'Sheet1')
    print('df_pasc_list.shape', df_pasc_list.shape)
    df_pasc_list['Code'] = df_pasc_list['Code'].apply(lambda x: x.strip().upper().replace('.', ''))
    df_pasc_list['Comorbidity'] = df_pasc_list['Comorbidity'].apply(lambda x: 'obc:' + x.strip())

    pasc_codes = df_pasc_list['Code']  # .str.upper().replace('.', '', regex=False)  # .to_list()
    pasc_codes_set = set(pasc_codes)
    print('Load compiled Obstetric Comorbidity list done from {}\nlen(pasc_codes)'.format(pasc_list_file),
          len(pasc_codes), 'len(pasc_codes_set):', len(pasc_codes_set))

    icd_pasc = {}
    pasc_index = {}

    for index, row in df_pasc_list.iterrows():
        comorbidity = row['Comorbidity']
        codetype = row['CodeType']
        icd = row['Code']
        icd_name = row['Description']
        icd_pasc[icd] = [comorbidity, codetype, icd_name]

    df_dim = df_pasc_list['Comorbidity'].value_counts()[df_pasc_list['Comorbidity'].unique()].reset_index()
    for index, row in df_dim.iterrows():
        ccsr_category = row[0]
        cnt = row[1]
        pasc_index[ccsr_category] = [index, cnt]

    print('len(pasc_index):', len(pasc_index))
    output_file = r'../data/mapping/icd_OBComorbidity_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(icd_pasc, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('len(pasc_index):', len(pasc_index))
    output_file = r'../data/mapping/OBComorbidity_index_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(pasc_index, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return icd_pasc, pasc_index, df_pasc_list


def _pre_ckd_codelist():
    df_ckd = pd.read_excel(r'../data/mapping/ckd_codes.xlsx', sheet_name=r'serum creatinine')
    df_ckd2 = pd.read_excel(r'../data/mapping/ckd_codes.xlsx', sheet_name=r'eGFR')

    df_loinc = pd.read_csv(r'../data/mapping/Loinc_2.76/LoincTableCore/LoincTableCore.csv', dtype=str)
    dfcom = pd.merge(df_ckd, df_loinc, left_on='code', right_on='LOINC_NUM', how='left')
    dfcom.to_csv(r'../data/mapping/ckd_codes-serum-creatinine.csv')

    dfcom2 = pd.merge(df_ckd2, df_loinc, left_on='code', right_on='LOINC_NUM', how='left')
    dfcom2.to_csv(r'../data/mapping/ckd_codes-eGFR.csv')

    # then build code list ckd_codes_revised.xlsx, move intermediate file into history
    return df_ckd, dfcom, df_ckd2, dfcom2


def zip_ruca_mapping():
    # 2010 Rural-Urban Commuting Area Codes, ZIP code file	8/17/2020
    df_ruca = pd.read_excel(r'../data/mapping/RUCA2010zipcode.xlsx', sheet_name=r'Data',
                            dtype={'ZIP_CODE': str})
    print('df_ruca.shape:', df_ruca.shape)
    print(df_ruca.dtypes)
    print(df_ruca.describe())
    print(df_ruca['ZIP_TYPE'].value_counts())

    zip_ruca = {}
    for index, row in tqdm(df_ruca.iterrows(), total=len(df_ruca)):
        zip = row['ZIP_CODE']
        state = row['STATE']
        type = row['ZIP_TYPE']
        r1 = row['RUCA1']
        r2 = row['RUCA2']
        zip_ruca[zip] = (r1, r2, state, type)

    print('len(zip_ruca)', len(zip_ruca))
    output_file = r'../data/mapping/zip_ruca_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(zip_ruca, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))
    return zip_ruca, df_ruca


def ICD_to_CCI_Comorbidity():
    # 2023-2-9
    start_time = time.time()
    dict_df_cci = pd.read_excel(r'../data/mapping/Quan-Charlson-Cormorbidity-Index.xlsx', dtype=str, sheet_name=None)
    print('len(icd_cci, cci_index, dict_df_cci)', len(dict_df_cci))

    icd_cci = {}
    cci_index = {}

    for ith, (key, df_cci) in enumerate(dict_df_cci.items()):
        cci = df_cci['Category'].unique()
        print(len(cci), cci)
        assert len(cci) == 1
        cci = cci[0]
        cci_index[cci] = [ith, len(df_cci)]
        print('cci:', cci, 'index:', ith, '#code:', len(df_cci), df_cci[['code type']].value_counts())
        for index, row in df_cci.iterrows():
            icd = row['dx code'].strip().upper().replace('.', '')
            name = row['LONG DESCRIPTION']
            type = row['code type']
            # cci = row['Category']
            order = row['CCI order']
            icd_cci[icd] = [cci, type, name, order]

    print('len(icd_cci):', len(icd_cci))
    output_file = r'../data/mapping/icd_cci_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(icd_cci, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('len(cci_index):', len(cci_index))
    output_file = r'../data/mapping/cci_index_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(cci_index, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return icd_cci, cci_index, dict_df_cci


def build_updated_covid_drugs():
    dict_df_all = pd.read_excel(r'../data/mapping/covid_drug.xlsx', sheet_name=None, dtype=str)  # read all sheets
    med_code = {}
    for ith, (key, df_all) in enumerate(dict_df_all.items()):
        print('drug:', key, 'index:', ith, '#code:', len(df_all), df_all[['codetype1']].value_counts())
        code_dict = {}
        for index, row in df_all.iterrows():
            code = row['code1'].strip()
            type = row['codetype1']
            name = row['descrip']
            code_dict[code] = [code, type, name]

        med_code[key] = code_dict

    print('med_code done,  len(med_code):', len(med_code))
    utils.dump(med_code, r'../data/mapping/covid_drugs_updated.pkl')
    return med_code, dict_df_all


def build_n3c_pax_contraindication():
    print('in build_n3c_pax_contraindication')
    df = pd.read_csv(r'../data/mapping/n3c_Pax_contraindications.csv', dtype=str)
    rx_info = {}
    for index, row in df.iterrows():
        rx = row['concept_code'].strip()
        name = row['concept_name']
        type = row['vocabulary_id']
        drug_type = row['concept_class_id']
        con_id = row['concept_id']
        rx_info[rx] = [rx, name, type, drug_type, con_id]

    print('rx_info done,  len(rx_info):', len(rx_info))
    utils.dump(rx_info, r'../data/mapping/n3c_pax_contraindication.pkl')
    return rx_info, df


def build_n3c_pax_indication():
    print('in build_n3c_pax_indication')
    df = pd.read_csv(r'../data/mapping/n3c_Pax_indication_codes.csv', dtype=str)
    """
    ICDO3               48987
    RxNorm Extension    20318
    SNOMED              13068
    RxNorm               1715
    ICD10CM               703
    Nebraska Lexicon      419
    HemOnc                 82
    HCPCS                  82
    CPT4                   76
    CIEL                   57
    OMOP Extension         55
    ICD10PCS               55
    LOINC                  32
    ATC                    31
    ICD9CM                 19
    ICD10                   6
    OPCS4                   5
    ICD9Proc                3
    PPI                     1
    """
    # ['ICDO3', 'RxNorm Extension', 'SNOMED', 'RxNorm', 'ICD10CM',
    # 'Nebraska Lexicon', 'HemOnc', 'HCPCS', 'CPT4', 'CIEL', 'OMOP Extension',
    #  'ICD10PCS', 'LOINC', 'ATC', 'ICD9CM', 'ICD10', 'OPCS4', 'ICD9Proc',
    #  'PPI']

    # remaining others:
    # ['RxNorm Extension',
    # 'Nebraska Lexicon', 'HemOnc',   'CIEL', 'OMOP Extension',
    # 'LOINC', 'ATC',
    # 'PPI']
    # steroid, and smoking are captured in other place
    # loinc codes are for smoking, explored later

    # # build snomed concept to icd codes set
    # df_snomed_map = pd.read_csv(r'../data/mapping/der2_iisssccRefset_ExtendedMapFull_US1000124_20230901.txt',
    #                             dtype=str, sep='\t', lineterminator='\r')
    # snomed_icd = defaultdict(set)
    # for index, row in tqdm(df_snomed_map.iterrows(), total=len(df_snomed_map)):
    #     snomed = row['referencedComponentId']
    #     icd = row['mapTarget']
    #     if pd.isna(icd) or not isinstance(icd, str):
    #         continue
    #     else:
    #         icd = icd.strip().upper().replace('.', '').strip('?')
    #
    #     snomed_icd[snomed].add(icd)
    #
    # print('len(snomed_icd)', len(snomed_icd))
    # utils.dump(snomed_icd, r'../data/mapping/snomed_icd.pkl')
    snomed_icd = utils.load(r'../data/mapping/snomed_icd.pkl')

    med_info = {}
    pro_info = {}
    dx_info = {}
    other_info = {}
    for index, row in df.iterrows():
        code = row['concept_code'].strip()
        type = row['vocabulary_id']
        name = row['concept_name']
        con_type = row['concept_class_id']
        con_id = row['concept_id']

        if type == 'ICDO3':
            code = code.split('-')
            code = code[1]
            code = code.strip().upper().replace('.', '')
            dx_info[code] = [code, name, type, con_type, con_id]
        elif type in ['ICD10CM', 'ICD10', 'ICD9CM', ]:
            code = code.strip().upper().replace('.', '')
            dx_info[code] = [code, name, type, con_type, con_id]
        elif type == 'RxNorm':
            med_info[code] = [code, name, type, con_type, con_id]
        elif type in ['ICD10PCS', 'ICD9Proc', 'CPT4', 'OPCS4', 'HCPCS']:
            pro_info[code] = [code, name, type, con_type, con_id]
        elif type == 'SNOMED':
            if code in snomed_icd:
                mapped_icd = snomed_icd[code]
                for x in mapped_icd:
                    dx_info[x] = [code, name, type, con_type, con_id]
        else:
            other_info[code] = [code, name, type, con_type, con_id]

    print('len(med_info):', len(med_info),
          'len(pro_info)', len(pro_info),
          'len(dx_info)', len(dx_info),
          'len(other_info)', len(other_info),
          )
    utils.dump((med_info, pro_info, dx_info, other_info), r'../data/mapping/n3c_pax_indication.pkl')
    return (med_info, pro_info, dx_info, other_info), df


def ICD_to_addedPaxRisk():
    # 2023-2-9
    start_time = time.time()
    dict_df_cci = pd.read_excel(r'../data/mapping/paxlovid_risk_added.xlsx', dtype=str, sheet_name=None)
    print('len(icd_cci, cci_index, dict_df_cci)', len(dict_df_cci))

    icd_cci = {}
    cci_index = {}

    for ith, (key, df_cci) in enumerate(dict_df_cci.items()):
        cci = df_cci['Category'].unique()
        print(len(cci), cci)
        assert len(cci) == 1
        cci = cci[0]
        cci_index[cci] = [ith, len(df_cci)]
        print('cci:', cci, 'index:', ith, '#code:', len(df_cci), df_cci[['code type']].value_counts())
        for index, row in df_cci.iterrows():
            icd = row['dx code'].strip().upper().replace('.', '')
            name = row['LONG DESCRIPTION']
            type = row['code type']
            # cci = row['Category']
            # order = row['CCI order']
            order = ith
            icd_cci[icd] = [cci, type, name, order]

    print('len(icd_cci):', len(icd_cci))
    output_file = r'../data/mapping/icd_addedPaxRisk_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(icd_cci, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('len(cci_index):', len(cci_index))
    output_file = r'../data/mapping/addedPaxRisk_index_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(cci_index, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return icd_cci, cci_index, dict_df_cci


def ICD_to_mental():
    # 2023-9-5
    start_time = time.time()

    icd_mental = {}
    mental_index = {}

    # part 1, from dsm related categories
    print('part 1, from dsm related categories')
    df_dsm = pd.read_excel(r'../data/mapping/dsm-5.xlsx', dtype=str, sheet_name='Sheet1')
    print('len(df_dsm)', len(df_dsm))

    cat_list_ = df_dsm['category'].unique()
    print(len(cat_list_), cat_list_)
    cat_list = [x for x in cat_list_ if pd.notna(x)]
    print(len(cat_list), cat_list)

    ith_cat = 0
    for cat in cat_list:
        df_cat = df_dsm.loc[df_dsm['category'] == cat, :]
        icd_mental_buff = {}
        for index, row in df_cat.iterrows():
            icd10 = row['ICD-10–CM'] #.strip().upper().replace('.', '')
            icd9 = row['ICD-9–CM']
            name = row['Disorder, condition, or problem']

            if pd.notna(icd10):
                icd10 = icd10.strip().upper().replace('.', '')
                icd_mental_buff[icd10] = [cat, 'icd10', name, ith_cat]

            if pd.notna(icd9):
                icd9 = icd9.strip().upper().replace('.', '')
                icd_mental_buff[icd9] = [cat, 'icd9', name, ith_cat]

        mental_index[cat] = [ith_cat, len(icd_mental_buff)]
        icd_mental.update(icd_mental_buff)
        ith_cat += 1

    # part 2, from revised elixhauser, Severe Mental Illnesses (SMI)  vs non SMI
    print('part 2, from revised elixhauser, Severe Mental Illnesses (SMI)  vs non SMI')
    df_elix= pd.read_excel(r'../data/mapping/mental_elix_tailored_SP.xlsx', dtype=str, sheet_name='Sheet1')
    print('len(df_elix)', len(df_elix))

    cat_list_ = df_elix['Category'].unique()
    print(len(cat_list_), cat_list_)
    cat_list = [x for x in cat_list_ if pd.notna(x)]
    print(len(cat_list), cat_list)
    for cat in cat_list:
        df_cat = df_elix.loc[df_elix['Category'] == cat, :]
        icd_mental_buff = {}
        for index, row in df_cat.iterrows():
            icd = row['code1']
            type = row['codetype1']
            name = row['long description']

            if pd.notna(icd):
                icd = icd.strip().strip('*').upper().replace('.', '')
                icd_mental_buff[icd] = [cat, type, name, ith_cat]

        mental_index[cat] = [ith_cat, len(icd_mental_buff)]
        icd_mental.update(icd_mental_buff)
        ith_cat += 1


    print('len(icd_mental):', len(icd_mental))
    output_file = r'../data/mapping/icd_mental_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(icd_mental, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('len(mental_index):', len(mental_index))
    output_file = r'../data/mapping/mental_index_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(mental_index, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return icd_mental, mental_index, [df_dsm, df_elix]


def ICD_to_CNS_LDN_covs():
    # 2025-4-8
    start_time = time.time()
    dict_df_cci = pd.read_excel(r'../data/mapping/LDN_CNS_Covariates_PASC.xlsx', dtype=str, sheet_name=None)
    print('len(dict_df_cci)', len(dict_df_cci))

    # warning: there are potnetial overlapping issue. If one icd code contribute two categories.
    # not a problem if no overlap
    icd_cci = {}
    cci_index = {}

    for ith, (key, df_cci) in enumerate(dict_df_cci.items()):
        print(ith, key, len(df_cci))
        if key == 'notes':
            break
        print('key of cov:', key, 'index:', ith, '#code:', len(df_cci))
        if key == 'MECFS':
            df_cci = df_cci.loc[df_cci['include']=='yes', :]
            print('key of cov:', key, 'index:', ith, '#code:', len(df_cci))

        cci_index[key] = [ith, len(df_cci)]
        for index, row in df_cci.iterrows():
            icd = row['ICD-10-CM Code'].strip().upper().replace('.', '')
            name = row['ICD-10-CM Code Description']
            type = 'icd10' # all ICD10 for the current ones, not code type columns. row['code type']
            ccsr_category_des =  row['CCSR CATEGORY 1 DESCRIPTION']
            # cci = row['Category']
            # order = row['CCI order']
            order = ith
            icd_cci[icd] = [key, type, name, ccsr_category_des]

    print('len(icd_cci):', len(icd_cci))
    output_file = r'../data/mapping/icd_covCNS-LDN_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(icd_cci, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('len(cci_index):', len(cci_index))
    output_file = r'../data/mapping/covCNS-LDN_index_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(cci_index, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return icd_cci, cci_index, dict_df_cci


def build_ssri_snri_drug_map():
    med_code = {}
    fname_dict = {'ssri_drug_list.xlsx': 'ssri',
                  'snri_drug_list.xlsx': 'snri',
                  'other_mental_drug_list.xlsx': 'other'}

    for fname, drugtype in fname_dict.items():
        df_all = pd.read_excel(r'../data/mapping/' + fname, sheet_name=None, dtype=str)
        print(fname, drugtype, 'len(df_all)', len(df_all))
        for ith, (key, df) in enumerate(df_all.items()):
            print('drug:', key, 'index:', ith, '#code:', len(df)) #, df[['code']].value_counts())
            code_dict = {}
            for index, row in df.iterrows():
                code = row['code'].strip()
                type = row['code type']
                name = row['name']
                drug_ingcat = row['drug']

                code_dict[code] = [code, type, name, drug_ingcat, drugtype]

            med_code[key] = code_dict

    print('med_code done,  len(med_code):', len(med_code))
    utils.dump(med_code, r'../data/mapping/ssri_snri_drugs_mapping.pkl')
    return med_code


def build_cns_naltrxone_drug_map():
    med_code = {}

    df_all = pd.read_excel(r'../data/mapping/CNS_stimulants_and_Naltrexone_code_list_25April7.xlsx', sheet_name=None, dtype=str)
    print('len(df_all)', len(df_all))
    for ith, (key, df) in enumerate(df_all.items()):
        print('drug:', key, 'index:', ith, '#code:', len(df)) #, df[['code']].value_counts())
        code_dict = {}
        for index, row in df.iterrows():
            code = row['code'].strip()
            type = row['code type']
            name = row['name']
            drug_ingcat = row['drug']

            code_dict[code] = [code, type, name, drug_ingcat, key]

        med_code[key] = code_dict

    print('med_code done,  len(med_code):', len(med_code))
    utils.dump(med_code, r'../data/mapping/cns_ldn_drugs_mapping.pkl')
    return med_code

def build_ADHD_ctrl_drug_map():
    med_code = {}

    df_all = pd.read_excel(r'../data/mapping/ADHD_control_code_list.xlsx', sheet_name=None, dtype=str)
    print('len(df_all)', len(df_all))
    for ith, (key, df) in enumerate(df_all.items()):
        print('drug:', key, 'index:', ith, '#code:', len(df)) #, df[['code']].value_counts())
        code_dict = {}
        for index, row in df.iterrows():
            code = row['code'].strip()
            type = row['code type']
            name = row['name']
            drug_ingcat = row['drug']

            code_dict[code] = [code, type, name, drug_ingcat, key]

        med_code[key] = code_dict

    print('med_code done,  len(med_code):', len(med_code))
    utils.dump(med_code, r'../data/mapping/ADHD_ctrl_drugs_mapping.pkl')
    return med_code


def build_pregnant_drugs_grt():
    dict_df_all = pd.read_excel(r'../data/mapping/preg_drug_of_interest.xlsx', sheet_name=None, dtype=str)  # read all sheets
    med_code = {}
    for ith, (key, df_all) in enumerate(dict_df_all.items()):
        print('drug:', key, 'index:', ith, '#code:', len(df_all), df_all[['codetype']].value_counts())
        code_dict = {}
        for index, row in df_all.iterrows():
            code = row['code'].strip()
            type = row['codetype']
            name = row['name']
            drug_group = row['drug group']
            code_dict[code] = [code, type, name, drug_group]

        med_code[key] = code_dict

    print('med_code done,  len(med_code):', len(med_code))
    utils.dump(med_code, r'../data/mapping/pregnant_drug_grt.pkl')
    return med_code, dict_df_all


def ICD_to_covNaltrexone_multiplemapping():
    # 2025-4-8
    start_time = time.time()
    dict_df_cci = pd.read_excel(r'../data/mapping/Naltrexone_Covariates_PASC_Q3version-v4-edit.xlsx', dtype=str, sheet_name=None)
    print('len(dict_df_cci)', len(dict_df_cci))

    # warning: there are potnetial overlapping issue. If one icd code contribute two categories.
    # not a problem if no overlap
    icd_cci = {}
    cci_index = {}

    for ith, (key, df_cci) in enumerate(dict_df_cci.items()):
        print(ith, key, len(df_cci))
        if key == 'notes':
            break
        print('key of cov:', key, 'index:', ith, '#code:', len(df_cci))
        if key == 'MECFS':
            df_cci = df_cci.loc[df_cci['include']=='yes', :]
            print('key of cov:', key, 'index:', ith, '#code:', len(df_cci))

        if key == 'Pain':
            df_cci = df_cci.loc[df_cci['exclude']!='1', :]
            print('key of cov:', key, 'index:', ith, '#code:', len(df_cci))

        cci_index[key] = [ith, len(df_cci)]
        for index, row in df_cci.iterrows():
            icd = row['code'].strip().upper().replace('.', '')
            name = row['description']
            type = row['codetype'] #'icd10' # all ICD10 for the current ones, not code type columns. row['code type']
            ccsr_category_des =  row['category']
            # cci = row['Category']
            # order = row['CCI order']
            order = ith
            # Notes: 2025-7-11
            # in this code list, one icd can map to different category, potential overwrite issues.
            # thus, revise this to list of list structure, and then revise encoding function as well
            # icd_cci[icd] = [key, type, name, ccsr_category_des]
            if icd not in icd_cci:
                icd_cci[icd] = []

            icd_cci[icd].append([key, type, name, ccsr_category_des])

    print('len(icd_cci):', len(icd_cci))
    output_file = r'../data/mapping/icd_covNaltrexone_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(icd_cci, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('len(cci_index):', len(cci_index))
    output_file = r'../data/mapping/covNaltrexone_index_mapping.pkl'
    utils.check_and_mkdir(output_file)
    pickle.dump(cci_index, open(output_file, 'wb'))
    print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return icd_cci, cci_index, dict_df_cci


def build_Naltrexone_drug_cov_map():
    med_code = {}

    df_all = pd.read_excel(r'../data/mapping/Naltrexone_Covariates_PASC_Q3version-v4-edit-drugpart.xlsx', sheet_name=None, dtype=str)
    print('len(df_all)', len(df_all))
    for ith, (key, df) in enumerate(df_all.items()):
        print('drug:', key, 'index:', ith, '#code:', len(df)) #, df[['code']].value_counts())
        code_dict = {}
        for index, row in df.iterrows():
            code = row['code'].strip()
            type = row['codetype']
            name = row['name']
            drug_ingcat = row['category']

            code_dict[code] = [code, type, name, drug_ingcat, key]

        med_code[key] = code_dict

    print('med_code done,  len(med_code):', len(med_code))
    utils.dump(med_code, r'../data/mapping/Naltrexone_drug_cov_mapping.pkl')
    return med_code


if __name__ == '__main__':
    # python pre_codemapping.py 2>&1 | tee  log/pre_codemapping_zip_adi.txt
    start_time = time.time()
    # 0. rxnorm to name
    # rx_name, atc_name = build_rxnorm_or_atc_to_name()

    # 1. Build rxnorm to atc mapping:
    # rxnorm_atcset, atc_rxnormset, atc3_index, df_rxrnom_atc = rxnorm_atc_from_NIH_UMLS()

    # 2. Build rxnorm to ingredient(s) mapping
    # rx_ing, df_rx_ing = rxnorm_ingredient_from_NIH_UMLS()
    # rx_ing_api, df_rx_ing_api = add_rxnorm_ingredient_by_umls_api()
    # rx_ing_combined, df_records_combined = combine_rxnorm_ingredients_dicts()
    # ing_index = selected_rxnorm_ingredient_to_index()
    # ndc_rx = build_NDC_to_rxnorm()

    # 3. Build zip5/9 to adi mapping
    # zip_adi, zip5_df = zip_adi_mapping_2020()
    # zip_adi, zip5_df = zip_adi_mapping_2021()
    # zip_adi, fips_adi, fips_zip, zip5_df = zip_fips_adi_mapping_2020() # updated use geocode for ADI

    # 4. Build ICD10 to CCSR mapping
    # icd_ccsr, ccsr_index, ccsr_df = ICD10_to_CCSR()

    # 5. Build ICD10 to elixhauser_comorbidity
    # icd_cmr, cmr_index, df_cmr = ICD_to_elixhauser_comorbidity()

    # 6. Build ICD10 to pasc
    # icd_pasc, pasc_index, df_pasc = ICD_to_PASC()
    # updated: 2023-11-9 to add more fine grained/selected categories
    # use other function/dimensions, instead of changing existing codes--> list of list, supporting overlapped categories,

    # icd_addedPASC, addedPASC_index, df_pasc = ICD_to_PASC_added_extension()
    # icd_brainfog, brainfog_index, dict_df_brainfog = ICD_to_PASC_brainfog()
    # icd_cfr, cfr_index, dict_df_cfr = ICD_to_PASC_cognitive_fatigue_respiratory()
    # 2024-12-14 add ME-CFS as outcome
    # icd_mecfs, mecfs_index, df_mecfs = ICD_to_PASC_ME_CFS()
    # 2024-12-14 add cvd to check death record with CVD dx
    # icd_cvddeath, cvddeath_index, dict_df_cvd = ICD_to_CVD_death()
    # zz

    # 7. Load CDC code mapping:
    # df_all, tailor_comorbidity, vent_dict = load_cdc_mapping()

    # 8. Load query 3 mapping:
    # df_all, med_code, vac_code = load_query3_vaccine_and_drug_mapping()
    #

    # 9 Load icd9 to icd10 mapping
    # icd9_icd10 = build_icd9_to_icd10()

    # 10 build ICD10 to negative outcome control of PASC
    # icd_pasc, pasc_index, df_pasc_list = ICD_to_negative_control_pasc()

    # 11 build comorbidities and outcome for pregnant women
    # 2023-2-9
    # icd_SMMpasc, SMMpasc_index, df_SMMpasc = ICD_to_PASC_Severe_Maternal_Morbidity()
    # icd_OBC, OBC_index, df_OBC = ICD_to_Obstetric_Comorbidity()

    # 12 build ckd code list for paxlovid (2023-10-17)
    # add more loinc info, then edited in excel
    # _pre_ckd_codelist()

    # 13 zip_ruca (2023-10-18)
    # zip_ruca, df_ruca = zip_ruca_mapping()

    # 14 CCI index, each dim, and then aggregated  (2023-10-18)
    # icd_cci, cci_index, dict_df_cci = ICD_to_CCI_Comorbidity()

    # 15 updated drug list for paxlovid and remdesivir (2023-10-18)
    # med_code, dict_df_all = build_updated_covid_drugs()

    # 16 build n3c pax drug list (2023-11-13)
    # rx_info, df_contraindication = build_n3c_pax_contraindication()
    # (med_info, pro_info, dx_info, other_info), df_indication = build_n3c_pax_indication()

    # 17 added covs for paxlovid risk  (2024-2-13)
    #icd_addedPaxRisk, addedPaxRisk_index, dict_df_addedPaxRisk = ICD_to_addedPaxRisk()

    # 18 ssri snri drugs map (2024-4-2)
    # updated 2024-7-12 by adding vilazodone
    # 2024-09-06 replace 'wellbutrin' with name bupropion
    # ssrisnrimed_code = build_ssri_snri_drug_map()

    # 19 add more mental categories 2024-09-05
    # icd_mental, mental_index, list_df_menta = ICD_to_mental()


    # 20 add pregnant related drug of interested, 2025-1-22
    #med_code, dict_df_all = build_pregnant_drugs_grt()

    # 21 CNS and Naltrxone/LDN related drugs, 2025-04-08
    # med_code = build_cns_naltrxone_drug_map()

    # 22 add more CNS LDN related covs, 2025-04-08
    # icd_covCNSLDN, covCNSLDN_index, list_df_covCNSLDN = ICD_to_CNS_LDN_covs()

    # 23 build ADHD ctrl drug, viloxazine, atomoxetine, nortriptyline, bupropion , 2025-5-20
    # ['viloxazine', 'atomoxetine', 'nortriptyline', 'bupropion']
    # med_code = build_ADHD_ctrl_drug_map()

    # 24 add Naltrexone related covs, 2025-07-11
    # multiple mapping this time, different from all above
    icd_covNaltrexone_multimap, covNaltrexone_index, list_df_covNaltrexone = ICD_to_covNaltrexone_multiplemapping()

    # 25 add Naltrexone related drugs, 2025-07-11
    Naltrexone_drug_cov_code = build_Naltrexone_drug_cov_map()

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
