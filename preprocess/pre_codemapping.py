import sys
# for linux env.
sys.path.insert(0,'..')
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
            n_has_return += 1
        else:
            n_no_return += 1

        if rx in rxnorm_ing:
            print(i, rx, ':already found:', ';'.join(rxnorm_ing[rx]), 'vs new found:', ';'.join(sorted(ings)))

        if i % 10000 == 0:
            print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
            print('Total search:', len(rxnorm_set), 'search:', i, 'n_has_return:', n_has_return, 'n_no_return:', n_no_return)

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
    with open(r'../data/mapping/rxnorm_ingredient_mapping_from_api_moiety.pkl', 'rb') as f: # with open(r'../data/mapping/rxnorm_ingredient_mapping_from_api.pkl', 'rb') as f:
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


def selected_rxnorm_ingredient_to_index():
    start_time = time.time()
    rx_df = pd.read_csv(r'../data/mapping/info_medication_cohorts_covid_4manuNegNoCovid_ALL_enriched.csv',
                        dtype={'rxnorm':str})
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
    return icd_ccsr, ccsr_index,  df


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


def ICD_to_PASC():
    # To get code mapping from icd10 to PASC our compiled list.
    # Data source: ../data/mapping/PASC_Adult_Combined_List_20220127_v3.xlsx May be updated later

    start_time = time.time()
    pasc_list_file = r'../data/mapping/PASC_Adult_Combined_List_20220127_v3.xlsx'
    df_pasc_list = pd.read_excel(pasc_list_file, sheet_name=r'PASC Screening List', usecols="A:N")
    print('df_pasc_list.shape', df_pasc_list.shape)
    df_pasc_list['ICD-10-CM Code'] = df_pasc_list['ICD-10-CM Code'].apply(lambda x : x.strip().upper().replace('.', ''))
    pasc_codes = df_pasc_list['ICD-10-CM Code'] #.str.upper().replace('.', '', regex=False)  # .to_list()
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


def load_cdc_mapping():
    input_file = r'../data/mapping/CDC_COVIDv22_CodeList_v1.xlsx'
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
    df_all = pd.read_excel(r'../data/mapping/RECOVER Query 3 Code List_2.28.22.xlsx', sheet_name=None, dtype=str)  # read all sheets
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
    ndc_rx = build_NDC_to_rxnorm()

    # 3. Build zip5/9 to adi mapping
    # zip_adi, zip5_df = zip_aid_mapping()

    # 4. Build ICD10 to CCSR mapping
    # icd_ccsr, ccsr_index, ccsr_df = ICD10_to_CCSR()

    # 5. Build ICD10 to elixhauser_comorbidity
    # icd_cmr, cmr_index, df_cmr = ICD_to_elixhauser_comorbidity()

    # 6. Build ICD10 to pasc
    # icd_pasc, pasc_index, df_pasc = ICD_to_PASC()

    # 7. Load CDC code mapping:
    # df_all, tailor_comorbidity, vent_dict = load_cdc_mapping()

    # 8. Load query 3 mapping:
    # df_all, med_code, vac_code = load_query3_vaccine_and_drug_mapping()
    #

    # 9 Load icd9 to icd10 mapping
    # icd9_icd10 = build_icd9_to_icd10()

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
