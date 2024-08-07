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

# https://lhncbc.nlm.nih.gov/RxNav/applications/RxNavViews.html#label:appendix
TTY_DICT = {
    "BN": "brand name",
    "SBD": "branded drug",
    "BPCK": "branded pack",
    "SBDC": "branded drug component",
    "DF": "dose form",
    "SBDF": "branded dose form",
    "DFG": "dose form group",
    "SBDG": "branded dose form group",
    "GPCK": "generic pack",
    "SCD": "clinical drug",
    "IN": "ingredient",
    "SCDC": "clinical drug component",
    "MIN": "multiple ingredients",
    "SCDF": "clinical dose form",
    "PIN": "precise ingredient",
    "SCDG": "clinical dose form group",
    # https://www.nlm.nih.gov/research/umls/rxnorm/docs/appendix5.html
    "SBDFP": "Branded Drug Form Precise",
    "SCDFP": "Clinical Drug Form Precise",
    "SCDGP": "Clinical Drug Form Group Precise",
}


def tty_descript(tty):
    if tty in TTY_DICT:
        return TTY_DICT[tty]
    else:
        return ''


def get_rx_property_by_api(rx):
    url_str = "https://rxnav.nlm.nih.gov/REST/rxcui/{}/properties.json".format(rx)
    r = requests.get(url_str)
    data = r.json()

    if "properties" in data:
        rxcui = data["properties"]['rxcui']
        assert rx == rxcui
        name = data["properties"]['name']
        syn = data["properties"]['synonym']
        tty = data["properties"]['tty']
        return (rx, name, tty, syn)
    else:
        raise ValueError


def get_rx_history_by_api(rx):
    url_str = "https://rxnav.nlm.nih.gov/REST/rxcui/{}/historystatus.json".format(rx)
    r = requests.get(url_str)
    data = r.json()

    try:
        d = data['rxcuiStatusHistory']['attributes']
        rxcui = d['rxcui']
        assert rx == rxcui
        name = d['name']
        tty = d['tty']
        return (tty, rx, name)
    except:
        return (np.nan, np.nan, np.nan)
        raise ValueError


#
def get_ndc_list_from_rxnorm_by_api(rx, history=1):
    """
    Depth of history to retrieve
    One of:
    0 NDCs presently directly associated
    1 NDCs ever directly associated
    2 NDCs ever (in)directly associated
    :param rx:
    :return:
    """
    url_str = "https://rxnav.nlm.nih.gov/REST/rxcui/{}/allhistoricalndcs.json?history={}".format(rx, history)
    r = requests.get(url_str)
    data = r.json()
    ndc_info = {}
    try:
        ndc_list = data['historicalNdcConcept']['historicalNdcTime'][0]['ndcTime']
        for x in ndc_list:
            ndc = x['ndc'][0]
            st = x['startDate']
            et = x['endDate']
            if ndc not in ndc_list:
                ndc_info[ndc] = [st, et, 'nih']
            else:
                print(ndc, 'in ndc_info map')
    except:
        print('error in reading', url_str)

    return ndc_info


def get_ndc_status_by_api(ndc):
    url_str = "https://rxnav.nlm.nih.gov/REST/ndcstatus.json?ndc={}".format(ndc)
    r = requests.get(url_str)
    data = r.json()
    info = []
    try:
        status = data['ndcStatus']['status']
        rxcui = data['ndcStatus']['rxcui']
        conceptName = data['ndcStatus']['conceptName']

        info.append(conceptName)
        info.append(status)
        info.append(rxcui)

        rx_list = []
        if 'ndcHistory' in data['ndcStatus']:
            for x in data['ndcStatus']['ndcHistory']:
                rx_list.append(x['activeRxcui'])
                rx_list.append(x['originalRxcui'])

            rx_set = set(rx_list)
            info.append(';'.join(rx_set))
            info.append(str(data['ndcStatus']['ndcHistory']))
        elif 'ndcSourceMapping' in data['ndcStatus']:
            for x in data['ndcStatus']['ndcSourceMapping']:
                rx_list.append(x['ndcRxcui'])

            rx_set = set(rx_list)
            info.append(';'.join(rx_set))
            info.append(str(data['ndcStatus']['ndcSourceMapping']))
        else:
            info.append(np.nan)
            info.append(np.nan)

    except:
        print('error in reading', url_str)
        info = [np.nan, ] * 5

    return info


def get_ndc_list_of_rxnorm_in_cardiology_cp_medication():
    fname = r'../data/V15_COVID19/output/character/cp_cardio/RECOVER Cardiology CP Code Lists_v6_8.10.22.xlsx'
    df = pd.read_excel(fname, sheet_name="Medications", engine='openpyxl', dtype=str)
    # from epic cards related codes by Mark
    fname_mark = r'../data/V15_COVID19/output/character/cp_cardio/cards Rxnorm NDC crosswalk.csv'
    df_mark = pd.read_csv(fname_mark, dtype=str)
    print(df_mark.shape, df_mark.columns)
    df_mark['ndc_normalized'] = df_mark['NDC_CODE'].apply(
        lambda x: utils.ndc_normalization(x) if pd.notna(x) else np.nan)
    df_mark['source'] = 'epic'

    result_list = []
    # ndc_list = get_ndc_list_from_rxnorm_by_api("213269")
    # info = get_ndc_status_by_api("00069420030")
    col_names = ['Category', 'Term Type', 'NDC', 'NDC Name', 'Code Type', 'status',
                 'rxcui', 'rxcui name', 'start time', 'end time', 'rxcui per ndcHistory', 'ndcHistory', 'datasource', ]

    n_rx_find_ndc_by_api = 0
    n_rx_find_ndc_by_epic = 0
    n_rx_find_additional_ndc_by_epic = 0

    n_ndc_from_api = 0
    n_ndc_additional_from_epic = 0
    n_ndc_no_info = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):
        category = row['Category']
        ttype = row['Term Type']
        rx = row['RxNorm CUI']
        name = row['Medication Name']

        ndc_list = get_ndc_list_from_rxnorm_by_api(rx)
        ndc_epic_df = df_mark.loc[df_mark['RXNORM_CODE'] == rx, :]
        if len(ndc_list) > 0:
            n_rx_find_ndc_by_api += 1
            n_ndc_from_api += len(ndc_list)

        if len(ndc_epic_df) > 0:
            n_rx_find_ndc_by_epic += 1
            print(rx, 'find in epic')
            _b_ = False
            for k, r in ndc_epic_df.iterrows():
                if pd.notna(r['ndc_normalized']) and (r['ndc_normalized'] not in ndc_list):
                    _b_ = True
                    ndc_list[r['ndc_normalized']] = [np.nan, np.nan, 'epic']
                    n_ndc_additional_from_epic += 1
                    print(n_ndc_additional_from_epic, rx, r)
            if _b_:
                n_rx_find_additional_ndc_by_epic += 1

        # ndc list can be enriched by Mark doc
        for key, value in ndc_list.items():
            result = []
            ndc = key
            st = value[0]
            et = value[1]
            source = value[2]
            info = get_ndc_status_by_api(ndc)
            print(rx, ndc, len(info))
            if len(info) == 0:
                n_ndc_no_info += 1
                print(n_ndc_no_info, 'ndc', ndc, 'not found information')
                continue

            result.extend([category, ttype, ndc, info[0], 'NDC'])
            result.extend(info[1:3])
            result.append(name)
            result.append(st)
            result.append(et)
            result.append(info[3])
            result.append(info[4])
            result.append(source)

            result_list.append(result)

    print('scan rx', len(df), '\n',
          "n_rx_find_ndc_by_api", n_rx_find_ndc_by_api, '\n',
          "n_rx_find_ndc_by_epic", n_rx_find_ndc_by_epic, '\n',
          "n_rx_find_additional_ndc_by_epic", n_rx_find_additional_ndc_by_epic, '\n',
          "n_ndc_from_api", n_ndc_from_api, '\n',
          "n_ndc_additional_from_epic", n_ndc_additional_from_epic, '\n',
          "n_ndc_no_info", n_ndc_no_info)

    df_result = pd.DataFrame(result_list, columns=col_names)
    print(df_result.shape)
    df_result.to_excel(
        r'../data/V15_COVID19/output/character/cp_cardio/medication_ndc_list_from_rxnorm_api-ever-direct_and_epic.xlsx')

    return df_result


def analyese_ndc_list_of_rxnorm_in_cardiology_cp_medication():
    fname = r'../data/V15_COVID19/output/character/cp_cardio/RECOVER Cardiology CP Code Lists_v6_8.10.22.xlsx'
    df_rx = pd.read_excel(fname, sheet_name="Medications", engine='openpyxl', dtype=str)
    print(df_rx.columns)

    # fname_ndc = r'../data/V15_COVID19/output/character/cp_cardio/medication_ndc_list_from_rxnorm_presently.xlsx'
    # fname_ndc = r'../data/V15_COVID19/output/character/cp_cardio/medication_ndc_list_from_rxnorm_ever-direct.xlsx'
    fname_ndc = r'../data/V15_COVID19/output/character/cp_cardio/medication_ndc_list_from_rxnorm_api-ever-direct_and_epic.xlsx'

    df_ndc = pd.read_excel(fname_ndc, sheet_name="Sheet1", engine='openpyxl', dtype=str)
    print(df_ndc.columns)

    fname_mark = r'../data/V15_COVID19/output/character/cp_cardio/cards Rxnorm NDC crosswalk.csv'
    df_mark = pd.read_csv(fname_mark, dtype=str)
    print(df_mark.columns)

    rx_in_rx = set(df_rx['RxNorm CUI'])
    rx_in_ndc = []
    for x in df_ndc['rxcui per ndcHistory']:
        vx = x.split(';')
        for item in vx:
            if item not in ['', ';']:
                rx_in_ndc.append(item)

    rx_in_ndc = set(rx_in_ndc)
    rx_in_mark = set(df_mark['RXNORM_CODE'])

    new_rx = list(rx_in_ndc - rx_in_rx)
    pd.DataFrame({'rxnorm': new_rx}).to_csv(
        r'../data/V15_COVID19/output/character/cp_cardio/new_rxnorm_from_ndc_to_rx.csv')

    new_rx_ndcinfo = df_ndc.loc[df_ndc['rxcui'].isin(new_rx), :]
    # new_rx_ndcinfo.to_csv(
    #     r'../data/V15_COVID19/output/character/cp_cardio/new_rxnorm_from_ndc_to_rx_moreInfo.csv')
    new_rx_ndcinfo.to_excel(
        r'../data/V15_COVID19/output/character/cp_cardio/new_rxnorm_from_ndc_to_rx_moreInfo-v2.xlsx')

    print('len(rx_in_rx)', len(rx_in_rx))
    print('len(rx_in_ndc)', len(rx_in_ndc))
    print('len(rx_in_mark)', len(rx_in_mark))

    print('len(rx_in_rx - rx_in_ndc)', len(rx_in_rx - rx_in_ndc))
    print('len(rx_in_ndc - rx_in_rx)', len(rx_in_ndc - rx_in_rx))
    print('len(rx_in_rx - rx_in_mark)', len(rx_in_rx - rx_in_mark))

    print('len(rx_in_rx & rx_in_ndc)', len(rx_in_rx & rx_in_ndc))
    print('len(rx_in_rx & rx_in_mark)', len(rx_in_rx & rx_in_mark))
    (rx_in_rx & rx_in_mark) - (rx_in_rx & rx_in_ndc)
    (rx_in_rx & rx_in_mark) & (rx_in_rx & rx_in_ndc)
    print('')


def new_rx_add_name():
    fname_in = r'../data/V15_COVID19/output/character/cp_cardio/new_rxnorm_from_ndc_to_rx.xlsx'
    df = pd.read_excel(fname_in, sheet_name="Sheet1", engine='openpyxl', dtype=str)
    print(df.columns)
    df['name'] = np.nan
    df['tty'] = np.nan

    for index, row in tqdm(df.iterrows(), total=len(df)):
        rx = row['rxnorm']
        tty, rx_cui, name = get_rx_history_by_api(rx)
        df.loc[index, 'name'] = name
        df.loc[index, 'tty'] = tty

    fname_out = r'../data/V15_COVID19/output/character/cp_cardio/new_rxnorm_from_ndc_to_rx-addName.xlsx'
    df.to_excel(fname_out)
    return df


def get_ndc_from_rxnorm_in_DM_cp_medication(tab):
    fname = r'../data/V15_COVID19/output/character/cp_dm/PASC-CP-Diabetes-Code-Lists-Version-6.6.22.xlsx'
    df = pd.read_excel(fname, sheet_name=tab, engine='openpyxl', dtype=str)
    print('read:', fname, tab)
    print(df.shape)

    result_list = []
    # ndc_list = get_ndc_list_from_rxnorm_by_api("213269")
    # info = get_ndc_status_by_api("00069420030")
    col_names = ['code', 'name', 'tty', 'drug_name', 'drug_class', 'UFL', 'REACHnet LEAD', 'CP Group',
                 'status', 'rxcui', 'rxcui name', 'rxcui per ndcHistory', 'ndcHistory']

    n_rx_find_ndc_by_api = 0
    n_ndc_from_api = 0
    n_ndc_no_info = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):
        rx = row['rxcui']
        name = row['name']
        tty = row['tty']
        name_simple = row['drug_name']
        drug_class = row['drug_class']
        ufl = row['UFL']
        reachnet = row['REACHnet LEAD']
        group = row['CP Group']

        ndc_list = get_ndc_list_from_rxnorm_by_api(rx)
        n_ndc_from_api += len(ndc_list)
        if len(ndc_list) > 0:
            n_rx_find_ndc_by_api += 1

        print(rx, ndc_list.keys(), len(ndc_list))
        # ndc list can be enriched by Mark doc
        for key, value in ndc_list.items():
            result = []
            ndc = key
            st = value[0]
            et = value[1]
            source = value[2]
            info = get_ndc_status_by_api(ndc)
            if len(info) == 0:
                n_ndc_no_info += 1
                print(n_ndc_no_info, 'ndc', ndc, 'not found information')
                continue

            result = [ndc, info[0], tty, name_simple, drug_class, ufl, reachnet, group, info[1], info[2], name, info[3],
                      info[4]]
            result_list.append(result)

    print('scan rx', len(df), '\n',
          "n_rx_find_ndc_by_api", n_rx_find_ndc_by_api, '\n',
          "n_ndc_from_api", n_ndc_from_api, '\n',
          "n_ndc_no_info", n_ndc_no_info)

    df_result = pd.DataFrame(result_list, columns=col_names)
    print(df_result.shape)
    df_result.to_excel(
        r'../data/V15_COVID19/output/character/cp_dm/ndc-crosswalk-tab-{}-raw.xlsx'.format(tab.replace('|', '-')))

    return df_result


def get_ndc_from_rxnorm_in_pulmonary_cp_medication():
    fname = r'../data/V15_COVID19/output/character/cp_pulmonary/rxnorm_to_crosswalk.xlsx'
    df = pd.read_excel(fname, sheet_name='Sheet1', engine='openpyxl', dtype=str)
    print('read:', fname)
    print(df.shape)
    print(df.columns)
    # Index(['Unnamed: 0', 'Class', 'Category', 'Code List', 'termType', 'rxcui',
    #        'name', 'isHumanDrug', 'isVetDrug', 'isPrescribable', 'Delete?',
    #        'Inclusion Parameters', 'ndc_code', 'RXNORM_CODE'],
    #       dtype='object')

    result_list = []
    # ndc_list = get_ndc_list_from_rxnorm_by_api("213269")
    # info = get_ndc_status_by_api("00069420030")
    col_names = ['code', 'name', 'tty', 'drug_name', 'drug_class', 'UFL', 'REACHnet LEAD', 'CP Group',
                 'status', 'rxcui', 'rxcui name', 'rxcui per ndcHistory', 'ndcHistory']

    n_rx_find_ndc_by_api = 0
    n_ndc_from_api = 0
    n_ndc_no_info = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):
        mclass = row['Class']
        cat = row['Category']
        codelist = row['Code List']
        tty = row['termType']
        rx = row['rxcui']
        name = row['name']
        ishuman = row['isHumanDrug']
        isVet = row['isVetDrug']
        isPre = row['isPrescribable']
        isDel = row['Delete?']
        inc = row['Inclusion Parameters']
        ndc = row['ndc_code']
        rx_code = row['RXNORM_CODE']

        ndc_list = get_ndc_list_from_rxnorm_by_api(rx)
        n_ndc_from_api += len(ndc_list)
        if len(ndc_list) > 0:
            n_rx_find_ndc_by_api += 1

        print(rx, ndc_list.keys(), len(ndc_list))
        # ndc list can be enriched by Mark doc
        for key, value in ndc_list.items():
            result = []
            ndc = key
            st = value[0]
            et = value[1]
            source = value[2]
            info = get_ndc_status_by_api(ndc)
            if len(info) == 0:
                n_ndc_no_info += 1
                print(n_ndc_no_info, 'ndc', ndc, 'not found information')
                continue

            result = row.copy(deep=True)
            result['ndc_code'] = ndc
            result['name'] = info[0]
            result['RXNORM_CODE'] = info[2]

            result_list.append(result)

    print('scan rx', len(df), '\n',
          "n_rx_find_ndc_by_api", n_rx_find_ndc_by_api, '\n',
          "n_ndc_from_api", n_ndc_from_api, '\n',
          "n_ndc_no_info", n_ndc_no_info)

    df_result = pd.DataFrame(result_list)
    print(df_result.shape)
    df_result.to_excel(r'../data/V15_COVID19/output/character/cp_pulmonary/rxnorm_to_crosswalk_ndc.xlsx')

    return df_result


def compare_t2dm_cdc_recover_med():
    fcdc = r'../data/V15_COVID19/output/character/cp_dm/Diabetes Code Lists.xlsx'
    df1 = pd.read_excel(fcdc, sheet_name='diab_meds', engine='openpyxl', dtype=str)
    df2 = pd.read_excel(fcdc, sheet_name='metformin_sglt2', engine='openpyxl', dtype=str)

    df_1m2 = df1.loc[df1['code1'].apply(lambda x: x not in df2['code1']), :]
    df_1m2 = df1.loc[~df1['code1'].isin(df2['code1']), :]

    a = df_1m2['descrip'].value_counts()
    b = df_1m2['descrip.1'].value_counts()

    fr = r'../data/V15_COVID19/output/character/cp_dm/PASC-CP-Diabetes-Code-Lists-Version-8.22.22 -NDC-crosswalk.xlsx'
    df3 = pd.read_excel(fr, sheet_name='5f. Oral Hypo No Met|Phen|SGLT2', engine='openpyxl', dtype=str)
    df4 = pd.read_excel(fr, sheet_name='5g. Oral Hy Met|Phen|SGLT2 Only', engine='openpyxl', dtype=str)

    c = df3['CP Group'].value_counts()
    c2 = df3['name'].value_counts()
    d = df4['CP Group'].value_counts()

    a.to_csv('../data/V15_COVID19/output/character/cp_dm/cdc_t2dm_med.csv')
    c2.to_csv('../data/V15_COVID19/output/character/cp_dm/recover_t2dm_med-5f.csv')

    print()
    return df1, df2, df_1m2


# 2023-08-31 for adrd drugs
def get_rx_from_namestr(drugname):
    """
    https://lhncbc.nlm.nih.gov/RxNav/APIs/api-RxNorm.getDrugs.html
    Get the drug products associated with a specified name. The name can be an ingredient, brand name,
    clinical dose form, branded dose form, clinical drug component, or branded drug component.
    The following table shows examples of input and the types of drug products returned.
    See default paths for the paths traveled to get concepts for each term type.
    https://rxnav.nlm.nih.gov/REST/drugs.json?name=donepezil

    """
    print('In get_rx_from_namestr({})'.format(drugname), 'Get the drug products associated with a specified name.')

    url_str = "https://rxnav.nlm.nih.gov/REST/drugs.json?name={}".format(drugname)
    r = requests.get(url_str)
    data = r.json()

    rlist = []
    if ("drugGroup" in data) and ('conceptGroup' in data['drugGroup']):
        data = data['drugGroup']['conceptGroup']
        for con in data:
            # tty = con['tty']
            if 'conceptProperties' in con:
                for c2 in con['conceptProperties']:
                    rx = c2['rxcui']
                    name = c2['name']
                    synonym = c2['synonym']
                    tty = c2['tty']
                    rlist.append((rx, 'rxcui', name, synonym, tty, tty_descript(tty), url_str))

    df = pd.DataFrame(rlist, columns=['code', 'code type', 'name', 'synonym', 'tty', 'tty_descript', 'query source'])
    return df


def find_rx_by_str(drugname):
    """
    https://lhncbc.nlm.nih.gov/RxNav/APIs/api-RxNorm.findRxcuiByString.html
    Searches for a name (from any vocabulary in RxNorm) and returns the RxCUIs associated with the name.
    The scope of the search is either Active concepts (allsrc=0) or Current concepts (allsrc=1).
    When the scope is Current concepts, only concepts that have at least one atom from a source in the srclist
    parameter are returned.
    The precision of the search is adjustable with the search parameter.
    An exact search ignores only upper/lowercase. A normalized search is less strict.
    A combination search tries an exact search, then a normalized search if nothing was found.
    https://rxnav.nlm.nih.gov/REST/rxcui.json?name=metoprolol&search=1

    :param drugname:
    :return:
    """
    print('In find_rx_by_str({})'.format(drugname),
          'Searches for a name (from any vocabulary in RxNorm) and returns the RxCUIs associated with the name.')
    url_str = "https://rxnav.nlm.nih.gov/REST/rxcui.json?name={}&search=1".format(drugname)
    r = requests.get(url_str)
    data = r.json()

    rlist = []
    if ("idGroup" in data) and ('rxnormId' in data['idGroup']):
        data = data['idGroup']['rxnormId']
        for con in data:
            rlist.append(con)

    print('len(rlist)', len(rlist), rlist)
    return rlist


def get_allrelated_from_rx(drugrx):
    """
    https://lhncbc.nlm.nih.gov/RxNav/APIs/api-RxNorm.getAllRelatedInfo.html
    Get RxNorm concepts related to the concept identified by rxcui.
    Related concepts may be of term types "IN", "MIN", "PIN", "BN", "SBD", "SBDC", "SBDF",
    "SBDG", "SCD", "SCDC", "SCDF", "SCDG", "DF", "DFG", "BPCK" and "GPCK".
    See default paths for the RxNorm relationship paths traveled to get concepts for each term type.
    The results include the concept identified by rxcui as well as related concepts.
    The GENERAL_CARDINALITY property of each concept is returned (in the genCard element) if requested with the expand parameter.
    https://rxnav.nlm.nih.gov/REST/drugs.json?name=donepezil
    """
    url_str = "https://rxnav.nlm.nih.gov/REST/rxcui/{}/allrelated.json".format(drugrx)
    r = requests.get(url_str)
    data = r.json()

    rlist = []
    if ("allRelatedGroup" in data) and ('conceptGroup' in data['allRelatedGroup']):
        data = data['allRelatedGroup']['conceptGroup']
        for con in data:
            # tty = con['tty']
            if 'conceptProperties' in con:
                for c2 in con['conceptProperties']:
                    rx = c2['rxcui']
                    name = c2['name']
                    synonym = c2['synonym']
                    tty = c2['tty']
                    rlist.append((rx, 'rxcui', name, synonym, tty, tty_descript(tty), url_str))

    df = pd.DataFrame(rlist, columns=['code', 'code type', 'name', 'synonym', 'tty', 'tty_descript', 'query source'])
    return df


def get_ndc_from_rxnorm(rxlist):
    print('len(rxlist):', len(rxlist), rxlist)

    n_rx_find_ndc_by_api = 0
    n_ndc_from_api = 0
    n_ndc_no_info = 0

    result_list = []
    col_names = ['code', 'code type', 'name', 'status', 'rxcui', 'rxcui per ndcHistory', 'ndcHistory', 'query source']

    for rx in tqdm(rxlist, total=len(rxlist)):
        ndc_list = get_ndc_list_from_rxnorm_by_api(rx)
        sourcequery = 'https://rxnav.nlm.nih.gov/REST/rxcui/{}/allhistoricalndcs.json?history=1'.format(rx)
        n_ndc_from_api += len(ndc_list)
        if len(ndc_list) > 0:
            n_rx_find_ndc_by_api += 1

        print(rx, len(ndc_list), ndc_list.keys())
        # ndc list can be enriched by Mark doc
        for key, value in ndc_list.items():
            result = []
            ndc = key
            st = value[0]
            et = value[1]
            source = value[2]
            info = get_ndc_status_by_api(ndc)
            if len(info) == 0:
                n_ndc_no_info += 1
                print(n_ndc_no_info, 'ndc', ndc, 'not found information')
                continue

            result = [ndc, 'ndc11', info[0], info[1], info[2], info[3], info[4], sourcequery]
            result_list.append(result)

    print('scan rx', len(rxlist), '\n',
          "n_rx_find_ndc_by_api", n_rx_find_ndc_by_api, '\n',
          "n_ndc_from_api", n_ndc_from_api, '\n',
          "n_ndc_no_info", n_ndc_no_info)

    df_result = pd.DataFrame(result_list, columns=col_names, dtype=str)
    print(df_result.shape)
    return df_result


def generate_demential_drug_list():
    ## 1. donepezil
    # codes for denopezil
    df = get_rx_from_namestr("donepezil")
    df.to_csv('../prehf/output/denepezil.csv')
    df2 = get_allrelated_from_rx("135446")
    df2.to_csv('../prehf/output/denepezil-135446.csv')
    df3 = get_allrelated_from_rx("236559")
    df3.to_csv('../prehf/output/denepezil-236559.csv')
    # df4 = get_allrelated_from_rx("1602583")
    # df4.to_csv('../prehf/output/denepezil-1602583.csv')
    df_rx = pd.concat([df, df2, df3], ignore_index=True, sort=False)
    df_rx_nodup = df_rx.drop_duplicates(['code', 'code type', "name", "synonym", "tty"])
    df_rx_nodup.to_csv('../prehf/output/denepezil-combined-nodup-rxcui.csv')

    df_ndc = get_ndc_from_rxnorm(df_rx_nodup['code'])
    df_ndc_nodup = df_ndc.drop_duplicates(['code', 'code type', "name", ])
    print('df_ndc.shape', df_ndc.shape, 'df_ndc_nodup.shape', df_ndc_nodup.shape, )
    df_ndc_nodup.to_csv('../prehf/output/denepezil-combined-nodup-NDC.csv', )

    df_merge = df_rx_nodup.merge(df_ndc_nodup, on=['code', 'code type', 'name', 'query source'], how='outer')
    df_merge['drug'] = 'denepezil'
    df_merge.to_excel('../prehf/output/denepezil-ndc-rxnom-merged.xlsx', )

    ## codes for tacrine
    df = get_rx_from_namestr("tacrine")
    df.to_csv('../prehf/output/tacrine.csv')
    df2 = get_allrelated_from_rx("10318")
    df2.to_csv('../prehf/output/tacrine-10318.csv')
    df3 = get_allrelated_from_rx("235972")
    df3.to_csv('../prehf/output/tacrine-235972.csv')

    df_rx = pd.concat([df, df2, df3], ignore_index=True, sort=False)
    df_rx_nodup = df_rx.drop_duplicates(['code', 'code type', "name", "synonym", "tty"])
    df_rx_nodup.to_csv('../prehf/output/tacrine-combined-nodup-rxcui.csv')

    df_ndc = get_ndc_from_rxnorm(df_rx_nodup['code'])
    df_ndc_nodup = df_ndc.drop_duplicates(['code', 'code type', "name", ])
    print('df_ndc.shape', df_ndc.shape, 'df_ndc_nodup.shape', df_ndc_nodup.shape, )
    df_ndc_nodup.to_csv('../prehf/output/tacrine-combined-nodup-NDC.csv', )

    df_merge = df_rx_nodup.merge(df_ndc_nodup, on=['code', 'code type', 'name', 'query source'], how='outer')
    df_merge['drug'] = 'tacrine'
    df_merge.to_excel('../prehf/output/tacrine-ndc-rxnom-merged.xlsx', )

    ## code for rivastigmine
    df = get_rx_from_namestr("rivastigmine")
    df.to_csv('../prehf/output/rivastigmine.csv')
    df2 = get_allrelated_from_rx("183379")
    df2.to_csv('../prehf/output/rivastigmine-183379.csv')

    df_rx = pd.concat([df, df2, ], ignore_index=True, sort=False)
    df_rx_nodup = df_rx.drop_duplicates(['code', 'code type', "name", "synonym", "tty"])
    df_rx_nodup.to_csv('../prehf/output/rivastigmine-combined-nodup-rxcui.csv')

    df_ndc = get_ndc_from_rxnorm(df_rx_nodup['code'])
    df_ndc_nodup = df_ndc.drop_duplicates(['code', 'code type', "name", ])
    print('df_ndc.shape', df_ndc.shape, 'df_ndc_nodup.shape', df_ndc_nodup.shape, )
    df_ndc_nodup.to_csv('../prehf/output/rivastigmine-combined-nodup-NDC.csv', )

    df_merge = df_rx_nodup.merge(df_ndc_nodup, on=['code', 'code type', 'name', 'query source'], how='outer')
    df_merge['drug'] = 'rivastigmine'
    df_merge.to_excel('../prehf/output/rivastigmine-ndc-rxnom-merged.xlsx', )

    ## code for galantamine
    df = get_rx_from_namestr("galantamine")
    df.to_csv('../prehf/output/galantamine.csv')
    df2 = get_allrelated_from_rx("4637")
    df2.to_csv('../prehf/output/galantamine-4637.csv')

    df_rx = pd.concat([df, df2, ], ignore_index=True, sort=False)
    df_rx_nodup = df_rx.drop_duplicates(['code', 'code type', "name", "synonym", "tty"])
    df_rx_nodup.to_csv('../prehf/output/galantamine-combined-nodup-rxcui.csv')

    df_ndc = get_ndc_from_rxnorm(df_rx_nodup['code'])
    df_ndc_nodup = df_ndc.drop_duplicates(['code', 'code type', "name", ])
    print('df_ndc.shape', df_ndc.shape, 'df_ndc_nodup.shape', df_ndc_nodup.shape, )
    df_ndc_nodup.to_csv('../prehf/output/galantamine-combined-nodup-NDC.csv', )

    df_merge = df_rx_nodup.merge(df_ndc_nodup, on=['code', 'code type', 'name', 'query source'], how='outer')
    df_merge['drug'] = 'galantamine'
    df_merge.to_excel('../prehf/output/galantamine-ndc-rxnom-merged.xlsx', )

    # code for memantine
    df = get_rx_from_namestr("memantine")
    df.to_csv('../prehf/output/memantine.csv')
    df2 = get_allrelated_from_rx("6719")
    df2.to_csv('../prehf/output/memantine-6719.csv')
    df3 = get_allrelated_from_rx("236685")
    df3.to_csv('../prehf/output/memantine-236685.csv')

    df_rx = pd.concat([df, df2, df3], ignore_index=True, sort=False)
    df_rx_nodup = df_rx.drop_duplicates(['code', 'code type', "name", "synonym", "tty"])
    df_rx_nodup.to_csv('../prehf/output/memantine-combined-nodup-rxcui.csv')

    df_ndc = get_ndc_from_rxnorm(df_rx_nodup['code'])
    df_ndc_nodup = df_ndc.drop_duplicates(['code', 'code type', "name", ])
    print('df_ndc.shape', df_ndc.shape, 'df_ndc_nodup.shape', df_ndc_nodup.shape, )
    df_ndc_nodup.to_csv('../prehf/output/memantine-combined-nodup-NDC.csv', )

    df_merge = df_rx_nodup.merge(df_ndc_nodup, on=['code', 'code type', 'name', 'query source'], how='outer')
    df_merge['drug'] = 'memantine'
    df_merge.to_excel('../prehf/output/memantine-ndc-rxnom-merged.xlsx', )

    # combine codes
    df1 = pd.read_excel('../prehf/output/denepezil-ndc-rxnom-merged.xlsx', dtype=str)
    df2 = pd.read_excel('../prehf/output/memantine-ndc-rxnom-merged.xlsx', dtype=str)
    df3 = pd.read_excel('../prehf/output/tacrine-ndc-rxnom-merged.xlsx', dtype=str)
    df4 = pd.read_excel('../prehf/output/rivastigmine-ndc-rxnom-merged.xlsx', dtype=str)
    df5 = pd.read_excel('../prehf/output/galantamine-ndc-rxnom-merged.xlsx', dtype=str)
    df_merge = pd.concat([df1, df2, df3, df4, df5], ignore_index=True, sort=False)
    df_merge_nodup = df_merge.drop_duplicates(['code', 'code type', "name", ])

    df_merge_nodup.to_excel('../prehf/output/dementia-drug-merged.xlsx', )


def generate_covid_drug_list():
    pass
    df = get_rx_from_namestr("paxlovid")
    df.to_csv('../prerecover/output/paxlovid.csv')
    df2 = get_allrelated_from_rx("2599543")
    df2.to_csv('../prerecover/output/paxlovid-2599543.csv')
    df3 = get_allrelated_from_rx("2587899")
    df3.to_csv('../prerecover/output/paxlovid-2587899.csv')
    df_rx = pd.concat([df, df2, df3], ignore_index=True, sort=False)
    df_rx_nodup = df_rx.drop_duplicates(['code', 'code type', "name", "synonym", "tty"])
    df_rx_nodup.to_csv('../prerecover/output/paxlovid-combined-nodup-rxcui.csv')

    df_ndc = get_ndc_from_rxnorm(df_rx_nodup['code'])
    df_ndc_nodup = df_ndc.drop_duplicates(['code', 'code type', "name", ])
    print('df_ndc.shape', df_ndc.shape, 'df_ndc_nodup.shape', df_ndc_nodup.shape, )
    df_ndc_nodup.to_csv('../prerecover/output/paxlovid-combined-nodup-NDC.csv', )

    df_merge = df_rx_nodup.merge(df_ndc_nodup, on=['code', 'code type', 'name', 'query source'], how='outer')
    df_merge['drug'] = 'paxlovid'
    df_merge.to_excel('../prerecover/output/paxlovid-ndc-rxnom-merged.xlsx', )

    ##
    df = get_rx_from_namestr("remdesivir")
    df.to_csv('../prerecover/output/remdesivir.csv')
    df2 = get_allrelated_from_rx("2284718")
    df2.to_csv('../prerecover/output/remdesivir-2284718.csv')

    df_rx = pd.concat([df, df2, ], ignore_index=True, sort=False)
    df_rx_nodup = df_rx.drop_duplicates(['code', 'code type', "name", "synonym", "tty"])
    df_rx_nodup.to_csv('../prerecover/output/remdesivir-combined-nodup-rxcui.csv')

    df_ndc = get_ndc_from_rxnorm(df_rx_nodup['code'])
    df_ndc_nodup = df_ndc.drop_duplicates(['code', 'code type', "name", ])
    print('df_ndc.shape', df_ndc.shape, 'df_ndc_nodup.shape', df_ndc_nodup.shape, )
    df_ndc_nodup.to_csv('../prerecover/output/remdesivir-combined-nodup-NDC.csv', )

    df_merge = df_rx_nodup.merge(df_ndc_nodup, on=['code', 'code type', 'name', 'query source'], how='outer')
    df_merge['drug'] = 'remdesivir'
    df_merge.to_excel('../prerecover/output/remdesivir-ndc-rxnom-merged.xlsx', )


def drug_code_list_generation_by_name(drugname: str, output_dir):
    # step 1
    df = get_rx_from_namestr(drugname)
    # step 2
    rlist = find_rx_by_str(drugname)
    # step 3
    df_list = [df, ]
    for rx in rlist:
        df_temp = get_allrelated_from_rx(rx)
        df_list.append(df_temp)

    df_rxcombined = pd.concat(df_list, ignore_index=True, sort=False)
    # df_rxcombined.to_csv(output_dir + '{}-combined-rxcui.csv'.format(drugname))

    df_rxcombined_nodup = df_rxcombined.drop_duplicates(['code', 'code type', "name", "synonym", "tty"])
    print('Rxnorm: df_rxcombined.shape', df_rxcombined.shape,
          'df_rxcombined_nodup.shape', df_rxcombined_nodup.shape, )
    df_rxcombined_nodup.to_csv(output_dir + '{}-combined-dedup-rxcui.csv'.format(drugname))

    # generate NDC
    df_ndc = get_ndc_from_rxnorm(df_rxcombined_nodup['code'])
    df_ndc_nodup = df_ndc.drop_duplicates(['code', 'code type', "name", ])
    print('NDC: df_ndc.shape', df_ndc.shape, 'df_ndc_nodup.shape', df_ndc_nodup.shape, )
    df_ndc_nodup.to_csv(output_dir + '{}-combined-dedup-NDC.csv'.format(drugname))
    #
    df_merge = df_rxcombined_nodup.merge(df_ndc_nodup, on=['code', 'code type', 'name', 'query source'], how='outer')
    df_merge['drug'] = drugname
    df_merge.to_excel(output_dir + '{}-ndc-rxnom-merged.xlsx'.format(drugname))

    print('df_merge.shape', df_merge.shape, )

    # Warning! Need human edit!
    # then, need to scan name, and delet some not related ones
    # rxrnom: check DF rxnorm, delete not relevant code, --> deleted DF, DFG
    # ndc: check name, delete drugs with different names. due to drug combination, expanding information of aux. single ingredient drug no this issue
    return df_merge, df_rxcombined_nodup, df_ndc_nodup


if __name__ == '__main__':
    start_time = time.time()
    ## 2023-11-4
    # generate HF drug list
    # df = get_rx_from_namestr("metoprolol")
    # df.to_csv('../prehf/output/HF_drug/metoprolol.csv')

    # rlist1 = find_rx_by_str("metoprolol")
    # rlist2 = find_rx_by_str("metoprolol tartrate")
    # rlist3 = find_rx_by_str("metoprolol succinate")
    # rlist4 = find_rx_by_str("lopressor")
    # rlist5 = find_rx_by_str("toprol")
    drugname = 'timolol' #'sotalol' #'propranolol'  # 'pindolol'
    # 'penbutolol' #'nebivolol' #'nadolol' #'labetalol' #'carteolol' #'betaxolol' #'atenolol' #'acebutolol' #'ertugliflozin' #'canagliflozin' #'bexagliflozin'  #'dapagliflozin' #'empagliflozin'  # 'sacubitril-valsartan'  # 'carvedilol ' # 'bisoprolol'  # "metoprolol"
    df_merge, df_rxcombined_nodup, df_ndc_nodup = drug_code_list_generation_by_name(drugname,
                                                                                    output_dir='../prehf/output/HF_drug/')

    # df2 = get_allrelated_from_rx("135446")
    # df2.to_csv('../prehf/output/denepezil-135446.csv')
    # df3 = get_allrelated_from_rx("236559")
    # df3.to_csv('../prehf/output/denepezil-236559.csv')
    # # df4 = get_allrelated_from_rx("1602583")
    # # df4.to_csv('../prehf/output/denepezil-1602583.csv')
    # df_rx = pd.concat([df, df2, df3], ignore_index=True, sort=False)
    # df_rx_nodup = df_rx.drop_duplicates(['code', 'code type', "name", "synonym", "tty"])
    # df_rx_nodup.to_csv('../prehf/output/denepezil-combined-nodup-rxcui.csv')
    #
    # df_ndc = get_ndc_from_rxnorm(df_rx_nodup['code'])
    # df_ndc_nodup = df_ndc.drop_duplicates(['code', 'code type', "name", ])
    # print('df_ndc.shape', df_ndc.shape, 'df_ndc_nodup.shape', df_ndc_nodup.shape, )
    # df_ndc_nodup.to_csv('../prehf/output/denepezil-combined-nodup-NDC.csv', )
    #
    # df_merge = df_rx_nodup.merge(df_ndc_nodup, on=['code', 'code type', 'name', 'query source'], how='outer')
    # df_merge['drug'] = 'denepezil'
    # df_merge.to_excel('../prehf/output/denepezil-ndc-rxnom-merged.xlsx', )

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
