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


if __name__ == '__main__':
    # python pre_codemapping.py 2>&1 | tee  log/pre_codemapping_zip_adi.txt
    start_time = time.time()

    # ndc_list = get_ndc_list_from_rxnorm_by_api("213269")
    # info = get_ndc_status_by_api("00069420030")
    # df_result = get_ndc_list_of_rxnorm_in_cardiology_cp_medication()
    # analyese_ndc_list_of_rxnorm_in_cardiology_cp_medication()
    # r = get_rx_history_by_api('93001')
    # df = new_rx_add_name()

    # df_5 = get_ndc_from_rxnorm_in_DM_cp_medication(tab='5. RxNorm CUI All')
    # df_a = get_ndc_from_rxnorm_in_DM_cp_medication(tab='5a. Insulin')
    # df_b = get_ndc_from_rxnorm_in_DM_cp_medication(tab='5b. GLP-1')
    # df_c = get_ndc_from_rxnorm_in_DM_cp_medication(tab='5c. Insulin|GLP-1')
    # df_d = get_ndc_from_rxnorm_in_DM_cp_medication(tab='5d. Pramlintide')
    # df_e = get_ndc_from_rxnorm_in_DM_cp_medication(tab='5e. All Oral Hypoglycemic')
    # df_f = get_ndc_from_rxnorm_in_DM_cp_medication(tab='5f. Oral Hypo No Met|Phen|SGLT2')
    # df_g = get_ndc_from_rxnorm_in_DM_cp_medication(tab='5g. Oral Hy Met|Phen|SGLT2 Only')

    # get_ndc_from_rxnorm_in_pulmonary_cp_medication()
    # df1, df2, df_1m2 = compare_t2dm_cdc_recover_med()

    # df = pd.read_csv(
    #     r'../data/V15_COVID19/output/character/cp_dm/matrix_cohorts_covid_4manuNegNoCovidV2_bool_ALL-anyPASC_diabetes_incidence-Sep2.csv',
    #     dtype={'patid': str, 'site': str, 'zip': str}, parse_dates=['index date'])
    #
    # cols = pd.read_csv(r'../data/V15_COVID19/output/character/cp_dm/select_cols_4_patient_list.csv')
    # cols_name = cols['name']
    # # df.loc[:, cols_name].to_excel(
    # #     r'../data/V15_COVID19/output/character/cp_dm/diabetes_incidence-Sep2.xlsx')
    #
    # df.loc[df['flag_diabetes']==1, cols_name].to_excel(
    #     r'../data/V15_COVID19/output/character/cp_dm/diabetes_incidence_cases-Sep2.xlsx')

    # df['N'] = 1
    #
    # print(df['site'].value_counts())
    # # pd.DataFrame(df.columns).to_csv(r'../data/V15_COVID19/output/character/cp_dm/select_cols.csv')
    # cols = pd.read_csv(r'../data/V15_COVID19/output/character/cp_dm/select_cols.csv')
    # cols_name = cols['name']
    #
    # print('df.shape:', df.shape)
    #
    # result = {}
    # for site in ['NYU', 'MSHS', 'MONTE', 'WCM', 'COL']:
    #     print('In site:', site)
    #     df_data = df.loc[df['site'] == site, cols_name]
    #     print('df_data.shape:', df_data.shape)
    #
    #     df_pos = df_data.loc[df_data["covid"] == 1, cols_name]
    #     df_neg = df_data.loc[df_data["covid"] == 0, cols_name]
    #
    #     print('df_pos.shape:', df_pos.shape)
    #     print('df_neg.shape:', df_neg.shape)
    #
    #
    #     def smd(m1, m2, v1, v2):
    #         VAR = np.sqrt((v1 + v2) / 2)
    #         smd = np.divide(
    #             m1 - m2,
    #             VAR, out=np.zeros_like(m1), where=VAR != 0)
    #         return smd
    #
    #
    #     result.update({'{}-Overall'.format(site): df_data.sum(),
    #                    '{}-Overall-mean'.format(site): df_data.mean(),
    #                    '{}-df_pos'.format(site): df_pos.sum(),
    #                    '{}-df_pos-mean'.format(site): df_pos.mean(),
    #                    '{}-df_neg'.format(site): df_neg.sum(),
    #                    '{}-df_neg-mean'.format(site): df_neg.mean(),
    #                    # 'smd': smd(df_pos.mean(), df_neg.mean(), df_pos.var(), df_neg.var())
    #                    })
    #
    # df_result = pd.DataFrame(result)
    # df_result.to_csv(
    #     r'../data/V15_COVID19/output/character/cp_dm/table_DM-All.csv')

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))