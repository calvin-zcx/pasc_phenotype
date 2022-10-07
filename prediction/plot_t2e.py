import os
import shutil
import zipfile

import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib as mpl
import time

import numpy as np
import csv
from collections import Counter, defaultdict
import pandas as pd
from misc.utils import check_and_mkdir, stringlist_2_str, stringlist_2_list
from scipy import stats
import re
import itertools
import functools
import random
import seaborn as sns

print = functools.partial(print, flush=True)
import zepid
from zepid.graphics import EffectMeasurePlot
import shlex

np.random.seed(0)
random.seed(0)
from misc import utils
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Rectangle
from html4vision import Col, imagetable

import os
import glob


def plot_image_html(df_pasc_info, folder='narrow'):
    html = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    * {
      box-sizing: border-box;
    }
    
    .column {
      float: left;
      width: 23%;
      padding: 5px;
    }
    
    /* Clearfix (clear floats) */
    .row::after {
      content: "";
      clear: both;
      display: table;
    }
    </style>
    </head>
    <body>
    
    <h2>Time-2-event distributions of PASC conditions in SARS-CoV-2 Infected Patients during 30 - 180 days after infection</h2>

            <div class="row">
            <div class="column">
            <p  align="left"> <b> Stratified by the acute status  </b></p>
            </div>
            <div class="column">
            <p align="center"> <b> Overall </b></p>
            </div>
            <div class="column">
            <p align="center"> <b> Not Hospitalized </b></p>
            </div>
            <div class="column">
            <p align="center"> <b> Hospitalized </b></p>
            </div>
    """
    a = [x.replace('output/t2e_figure/', '') for x in glob.glob("output/t2e_figure/{}/*.png".format(folder))]
    a = sorted(a, reverse=True)

    a_revised = []
    for i in range(0, len(a), 3):
        # html += f"<center><img src='{file}'/ height=30%></center><br>"
        pasc = a[i].split('\\')[-1].replace('-Overall.png', '')
        row = df_pasc_info.loc[pasc, :]

        hr = row['hr-w']
        ci = stringlist_2_list(row['hr-w-CI'])
        p = row['hr-w-p']
        domain = row['Organ Domain']

        ncum1 = stringlist_2_list(row['cif_1_w'])[-1] * 1000
        ncum_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1] * 1000,
                   stringlist_2_list(row['cif_1_w_CIUpper'])[-1] * 1000]
        cifdif = stringlist_2_list(row['cif-w-diff'])[-1] * 1000

        a_revised.append((a[i], a[i + 1], a[i + 2], ncum1))
        # a_revised.append((a[i], a[i+1], a[i+2], cifdif))

    a_revised = sorted(a_revised, reverse=True, key=lambda tup: tup[3])

    for i in range(len(a_revised)):
        # html += f"<center><img src='{file}'/ height=30%></center><br>"
        pasc = a_revised[i][0].split('\\')[-1].replace('-Overall.png', '')

        row = df_pasc_info.loc[pasc, :]

        hr = row['hr-w']
        ci = stringlist_2_list(row['hr-w-CI'])
        p = row['hr-w-p']
        domain = row['Organ Domain']

        ncum1 = stringlist_2_list(row['cif_1_w'])[-1] * 1000
        ncum_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1] * 1000,
                   stringlist_2_list(row['cif_1_w_CIUpper'])[-1] * 1000]

        # reuse- nabs for ci in neg
        ncum0 = stringlist_2_list(row['cif_0_w'])[-1] * 1000

        cifdif = stringlist_2_list(row['cif-w-diff'])[-1] * 1000

        html += """
            <div class="row">
            <div class="column">
            <p> {} - {}</p>
            <p> <b> {} </b></p>
            <p> Organ: {} </p>
            <p> aHR: {:.2f} ({:.2f}, {:.2f}), Pvalue: {:.1g} </p>
            <p> CIF+: {:.1f}, CIF-: {:.1f} (Per 1,000)</p>
            <p> Excess burden: {:.1f}   (Per 1,000)</p>
          </div>
          <div class="column">
            <img src="{}" alt="Overall" style="width:100%">
          </div>
          <div class="column">
            <img src="{}" alt="Inpatient" style="width:100%">
          </div>
          <div class="column">
            <img src="{}" alt="Outpatient" style="width:100%">
          </div>
        </div>
        """.format(i + 1, a_revised[i][0].replace('-Overall', '-*'),
                   pasc, domain,
                   hr, ci[0], ci[1], p,
                   ncum1, ncum0, cifdif,
                   a_revised[i][0], a_revised[i][1], a_revised[i][2])

    html += "</body></html>"

    # outfile = "output/t2e_figure/{}-index_excess_sorted.html".format(folder)
    outfile = "output/t2e_figure/{}-index.html".format(folder)

    with open(outfile, "w") as outputfile:
        outputfile.write(html)

    # os.startfile(outfile)


def fig_plot_t2e(t2e, title, outfile):
    ax = plt.subplot(111)
    sns.histplot(
        t2e,
        stat="proportion", common_norm=False, bins=30, kde=True
    )
    plt.title(title)
    plt.xlim(left=30, right=180)
    plt.savefig(outfile)
    # plt.show()
    plt.close()


def plot_anypasc_t2e(pasc_type='1dx', pasc_n_type='broad'):
    if pasc_type == '1dx':
        data_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_ALL.csv'
    elif pasc_type == '2dx30days':
        data_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_2dx30daysAnyPASC_ALL.csv'
    elif pasc_type == '2dx1days':
        data_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_2dx1daysAnyPASC_ALL.csv'
    else:
        raise ValueError

    df = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str},
                     parse_dates=['index date'])  # , nrows=100
    # df_pasc_info = pd.read_excel('output/causal_effects_specific_withMedication_v3.xlsx', sheet_name='diagnosis')
    df_pasc_info = pd.read_excel('output/causal_effects_specific_dx_insight-MultiPval-DXMEDALL_withOldSelection.xlsx',
                                 sheet_name='dx')

    name_map = {}
    for index, rows in df_pasc_info.iterrows():
        name_map[rows['pasc']] = rows['PASC Name Simple']

    selected_pasc_list = df_pasc_info.loc[df_pasc_info['selected_old'] == 1, 'pasc']
    print('len(selected_pasc_list)', len(selected_pasc_list))
    print(selected_pasc_list)

    selected_pasc_list_narrow = df_pasc_info.loc[df_pasc_info['selected_narrow_old'] == 1, 'pasc']
    print('len(selected_pasc_list_narrow)', len(selected_pasc_list_narrow))

    exclude_DX_list = {
        'Neurocognitive disorders': ['DX: Dementia'],
        'Diabetes mellitus with complication': ['DX: Diabetes Type 2'],
        'Chronic obstructive pulmonary disease and bronchiectasis': ['DX: Chronic Pulmonary Disorders', 'DX: COPD'],
        'Circulatory signs and symptoms': ['DX: Arrythmia'],
        'Anemia': ['DX: Anemia'],
        'Heart failure': ["DX: Congestive Heart Failure"]
    }

    print('Labeling INCIDENT pasc in {0,1}')
    # flag@pascname  for incidence label, dx-t2e@pascname for original shared t2e
    for pasc in selected_pasc_list:
        flag = df['dx-out@' + pasc] - df['dx-base@' + pasc]
        # if pasc in exclude_DX_list:
        #     ex_DX_list = exclude_DX_list[pasc]
        #     print(pasc, 'further exclude', ex_DX_list)
        #     for ex_DX in ex_DX_list:
        #         flag -= df[ex_DX]

        df['flag@' + pasc] = (flag > 0).astype('int')

    list_t2e_all = []
    list_t2e_outpatient = []
    list_t2e_inpatient = []

    if pasc_n_type == 'narrow':
        pasc_list = selected_pasc_list_narrow
    elif pasc_n_type == 'broad':
        pasc_list = selected_pasc_list
    else:
        raise ValueError

    for ith, pasc in enumerate(pasc_list):
        print(ith, pasc, name_map[pasc])

        df_sub = df.loc[(df['flag@' + pasc] == 1) & (df['covid'] == 1), :]
        t2e_all = df_sub['dx-t2e@' + pasc]
        t2e_outpatient = df_sub.loc[(df_sub['hospitalized'] == 0) & (df_sub['criticalcare'] == 0), 'dx-t2e@' + pasc]
        t2e_inpatient = df_sub.loc[(df_sub['hospitalized'] == 1) | (df_sub['criticalcare'] == 1), 'dx-t2e@' + pasc]

        list_t2e_all.append(t2e_all)
        list_t2e_outpatient.append(t2e_outpatient)
        list_t2e_inpatient.append(t2e_inpatient)

        continue
        # Overall
        fig_plot_t2e(t2e_all,
                     name_map[pasc] + '-Overall',
                     'output/t2e_figure/' + name_map[pasc].replace('/', ' ') + '-Overall.png')

        fig_plot_t2e(t2e_outpatient,
                     name_map[pasc] + '-Not-Hospitalized',
                     'output/t2e_figure/' + name_map[pasc].replace('/', ' ') + '-Not-Hospitalized.png')

        fig_plot_t2e(t2e_inpatient,
                     name_map[pasc] + '-Hospitalized',
                     'output/t2e_figure/' + name_map[pasc].replace('/', ' ') + '-Hospitalized.png')

    all_t2e_all = pd.concat(list_t2e_all, axis=0, ignore_index=True)
    all_t2e_outpatient = pd.concat(list_t2e_outpatient, axis=0, ignore_index=True)
    all_t2e_inpatient = pd.concat(list_t2e_inpatient, axis=0, ignore_index=True)

    fig_plot_t2e(all_t2e_all,
                 'AnyPASC-{}-{}-Overall'.format(pasc_type, pasc_n_type),
                 'output/t2e_figure/AnyPASC-{}-{}-Overall.png'.format(pasc_type, pasc_n_type))

    fig_plot_t2e(all_t2e_outpatient,
                 'AnyPASC-{}-{}-Not-Hospitalized'.format(pasc_type, pasc_n_type),
                 'output/t2e_figure/AnyPASC-{}-{}-Not-Hospitalized.png'.format(pasc_type, pasc_n_type))

    fig_plot_t2e(all_t2e_inpatient,
                 'AnyPASC-{}-{}-Hospitalized'.format(pasc_type, pasc_n_type),
                 'output/t2e_figure/AnyPASC-{}-{}-Hospitalized.png'.format(pasc_type, pasc_n_type))


if __name__ == '__main__':
    plot_anypasc_t2e(pasc_type='1dx', pasc_n_type='broad')
    plot_anypasc_t2e(pasc_type='1dx', pasc_n_type='narrow')
    plot_anypasc_t2e(pasc_type='2dx30days', pasc_n_type='broad')
    plot_anypasc_t2e(pasc_type='2dx30days', pasc_n_type='narrow')
    plot_anypasc_t2e(pasc_type='2dx1days', pasc_n_type='broad')
    plot_anypasc_t2e(pasc_type='2dx1days', pasc_n_type='narrow')

    # df_pasc_info = pd.read_excel('output/causal_effects_specific_dx_insight-MultiPval-DXMEDALL_withOldSelection.xlsx',
    #                              sheet_name='dx')
    # df_pasc_info['PASC Name Simple'] = df_pasc_info['PASC Name Simple'].apply(lambda x:x.replace('/', ' '))
    # df_pasc_info = df_pasc_info.set_index('PASC Name Simple')
    # # plot_image_html(df_pasc_info, 'narrow')
    # # plot_image_html(df_pasc_info, 'broad')
    # # plot_image_html(df_pasc_info, 'narrow_2dx30days')
    # # plot_image_html(df_pasc_info, 'broad_2dx30days')
    # plot_image_html(df_pasc_info, 'narrow_2dx1day')
    # plot_image_html(df_pasc_info, 'broad_2dx1day')
