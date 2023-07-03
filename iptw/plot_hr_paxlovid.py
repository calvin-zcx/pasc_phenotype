import os
import shutil
import zipfile

import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import re

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


def plot_forest_for_dx_organ_pax(database='recover', star=True, text_right=False):

    # df = pd.read_excel(
    #         r'../data/recover/output/results/Paxlovid-allnarrow-V5/causal_effects_specific.xlsx',
    #         sheet_name='dx')
    # df = pd.read_excel(
    #     r'../data/recover/output/results/Paxlovid-malenarrow/causal_effects_specific_male.xlsx',
    #     sheet_name='dx')
    df = pd.read_excel(
        r'../data/recover/output/results/Paxlovid-above65narrow/causal_effects_specific_above65.xlsx',
        sheet_name='dx')

    df_select = df.sort_values(by='hr-w', ascending=True)
    # df_select = df_select.loc[df_select['selected'] == 1, :]  #
    print('df_select.shape:', df_select.shape)

    organ_list = df_select['Organ Domain'].unique()
    print(organ_list)
    organ_list = [
        'Overall',
        'Diseases of the Nervous System',
        'Diseases of the Skin and Subcutaneous Tissue',
        'Diseases of the Respiratory System',
        'Diseases of the Circulatory System',
        'Diseases of the Blood and Blood Forming Organs and Certain Disorders Involving the Immune Mechanism',
        'Endocrine, Nutritional and Metabolic Diseases',
        'Diseases of the Digestive System',
        'Diseases of the Genitourinary System',
        'Diseases of the Musculoskeletal System and Connective Tissue',
        # 'Certain Infectious and Parasitic Diseases',
        'General']
    # 'Injury, Poisoning and Certain Other Consequences of External Causes']
    organ_n = np.zeros(len(organ_list))
    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    pasc_row = []

    nabsv = []
    ncumv = []

    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple'].strip('*')
            pasc = row['pasc']
            hr = row['hr-w']
            ci = stringlist_2_list(row['hr-w-CI'])
            p = row['hr-w-p']
            domain = row['Organ Domain']

            # nabs = row['no. pasc in +']
            ncum = stringlist_2_list(row['cif_1_w'])[-1] * 1000
            ncum_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1] * 1000,
                       stringlist_2_list(row['cif_1_w_CIUpper'])[-1] * 1000]

            # use nabs for ncum_ci_negative
            nabs = stringlist_2_list(row['cif_0_w'])[-1] * 1000

            if star:
                if p <= 0.001:
                    name += '***'
                elif p <= 0.01:
                    name += '**'
                elif p <= 0.05:
                    name += '*'

            # if (database == 'V15_COVID19') and (row['selected'] == 1) and (row['selected oneflorida'] == 1):
            #     name += r'$^{‡}$'
            #
            # if (database == 'oneflorida') and (row['selected'] == 1) and (row['selected insight'] == 1):
            #     name += r'$^{‡}$'

            # if pasc == 'PASC-General':
            #     pasc_row = [name, hr, ci, p, domain, nabs, ncum]
            #     continue
            if domain == organ:
                organ_n[i] += 1
                if len(name.split()) >= 5:
                    name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])
                labs.append(name)
                measure.append(hr)
                lower.append(ci[0])
                upper.append(ci[1])
                pval.append(p)

                nabsv.append(nabs)
                ncumv.append(ncum)



    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper,
                          nabs=nabsv, ncumIncidence=ncumv)
    p.labels(scale='log')

    # organ = 'ALL'
    # p.labels(effectmeasure='aHR', add_label1='CIF per\n1000', add_label2='No. of\nCases')  # aHR
    p.labels(effectmeasure='aHR', add_label1='CIF per\n1000\nin Pos', add_label2='CIF per\n1000\nin Neg')

    # p.colors(pointcolor='r')
    # '#F65453', '#82A2D3'
    # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']

    c = '#F65453'
    p.colors(pointshape="s", errorbarcolor=c, pointcolor=c)  # , linecolor='black'),   # , linecolor='#fcb2ab')
    ax = p.plot_with_incidence(figsize=(8.6, .42 * len(labs)), t_adjuster=0.0108, max_value=2.5, min_value=0.15,
                               size=5, decimal=2,
                               text_right=text_right)


    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)

    organ_n_cumsum = np.cumsum(organ_n)
    for i in range(len(organ_n) - 1):
        ax.axhline(y=organ_n_cumsum[i] - .5, xmin=0.09, color=p.linec, zorder=1, linestyle='--')  # linewidth=1,

    # ax.set_yticklabels(labs, fontsize=11.5)
    ax.set_yticklabels(labs, fontsize=14)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    output_dir = r'../data/recover/output/results/Paxlovid-above65narrow/figure/'
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'hr_2CIF-V5.png', bbox_inches='tight', dpi=600)

    plt.savefig(output_dir + 'hr_2CIF-V5.pdf', bbox_inches='tight', transparent=True)
    plt.show()
    print()
    # plt.clf()
    plt.close()


if __name__ == '__main__':

    plot_forest_for_dx_organ_pax(database='recover')
    print('Done!')
