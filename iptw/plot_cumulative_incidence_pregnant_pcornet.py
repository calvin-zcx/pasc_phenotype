# import torch
import numpy as np
from sklearn.metrics import roc_auc_score
# import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import softmax
from lifelines import KaplanMeierFitter, CoxPHFitter, AalenJohansenFitter
from lifelines.statistics import survival_difference_at_fixed_point_in_time_test, proportional_hazard_test, logrank_test
import pandas as pd
from misc.utils import check_and_mkdir, stringlist_2_str, stringlist_2_list
import pickle
import os
from lifelines.plotting import add_at_risk_counts
from misc import utils


def _clean_name_(s, maxlen=50):
    s = s.replace(':', '-').replace('/', '-')
    s_trunc = (s[:maxlen] + '..') if len(s) > maxlen else s
    return s_trunc


def pformat_symbol(p):
    if p <= 0.001:
        sigsym = '$^{***}$'
        p_format = '{:.1e}'.format(p)
    elif p <= 0.01:
        sigsym = '$^{**}$'
        p_format = '{:.3f}'.format(p)
    elif p <= 0.05:
        sigsym = '$^{*}$'
        p_format = '{:.3f}'.format(p)
    else:
        sigsym = '$^{ns}$'
        p_format = '{:.3f}'.format(p)
    return p_format, sigsym


def plot_cif_primary():
    # need to tune y limit
    cohorttype = 'atrisk'
    severity = 'all'
    # i = 1
    # pasc = 'any_pasc'
    # i = 8
    # pasc = 'hospitalization_postacute'
    # i = 5
    # pasc = 'death_postacute'
    i = 6
    pasc = 'any_CFR'
    infile = r'../data/recover/output/results/Paxlovid-{}-{}-{}-V3/{}-{}-cumIncidence-ajf1w-ajf0w.pkl'.format(
        cohorttype,
        severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
        'pcornet',
        i, pasc.replace(':', '-').replace('/', '-'))

    output_file = r'../data/recover/output/results/Paxlovid-{}-{}-{}-V3/figure/{}-{}-cumIncidence.png'.format(
        cohorttype,
        severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
        'pcornet',
        i, pasc.replace(':', '-').replace('/', '-'))

    print('infile:', infile)
    ajf1w, ajf0w = utils.load(infile)
    # results_w = survival_difference_at_fixed_point_in_time_test(180, ajf1w, ajf0w)

    plt.rc('font', family='serif')
    fig = plt.figure(figsize=(5, 3.2))  # figsize=(12, 9)
    ax = fig.add_subplot(1, 1, 1)

    # ax = plt.subplot(111)
    # ajf1.plot(ax=ax)
    ajf1w.plot(ax=ax, loc=slice(0., 180), color='#d62728')  # 0, 180  loc=slice(0., controlled_t2e.max())
    # ajf0.plot(ax=ax)
    ajf0w.plot(ax=ax, loc=slice(0., 180), color='#1f77b4', linestyle='dashed', )  # loc=slice(0., controlled_t2e.max())
    # add_at_risk_counts(ajf1w, ajf0w, ax=ax, )
    ax.grid(which='major', axis='both', linestyle='--')
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.xlim(30, 180)
    # plt.ylim(0, 0.006)
    plt.ylim(0, 0.1)

    ax.set_xticks(list(range(30, 190, 30)))
    plt.xticks(fontsize=12)

    ax.set_yticklabels(['{:.1%}'.format(tick).strip('%') for tick in ax.get_yticks()])
    plt.yticks(fontsize=12)

    ax.set_xlabel('Days', fontsize=14)
    ax.set_ylabel('Cumulative Incidence (%)', fontsize=14)

    plt.tight_layout()
    # plt.ylim([0, ajf0w.cumulative_density_.loc[180][0] * 3])
    # plt.title(pasc, fontsize=12)

    check_and_mkdir(output_file)
    plt.savefig(output_file, bbox_inches='tight', dpi=600)
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight', transparent=True)
    plt.show()


def plot_cif_negctrl():
    cohorttype = 'norisk'  # 'atrisk'
    severity = 'all'
    i = 6
    pasc = 'Skin cancers'
    # i = 13
    # pasc = 'Benign neoplasms'
    indir = r'../data/recover/output/results/Paxlovid-{}-{}-pcornet-NEGCTRL-V3/'.format(
        cohorttype, _clean_name_(severity))

    infile = indir + r'{}-{}-cumIncidence-ajf1w-ajf0w.pkl'.format(i, _clean_name_(pasc))
    output_file = indir + r'figure/{}-{}-cumIncidence.png'.format(i, _clean_name_(pasc))

    df = pd.read_csv(indir + 'causal_effects_specific-snapshot-30.csv')
    row = df.loc[df['pasc'] == pasc, :].squeeze()

    print('infile:', infile)
    ajf1w, ajf0w = utils.load(infile)
    # results_w = survival_difference_at_fixed_point_in_time_test(180, ajf1w, ajf0w)

    plt.rc('font', family='serif')
    fig = plt.figure(figsize=(6, 3.5))  # figsize=(12, 9)
    ax = fig.add_subplot(1, 1, 1)

    # ax = plt.subplot(111)
    # ajf1.plot(ax=ax)
    ajf1w.plot(ax=ax, loc=slice(0., 180), color='#d62728')  # 0, 180  loc=slice(0., controlled_t2e.max())
    # ajf0.plot(ax=ax)
    ajf0w.plot(ax=ax, loc=slice(0., 180), color='#1f77b4', linestyle='dashed', )  # loc=slice(0., controlled_t2e.max())
    # add_at_risk_counts(ajf1w, ajf0w, ax=ax, )
    # ax.grid(which='major', axis='both', linestyle='--')
    # ax.grid(which='major', axis='both', linestyle='--')

    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.xlim(30, 180)
    if i == 6:
        plt.ylim(0, 0.006)
    elif i == 13:
        plt.ylim(0, 0.05)
    else:
        plt.ylim(0, 0.1)

    ax.set_xticks(list(range(30, 190, 30)))
    plt.xticks(fontsize=12)

    ax.set_yticklabels(['{:.1%}'.format(tick).strip('%') for tick in ax.get_yticks()])
    plt.yticks(fontsize=12)

    ax.set_xlabel('Days', fontsize=14)
    ax.set_ylabel('Cumulative Incidence (%)', fontsize=14)

    hr = row['hr-w']
    ci = stringlist_2_list(row['hr-w-CI'])
    p = row['hr-w-p']

    cif1 = stringlist_2_list(row['cif_1_w'])[-1] * 100
    cif1_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1] * 100,
               stringlist_2_list(row['cif_1_w_CIUpper'])[-1] * 100]

    # use nabs for ncum_ci_negative
    cif0 = stringlist_2_list(row['cif_0_w'])[-1] * 100
    cif0_ci = [stringlist_2_list(row['cif_0_w_CILower'])[-1] * 100,
               stringlist_2_list(row['cif_0_w_CIUpper'])[-1] * 100]

    cif_diff = stringlist_2_list(row['cif-w-diff-2'])[-1] * 100
    cif_diff_ci = [stringlist_2_list(row['cif-w-diff-CILower'])[-1] * 100,
                   stringlist_2_list(row['cif-w-diff-CIUpper'])[-1] * 100]
    cif_diff_p = stringlist_2_list(row['cif-w-diff-p'])[-1]

    cif_diff_pformat, cif_diff_psym = pformat_symbol(cif_diff_p)
    ahr_pformat, ahr_psym = pformat_symbol(p)

    if i == 6:
        ax.annotate('Paxlovid vs. control:\n'
                    'DIFF, {:.2f} (95% CI, {:.2f} to {:.2f}) P, {}\n'
                    'HR,    {:.2f} (95% CI,  {:.2f} to {:.2f}) p, {}'.format(
            cif_diff, cif_diff_ci[0], cif_diff_ci[1], cif_diff_pformat,
            hr, ci[0], ci[1], ahr_pformat),
            xy=(0.3, 0.025), xycoords='axes fraction', fontsize=11,
            # horizontalalignment='right',
            verticalalignment='bottom'
        )
        pass
    elif i == 13:
        ax.annotate('Paxlovid vs. control:\n'
                    'DIFF, {:.2f} (95% CI, {:.2f} to {:.2f}) P, {}\n'
                    'HR,    {:.2f} (95% CI, {:.2f} to {:.2f}) p, {}'.format(
            cif_diff, cif_diff_ci[0], cif_diff_ci[1], cif_diff_pformat,
            hr, ci[0], ci[1], ahr_pformat),
            xy=(0.3, 0.025), xycoords='axes fraction', fontsize=11,
            # horizontalalignment='right',
            verticalalignment='bottom'
        )

    plt.tight_layout()
    # plt.ylim([0, ajf0w.cumulative_density_.loc[180][0] * 3])
    # plt.title(pasc, fontsize=12)

    check_and_mkdir(output_file)
    plt.savefig(output_file, bbox_inches='tight', dpi=600)
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight', transparent=True)
    plt.show()


if __name__ == "__main__":
    cohorttype = 'atrisk'  # 'atrisk'
    severity = 'all'
    i = 1
    pasc = 'any_pasc'

    fullyscale = True
    fullyscale = False
    indir = r'../data/recover/output/results-20230825/DX-pospreg-posnonpreg-Rev2RerunOri/'
    indir = r'../data/recover/output/results-20230825/DX-pospreg-posnonpreg-Rev2PSM1to1/'

    infile = indir + r'{}-{}-cumIncidence-ajf1w-ajf0w.pkl'.format(i, _clean_name_(pasc))
    output_file = indir + r'figure/{}-{}-cumIncidence-nogrid{}.png'.format(
        i,
        _clean_name_(pasc),
        '-fullyscale' if fullyscale else '')

    df = pd.read_csv(indir + 'causal_effects_specific-snapshot-2.csv')
    row = df.loc[df['pasc'] == pasc, :].squeeze()

    print('infile:', infile)
    ajf1w, ajf0w = utils.load(infile)
    # results_w = survival_difference_at_fixed_point_in_time_test(180, ajf1w, ajf0w)

    plt.rc('font', family='serif')
    fig = plt.figure(figsize=(6 , 3.5))  # figsize=(12, 9)
    ax = fig.add_subplot(1, 1, 1)

    # ax = plt.subplot(111)
    # ajf1.plot(ax=ax)
    ajf1w.plot(ax=ax, loc=slice(0., 180), color='#d62728')  # 0, 180  loc=slice(0., controlled_t2e.max())
    # ajf0.plot(ax=ax)
    ajf0w.plot(ax=ax, loc=slice(0., 180), color='#1f77b4', linestyle='dashed', )  # loc=slice(0., controlled_t2e.max())
    # add_at_risk_counts(ajf1w, ajf0w, ax=ax, )
    # ax.grid(which='major', axis='both', linestyle='--')
    # ax.grid(which='major', axis='both', linestyle='--')

    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.xlim(30, 180)
    if fullyscale:
        plt.ylim(0, 1)
    else:
        if i == 1:  # any_pasc
            plt.ylim(0, 0.5)
        # if i == 8:  # 'hospitalization_postacute'
        #     plt.ylim(0, 0.1)
        # elif i == 5:  # 'death_postacute'
        #     plt.ylim(0, 0.006)
        # elif i == 6:  # 'any_CFR'
        #     plt.ylim(0, 0.1)
        else:
            # plt.ylim(0, 0.1)
            pass

    ax.set_xticks(list(range(30, 190, 30)))
    plt.xticks(fontsize=12)
    if i in [1, 8, 6]:
        ax.set_yticklabels(['{:.0%}'.format(tick).strip('%') for tick in ax.get_yticks()])
    else:
        ax.set_yticklabels(['{:.1%}'.format(tick).strip('%') for tick in ax.get_yticks()])

    plt.yticks(fontsize=12)

    ax.set_xlabel('Days', fontsize=14)
    ax.set_ylabel('Cumulative Incidence (%)', fontsize=14)

    hr = row['hr-w']
    ci = stringlist_2_list(row['hr-w-CI'])
    p = row['hr-w-p']

    cif1 = stringlist_2_list(row['cif_1_w'])[-1] * 100
    cif1_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1] * 100,
               stringlist_2_list(row['cif_1_w_CIUpper'])[-1] * 100]

    # use nabs for ncum_ci_negative
    cif0 = stringlist_2_list(row['cif_0_w'])[-1] * 100
    cif0_ci = [stringlist_2_list(row['cif_0_w_CILower'])[-1] * 100,
               stringlist_2_list(row['cif_0_w_CIUpper'])[-1] * 100]

    cif_diff = stringlist_2_list(row['cif-w-diff-2'])[-1] * 100
    cif_diff_ci = [stringlist_2_list(row['cif-w-diff-CILower'])[-1] * 100,
                   stringlist_2_list(row['cif-w-diff-CIUpper'])[-1] * 100]
    cif_diff_p = stringlist_2_list(row['cif-w-diff-p'])[-1]

    cif_diff_pformat, cif_diff_psym = pformat_symbol(cif_diff_p)
    ahr_pformat, ahr_psym = pformat_symbol(p)

    # if i == 6:
    #     ax.annotate('Paxlovid vs. control:\n'
    #                 'DIFF, {:.2f} (95% CI, {:.2f} to {:.2f}) P, {}\n'
    #                 'HR,    {:.2f} (95% CI,  {:.2f} to {:.2f}) p, {}'.format(
    #         cif_diff, cif_diff_ci[0], cif_diff_ci[1], cif_diff_pformat,
    #         hr, ci[0], ci[1], ahr_pformat),
    #         xy=(0.3, 0.025), xycoords='axes fraction', fontsize=11,
    #         # horizontalalignment='right',
    #         verticalalignment='bottom'
    #     )
    #     pass
    # elif i == 13:
    #     ax.annotate('Paxlovid vs. control:\n'
    #                 'DIFF, {:.2f} (95% CI, {:.2f} to {:.2f}) P, {}\n'
    #                 'HR,    {:.2f} (95% CI, {:.2f} to {:.2f}) p, {}'.format(
    #         cif_diff, cif_diff_ci[0], cif_diff_ci[1], cif_diff_pformat,
    #         hr, ci[0], ci[1], ahr_pformat),
    #         xy=(0.3, 0.025), xycoords='axes fraction', fontsize=11,
    #         # horizontalalignment='right',
    #         verticalalignment='bottom'
    #     )

    if not fullyscale:
        # ax.annotate('COVID Infected Pregnant vs. NonPregnant:\n'
        #             'Difference, {:.2f} (95% CI, {:.2f} to {:.2f})\nPval, {}'.format(
        #     cif_diff, cif_diff_ci[0], cif_diff_ci[1], cif_diff_pformat,
        # ),
        #     xy=(0.35, 0.03), xycoords='axes fraction', fontsize=11,
        #     # horizontalalignment='right',
        #     verticalalignment='bottom'
        # )
        ax.annotate('COVID Infected Pregnant vs. PS-matched NonPregnant:\n'
                    'HR, {:.2f} (95% CI, {:.2f} to {:.2f})\nPval, {}\n'
                    'Difference, {:.2f} (95% CI, {:.2f} to {:.2f})\nPval, {}'.format(
            hr, ci[0], ci[1], ahr_pformat,
            cif_diff, cif_diff_ci[0], cif_diff_ci[1], cif_diff_pformat,
        ),
            xy=(0.03, 0.5), xycoords='axes fraction', fontsize=11,
            # horizontalalignment='right',
            verticalalignment='bottom'
        )

    else:
        # ax.annotate('COVID Infected Pregnant vs. NonPregnant:\n'
        #             'Difference, {:.2f} (95% CI, {:.2f} to {:.2f})\nPval, {}'.format(
        #     cif_diff, cif_diff_ci[0], cif_diff_ci[1], cif_diff_pformat,
        # ),
        #     xy=(0.02, 0.6), xycoords='axes fraction', fontsize=11,
        #     # horizontalalignment='right',
        #     verticalalignment='bottom'
        # )
        ax.annotate('COVID Infected Pregnant vs. PS-matched NonPregnant:\n'
                    'HR, {:.2f} (95% CI, {:.2f} to {:.2f})\nPval, {}\n'
                    'Difference, {:.2f} (95% CI, {:.2f} to {:.2f})\nPval, {}'.format(
            hr, ci[0], ci[1], ahr_pformat,
            cif_diff, cif_diff_ci[0], cif_diff_ci[1], cif_diff_pformat,
        ),
            xy=(0.03, 0.5), xycoords='axes fraction', fontsize=11,
            # horizontalalignment='right',
            verticalalignment='bottom'
        )
    ax.legend(frameon=False, loc='upper center', ncol=2)

    plt.tight_layout()
    # plt.ylim([0, ajf0w.cumulative_density_.loc[180][0] * 3])
    # plt.title(pasc, fontsize=12)

    check_and_mkdir(output_file)
    plt.savefig(output_file, bbox_inches='tight', dpi=600)
    plt.savefig(output_file.replace('.png', '.pdf'), bbox_inches='tight', transparent=True)
    plt.show()
