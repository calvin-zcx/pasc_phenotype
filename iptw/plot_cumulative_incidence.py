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
from misc.utils import check_and_mkdir
import pickle
import os
from lifelines.plotting import add_at_risk_counts
from misc import utils

if __name__ == "__main__":
    cohorttype = 'atrisknopreg'
    severity = 'all'
    i = 1
    pasc = 'any_pasc'
    i = 8
    pasc = 'hospitalization_postacute'
    i = 5
    pasc = 'death_postacute'
    infile = r'../data/recover/output/results/Paxlovid-{}-{}-{}-V2/{}-{}-cumIncidence-ajf1w-ajf0w.pkl'.format(
        cohorttype,
        severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
        'pcornet',
        i, pasc.replace(':', '-').replace('/', '-'))

    output_file = r'../data/recover/output/results/Paxlovid-{}-{}-{}-V2/figure/{}-{}-cumIncidence.png'.format(
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
    plt.ylim(0, 0.006)

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