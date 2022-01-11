import numpy as np
import pylab as plt
import seaborn as sns; sns.set()
import pandas as pd
import pickle
import scipy
import matplotlib
from scipy import sparse
import sklearn

# import tsne
import sys
import pickle


# df = pd.read_excel('users.xlsx', sheet_name = [0,1,2])
# df = pd.read_excel('users.xlsx', sheet_name = ['User_info','compound'])
df_all = pd.read_excel(r'C:\Users\zangc\Documents\Boston\workshop\2021-PASC\diagnosis\PASC Adult Master Diagnosis List.xlsx', sheet_name = None) # read all sheets
# dict_keys(['Methods and Notes', , 'HD to CCSR Comparison', 'Potential Additional Codes'])
df = df_all['PASC Adult Master Dx List']
# columns: Index(['HD Domain', 'CCSR Category', 'CCSR Category Description',
#        'CCSR ICD-10-CM Code', 'CCSR ICD-10-CM Code Description',
#        'Inpatient Default CCSR (Y/N/X)', 'Outpatient Default CCSR (Y/N/X)',
#        'CCSR Rationale', 'Presence in Pediatric List', 'Pediatric Category',
#        'Pediatric Syndromic', 'Pediatric Systemic',
#        'Taquet Incidence Study Inclusion', 'Taquet Incidence Labels & Burden',
#        'Daugherty Risk Study Inclusion',
#        'Daugherty Risk Study Category & Burden'],
#       dtype='object')
print("df.shape", df.shape)
print("len(df['CCSR ICD-10-CM Code'].unique())", len(df['CCSR ICD-10-CM Code'].unique()))