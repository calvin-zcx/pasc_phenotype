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


def load_ccsr():
    df = pd.read_excel(r'data\DXCCSR_v2022-1.xlsx', sheet_name='DXCCSR_v2022-1', dtype=str)  # read all sheets
    icd_info = {}
    for index, row in df.iterrows():
        icd = row[r"'ICD-10-CM CODE'"].strip(r'"').strip(r"'")
        icd_des = row[r"'ICD-10-CM CODE DESCRIPTION'"].strip(r'"').strip(r"'")
        cat = row[r"'CCSR CATEGORY 1'"].strip(r'"').strip(r"'")
        cat_des = row[r"'CCSR CATEGORY 1 DESCRIPTION'"].strip(r'"').strip(r"'")
        if icd not in icd_info:
            icd_info[icd] = (icd_des, cat, cat_des)
        else:
            print(icd, 'in ', 'icd_info:', icd_info[icd])

    return icd_info

icd_info = load_ccsr()


# df = pd.read_excel('users.xlsx', sheet_name = [0,1,2])
# df = pd.read_excel('users.xlsx', sheet_name = ['User_info','compound'])
df_all = pd.read_excel(r'data\PASC Adult Master Diagnosis List-ADDED-V2-Chengxi_YZ.xlsx', sheet_name = None) # read all sheets
# dict_keys(['Methods and Notes', , 'HD to CCSR Comparison', 'Potential Additional Codes'])
df = df_all['PASC Adult Master Dx List']
# Index(['HD Domain', 'CCSR Category', 'CCSR Category Description',
#        'CCSR ICD-10-CM Code', 'CCSR ICD-10-CM Code Description',
#        'Inpatient Default CCSR (Y/N/X)', 'Outpatient Default CCSR (Y/N/X)',
#        'CCSR Rationale', 'Presence in Pediatric List', 'Pediatric Category',
#        'Pediatric Syndromic', 'Pediatric Systemic',
#        'Taquet Incidence Study Inclusion', 'Taquet Incidence Labels & Burden',
#        'Daugherty Risk Study Inclusion',
#        'Daugherty Risk Study Category & Burden', 'Yan Burden Study Inclusion',
#        'Yan Burden Study Category', 'Select Any Of 3', 'Jason's study',
#        'Jason's study category', 'Seelct Any of 4',
#        'Lars Danish Study Inclusion', 'Lars Danish Study Category',
#        'Arch UF Study Inclusion', 'Arch UF Study Category',
#        'Kelly NYC Study Inclusion', 'Kelly NYC Study Category',
#        'Elissa Canada Study Inclusion', 'Elissa Canada Study Category'],
#       dtype='object')
print("df.shape", df.shape)
print("len(df['CCSR ICD-10-CM Code'].unique())", len(df['CCSR ICD-10-CM Code'].unique()))
df = df.drop_duplicates(subset=['CCSR ICD-10-CM Code'], keep='last')
print("df.shape", df.shape)
print("len(df['CCSR ICD-10-CM Code'].unique())", len(df['CCSR ICD-10-CM Code'].unique()))

df['CCSR CATEGORY 1'] = ''
df['CCSR CATEGORY 1 DESCRIPTION'] = ''
for index, row in df.iterrows():
    icd = row[r"CCSR ICD-10-CM Code"].strip()
    # icd_info[icd] = (icd_des, cat, cat_des)
    record = icd_info[icd]
    icd_des = record[0]
    cat = record[1]
    cat_des = record[2]
    df.loc[index, 'CCSR CATEGORY 1'] = cat
    df.loc[index, 'CCSR CATEGORY 1 DESCRIPTION'] = cat_des

# Step 1: selected ALL ICD code, count
# Step 2: choose major CCSR category (Or just delet, use unique code, then add CCSR if necessary)
v_flag = [x for x in df.columns if 'Inclusion' in x]

a = df.loc[:, v_flag].apply(lambda x: x=='X')
b = a.sum(axis=1)
df['Number_of_Paper_Support'] = df.loc[:, v_flag].apply(lambda x: x=='X').sum(axis=1)
df.to_excel(r'data\PASC Adult Master Diagnosis List-ADDED-V2-Chengxi_YZ_Parsimonious_v2.xlsx')
