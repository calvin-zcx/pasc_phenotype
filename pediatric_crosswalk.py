import numpy as np
import pylab as plt
import seaborn as sns; sns.set()
import pandas as pd
import pickle
import scipy
import matplotlib
from scipy import sparse
import sklearn
import sys
import pickle
import openpyxl
from openpyxl import load_workbook
from collections import Counter

# step 1: read pediatric list clusters.csv file
print('1-Load pediatric list')
df_ped = pd.read_csv(r'data/Pediatric_Crosswalk/clusters.csv', dtype=str)  # read all sheets
print('df_ped.shape:', df_ped.shape)  # (66286, 9)
print('df_ped.columns:', df_ped.columns)
# ['concept_id', 'concept_name', 'concept_code',
# 'vocabulary_id', 'cluster', 'syndromic',
# 'systemic', 'clincian_interest', 'cluster_category']
df_ped['ICD-10-CM Code'] = df_ped.loc[:, 'concept_code'].apply(lambda x: x.strip().replace('.', ''))
df_ped['in_pediatric_list'] = 'X'

# step 2: read parsimonious list with Jason color (can we read color? or do it by hands?)
print('2-Load our Parsimonious list')
excel_file = r'data/Pediatric_Crosswalk/PASC Adult Master Diagnosis List-Parsimonious-20220104jb.xlsx'
# wb = load_workbook(excel_file, data_only = True)
# sh = wb['Code List']
# color_in_hex = sh['A2'].fill.start_color.index # this gives you Hexadecimal value of the color
# print ('HEX =',color_in_hex)
# print('RGB =', tuple(int(color_in_hex[i:i+2], 16) for i in (0, 2, 4))) # Color in RGB
df_pars = pd.read_excel(excel_file, sheet_name='Code List') # read all sheets
print('df_pars.shape:', df_pars.shape)  # (4661, 8)
print('df_pars.columns:', df_pars.columns)
# ['HD Domain', 'CCSR CATEGORY 1', 'CCSR CATEGORY 1 DESCRIPTION',
#  'ICD-10-CM Code', 'ICD-10-CM Code Description', 'Number_of_Paper_Support',
#  'Jason's study Inclusion', 'Jason's study category']
df_pars['in_master_list'] = 'X'

# step 3: outer join two list, For codes that are in the pediatric list but not currently in the parsimonious list,
# please still integrate them as line items with a ‘0’ noted in the column for ‘Number_of_Paper_Support’.
print('3-Combine')
df_outer = pd.merge(df_pars, df_ped, on='ICD-10-CM Code', how='outer')
# df_pars.merge(df_ped, left_on='ICD-10-CM Code', right_on='ICD_no_dot_4_mapping')
print('df_outer.shape:', df_outer.shape)
print('df_outer.columns:', df_outer.columns)
# df_outer.to_excel(r'data/Pediatric_Crosswalk/PASC Adult Master Diagnosis List-Parsimonious-jb-crosswalkPed.xlsx')

# step 4 add mark's count from WCM
df_mark = pd.read_excel(r'data/Pediatric_Crosswalk/WCM diagnoses CCS categories and ICD codes in covid pos pts -parsimonious flags.xlsx',
                       sheet_name='Sheet1')
print('df_mark.shape:', df_mark.shape)  # (11489, 9)
print('df_mark.columns:', df_mark.columns)
# ['Unnamed: 0', 'Default CCSR CATEGORY DESCRIPTION OP', 'Group total',
#        'ICD-10-CM CODE', 'ICD-10-CM CODE DESCRIPTION', 'count',
#        'Number_of_Paper_Support', 'Jason's study Inclusion',
#        'Jason's study category']
# df_mark_subcols = df_mark[['Group total', 'count', 'ICD-10-CM CODE']]
# df_mark_subcols =
df_mark['in_markwcm_list'] = 'X'

df_outer = pd.merge(df_outer, df_mark, on='ICD-10-CM Code', how='outer')
print('df_outer.shape:', df_outer.shape)  # (11489, 9)
print('df_outer.columns:', df_outer.columns)

# step 5: add additional CCSR info which were not used in the df_ped
_df_2 = df_pars[['HD Domain', 'CCSR CATEGORY 1']]
_df_2.loc[:, 'CCSR CATEGORY 1'] = _df_2.loc[:, 'CCSR CATEGORY 1'].apply(lambda x:x[:3] if isinstance(x,str) else x)
_df_3 = _df_2.drop_duplicates()
# Caution: END corrspond to multiple HD domain. So currently do not assign HD domain to Ped codes
# Diseases of the Blood and Blood Forming Organs and Certain Disorders Involving the Immune Mechanism,BLD
# Diseases of the Circulatory System,CIR
# Diseases of the Digestive System,DIG
# Diseases of the Ear and Mastoid Process,EAR
# Endocrine, Nutritional and Metabolic Diseases,END
# Diseases of the Eye and Adnexa,END
# Diseases of the Eye and Adnexa,EYE
# Factors Influencing Health Status and Contact with Health Services,FAC
# Diseases of the Genitourinary System,END
# Diseases of the Genitourinary System,GEN
# Diseases of the Genitourinary System,BLD
# Certain Infectious and Parasitic Diseases,INF
# Certain Infectious and Parasitic Diseases,CIR
# Mental, Behavioral and Neurodevelopmental Disorders,MBD
# Diseases of the Musculoskeletal System and Connective Tissue,INF
# Diseases of the Musculoskeletal System and Connective Tissue,MUS
# Diseases of the Musculoskeletal System and Connective Tissue,CIR
# Diseases of the Musculoskeletal System and Connective Tissue,END
# Diseases of the Musculoskeletal System and Connective Tissue,INJ
# Diseases of the Nervous System,INF
# Diseases of the Nervous System,NVS
# Diseases of the Nervous System,CIR
# Diseases of the Nervous System,INJ
# Diseases of the Nervous System,END
# Diseases of the Nervous System,MBD
# Diseases of the Respiratory System,INF
# Diseases of the Respiratory System,RSP
# Diseases of the Skin and Subcutaneous Tissue,SKN
# Diseases of the Skin and Subcutaneous Tissue,END
# Diseases of the Skin and Subcutaneous Tissue,INJ
# Symptoms, Signs and Abnormal Clinical and Laboratory Findings, Not Elsewhere Classified,NVS
# Symptoms, Signs and Abnormal Clinical and Laboratory Findings, Not Elsewhere Classified,INF
# Symptoms, Signs and Abnormal Clinical and Laboratory Findings, Not Elsewhere Classified,INJ
# Symptoms, Signs and Abnormal Clinical and Laboratory Findings, Not Elsewhere Classified,SYM
ccsr3d_domain = {}
for index, row in _df_3.iterrows():
    domain = row[0]
    ccsr3d = row[1]
    if (domain is not np.nan) and (ccsr3d is not np.nan):
        ccsr3d_domain[ccsr3d] = domain


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

for index, row in df_outer.iterrows():
    icd = row[r"ICD-10-CM Code"].strip()
    # icd_info[icd] = (icd_des, cat, cat_des)
    if icd in icd_info:
        record = icd_info[icd]
        icd_des = record[0]
        cat = record[1]
        cat_des = record[2]
        df_outer.loc[index, 'CCSR CATEGORY 1'] = cat
        df_outer.loc[index, 'CCSR CATEGORY 1 DESCRIPTION'] = cat_des
    else:
        print(icd, 'not found in CCSR maps, index:', index)


# step

df_outer.to_excel(r'data/Pediatric_Crosswalk/auxiliry-3-PASC Adult Master Diagnosis List-Parsimonious-jb-crosswalkPed.xlsx')
print('Done')