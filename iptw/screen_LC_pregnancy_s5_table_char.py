import fnmatch
import sys

# for linux env.
sys.path.insert(0, '..')
import time
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import datetime
from misc import utils
# import eligibility_setting as ecs
import functools
import fnmatch
from lifelines import KaplanMeierFitter, CoxPHFitter
import random

print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    # parser.add_argument('--dataset', choices=['oneflorida', 'V15_COVID19'], default='V15_COVID19',
    #                     help='data bases')
    # parser.add_argument('--site', default='all',  # choices=['COL', 'MSHS', 'MONTE', 'NYU', 'WCM', 'ALL', 'all'],
    #                     help='one particular site or all')
    # parser.add_argument('--severity', choices=['all',
    #                                            'outpatient', 'inpatient', 'icu', 'inpatienticu',
    #                                            'female', 'male',
    #                                            'white', 'black',
    #                                            'less65', '65to75', '75above', '20to40', '40to55', '55to65', 'above65',
    #                                            'Anemia', 'Arrythmia', 'CKD', 'CPD-COPD', 'CAD',
    #                                            'T2D-Obesity', 'Hypertension', 'Mental-substance', 'Corticosteroids',
    #                                            'healthy',
    #                                            '03-20-06-20', '07-20-10-20', '11-20-02-21',
    #                                            '03-21-06-21', '07-21-11-21',
    #                                            '1stwave', 'delta', 'alpha', 'preg-pos-neg',
    #                                            'pospreg-posnonpreg',
    #                                            'fullyvac', 'partialvac', 'anyvac', 'novacdata',
    #                                            ],
    #                     default='all')
    parser.add_argument("--random_seed", type=int, default=0)
    # parser.add_argument('--negative_ratio', type=int, default=10)  # 5
    # parser.add_argument('--selectpasc', action='store_true')

    parser.add_argument("--kmatch", type=int, default=5)
    parser.add_argument('--replace', action='store_true')

    # parser.add_argument("--usedx", type=int, default=1)  # useacute
    # parser.add_argument("--useacute", type=int, default=1)

    args = parser.parse_args()

    # More args

    if args.random_seed < 0:
        from datetime import datetime
        args.random_seed = int(datetime.now())

    # args.save_model_filename = os.path.join(args.output_dir, '_S{}{}'.format(args.random_seed, args.run_model))
    # utils.check_and_mkdir(args.save_model_filename)
    return args


# from iptw.PSModels import ml
# from iptw.evaluation import *
def _t2eall_to_int_list_dedup(t2eall):
    t2eall = t2eall.strip(';').split(';')
    t2eall = set(map(int, t2eall))
    t2eall = sorted(t2eall)

    return t2eall



def build_exposure_group_and_table1_print_explore():
    # %% Step 1. Load  Data
    start_time = time.time()
    np.random.seed(0)
    random.seed(0)

    args = parse_args()
    in_file_df = r'../data/recover/output/pregnancy_output_y4/pregnant_yr4_exposureBuilt-pregafter180-all-anypasc.csv'
    out_file_df = in_file_df.replace('.csv', '-CharacterTable.xlsx')

    print('in read: ', in_file_df)
    df = pd.read_csv(in_file_df,
                     dtype={'patid': str, 'site': str, 'zip': str},
                     parse_dates=['index date', 'dob',
                                  'flag_delivery_date',
                                  'flag_pregnancy_start_date',
                                  'flag_pregnancy_end_date'
                                  ],
                     )
    print('df.shape:', df.shape)
    print('Read Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    case_label = 'LC Exposed'
    ctrl_label = 'Non-Exposed'
    df_pos = df.loc[df['exposed'] == 1, :]
    df_neg = df.loc[df['exposed'] == 0, :]

    print('len(df)', len(df),
          'len(df_pos)', len(df_pos),
          'len(df_neg)', len(df_neg),
          )

    output_columns = ['All', case_label, ctrl_label, 'SMD']

    print('treated df_pos.shape', df_pos.shape,
          'control df_neg.shape', df_neg.shape,
          'combined df.shape', df.shape, )

    # print('Dump file for updating covs at drug index ', out_file_df)
    # df.to_csv(out_file_df, index=False)
    # print('Dump file done ', out_file_df)

    # step 2: generate initial table!
    # return df, df
    def _n_str(n):
        return '{:,}'.format(n)

    def _quantile_str(x):
        v = x.quantile([0.25, 0.5, 0.75]).to_list()
        return '{:.0f} ({:.0f}—{:.0f})'.format(v[1], v[0], v[2])

    def _percentage_str(x):
        n = x.sum()
        per = x.mean()
        return '{:,} ({:.1f})'.format(n, per * 100)

    def _smd(x1, x2):
        m1 = x1.mean()
        m2 = x2.mean()
        v1 = x1.var()
        v2 = x2.var()

        VAR = np.sqrt((v1 + v2) / 2)
        smd = np.divide(
            m1 - m2,
            VAR, out=np.zeros_like(m1), where=VAR != 0)
        return smd

    row_names = []
    records = []

    # N
    row_names.append('N')
    records.append([
        _n_str(len(df)),
        _n_str(len(df_pos)),
        _n_str(len(df_neg)),
        np.nan
    ])

    # Sex
    row_names.append('Sex-no. (%)')
    records.append([])
    sex_col = ['Female', 'Male', 'Other/Missing']
    # sex_col = ['Female', 'Male']

    row_names.extend(sex_col)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in sex_col])

    # age
    row_names.append('Median age (IQR)-yr')
    records.append([
        _quantile_str(df['age']),
        _quantile_str(df_pos['age']),
        _quantile_str(df_neg['age']),
        _smd(df_pos['age'], df_neg['age'])
    ])

    row_names.append('Age group-no. (%)')
    records.append([])

    age_col = ['pregage:18-<25 years', 'pregage:25-<30 years', 'pregage:30-<35 years',
               'pregage:35-<40 years', 'pregage:40-<45 years', 'pregage:45-50 years',]
    age_col_output = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-50',]
    row_names.extend(age_col_output)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in age_col])

    # Race
    row_names.append('Race-no. (%)')
    records.append([])
    # col_names = ['Asian', 'Black or African American', 'White', 'Other', 'Missing']
    col_names = ['RE:Asian Non-Hispanic', 'RE:Black or African American Non-Hispanic', 'RE:Hispanic or Latino Any Race',
                 'RE:White Non-Hispanic', 'RE:Other Non-Hispanic', 'RE:Unknown']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # # Ethnic group
    # row_names.append('Ethnic group-no. (%)')
    # records.append([])
    # col_names = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])

    row_names.append('Acute severity-no. (%)')
    records.append([])
    col_names = ['outpatient', 'inpatient', 'icu', 'inpatienticu']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # # follow-up
    # row_names.append('Follow-up days (IQR)')
    # records.append([
    #     _quantile_str(df['maxfollowup']),
    #     _quantile_str(df_pos['maxfollowup']),
    #     _quantile_str(df_neg['maxfollowup']),
    #     _smd(df_pos['maxfollowup'], df_neg['maxfollowup'])
    # ])
    #
    # row_names.append('T2 Death days (IQR)')
    # records.append([
    #     _quantile_str(df['death t2e']),
    #     _quantile_str(df_pos['death t2e']),
    #     _quantile_str(df_neg['death t2e']),
    #     _smd(df_pos['death t2e'], df_neg['death t2e'])
    # ])
    # col_names = ['death', 'death in acute', 'death post acute']
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in
    #      col_names])

    # # utilization
    # row_names.append('No. of hospital visits in the past 3 yr-no. (%)')
    # records.append([])
    # # part 1
    # col_names = ['No. of Visits:0', 'No. of Visits:1-3', 'No. of Visits:4-9', 'No. of Visits:10-19',
    #              'No. of Visits:>=20',
    #              'No. of hospitalizations:0', 'No. of hospitalizations:1', 'No. of hospitalizations:>=1']
    # col_names_out = col_names
    # row_names.extend(col_names_out)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in
    #      col_names])

    # # utilization
    # row_names.append('No. of hospital visits in the past 3 yr-no. (%)')
    # records.append([])
    # # part 1
    # col_names = ['inpatient no.', 'outpatient no.', 'emergency visits no.', 'other visits no.']
    # col_names_out = ['No. of Inpatient Visits', 'No. of Outpatient Visits',
    #                  'No. of Emergency Visits', 'No. of Other Visits']
    #
    # row_names.extend(col_names_out)
    # records.extend(
    #     [[_quantile_str(df[c]), _quantile_str(df_pos[c]), _quantile_str(df_neg[c]), _smd(df_pos[c], df_neg[c])] for c in
    #      col_names])
    #
    # # part2
    # df_pos['Inpatient >=3'] = df_pos['inpatient visits 3-4'] + df_pos['inpatient visits >=5']
    # df_neg['Inpatient >=3'] = df_neg['inpatient visits 3-4'] + df_neg['inpatient visits >=5']
    # df_pos['Outpatient >=3'] = df_pos['outpatient visits 3-4'] + df_pos['outpatient visits >=5']
    # df_neg['Outpatient >=3'] = df_neg['outpatient visits 3-4'] + df_neg['outpatient visits >=5']
    # df_pos['Emergency >=3'] = df_pos['emergency visits 3-4'] + df_pos['emergency visits >=5']
    # df_neg['Emergency >=3'] = df_neg['emergency visits 3-4'] + df_neg['emergency visits >=5']
    #
    # df['Inpatient >=3'] = df['inpatient visits 3-4'] + df['inpatient visits >=5']
    # df['Outpatient >=3'] = df['outpatient visits 3-4'] + df['outpatient visits >=5']
    # df['Emergency >=3'] = df['emergency visits 3-4'] + df['emergency visits >=5']
    #
    # col_names = ['inpatient visits 0', 'inpatient visits 1-2', 'Inpatient >=3',
    #              'outpatient visits 0', 'outpatient visits 1-2', 'Outpatient >=3',
    #              'emergency visits 0', 'emergency visits 1-2', 'Emergency >=3']
    # col_names_out = ['Inpatient 0', 'Inpatient 1-2', 'Inpatient >=3',
    #                  'Outpatient 0', 'Outpatient 1-2', 'Outpatient >=3',
    #                  'Emergency 0', 'Emergency 1-2', 'Emergency >=3']
    # row_names.extend(col_names_out)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])

    # Days between preg onset and infection
    row_names.append('Days between infection and pregnancy onset (IQR)-days')
    records.append([
        _quantile_str(df['days_between_covid_pregnant_onset']),
        _quantile_str(df_pos['days_between_covid_pregnant_onset']),
        _quantile_str(df_neg['days_between_covid_pregnant_onset']),
        _smd(df_pos['days_between_covid_pregnant_onset'], df_neg['days_between_covid_pregnant_onset'])
    ])

    # ADI
    row_names.append('Median area deprivation index (IQR)-rank')
    records.append([
        _quantile_str(df['adi']),
        _quantile_str(df_pos['adi']),
        _quantile_str(df_neg['adi']),
        _smd(df_pos['adi'], df_neg['adi'])
    ])

    # col_names = ['ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49',
    #              'ADI50-59', 'ADI60-69', 'ADI70-79', 'ADI80-89', 'ADI90-100',
    #              'ADIMissing']
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])

    # BMI
    row_names.append('BMI (IQR)')
    records.append([
        _quantile_str(df['bmi']),
        _quantile_str(df_pos['bmi']),
        _quantile_str(df_neg['bmi']),
        _smd(df_pos['bmi'], df_neg['bmi'])
    ])

    col_names = ['BMI: <18.5 under weight', 'BMI: 18.5-<25 normal weight',
                 'BMI: 25-<30 overweight ', 'BMI: >=30 obese ', 'BMI: missing']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # Smoking:
    col_names = ['Smoker: never', 'Smoker: current', 'Smoker: former', 'Smoker: missing',
                 'PaxRisk:Smoking-Tobacco', 'PaxRisk:Smoking-Tobacco-Disorder']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # Vaccine:
    col_names = ['Fully vaccinated - Pre-index',
                 # 'Fully vaccinated - Post-index',
                 'Partially vaccinated - Pre-index',
                 # 'Partially vaccinated - Post-index',
                 'No evidence - Pre-index',
                 # 'No evidence - Post-index',
                 ]
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # time of index period
    row_names.append('Index time period of patients-no. (%)')
    records.append([])

    # part 1
    col_names = ['03/20-06/20', '07/20-10/20', '11/20-02/21',
                 '03/21-06/21', '07/21-10/21', '11/21-02/22',
                 '03/22-06/22', '07/22-10/22', '11/22-02/23',
                 '03/23-06/23', '07/23-10/23', '11/23-02/24',
                 '03/24-06/24', '07/24-10/24', '11/24-02/25',
                 ]
    # col_names = [
    #              '03/22-06/22', '07/22-10/22', '11/22-02/23',
    #              ]
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # Coexisting coditions
    row_names.append('Coexisting conditions-no. (%)')

    records.append([])
    col_names = (
            ['PaxRisk:Obesity',
            'PaxRisk:Chronic kidney disease',
            'PaxRisk:Hypertension',
            'PaxRisk:Immunocompromised condition or weakened immune system',
            'PaxRisk:Smoking current',
            'PaxRisk:Substance use disorders'] +
            ['PaxRisk:Cancer', 'PaxRisk:Chronic kidney disease', 'PaxRisk:Chronic liver disease',
             'PaxRisk:Chronic lung disease', 'PaxRisk:Cystic fibrosis',
             'PaxRisk:Dementia or other neurological conditions', 'PaxRisk:Diabetes', 'PaxRisk:Disabilities',
             'PaxRisk:Heart conditions', 'PaxRisk:Hypertension', 'PaxRisk:HIV infection',
             'PaxRisk:Immunocompromised condition or weakened immune system', 'PaxRisk:Mental health conditions',
             'PaxRisk:Overweight and obesity', 'PaxRisk:Obesity', 'PaxRisk:Pregnancy',
             'PaxRisk:Sickle cell disease or thalassemia',
             'PaxRisk:Smoking current', 'PaxRisk:Stroke or cerebrovascular disease',
             'PaxRisk:Substance use disorders', 'PaxRisk:Tuberculosis'] +
            ["DX: Coagulopathy", "DX: Peripheral vascular disorders ", "DX: Seizure/Epilepsy", "DX: Weight Loss",
             'DX: Obstructive sleep apnea', 'DX: Epstein-Barr and Infectious Mononucleosis (Mono)', 'DX: Herpes Zoster',
             'mental-base@Schizophrenia Spectrum and Other Psychotic Disorders',
             'mental-base@Depressive Disorders',
             'mental-base@Bipolar and Related Disorders',
             'mental-base@Anxiety Disorders',
             'mental-base@Obsessive-Compulsive and Related Disorders',
             'mental-base@Post-traumatic stress disorder',
             'mental-base@Bulimia nervosa',
             'mental-base@Binge eating disorder',
             'mental-base@premature ejaculation',
             'mental-base@Autism spectrum disorder',
             'mental-base@Premenstrual dysphoric disorder',
             'mental-base@SMI',
             'mental-base@non-SMI', ] +
            ['obc:Placenta accreta spectrum', 'obc:Pulmonary hypertension', 'obc:Chronic renal disease',
             'obc:Cardiac disease, preexisting', 'obc:HIV/AIDS', 'obc:Preeclampsia with severe features',
             'obc:Placental abruption', 'obc:Bleeding disorder, preexisting', 'obc:Anemia, preexisting',
             'obc:Twin/multiple pregnancy', 'obc:Preterm birth (< 37 weeks)',
             'obc:Placenta previa, complete or partial', 'obc:Neuromuscular disease',
             'obc:Asthma, acute or moderate/severe',
             'obc:Preeclampsia without severe features or gestational hypertension',
             'obc:Connective tissue or autoimmune disease', 'obc:Uterine fibroids', 'obc:Substance use disorder',
             'obc:Gastrointestinal disease', 'obc:Chronic hypertension', 'obc:Major mental health disorder',
             'obc:Preexisting diabetes mellitus', 'obc:Thyrotoxicosis', 'obc:Previous cesarean birth',
             'obc:Gestational diabetes mellitus', 'obc:Delivery BMI\xa0>\xa040']
    )

    col_names_out = (['Obesity',
            'Chronic kidney disease',
            'Hypertension',
            'Immunocompromised condition or weakened immune system',
            'Smoking current',
            'Substance use disorders'] + ['Cancer', 'Chronic kidney disease',
                     'Chronic liver disease',
                     'Chronic lung disease', 'Cystic fibrosis',
                     'Dementia or other neurological conditions', 'Diabetes',
                     'Disabilities',
                     'Heart conditions', 'Hypertension', 'HIV infection',
                     'Immunocompromised condition or weakened immune system',
                     'Mental health conditions',
                     'Overweight and obesity', 'Obesity', 'Pregnancy',
                     'Sickle cell disease or thalassemia',
                     'Smoking current or former',
                     'Stroke or cerebrovascular disease',
                     'Substance use disorders', 'Tuberculosis', ] +
                     ["Coagulopathy", "Peripheral vascular disorders ", "Seizure/Epilepsy", "Weight Loss",
                      'Obstructive sleep apnea', 'Epstein-Barr and Infectious Mononucleosis (Mono)', 'Herpes Zoster',
                      'Schizophrenia Spectrum and Other Psychotic Disorders',
                      'Depressive Disorders',
                      'Bipolar and Related Disorders',
                      'Anxiety Disorders',
                      'Obsessive-Compulsive and Related Disorders',
                      'Post-traumatic stress disorder',
                      'Bulimia nervosa',
                      'Binge eating disorder',
                      'premature ejaculation',
                      'Autism spectrum disorder',
                      'Premenstrual dysphoric disorder',
                      'SMI',
                      'non-SMI', ]  +
                     ['obc:Placenta accreta spectrum', 'obc:Pulmonary hypertension', 'obc:Chronic renal disease',
                      'obc:Cardiac disease, preexisting', 'obc:HIV/AIDS', 'obc:Preeclampsia with severe features',
                      'obc:Placental abruption', 'obc:Bleeding disorder, preexisting', 'obc:Anemia, preexisting',
                      'obc:Twin/multiple pregnancy', 'obc:Preterm birth (< 37 weeks)',
                      'obc:Placenta previa, complete or partial', 'obc:Neuromuscular disease',
                      'obc:Asthma, acute or moderate/severe',
                      'obc:Preeclampsia without severe features or gestational hypertension',
                      'obc:Connective tissue or autoimmune disease', 'obc:Uterine fibroids',
                      'obc:Substance use disorder',
                      'obc:Gastrointestinal disease', 'obc:Chronic hypertension', 'obc:Major mental health disorder',
                      'obc:Preexisting diabetes mellitus', 'obc:Thyrotoxicosis', 'obc:Previous cesarean birth',
                      'obc:Gestational diabetes mellitus', 'obc:Delivery BMI\xa0>\xa040']
                     )

    row_names.extend(col_names_out)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # col_names = ['score_cci_charlson', 'score_cci_quan']
    # col_names_out = ['score_cci_charlson', 'score_cci_quan']
    # row_names.extend(col_names_out)
    # records.extend(
    #     [[_quantile_str(df[c]), _quantile_str(df_pos[c]), _quantile_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])
    # row_names.append('CCI Score-no. (%)')
    # records.append([])

    col_names = ['cci_quan:0', 'cci_quan:1-2', 'cci_quan:3+', 'cci_quan:3-4', 'cci_quan:5-10', 'cci_quan:11+']
    col_names_out = ['cci_quan:0', 'cci_quan:1-2', 'cci_quan:3+', 'cci_quan:3-4', 'cci_quan:5-10', 'cci_quan:11+']
    row_names.extend(col_names_out)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    df_out = pd.DataFrame(records, columns=output_columns, index=row_names)
    df_out['SMD'] = df_out['SMD'].astype(float)

    df_out.to_excel(out_file_df)
    print('Dump done ', df_out)

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return df, df_out


if __name__ == '__main__':
    # python screen_build_naltrexone_treat_table_playwithEC_match.py  --replace --kmatch 10 2>&1 | tee  log/screen_build_naltrexone_treat_table_playwithEC_match-k10-v3-replace.txt
    # python screen_build_naltrexone_treat_table_playwithEC_match.py  --replace --kmatch 5 2>&1 | tee  log/screen_build_naltrexone_treat_table_playwithEC_match-k5-v3-replace.txt
    # python screen_build_naltrexone_treat_table_playwithEC_match.py  --replace --kmatch 15 2>&1 | tee  log/screen_build_naltrexone_treat_table_playwithEC_match-k15-v3-replace.txt
    # python screen_build_naltrexone_treat_table_playwithEC_match.py  --replace --kmatch 3 2>&1 | tee  log/screen_build_naltrexone_treat_table_playwithEC_match-k3-v3-replace.txt
    # python screen_build_naltrexone_treat_table_playwithEC_match.py  --replace --kmatch 1 2>&1 | tee  log/screen_build_naltrexone_treat_table_playwithEC_match-k1-v3-replace.txt

    #  timeout /t 21600;
    # timeout /t 28800; python screen_build_naltrexone_treat_table_playwithEC_match.py  --kmatch 1 2>&1 | tee  log/screen_build_naltrexone_treat_table_playwithEC_match-k1-v2.txt
    # timeout /t 28800; python screen_build_naltrexone_treat_table_playwithEC_match.py  --kmatch 2 2>&1 | tee  log/screen_build_naltrexone_treat_table_playwithEC_match-k2-v2.txt

    start_time = time.time()

    # 2025-07-15
    df, df_out = build_exposure_group_and_table1_print_explore()

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
