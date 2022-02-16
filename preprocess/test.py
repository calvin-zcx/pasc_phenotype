import pandas as pd


in_file = r'../data/V15_COVID19/output/character/zip/matrix_cohorts_covid_query12_encoding_bool_ALL.csv'
print('Try to load:', in_file)
df_data = pd.read_csv(in_file, dtype={'patid': str})
print('df_data.shape:', df_data.shape)
df_pos = df_data.loc[df_data["covid"], :]
print('df_pos.shape:', df_pos.shape)

print('Choose cohorts pasc_incidence')
pasc = r"Diabetes mellitus with complication"
df_data = df_data.loc[(df_data['flag@' + pasc] >= 1) & (df_data['baseline@' + pasc] == 0), :]
print('df_data.shape:', df_data.shape)
df_pos = df_data.loc[df_data["covid"], :]
print('df_pos.shape:', df_pos.shape)

print('diabete cohorts')