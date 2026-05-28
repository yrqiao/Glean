import pandas as pd
import numpy as np

df = pd.read_csv('list_attr_celeba.txt', delimiter='\s+')
df = df.reset_index()
df.rename(columns={'index': 'filename'}, inplace=True)
df.replace(-1, 0, inplace=True)
random_flip = np.random.rand(len(df)) < 0.25
df['Smiling'] = df['Smiling'] ^ random_flip

df1 = df[(df['Smiling'] == 1) & (df['Blond_Hair'] == 1)]
df2 = df[(df['Smiling'] == 1) & (df['Blond_Hair'] == 0)]
df3 = df[(df['Smiling'] == 0) & (df['Blond_Hair'] == 1)]
df4 = df[(df['Smiling'] == 0) & (df['Blond_Hair'] == 0)]

concat_df1 = pd.concat([df1.head(500), df2.head(4500), df3.head(4500), df4.head(500)], ignore_index=True)
env1 = pd.DataFrame()
env1['filename'] = concat_df1['filename']
env1['labels'] = concat_df1.drop('filename', axis=1).apply(lambda row: row.tolist(), axis=1)
env1.to_pickle('train_env1_smile.pickle')

concat_df2 = pd.concat([df1.iloc[500:1500], df2.iloc[4500:8500], df3.iloc[4500:8500], df4.iloc[500:1500]], ignore_index=True)
env2 = pd.DataFrame()
env2['filename'] = concat_df2['filename']
env2['labels'] = concat_df2.drop('filename', axis=1).apply(lambda row: row.tolist(), axis=1)
env2.to_pickle('train_env2_smile.pickle')

concat_df3 = pd.concat([df1.iloc[1500:3300], df2.iloc[8500:8700], df3.iloc[8500:8700], df4.iloc[1500:3300]], ignore_index=True)
test_env = pd.DataFrame()
test_env['filename'] = concat_df3['filename']
test_env['labels'] = concat_df3.drop('filename', axis=1).apply(lambda row: row.tolist(), axis=1)
test_env.to_pickle('test_env_smile.pickle')

print(df.head())