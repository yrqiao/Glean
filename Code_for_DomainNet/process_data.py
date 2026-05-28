import pandas as pd
import numpy as np
from utils import *

set_deterministic(0)
df = pd.read_excel('DomainNet_attr.xlsx')
# df = df.reset_index()
# df.rename(columns={'index': 'filename'}, inplace=True)
# df.replace(-1, 0, inplace=True)
random_flip = np.random.rand(len(df)) < 0.25
df['Animal'] = df['Animal'] ^ random_flip

df1 = df[(df['Animal'] == 1) & (df['Clipart'] == 1)]
df2 = df[(df['Animal'] == 1) & (df['Clipart'] == 0)]
df3 = df[(df['Animal'] == 0) & (df['Clipart'] == 1)]
df4 = df[(df['Animal'] == 0) & (df['Clipart'] == 0)]

concat_df1 = pd.concat([
    df1.head(450), df2.head(50), df3.head(50), df4.head(450),
    # df5.head(50), df6.head(450), df7.head(50), df8.head(450),
    ], ignore_index=True)
env1 = pd.DataFrame()
env1['Filename'] = concat_df1['Filename']
env1['labels'] = concat_df1.drop('Filename', axis=1).apply(lambda row: row.tolist(), axis=1)
env1.to_pickle('train_env1_DomainNet.pickle')

concat_df2 = pd.concat([
    df1.iloc[450:850], df2.iloc[50:150], df3.iloc[50:150], df4.iloc[450:850],
    # df5.iloc[50:150], df6.iloc[450:850], df7.iloc[50:150], df8.iloc[450:850],
    ], ignore_index=True)
env2 = pd.DataFrame()
env2['Filename'] = concat_df2['Filename']
env2['labels'] = concat_df2.drop('Filename', axis=1).apply(lambda row: row.tolist(), axis=1)
env2.to_pickle('train_env2_DomainNet.pickle')

concat_df3 = pd.concat([
    df1.iloc[850:860], df2.iloc[150:240], df3.iloc[150:240], df4.iloc[850:860],
    # df5.iloc[150:240], df6.iloc[850:860], df7.iloc[150:240], df8.iloc[850:860],
    ], ignore_index=True)
test_env = pd.DataFrame()
test_env['Filename'] = concat_df3['Filename']
test_env['labels'] = concat_df3.drop('Filename', axis=1).apply(lambda row: row.tolist(), axis=1)
test_env.to_pickle('test_env_DomainNet.pickle')
