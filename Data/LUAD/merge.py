import pandas as pd

df1 = pd.read_table('expression_matrix.1.txt.gz')
df2 = pd.read_table('expression_matrix.2.txt.gz')
df3 = pd.read_table('expression_matrix.3.txt.gz')

df = pd.concat([df1, df2, df3])
df.to_csv('expression_matrix.txt', index=False, sep='\t')
