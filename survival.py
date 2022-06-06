import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


df_cli = pd.read_table('Data/LUAD/clean/clinical.txt')
df_predict = pd.read_table('Results/LUAD/feature_importance.tsv.fold5.tsv')
project = 'LUAD'

df_predict = df_predict[df_predict.Score_zscore.abs() > 1.5]
smps = df_predict.columns.to_list()[3:]
for idx, row in df_predict.iterrows():
    pathway = row[0]
    score_median = row[1]
    smp_idx = 0
    group1 = []
    group2 = []
    for value in row[3:]:
        if value < score_median:
            group1.append(smps[smp_idx])
        else:
            group2.append(smps[smp_idx])
        smp_idx += 1
    
    df_cli_group1 = df_cli[df_cli.Patient_ID.isin(group1)]
    df_cli_group2 = df_cli[df_cli.Patient_ID.isin(group2)]

    # pvalue: log-rank test
    results=logrank_test(df_cli_group1.Time, df_cli_group2.Time,
                         event_observed_A=df_cli_group1.Status, 
                         event_observed_B=df_cli_group2.Status)
    pvalue = results.p_value

    # plot
    kmf = KaplanMeierFitter()
    ax = plt.subplot(111)
    kmf.fit(df_cli_group1.Time, event_observed=df_cli_group1.Status,label='Group1')
    kmf.plot(ax=ax)
    kmf.fit(df_cli_group2.Time, event_observed=df_cli_group2.Status,label='Group2')
    kmf.plot(ax=ax)
    plt.title(pathway + ' , log-rank test: {}'.format(pvalue))
    plt.xlabel('Time (Days)')
    pathway = pathway.replace(' ', '')
    pathway = pathway.replace('/', '_')
    plt.savefig('Results/{}/'.format(project) + pathway.replace(' ', '') + '.pdf')
    plt.show()

