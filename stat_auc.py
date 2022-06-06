import pandas as pd
import numpy as np
import sys

project = sys.argv[1]

file_ = './Results/{}/train_model.log'.format(project)
dic_ = {}
with open(file_, 'r') as F:
    for line in F.readlines():
        line = line.strip()
        if line.startswith('##'):
            nam = line[2:]
            dic_[nam] = []
        else:
            valu = float(line.split(',')[-1].split(': ')[-1])
            dic_[nam].append(valu)

total_auc = []
for k,v in dic_.items():
    total_auc += v[-50:]
#     print(k, np.mean(v[-50:]), np.std(v[-50:]))
print('\nPathGNN AUC: ', np.mean(total_auc), np.std(total_auc))



file_dnn = './Results/{}/dnn.log'.format(project)
dic_dnn = {}
with open(file_dnn, 'r') as F:
    for line in F.readlines():
        line = line.strip()
        if not line.startswith('KF'):
            continue
        sp = [i.strip() for i in line.split(';')]
        auc = float(sp[-1].split(':')[-1].strip())
        if sp[0] not in dic_dnn.keys():
            dic_dnn[sp[0]] = [auc]
        else:
            dic_dnn[sp[0]].append(auc)

total_auc = []
for k,v in dic_dnn.items():
    total_auc += v[-10:]
#     print(k, np.mean(v[-10:]), np.std(v[-10:]))
print('\nadj-DNN AUC: ', np.mean(total_auc), np.std(total_auc))


file_lg = './Results/{}/lg.log'.format(project)
total_auc = []
with open(file_lg, 'r') as F:
    for line in F.readlines():
        line = line.strip()
        if not line.startswith('KF'):
            continue
        sp = [i.strip() for i in line.split(';')]
        auc = float(sp[-1].split(':')[-1].strip())
        total_auc.append(auc)
print('\nLG AUC: ', np.mean(total_auc), np.std(total_auc))


file_lg = './Results/{}/rf.log'.format(project)
total_auc = []
with open(file_lg, 'r') as F:
    for line in F.readlines():
        line = line.strip()
        if not line.startswith('KF'):
            continue
        sp = [i.strip() for i in line.split(';')]
        auc = float(sp[-1].split(':')[-1].strip())
        total_auc.append(auc)
        
print('\nRF AUC: ', np.mean(total_auc), np.std(total_auc))