import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from utils import get_config, seed_torch
import os
import time



# construct model
class DNN(torch.nn.Module):
    def __init__(self, pathway_mask):
        super(DNN, self).__init__()
        self.pathway = torch.nn.Linear(8613, 857)
        self.hiden1 = torch.nn.Linear(857, 128)
        self.hiden2 = torch.nn.Linear(128, 32)
        self.hiden3 = torch.nn.Linear(32, 2)
        self.pathway_mask = pathway_mask

    def forward(self, data):
        self.pathway.weight.data = self.pathway.weight.data.mul(self.pathway_mask)
        x = self.pathway(data)
        x = torch.nn.ReLU()(x)
        x = self.hiden1(x)
        x = torch.nn.ReLU()(x)
        x = self.hiden2(x)
        torch.nn.Dropout(0.2)
        x = torch.nn.ReLU()(x)
        x = self.hiden3(x)
        return x

# evaluate funciton
def get_auc(outputs, labels):
    auc = roc_auc_score(labels, outputs)
    return auc    


def train(X_train, X_test, Y_train, Y_test, pathway_mask):
    model = DNN(pathway_mask=pathway_mask)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        predict_ = model(X_train)
        loss = criterion(predict_, Y_train)
        loss.backward()
        optimizer.step()
        
        # train metrics
        auc = get_auc(predict_.detach().numpy()[:,1], Y_train)
        out_classes = np.argmax(predict_.detach().numpy(), axis=1)
        acc = sum(out_classes == Y_train.detach().numpy()) / len(Y_train)
        
        # test metrics
        model.eval()
        predic_test = model(X_test)
        test_auc = get_auc(predic_test.detach().numpy()[:,1], Y_test)
        out_classes = np.argmax(predic_test.detach().numpy(), axis=1)
        test_acc = sum(out_classes == Y_test.detach().numpy()) / len(Y_test)
        
        print('KFlod:{:02d} ; Epoch: {:03d}; Loss: {:.5f}; ACC: {:.5f}; AUC: {:.5f}; ACC_test: {:5f}; AUC_test: {:.5f}\n'.format(fold_num, epoch, loss.detach().numpy(), acc, auc,test_acc ,test_auc))
        with open(dnn_log, 'a') as F:
            outline = 'KFlod:{:02d} ; Epoch: {:03d}; Loss: {:.5f}; ACC: {:.5f}; AUC: {:.5f}; ACC_test: {:5f}; AUC_test: {:.5f}\n'.format(fold_num, epoch, loss.detach().numpy(), acc, auc,test_acc ,test_auc)
            F.writelines(outline)

def svm(X_train, X_test, Y_train, Y_test):
    # init svm
    clf = SVC(C=0.6, kernel='rbf', gamma=3, decision_function_shape='ovr')
    # train svm
    clf.fit(X_train.detach().numpy(), Y_train.detach().numpy())
    # predict svm
    predic_test = clf.predict(X_test.detach().numpy())

    # metric: test
    fpr,tpr,threshold = roc_curve(Y_test.detach().numpy(), predic_test)
    test_auc = auc(fpr,tpr)
    test_acc = accuracy_score(predic_test, Y_test.detach().numpy())

    # metric: train
    predict_train = clf.predict(X_train.detach().numpy())
    fpr,tpr,threshold = roc_curve(Y_train.detach().numpy(), predict_train)
    train_auc = auc(fpr,tpr)
    train_acc = accuracy_score(predict_train, Y_train.detach().numpy())

    with open(svm_log, 'a') as F:
        outline = 'KFlod:{:02d} ; Train_ACC: {:.5f}; Train_AUC: {:.5f}; ACC_test: {:5f}; AUC_test: {:.5f}\n'.format(fold_num, train_acc, train_auc,test_acc ,test_auc)
        F.writelines(outline)


def rf(X_train, X_test, Y_train, Y_test):
    # init svm
    clf = RandomForestClassifier(random_state=0, n_estimators=100, max_depth=200)
    # train svm
    clf.fit(X_train.detach().numpy(), Y_train.detach().numpy())
    # predict svm
    predic_test = clf.predict(X_test.detach().numpy())

    # metric: test
    fpr,tpr,threshold = roc_curve(Y_test.detach().numpy(), predic_test)
    test_auc = auc(fpr,tpr)
    test_acc = accuracy_score(predic_test, Y_test.detach().numpy())

    # metric: train
    predict_train = clf.predict(X_train.detach().numpy())
    fpr,tpr,threshold = roc_curve(Y_train.detach().numpy(), predict_train)
    train_auc = auc(fpr,tpr)
    train_acc = accuracy_score(predict_train, Y_train.detach().numpy())

    with open(rf_log, 'a') as F:
        outline = 'KFlod:{:02d} ; Train_ACC: {:.5f}; Train_AUC: {:.5f}; ACC_test: {:5f}; AUC_test: {:.5f}\n'.format(fold_num, train_acc, train_auc,test_acc ,test_auc)
        F.writelines(outline)
    

def lg(X_train, X_test, Y_train, Y_test):
    from sklearn.linear_model import LogisticRegression
    lg = LogisticRegression(C=1.0, max_iter=10000)  # 默认使用L2正则化避免过拟合，C=1.0表示正则力度(超参数，可以调参调优)
    lg.fit(X_train.detach().numpy(), Y_train.detach().numpy())
    predic_test = lg.predict(X_test.detach().numpy())
    # metric: test
    fpr,tpr,threshold = roc_curve(Y_test.detach().numpy(), predic_test)
    test_auc = auc(fpr,tpr)
    test_acc = accuracy_score(predic_test, Y_test.detach().numpy())
    # metric: train
    predict_train = lg.predict(X_train.detach().numpy())
    fpr,tpr,threshold = roc_curve(Y_train.detach().numpy(), predict_train)
    train_auc = auc(fpr,tpr)
    train_acc = accuracy_score(predict_train, Y_train.detach().numpy())
    with open(lg_log, 'a') as F:
        outline = 'KFlod:{:02d} ; Train_ACC: {:.5f}; Train_AUC: {:.5f}; ACC_test: {:5f}; AUC_test: {:.5f}\n'.format(fold_num, train_acc, train_auc,test_acc ,test_auc)
        F.writelines(outline)

# construct model
class CommonDNN(torch.nn.Module):
    def __init__(self):
        super(CommonDNN, self).__init__()
        self.hiden0 = torch.nn.Linear(8613, 857)
        self.hiden1 = torch.nn.Linear(857, 128)
        self.hiden2 = torch.nn.Linear(128, 32)
        self.hiden3 = torch.nn.Linear(32, 2)

    def forward(self, data):
        x = self.hiden0(data)
        x = torch.nn.ReLU()(x)
        x = self.hiden1(x)
        x = torch.nn.ReLU()(x)
        x = self.hiden2(x)
        torch.nn.Dropout(0.2)
        x = torch.nn.ReLU()(x)
        x = self.hiden3(x)
        return x


def common_train(X_train, X_test, Y_train, Y_test, pathway_mask):
    model = CommonDNN().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        predict_ = model(X_train.cuda())
        loss = criterion(predict_, Y_train.cuda())
        loss.backward()
        optimizer.step()
        
        # train metrics
        auc = get_auc(predict_.cpu().detach().numpy()[:,1], Y_train)
        out_classes = np.argmax(predict_.cpu().detach().numpy(), axis=1)
        acc = sum(out_classes == Y_train.cpu().detach().numpy()) / len(Y_train)
        
        # test metrics
        model.eval()
        predic_test = model(X_test.cuda())
        test_auc = get_auc(predic_test.cpu().detach().numpy()[:,1], Y_test)
        out_classes = np.argmax(predic_test.cpu().detach().numpy(), axis=1)
        test_acc = sum(out_classes == Y_test.cpu().detach().numpy()) / len(Y_test)
        
        print('KFlod:{:02d} ; Epoch: {:03d}; Loss: {:.5f}; ACC: {:.5f}; AUC: {:.5f}; ACC_test: {:5f}; AUC_test: {:.5f}\n'.format(fold_num, epoch, loss.cpu().detach().numpy(), acc, auc,test_acc ,test_auc))
        with open(dnn_log+'.common.txt', 'a') as F:
            outline = 'KFlod:{:02d} ; Epoch: {:03d}; Loss: {:.5f}; ACC: {:.5f}; AUC: {:.5f}; ACC_test: {:5f}; AUC_test: {:.5f}\n'.format(fold_num, epoch, loss.cpu().detach().numpy(), acc, auc,test_acc ,test_auc)
            F.writelines(outline)


# get configure
# set seed
seed_torch(1024)

# Get configures
config = get_config('config_LGG.ini')
## device
device = torch.device(config['DEFAULT']['device'])
## project root dir
project_root_dir = config['DEFAULT']['root_dir']
## project name
project_name = config['DEFAULT']['project'] 
## cilical file path
path_gmt = os.path.join(project_root_dir, config['DNN']['pathway_gmt'])
## cilical file path
path_cli = os.path.join(project_root_dir, config['DATA']['data_dir'], 'clean/clinical.txt')
## KM group1
km_group1 = os.path.join(project_root_dir, config['MODEL']['kmgroup1'])
## KM group2
km_group2 = os.path.join(project_root_dir, config['MODEL']['kmgroup2'])
## log
dnn_log = config['DNN']['log']
## svm_log
svm_log = config['SVM']['log']
## rf_log
rf_log = config['RF']['log']
lg_log = config['LG']['log']
# init log
local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
with open(dnn_log, 'w') as F:
    F.writelines(project_name + '\t' + str(local_time) + '\n\n')

with open(svm_log, 'w') as F:
    F.writelines(project_name + '\t' + str(local_time) + '\n\n')

with open(rf_log, 'w') as F:
    F.writelines(project_name + '\t' + str(local_time) + '\n\n')

# read pathway gmt format
with open(path_gmt, 'r') as F:
    lines = [line.strip() for line in F.readlines()]
genes = []
for line in lines:
    gene = line.split('\t')[1:]
    genes += gene
genes = list(set(genes))


# read expression matrix
df = pd.read_table(os.path.join(project_root_dir, config['DATA']['data_dir'], 'clean/expression_matrix.txt'))
df_cli = pd.read_table(os.path.join(project_root_dir, config['DATA']['data_dir'], 'clean/clinical.txt'))
dic = {}
for i in df.columns.to_list():
    if i == 'Entrez_Gene_Id':
        continue
    dic[i] = df_cli.loc[df_cli.Patient_ID == i, 'age_norm'].to_list()[0]
dic['Entrez_Gene_Id'] = 999999999999999
df = df.append(dic, ignore_index=True)

dic = {}
for i in df.columns.to_list():
    if i == 'Entrez_Gene_Id':
        continue
    dic[i] = df_cli.loc[df_cli.Patient_ID == i, 'ajcc_pathologic_stage_score'].to_list()[0]
dic['Entrez_Gene_Id'] = 999999999999998
df = df.append(dic, ignore_index=True)

Entrez_Gene_Id = df.Entrez_Gene_Id.to_list()
for gene in set(genes):
    if int(gene) not in Entrez_Gene_Id:
        df = df.append({'Entrez_Gene_Id': int(gene)}, ignore_index=True)
df=df.fillna(0.1)

# calculate pathway mask matrix
path_layer_mat = np.zeros((len(lines), len(genes)))
Entrez_Gene_Id = df.Entrez_Gene_Id.to_list()
i = 0
for line in lines:
    gene = line.split('\t')[1:]
    gene_idx = [Entrez_Gene_Id.index(int(g)) for g in gene]
    path_layer_mat[i, gene_idx] = 1
    i += 1
pathway_mask = torch.tensor(path_layer_mat).to(torch.float)


# prepare train data
if os.path.exists(km_group1) and os.path.exists(km_group2):
        df_os_1 = pd.read_table(km_group1)
        df_os_2 = pd.read_table(km_group2)
else:
    df_os = pd.read_table(path_cli)
    df_os_1 = df_os[df_os.Time > 3*365]
    df_os_2 = df_os[(df_os.Time < 3*365) & (df_os.Status == 1)]
group1 = df_os_1.Patient_ID.to_list()
group2 = df_os_2.Patient_ID.to_list()
data = torch.tensor(df[group1 + group2].T.values).to(torch.float)
real_labels = torch.tensor(np.concatenate([np.ones(len(group1)), np.zeros(len(group2))])).to(torch.long)
data, real_labels = shuffle(data, real_labels)

kf = KFold(n_splits=5)
fold_num = 1
for train_index, test_index in kf.split(data):
    X_train, Y_train = data[train_index], real_labels[train_index]
    X_test, Y_test = data[test_index], real_labels[test_index]

    # ## training DNN
    # train(X_train, X_test, Y_train, Y_test, pathway_mask)

    # # SVM
    # svm(X_train, X_test, Y_train, Y_test)

    # # RF
    # rf(X_train, X_test, Y_train, Y_test)

    # lg(X_train, X_test, Y_train, Y_test)

    common_train(X_train, X_test, Y_train, Y_test, pathway_mask)

    fold_num += 1
    




