from operator import index
import sys
import os
import pandas as pd
import numpy as np
from pandas.io.parsers import read_table
from sklearn.utils import shuffle
import tqdm

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch_geometric.nn import TopKPooling, SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

from utils import seed_torch
from utils import get_config
from submodel import Pathway_Score
import random

from captum.attr import IntegratedGradients


###############################################################################
#                                                                             #
#               　　　　　　　  Ｄefine metrics     　　                         #
#                                                                             #
###############################################################################
def get_auc(outputs, labels):
    auc = roc_auc_score(labels, outputs)
    return auc


def get_acc(output, target):
    output = torch.sigmoid(output) >= 0.5
    target = target == 1.0
    return torch.true_divide((target == output).sum(dim=0), output.size(0)).item()


###############################################################################
#                                                                             #
#               　　　　　　　  Get sample's label  　　                         #
#                                                                             #
###############################################################################
def get_labels(dataset, df_os_1):
    real_labels = []
    for smp, data in dataset.items():
        if smp in df_os_1.Patient_ID.values:
            real_labels.append(0.0)
        else:
            real_labels.append(1.0)
    real_labels = torch.tensor(real_labels)

    return real_labels



###############################################################################
#                                                                             #
#               　　　　　　　  Split: Train and Test data                      #
#                                                                             #
###############################################################################

def split_dataset(dataset, df_os_1, df_os_2='N', test_perc=0.2):
    seed_torch()
    total_smp = list(dataset.keys())
    group1_smp = df_os_1.Patient_ID.to_list()
    group2_smp = df_os_2.Patient_ID.to_list()
    group1_smp_test = list(np.random.choice(group1_smp, round(len(group1_smp)*test_perc), replace=False))
    group1_smp_train = [ i for i in group1_smp if i not in group1_smp_test]

    group2_smp_test = list(np.random.choice(group2_smp, round(len(group2_smp)*test_perc), replace=False))
    group2_smp_train = [ i for i in group2_smp if i not in group2_smp_test]

    # train_dataset
    train_dataset = {}
    train_labels = []
    labels_ = group1_smp_train + group2_smp_train
    random.shuffle(labels_)
    for l in labels_:
        train_dataset[l] = dataset[l]
        if l in group1_smp_train:
            train_labels.append(0.0)
        else:
            train_labels.append(1.0)
    train_labels = torch.tensor(train_labels)

    # test_dataset
    test_dataset = {}
    test_labels = []
    labels_ = group1_smp_test + group2_smp_test
    random.shuffle(labels_)
    for l in labels_:
        test_dataset[l] = dataset[l]
        if l in group1_smp_test:
            test_labels.append(0.0)
        else:
            test_labels.append(1.0)
    test_labels = torch.tensor(test_labels)

    return train_labels, train_dataset, test_labels, test_dataset


def split_dataset_fold(dataset, df_os_1, df_os_2, folds=5 ,test_fold=1):
    if test_fold > folds:
        print('error: Test fold should be less than folds.\n')
        sys.exit(0)
    group1_smp = df_os_1.Patient_ID.to_list()
    group2_smp = df_os_2.Patient_ID.to_list()
    seed_torch(42)
    random.shuffle(group1_smp)
    seed_torch(42)
    random.shuffle(group2_smp)
    group1_smp_ = group1_smp + group1_smp
    group2_smp_ = group2_smp + group2_smp
    group1_step = round(len(group1_smp) / folds)
    group2_step = round(len(group2_smp) / folds)

    group1_smp_test = group1_smp_[group1_step*(test_fold-1): group1_step*test_fold]
    group1_smp_train = [ i for i in group1_smp if i not in group1_smp_test]

    group2_smp_test = group2_smp_[group2_step*(test_fold-1): group2_step*test_fold]
    group2_smp_train = [ i for i in group2_smp if i not in group2_smp_test]

    # train_dataset
    train_dataset = {}
    train_labels = []
    labels_ = group1_smp_train + group2_smp_train
    random.shuffle(labels_)
    for l in labels_:
        train_dataset[l] = dataset[l]
        if l in group1_smp_train:
            train_labels.append(0.0)
        else:
            train_labels.append(1.0)
    train_labels = torch.tensor(train_labels)

    # test_dataset
    test_dataset = {}
    test_labels = []
    labels_ = group1_smp_test + group2_smp_test
    random.shuffle(labels_)
    for l in labels_:
        test_dataset[l] = dataset[l]
        if l in group1_smp_test:
            test_labels.append(0.0)
        else:
            test_labels.append(1.0)
    test_labels = torch.tensor(test_labels)

    return train_labels, train_dataset, test_labels, test_dataset


###############################################################################
#                                                                             #
#               　　　　　　　  Define Generator    　　                         #
#                                                                             #
###############################################################################

def generator(df_os, pvalue=0.05, Mini_percent=0.2, Max_sample_number=5000, plot=False, 
                   figname='km_survival.pdf', km_group1= None, km_group2= None, seed=42):
    """
    df_os: A dataframe type with two columns: Patient_ID, Status, Time.  [DataFrame]
    
    pvalue: The generator uses log-rank test method to perform statistical test on subtype survival time. This parameter refers to the threshold for passing the test.  [Float]
    
    Mini_percent: The minimum number of each subtype sample.  [float]

    Max_sample_number: The max number for generator, if sample number more than the number, generator will be stop. [int]

    plot: if set True, will plot K-M survival figure.

    figname: K-M survival figure name, if plot set True.

    seed: random seed. default, 42.
    """
    np.random.seed(seed)
    sample_number = 0
    p_value = 1
    while p_value > pvalue:
        total_number = df_os.shape[0]
        mini_number = np.random.randint(int(total_number*Mini_percent), total_number-int(total_number*Mini_percent), 1)
        choice_idx = np.random.choice(total_number, mini_number, replace=False)
        df_os_1 = df_os.loc[choice_idx, :]
        df_os_2 = df_os[~df_os.Patient_ID.isin(df_os_1.Patient_ID)]

        T1 = df_os_1.loc[:, 'Time'].values
        T2 = df_os_2.loc[:, 'Time'].values
        E1 = df_os_1.loc[:, 'Status'].values
        E2 = df_os_2.loc[:, 'Status'].values

        # perform logrank_test between subtypes
        results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
        p_value = results.p_value
        sample_number += 1

        if sample_number > Max_sample_number:
            print('Generator stop! Please rerun through increasing the number of samples with Max_sample_number parameter OR decreasing the p-value threshold with pvalue parameter. \n')
            sys.exit(0)

    if plot:
        fig = plt.figure(figsize=(8,6))
        kmf = KaplanMeierFitter()
        ax = plt.subplot(111)
        kmf.fit(T1, event_observed=E1, label="Group 1")
        kmf.plot(ax=ax)
        kmf.fit(T2, event_observed=E2, label="Group 2")
        kmf.plot(ax=ax)
        plt.title('K-M Survival plot. \nGroup 1 number is: {}; \nGroup 2 number is: {}.\nP-value with log-rank test is : {:.3e}.'.format(len(T1), len(T2),p_value))
        plt.savefig(figname)
    print('[INFO] Sample Number is: {}\n'.format(sample_number))

    # output data
    if km_group1 != None:
        df_os_1.to_csv(km_group1, index=False, sep='\t')
    if km_group2 != None:
        df_os_2.to_csv(km_group2, index=False, sep='\t')

    return df_os_1, df_os_2


###############################################################################
#                                                                             #
#               　　　　　　　  Define Model        　　                         #
#                                                                             #
###############################################################################
# class Model(torch.nn.Module):
#     def __init__(self, submodel, dataset):
#         super(Model, self).__init__()
#         self.submodel = submodel
#         self.lin = torch.nn.Sequential(
#             torch.nn.Linear(len(dataset[list(dataset.keys())[0]]) + 2, 256) , # Two-class task
#             torch.nn.ReLU(),
#             # torch.nn.BatchNorm1d(128),
#             # torch.nn.Dropout(p=0.4),
#             torch.nn.Linear(256, 128),
#             torch.nn.Dropout(p=0.4),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 32),
#             torch.nn.ReLU(),
#             torch.nn.Linear(32, 2),
#             torch.nn.Softmax(dim=0)
#         )
    
#     def forward(self, data, device, cli_features):   
#         """
#         cli_features: a list, value is tensor
#         """     
#         pathway_score = []  # save pathway score

#         loader = DataLoader(data, batch_size=60, shuffle=False)
#         for dat in loader:
#             dat = dat.to(device)
#             x = self.submodel(dat)
#             pathway_score.append(x)

#         pathway_score += cli_features
#         x = torch.cat(pathway_score, dim=0)
#         x = self.lin(x)
#         return x

class Model(torch.nn.Module):
    def __init__(self, submodel, dataset):
        super(Model, self).__init__()
        self.submodels = torch.nn.ModuleList()
        for _ in range(855):
            self.submodels.append(Pathway_Score())
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(len(dataset[list(dataset.keys())[0]]) + 2, 512) , # Two-class task
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.Dropout(p=0.4),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
            torch.nn.Softmax(dim=0)
        )
    
    def forward(self, data, device, cli_features):   
        """
        cli_features: a list, value is tensor
        """     
        pathway_score = []  # save pathway score
        loader = DataLoader(data, batch_size=60, shuffle=False)
        i = 0
        for dat in loader:
            dat = dat.to(device)
            x = self.submodels[i](dat)
            pathway_score.append(x)
            i += 1
        pathway_score += cli_features
        x = torch.cat(pathway_score, dim=0)
        x = self.lin(x)
        return x

###############################################################################
#                                                                             #
#               　　　　　　　  Define Train        　　                         #
#                                                                             #
###############################################################################

def train(train_labels, train_dataset, test_labels, test_dataset, minibatch=48):
    ## forward per pathway for one sample
    out_probs = []  # model output
    real_label = []  # train labels
    loss_all = []
    auc_all = [] 
    acc_all = []
    idx = 0

    for smp, data in tqdm.tqdm(train_dataset.items()):
        # get cli_features
        tb = pd.read_table(path_cli)
        tb = tb.set_index('Patient_ID')
        ajcc_score = tb.loc[smp, 'ajcc_pathologic_stage_score']
        age_norm = tb.loc[smp, 'age_norm']
        cli_features = [torch.tensor([ajcc_score]).to(torch.float), torch.tensor([age_norm]).to(torch.float)]

        # forward
        out_probs.append(model(data, device=device, cli_features=cli_features))
        real_label.append(train_labels[idx])

        # backward
        idx += 1
        if idx % minibatch == 0 or smp==list(train_dataset.keys())[-1]:
            model.train()
            optimizer.zero_grad()

            ## get batch loss 
            real_label = torch.tensor(real_label)
            out_probs = torch.cat(out_probs, dim=0).reshape(-1,2).cpu()
            loss = criterion(out_probs, real_label.type(torch.long))
            loss.backward()
            optimizer.step()

            # output batch loss
            loss_all.append(loss.item())

            # output batch auc
            auc = get_auc(out_probs.detach().numpy()[:,1], real_label)
            auc_all.append(auc)

            # output batch acc
            out_classes = np.argmax(out_probs.detach().numpy(), axis=1)
            acc = sum(out_classes == real_label.detach().numpy()) / len(real_label)
            acc_all.append(acc)

            # reset out_probs and real_label
            out_probs = []
            real_label = []
    
    # validation
    test_auc, test_acc = test_func(test_labels, test_dataset)
    train_auc, train_acc = test_func(train_labels, train_dataset)
    print('Fold: ', fold)
    # print('Epoch: {:03d}, Loss: {:.5f};  Train_Acc: {:.5f}, Train_Auc: {:.5f}, Test_Acc: {:.5f}, Test_Auc: {:.5f}\n'.format(
    #     epoch, np.mean(loss_all), np.mean(acc_all), np.mean(auc_all), test_acc, test_auc))
    print('Epoch: {:03d}, Loss: {:.5f};  Train_Acc: {:.5f}, Train_Auc: {:.5f}, Test_Acc: {:.5f}, Test_Auc: {:.5f}\n'.format(
        epoch, np.mean(loss_all), train_acc, train_auc, test_acc, test_auc))

    # output training log
    with open(save_log, 'a') as F:
        F.writelines('Fold: {:02d}, Epoch: {:03d}, Loss: {:.5f};  Train_Acc: {:.5f}, Train_Auc: {:.5f}, Test_Acc: {:.5f}, Test_Auc: {:.5f}\n'.format(
            fold, epoch, np.mean(loss_all), train_acc, train_auc, test_acc, test_auc))

    return model


###############################################################################
#                                                                             #
#               　　　　　　　  Define Test         　　                         #
#                                                                             #
###############################################################################

def test_func(test_labels, test_dataset):
    out_probs = []
    for smp, data in tqdm.tqdm(test_dataset.items()):
        # get cli_features
        tb = pd.read_table(path_cli)
        tb = tb.set_index('Patient_ID')
        ajcc_score = tb.loc[smp, 'ajcc_pathologic_stage_score']
        age_norm = tb.loc[smp, 'age_norm']
        cli_features = [torch.tensor([ajcc_score]).to(torch.float), torch.tensor([age_norm]).to(torch.float)]
        model.eval()
        with torch.no_grad():
            out_probs.append(model(data, device=device, cli_features=cli_features))

    out_probs = torch.cat(out_probs, dim=0).reshape(-1,2).cpu()

    # output batch auc
    auc = get_auc(out_probs.detach().numpy()[:,1], test_labels)

    # output batch acc
    out_classes = np.argmax(out_probs.detach().numpy(), axis=1)
    acc = sum(out_classes == test_labels.detach().numpy()) / len(test_labels)

    return auc, acc


###############################################################################
#                                                                             #
#               　　　　　　　  Define IG           　　                         #
#                                                                             #
###############################################################################

def ig(path_model, test_labels, test_dataset, figname, output):
    tb = pd.read_table(path_cli)
    tb = tb.set_index('Patient_ID')
    model = torch.load(path_model)
    model.eval()
    S = []
    smps = ['Pathway', 'Score_median', 'Score_zscore']
    with torch.no_grad():
        for smp, data in tqdm.tqdm(test_dataset.items()):
            smps.append(smp)
            # get cli_features
            ajcc_score = tb.loc[smp, 'ajcc_pathologic_stage_score']
            age_norm = tb.loc[smp, 'age_norm']
            i = 0
            for dat in data:
                S.append(float(model.submodels[i](dat).detach().numpy()))
                i += 1
            S = S + [ajcc_score, age_norm]

    input_tensor_ = torch.tensor(S, dtype=torch.float32).reshape(-1, 857)
    input_tensor_.requires_grad_()
    net = model.lin
    ig = IntegratedGradients(net)

    # run ig
    attr, delta = ig.attribute(input_tensor_, target=1, return_convergence_delta=True)
    attr = attr.detach().numpy()

    # get features
    df_pathway = pd.read_table('Pathway/keep_pathways_details.tsv')
    features = np.array(df_pathway.pathway.to_list() + ['Stage', 'Age'])

    # stats
    df_attr = pd.DataFrame(attr).T
    df_median = pd.DataFrame(df_attr.T.median())
    df_median.columns = ['median']
    df_zscore = (df_median - df_median.mean() ) / df_median.std()
    df_zscore.columns = ['zscore']
    df_feature = pd.DataFrame(features)
    df_feature.columns = ['feature']
    df_attr = pd.DataFrame(attr).T
    df_attr_out = pd.concat([df_feature, df_median, df_zscore, df_attr], axis=1)
    df_attr_out['sort_'] = df_attr_out['median'].map(abs)
    df_attr_out = df_attr_out.sort_values('sort_', ascending=False)
    df_attr_out = df_attr_out.drop(columns=['sort_']).reset_index(drop=True)
    df_attr_out.columns = smps
    df_attr_out.to_csv(output, index=False, sep='\t')

    # plot
    # print(df_attr_out)
    # df_plot = df_attr_out[df_attr_out.Score_zscore.abs() > 1.5]
    # fig, ax = plt.subplots(1,1,figsize=(8,6))
    # plt.barh(df_plot.Pathway, df_plot.Score_zscore)
    # ax.invert_yaxis()
    # plt.xlabel('Feature importance')
    # plt.savefig(figname, bbox_inches = 'tight')


if __name__ == '__main__':
    # set seed
    seed_torch(1024)

    train_step = True
    # Get configures
    config = get_config(sys.argv[1])
    ## device
    device = torch.device(config['DEFAULT']['device'])
    ## project root dir
    project_root_dir = config['DEFAULT']['root_dir']
    ## cilical file path
    path_cli = os.path.join(project_root_dir, config['DATA']['data_dir'], 'clean/clinical.txt')
    ## dataset file path
    path_dataset = os.path.join(project_root_dir, config['DATA']['data_dir'], 'clean/pathway_reactome_ssgsea.pt')
    ## submodel file path
    path_submodel = os.path.join(project_root_dir, config['SUBMODEL']['submodel'])
    ## KM figname
    km_fig = os.path.join(project_root_dir, config['MODEL']['kmfig'])
    ## KM group1
    km_group1 = os.path.join(project_root_dir, config['MODEL']['kmgroup1'])
    ## KM group2
    km_group2 = os.path.join(project_root_dir, config['MODEL']['kmgroup2'])
    ## model
    path_model = os.path.join(project_root_dir, config['MODEL']['model'])
    ## model_run
    run_model = config['MODEL']['run']
    ## ig run
    run_ig = config['IG']['run']
    ## ig_fig
    ig_fig = config['IG']['figname']
    ## output
    ig_output = config['IG']['output']
    ## save log
    save_log = config['MODEL']['log']

    # Run Group for Patients
    print('[INFO] Run generator...\n')
    if os.path.exists(km_group1) and os.path.exists(km_group2):
        df_os_1 = pd.read_table(km_group1)
        df_os_2 = pd.read_table(km_group2)
    else:
        df_os = pd.read_table(path_cli)
        df_os_1 = df_os[df_os.Time > 3*365]
        df_os_2 = df_os[(df_os.Time < 3*365) & (df_os.Status == 1)]
    print('LTS number: {}; Non-LTS number: {}'.format(df_os_1.shape[0], df_os_2.shape[0])) 
    cross_weight = torch.tensor([df_os_1.shape[0],df_os_2.shape[0]], dtype=torch.float32)
    cross_weight = torch.tensor([max(cross_weight)/x for x in cross_weight], dtype=torch.float32)

    # Read dataset
    print('[INFO] Load dataset...\n')
    dataset = torch.load(path_dataset)
    # train_labels, train_dataset, test_labels, test_dataset = split_dataset(dataset=dataset, df_os_1=df_os_1,df_os_2=df_os_2, test_perc=0.2)
    
    fold = int(sys.argv[2])
    train_labels, train_dataset, test_labels, test_dataset = split_dataset_fold(dataset, df_os_1, df_os_2, folds=5 ,test_fold=fold)
    with open(save_log, 'a') as F:
        F.writelines('##Fold: {}\n'.format(fold))

    if run_model == "True":
        print('[INFO] Training model...\n')
        submodel_ = Pathway_Score().to(device)
        real_labels = train_labels
        dataset = train_dataset
        criterion = torch.nn.CrossEntropyLoss(weight=cross_weight)  # two class

        if os.path.exists(path_model):
            print('[INFO] Training (continue)\n')
            # model = torch.load(path_model).to(device)
            # for parma in model.parameters():
            #     parma.requires_grad = True
            # optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0005)
            # criterion = torch.nn.CrossEntropyLoss(weight=cross_weight)
            # for epoch in range(50):
            #     predict_label = train(train_labels, train_dataset, test_labels, test_dataset)
        
        else:
            print('[INFO] Denova Training\n')
            model = Model(submodel=submodel_, dataset=dataset).to(device)
            loss_fun = torch.nn.BCEWithLogitsLoss()

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
            for epoch in range(150):
                predict_label = train(train_labels, train_dataset, test_labels, test_dataset)
                torch.save(model, path_model + '.epoch{}.{}.pt'.format(epoch, fold))
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001) 
            for epoch in range(100):
                predict_label = train(train_labels, train_dataset, test_labels, test_dataset)
                torch.save(model, path_model + '.epoch{}.{}.pt'.format(epoch+150, fold))
            
            ig(path_model + '.epoch{}.{}.pt'.format(epoch+50, fold), test_labels, test_dataset, figname=ig_fig + 'fold{}'.format(fold), output=ig_output + 'fold{}'.format(fold))

        # save model
        # torch.save(model, path_model)
    # ig("/home/PJLAB/liangbilin/Projects/DeepSurvial/Model/KIRC/KIRC.PathGNN.Fold{}.epoch249.pt".format(fold), '-', dataset, figname=ig_fig + '.fold{}.pdf'.format(fold), output=ig_output + '.fold{}.tsv'.format(fold))

    # if run_ig == "True":
    #     print('[INFO] Runing IG for important features.\n')
    #     ig(path_model, test_labels, test_dataset, figname=ig_fig, output=ig_output)

