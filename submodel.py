import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, SAGPooling, Set2Set, GraphNorm
from utils import seed_torch, get_config
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr


###############################################################################
#                                                                             #
#                   Calculate Pathway Enrichment score                        #
#                                                                             #
###############################################################################

class Pathway_Score(torch.nn.Module):
    def __init__(self):
        super(Pathway_Score, self).__init__()
        self.conv1 = SAGEConv(1, 8)
        self.pool1 = SAGPooling(8, ratio=0.8)
        self.sns1 = Set2Set(8,3)
        self.conv2 = SAGEConv(8, 8)
        self.pool2 = SAGPooling(8, ratio=0.8)
        self.sns2 = Set2Set(8,3)
        self.conv3 = SAGEConv(8, 8)
        self.pool3 = SAGPooling(8, ratio=0.8)
        self.sns3 = Set2Set(8,3)
        self.lin = torch.nn.Sequential(
                                        torch.nn.Linear(48,16), 
                                        torch.nn.Tanh(),
                                        # torch.nn.Dropout(p=0.2),
                                        torch.nn.Linear(16,1),
                                        torch.nn.Tanh()
                                    )
    def forward(self, dat):      
        x, edge_index, batch = dat.x, dat.edge_index, dat.batch

        # GNN-1
        x =torch.tanh(self.conv1(x, edge_index))
        x, edge_index, _,batch , _, _ = self.pool1(x, edge_index, None, batch)
        x1 = self.sns1(x, batch)

        # GNN-2
        x = GraphNorm(8)(x)
        x = torch.tanh(self.conv2(x, edge_index))  
        x, edge_index, _,batch , _, _ = self.pool2(x, edge_index, None, batch)
        x2 = self.sns2(x, batch)

        # GNN-3
        x = GraphNorm(8)(x)
        x = torch.tanh(self.conv3(x, edge_index))  
        x, edge_index, _,batch , _, _ = self.pool3(x, edge_index, None, batch)
        x3 = self.sns3(x, batch)

        x = torch.cat([x1,x2,x3], dim=1)

        # MLP
        x = self.lin(x).squeeze()
        
        return x


###############################################################################
#                                                                             #
#               　　　　　　　  Ｄefine training    　　                         #
#                                                                             #
###############################################################################
def train():
    """
    Training model for predicting pathway activated level.

    model: model
    dataset: graph dataset
    device: 
    """
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fun(output.type(torch.float), data.y.type(torch.float))
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


###############################################################################
#                                                                             #
#               　　　　　　Ｄefine test function    　　                        #
#                                                                             #
###############################################################################
def test(loader):
    predicts = []
    real_y = []
    model.eval()
    for data in loader:
        data = data.to('cpu')
        with torch.no_grad():
            pred = model(data)
        predicts += list(pred.detach().numpy())
        real_y += list(data.y)
    pccs = pearsonr(pred.detach().numpy(), data.y)
    spear = spearmanr(pred.detach().numpy(), data.y)
    mse_ = mean_squared_error(data.y, pred.detach().numpy())
    rmse_ = np.sqrt(mse_)
    mae_ = mean_absolute_error(data.y, pred.detach().numpy())
    r2_ = r2_score(data.y, pred.detach().numpy())
    return pccs[0], spear[0], rmse_, mae_, r2_



if __name__ == '__main__':
    # set seed
    seed_torch(1024)

    # get configures
    config = get_config('config_KIRC.ini')
    ## project name
    project_name = config['DEFAULT']['project'] 
    ## root dir
    project_root_dir = config['DEFAULT']['root_dir'] 
    ## model saving path
    model_path = os.path.join(project_root_dir, config['SUBMODEL']['submodel'])
    ## dataset path
    dataset_path = os.path.join(project_root_dir, config['DATA']['data_dir'], 'clean/pathway_reactome_ssgsea.pt')
    ## train log
    save_log = config['SUBMODEL']['log']


    # init log file
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(save_log, 'w') as F:
        F.writelines(project_name + '\t' + str(local_time) + '\n\n')


    # read dataset, split train dataset and test dataset.
    print('[INFO] Load dataset... \n')
    dataset_ = torch.load(dataset_path)
    dataset = []
    for k,v in dataset_.items():
        dataset += v
    random.shuffle(dataset)

    kf = KFold(n_splits=5)
    fold_num = 1
    for train_index, test_index in kf.split(dataset):
        print('[INFO] Run {} fold.\n'.format(fold_num))
        train_dataset = [dataset[idx_] for idx_ in train_index]
        test_dataset = [dataset[idx_] for idx_ in test_index]
        test_loader = DataLoader(test_dataset, batch_size=60)
        train_loader = DataLoader(train_dataset, batch_size=60)

        print('[INFO] Training... \n')
        device = torch.device('cpu')
        if os.path.exists(model_path):
            model = torch.load(model_path).to(device)
        else:
            model = Pathway_Score().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        loss_fun = torch.nn.MSELoss()
        
        for epoch in range(30):
            loss_ = train()            
            train_person, train_spear, train_rmse, train_mae, train_r2 = test(train_loader)
            test_person, test_spear, test_rmse, test_mae, test_r2 = test(test_loader)

            print('kfold: {:03d}, epoch: {:03d}, Loss: {:.5f}, Train Pearson: {:.5f}, Train Spearman: {:.5f}, Train RMSE: {:.5f}, Train MAE: {:.5f}, Train R2: {:.5f}, Test Pearson: {:.5f}, Test Spearman: {:.5f}, Test RMSE: {:.5f}, Test MAE: {:.5f}, Test R2: {:.5f} \n'.format(
                fold_num, epoch, loss_, train_person, train_spear, train_rmse, train_mae, train_r2, test_person, test_spear, test_rmse, test_mae, test_r2))
            
            outline = 'kfold: {:03d}, epoch: {:03d}, Loss: {:.5f}, Train Pearson: {:.5f}, Train Spearman: {:.5f}, Train RMSE: {:.5f}, Train MAE: {:.5f}, Train R2: {:.5f}, Test Pearson: {:.5f}, Test Spearman: {:.5f}, Test RMSE: {:.5f}, Test MAE: {:.5f}, Test R2: {:.5f} \n'.format(
                fold_num, epoch, loss_, train_person, train_spear, train_rmse, train_mae, train_r2, test_person, test_spear, test_rmse, test_mae, test_r2)
            
            with open(save_log, 'a') as F:
                F.writelines(outline)

        torch.save(model, model_path + '_fold' +str(fold_num)+'.pt')
        fold_num += 1

        # donot cross validation
        # torch.save(model, model_path)
