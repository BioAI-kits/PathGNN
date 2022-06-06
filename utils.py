import torch
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from torch_geometric.data import Data
from captum.attr import IntegratedGradients
# from model import Net
import configparser


###############################################################################
#                                                                             #
#                               设置随机数种子                                  #
#                                                                             #
###############################################################################
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


###############################################################################
#                                                                             #
#                              get configures                                 #
#                                                                             #
###############################################################################
def get_config(config_file = 'config_LUAD.ini'):
    config = configparser.ConfigParser()
    config.read(config_file)
    print('[INFO] Perform {} Project.\n'.format(config['DEFAULT']['project']))
    return config


###############################################################################
#                                                                             #
#                             integrated_gradients                            #
#                                                                             #
###############################################################################
def integrated_gradients(model_path, 
                         pathways='/home/PJLAB/liangbilin/Projects/DeepSurvial/Pathway/pathways.names.txt', 
                         top_num=5,
                         plot='pathways_importance.png'):
    """
    model_path: model file  (pytorch model, pt5)
    pathways: pathway names
    top: keep top importance pathway (int)
    """
    # read pathways name to a list
    with open(pathways) as F:
        lines = [line.strip() for line in F.readlines()]
        lines = np.array(lines)

    # load model
    model = torch.load(model_path)

    # extract calss layers
    net = model.lin1

    # ig
    ig = IntegratedGradients(net)

    # get net input:~~~~~~~~~~~~~~~~~~~~~~~~~~使用了模拟数据~~~~~~~~~~~~~~~~~~~修改
    input_tensor_ = torch.rand(100, 300)
    input_tensor_.requires_grad_()

    # run ig
    attr, delta = ig.attribute(input_tensor_, target=0, return_convergence_delta=True)
    attr = attr.detach().numpy()

    # get top importance features
    importances_ = list(np.mean(attr, axis=0))
    importances_abs = [abs(i) for i in importances_]
    max_index = map(importances_abs.index, heapq.nlargest(top_num, importances_abs))
    idx = list(max_index)  # top importance pathways index
    importance_score = importances_[idx]  # positive and negetive
    print('Important pathway score is: ', importance_score)

    # plot
    if plot:
        x_pos = np.array(lines)[idx]
        importances = np.mean(attr, axis=0)[idx]
        plt.figure(figsize=(12,6))
        plt.bar(x_pos, importances, align='center')
        plt.title("Average Feature Importances")
        plt.xticks(rotation='vertical')
        plt.savefig(plot)

    return lines[idx]


if __name__ == '__main__':
    pass

