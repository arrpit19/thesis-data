import random
import models3pgn
import training3pgn
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
import csv
import argparse





def eval2mlp():
    
    # select NAS data
    train_x = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/multisiteclimS2008.csv")
    train_y = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/multi3sitegpp2008.csv")

    
    test_x =pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/xtesthyy2008.csv")
    test_y = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/gpptest2008.csv" )
   

    splits = 3
    print(splits)
    print('TRAINTEST', train_x, train_y, test_x, test_y)


    train_x.index, train_y.index = np.arange(0, len(train_x)), np.arange(0, len(train_y))
    # Load Architecture & HP
    d = pd.read_csv("N2mlpHP.csv")
    a = d.loc[d.ind_mini.idxmin()]
    layersizes = np.array(np.matrix(a.layersizes)).ravel().astype(int)
    parms = np.array(np.matrix(a.parameters)).ravel()
    lr = parms[0]
    bs = int(parms[1])
    model_design = {'layersizes': layersizes}
    print('layersizes', layersizes)
    model_design = {'layersizes': layersizes}


    hp = {'epochs': 5000,
      'batchsize': int(bs),
            'lr': lr}
    print('HYPERPARAMETERS', hp)
    data_dir = "C:/Users/Arpit/"
    data = "2mlp"

    td, se, ae = training3pgn.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data, reg=None, emb=False, exp=2)
    pd.DataFrame.from_dict(td).to_csv('2mlp_{data_use}_trainloss.csv')
    pd.DataFrame.from_dict(se).to_csv('2mlp_{data_use}_vseloss.csv')
    pd.DataFrame.from_dict(ae).to_csv('2mlp_{data_use}_vaeloss.csv')
    
    
     # Evaluation
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    x_train, y_train = torch.tensor(train_x.to_numpy(), dtype=torch.float32), torch.tensor(train_y.to_numpy(), dtype=torch.float32)
    x_test, y_test = torch.tensor(test_x.to_numpy(), dtype=torch.float32), torch.tensor(test_y.to_numpy(), dtype=torch.float32)

    train_rmse = []
    train_mae = []
    test_rmse = []
    test_mae = []
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    preds_train = {}
    preds_test = {}

    for i in range(splits):
        i += 1
        #import model
        model = models3pgn.NMLP(test_x.shape[1], 1, model_design['layersizes'])
        model.load_state_dict(torch.load(''.join((data_dir, f"22mlp_model{i}.pth"))))
        model.eval()
        with torch.no_grad():
            p_train = model(x_train)
            p_test = model(x_test)
            preds_train.update({f'train_mlp{i}': p_train.flatten().numpy()})
            preds_test.update({f'test_mlp{i}': p_test.flatten().numpy()})
            train_rmse.append(mse(p_train, y_train).tolist())
            train_mae.append(mae(p_train, y_train).tolist())
            test_rmse.append(mse(p_test, y_test).tolist())
            test_mae.append(mae(p_test, y_test).tolist())

    performance = {'train_RMSE': train_rmse,
               'train_MAE': train_mae,
               'test_RMSE': test_rmse,
               'test_mae': test_mae}


    print(preds_train)



    pd.DataFrame.from_dict(performance).to_csv('2mlp_evalperformance.csv')
    pd.DataFrame.from_dict(preds_test).to_csv('2mlp_eval_preds_test.csv')


    

eval2mlp()



import matplotlib as plt
test_y = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/gpptest2008.csv" )
multimlp = pd.read_csv("C:/Users/Arpit/2mlp_{data_use}_eval_preds_test.csv")
plt.plot(range(len(test_y)), test_y['GPP'], label='Observed GPP', marker='o')
plt.plot(range(len(multimlp)), multimlp['test_mlp2'], label='Predicted GPP by mlp', marker='o')

# Customize the plot
plt.title("GPP plot multisite mlp", fontsize=16)
plt.xlabel("Month", fontsize=14)
plt.ylabel("Values    (tonnes/hectare)", fontsize=14)
plt.legend()

print(multimlp)






