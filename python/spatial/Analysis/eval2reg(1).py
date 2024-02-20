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


def eval2reg():
    
    train_x = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/multisiteclimS2008.csv")
    train_y = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/multi3sitegpp2008.csv")

    
    test_x =pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/xtesthyy2008.csv")
    test_y = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/gpptest2008.csv" )
   
    print('TrainTest',train_x, train_y, test_x, test_y)
   
  
    reg =pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/gpptestmulti2008.csv" )
    
    splits = 3
    print("XYREG", train_x, train_y, reg)
    
   
    train_x.index, train_y.index, reg.index = np.arange(0, len(train_x)), np.arange(0, len(train_y)), np.arange(0, len(reg))
    print(len(train_x), len(train_y), len(reg))
    d = pd.read_csv("N2regHP.csv")
    a = d.loc[d.ind_mini.idxmin()]
    layersizes = np.array(np.matrix(a.layersizes)).ravel().astype(int)
    parms = np.array(np.matrix(a.parameters)).ravel()
    lr = parms[0]
    bs = int(parms[1])
    eta = parms[2]
    model_design = {'layersizes': layersizes}
    print('layersizes', layersizes)

    hp = {'epochs': 5000,
          'batchsize': int(bs),
      'lr': lr,
        'eta': eta}
    print('HYPERPARAMETERS', hp)
    data_dir = "C:/Users/Arpit/"
    data = "reg"
    td, se, ae = training3pgn.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data, reg=reg, emb=False, exp=2)
    pd.DataFrame.from_dict(td).to_csv('2reg_{data_use}_eval_tloss.csv')
    pd.DataFrame.from_dict(se).to_csv('2reg_{data_use}_eval_vseloss.csv')
    pd.DataFrame.from_dict(ae).to_csv('2reg_{data_use}_eval_vaeloss.csv')


    mse = nn.MSELoss()
    mae = nn.L1Loss()

    x_train, y_train = torch.tensor(train_x.to_numpy(), dtype=torch.float32), torch.tensor(train_y.to_numpy(), dtype=torch.float32)
    x_test, y_test = torch.tensor(test_x.to_numpy(), dtype=torch.float32), torch.tensor(test_y.to_numpy(), dtype=torch.float32)

    train_rmse = []
    train_mae = []
    test_rmse = []
    test_mae = []

    preds_tr = {}
    preds_te = {}
    for i in range(splits):
        i += 1
        #import model
        model = models3pgn.NMLP(x_train.shape[1], y_train.shape[1], model_design['layersizes'])
        model.load_state_dict(torch.load(''.join((data_dir, f"2reg_model{i}.pth"))))
        model.eval()
        with torch.no_grad():
            p_train = model(x_train)
            p_test = model(x_test)
            preds_tr.update({f'train_reg{i}':  p_train.flatten().numpy()})
            preds_te.update({f'test_reg{i}':  p_test.flatten().numpy()})
            train_rmse.append(mse(p_train, y_train).tolist())
            train_mae.append(mae(p_train, y_train).tolist())
            test_rmse.append(mse(p_test, y_test).tolist())
            test_mae.append(mae(p_test, y_test).tolist())


    performance = {'train_RMSE': train_rmse,
                   'train_MAE': train_mae,
                   'test_RMSE': test_rmse,
                   'test_mae': test_mae}


    print(performance)


    pd.DataFrame.from_dict(performance).to_csv('2reg_eval_{data_use}_performance.csv')
    pd.DataFrame.from_dict(preds_tr).to_csv('2reg_eval_preds_{data_use}_train.csv')
    pd.DataFrame.from_dict(preds_te).to_csv('2reg_eval_preds_{data_use}_test.csv')





eval2reg()

import matplotlib as plt
test_y = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/gpptest2008.csv" )
regmlp = pd.read_csv("C:/Users/Arpit/2reg_eval_preds_{data_use}_test.csv")
plt.plot(range(len(test_y)), test_y['GPP'], label='Observed GPP', marker='o')
plt.plot(range(len(regmlp)), regmlp['test_reg2'], label='Predicted GPP by reg', marker='o')

# Customize the plot
plt.title("GPP plot multisite reg", fontsize=16)
plt.xlabel("Month", fontsize=14)
plt.ylabel("Values    (tonnes/hectare)", fontsize=14)
plt.legend()

print(regmlp)



