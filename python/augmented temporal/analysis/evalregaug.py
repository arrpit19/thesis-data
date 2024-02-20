

import random
import models3pgn
import training3pgn
import torch
import pandas as pd
import numpy as np
import argparse
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
import csv
import argparse



def evalreg():
    x= pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/augtraintemp.csv")
    y = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/auggpptrain.csv")
    train_x = x
    train_y = y

    yptr = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/augr3pgntrainpred.csv")
    ypte = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/ypred2008cali.csv")
    
    #y = y.to_frame()

    reg_tr = yptr
    reg_te = ypte

    train_x = x
    train_y = y

    splits = 3

    test_x = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/xtesthyy2008.csv")
    test_y = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/gpptest2008.csv")

    print('XYREG', train_x, train_y, reg_tr, test_x, test_y, reg_te)
    
    splits = 3
    train_x.index, train_y.index, reg_tr.index = np.arange(0, len(train_x)), np.arange(0, len(train_y)), np.arange(0, len(reg_tr)) 
    test_x.index, test_y.index, reg_te.index = np.arange(0, len(test_x)), np.arange(0, len(test_y)), np.arange(0, len(reg_te))

    d = pd.read_csv("f-regNAS.csv")
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
    print('Hyperp', hp)

    data_dir = "C:/Users/Arpit/"
    data = "regaug"
    tloss = training3pgn.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data, reg=reg_tr, emb=False)
    # Evaluation
    print(tloss)
    train_loss = tloss['train_loss']
    val_loss = tloss['val_loss']
    t1 = []
    t2 = []
    t3 = []
   
    for i in range(5000):
        t1.append(train_loss[0][i])
        t2.append(train_loss[1][i])
        t3.append(train_loss[2][i])
      
    pd.DataFrame({"f1": t1, "f2": t2, "f3":t3}).to_csv('regaug_trainloss.csv')
    v1 = []
    v2 = []
    v3 = []
    
    for i in range(5000):
        v1.append(val_loss[0][i])
        v2.append(val_loss[1][i])
        v3.append(val_loss[2][i])
       

    pd.DataFrame({"f1": v1, "f2": v2, "f3":v3}).to_csv('regaug_vloss_.csv')

    mse = nn.MSELoss()
    mae = nn.L1Loss()

    x_train, y_train, tr_reg = torch.tensor(train_x.to_numpy(), dtype=torch.float32), torch.tensor(train_y.to_numpy(), dtype=torch.float32), torch.tensor(reg_tr.to_numpy(), dtype=torch.float32)
    x_test, y_test, te_reg = torch.tensor(test_x.to_numpy(), dtype=torch.float32), torch.tensor(test_y.to_numpy(), dtype=torch.float32), torch.tensor(reg_te.to_numpy(), dtype=torch.float32)

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
        model.load_state_dict(torch.load(''.join((data_dir, f"regaug_model{i}.pth"))))
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


    pd.DataFrame.from_dict(performance).to_csv('regaug_eval_performance.csv')
    pd.DataFrame.from_dict(preds_tr).to_csv('regaug_eval_preds_train.csv')
    pd.DataFrame.from_dict(preds_te).to_csv('regaug_eval_preds_test.csv')


evalreg()


import matplotlib.pyplot as plt
test_y = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/gpptest2008.csv" )
regmlpaug = pd.read_csv("C:/Users/Arpit/regaug_eval_preds_test.csv")

plt.plot(range(len(test_y)), test_y['GPP'], label='Observed GPP', marker='o')
plt.plot(range(len(regmlpaug)), regmlpaug['test_reg2'], label='Predicted GPP by Reg ', marker='o')
plt.title("GPP plot 2008 Augmented data", fontsize=16)
plt.xlabel("Month", fontsize=14)
plt.ylabel("Values    (tonnes/hectare)", fontsize=14)
plt.legend()
