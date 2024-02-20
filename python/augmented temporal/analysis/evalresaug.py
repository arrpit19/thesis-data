
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



def evalres2():
    
    x= pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/augtraintemp.csv")
    y = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/auggpptrain.csv")
    
    
    yptr = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/augr3pgntrainpred.csv")
    ypte = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/ypred2008cali.csv")
    yp_tr = yptr
    yp_te = ypte
    #y = y.to_frame()
    train_x = x
    train_y = y
    splits = 3

    test_x = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/xtesthyy2008.csv")
    test_y = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/gpptest2008.csv")
    print('TEST X und YP test', train_x, train_y, yp_tr, test_x, test_y, yp_te)
    splits = 3
    print(splits)
    train_x.index, train_y.index, yp_tr.index = np.arange(0, len(train_x)), np.arange(0, len(train_y)), np.arange(0, len(yp_tr)) 
    test_x.index, test_y.index, yp_te.index = np.arange(0, len(test_x)), np.arange(0, len(test_y)), np.arange(0, len(yp_te))

    # Load results from NAS
    d = pd.read_csv("Nres2HPNAS.csv")
    a = d.loc[d.ind_mini.idxmin()]
    layersizes = np.array(np.matrix(a.layersizes)).ravel().astype(int)
    parms = np.array(np.matrix(a.parameters)).ravel()
    lr = parms[0]
    bs = int(parms[1])
    model_design = {'layersizes': layersizes}
    print('layersizes', layersizes)

    hp = {'epochs': 5000,
           'batchsize': int(bs),
           'lr': lr
           }

    data_dir ="C:/Users/Arpit/"
    data = "res2aug"
    tloss = training3pgn.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data, reg=None, emb=False, res=2, y3pgn=yp_tr, exp=1)
    train_loss = tloss['train_loss']
    val_loss = tloss['val_loss']
    t1 = []
    t2 = []
    t3 = []
    
    for i in range(5000):
        t1.append(train_loss[0][i])
        t2.append(train_loss[1][i])
        t3.append(train_loss[2][i])
      
    pd.DataFrame({"f1": t1, "f2": t2, "f3":t3}).to_csv('res2aug_trainloss.csv')
    v1 = []
    v2 = []
    v3 = []
    
    for i in range(5000):
        v1.append(val_loss[0][i])
        v2.append(val_loss[1][i])
        v3.append(val_loss[2][i])
        

    pd.DataFrame({"f1": v1, "f2": v2, "f3":v3}).to_csv('res2aug_vloss.csv')

    # Evaluation
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    x_train, y_train, tr_yp = torch.tensor(train_x.to_numpy(), dtype=torch.float32), torch.tensor(train_y.to_numpy(), dtype=torch.float32), torch.tensor(yp_tr.to_numpy(), dtype=torch.float32)
    x_test, y_test, te_yp = torch.tensor(test_x.to_numpy(), dtype=torch.float32), torch.tensor(test_y.to_numpy(), dtype=torch.float32), torch.tensor(yp_te.to_numpy(), dtype=torch.float32)
    train_rmse = []
    train_mae = []
    test_rmse = []
    test_mae = []
    preds_tr = {}
    preds_te = {}
    for i in range(splits):
        i += 1
        #import model
        model = models3pgn.RES(x_train.shape[1], y_train.shape[1], model_design['layersizes'])
        model.load_state_dict(torch.load(''.join((data_dir, f"res2aug_model{i}.pth"))))
        model.eval()
        with torch.no_grad():
            p_train = model(x_train, tr_yp)
            p_test = model(x_test, te_yp)
            
            preds_tr.update({f'train_res2{i}':  p_train.flatten().numpy()})
            preds_te.update({f'test_res2{i}':  p_test.flatten().numpy()})

            train_rmse.append(mse(p_train, y_train).tolist())
            train_mae.append(mae(p_train, y_train).tolist())
            test_rmse.append(mse(p_test, y_test).tolist())
            test_mae.append(mae(p_test, y_test).tolist())


    performance = {'train_RMSE': train_rmse,
                    'train_MAE': train_mae,
                    'test_RMSE': test_rmse,
                    'test_mae': test_mae}


    pd.DataFrame.from_dict(performance).to_csv('res2aug_evalperformance.csv')
    pd.DataFrame.from_dict(preds_tr).to_csv('res2aug_eval_preds_train.csv')
    pd.DataFrame.from_dict(preds_te).to_csv('res2aug_eval_preds_test.csv')



evalres2()


import matplotlib.pyplot as plt
test_y = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/gpptest2008.csv" )
resmlpaug = pd.read_csv("C:/Users/Arpit/res2aug_eval_preds_test.csv")
plt.plot(range(len(test_y)), test_y['GPP'], label='Observed GPP', marker='o')
plt.plot(range(len(resmlpaug)), resmlpaug['test_res22'], label='Predicted GPP by Res ', marker='o')
plt.title("GPP plot 2008 Augmented data", fontsize=16)
plt.xlabel("Month", fontsize=14)
plt.ylabel("Values    (tonnes/hectare)", fontsize=14)
plt.legend()




