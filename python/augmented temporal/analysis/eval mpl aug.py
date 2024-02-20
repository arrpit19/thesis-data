
import torch
import pandas as pd
import numpy as np
import argparse
import random
import models3pgn
import training3pgn
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
import csv
import argparse




def evalmlp():
    
    data_file1 = "C:/Users/Arpit/Documents/ProfoundData/augtraintemp.csv"
    data_file2 = "C:/Users/Arpit/Documents/ProfoundData/auggpptrain.csv"


    x = pd.read_csv(data_file1)

    


    y = pd.read_csv(data_file2)
    train_x = x
    train_y = y

    splits = 3
    test_x = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/xtesthyy2008.csv")
    test_y = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/gpptest2008.csv")
    train_x.index, train_y.index = np.arange(0, len(train_x)), np.arange(0, len(train_y)) 
    test_x.index, test_y.index = np.arange(0, len(test_x)), np.arange(0, len(test_y))
    
    # Architecture
    d = pd.read_csv("C:/Users/Arpit/faug.csv")
    a = d.loc[d.ind_mini.idxmin()]
    layersizes = np.array(np.matrix(a.layersizes)).ravel().astype(int)
    parms = np.array(np.matrix(a.parameters)).ravel()
    lr = parms[0]
    bs = int(parms[1])
    model_design = {'layersizes': layersizes}
    print('layersizes', layersizes)

    hp = {'epochs': 5000,
      'batchsize': int(bs),
          'lr': lr}
    print('HYPERPARAMETERS', hp)
    data_dir = "C:/Users/Arpit/"
    data = "mlpaug"
    
    tloss = training3pgn.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data, reg=None, emb=False, hp=False)
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
        
    v1 = []
    v2 = []
    v3 = []
    
    for i in range(5000):
        v1.append(val_loss[0][i])
        v2.append(val_loss[1][i])
        v3.append(val_loss[2][i])
        

    pd.DataFrame({"f1": v1, "f2": v2, "f3":v3 }).to_csv('mlp_vlossaug.csv')
    pd.DataFrame({"f1": t1, "f2": t2, "f3":t3 }).to_csv('mlp_trainlossaug_.csv')

    # Evaluation
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    x_train, y_train = torch.tensor(train_x.to_numpy(), dtype=torch.float32), torch.tensor(train_y.to_numpy(), dtype=torch.float32)
    x_test, y_test = torch.tensor(test_x.to_numpy(), dtype=torch.float32), torch.tensor(test_y.to_numpy(), dtype=torch.float32)

    train_rmse = []
    train_mae = []
    test_rmse = []
    test_mae = []
    
    preds_train = {}
    preds_test = {}

    for i in range(splits):
        i += 1
        #import model
        model = models3pgn.NMLP(x.shape[1], y.shape[1], model_design['layersizes'])
        model.load_state_dict(torch.load(''.join((data_dir, f"mlpaug_model{i}.pth"))))
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



    pd.DataFrame.from_dict(performance).to_csv('mlp_eval_performanceaug.csv')
    pd.DataFrame.from_dict(preds_train).to_csv('mlp_eval_preds_trainaug.csv')
    pd.DataFrame.from_dict(preds_test).to_csv('mlp_eval_preds_testaug.csv')



evalmlp()
    


import matplotlib.pyplot as plt
test_y = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/gpptest2008.csv" )
augmlp = pd.read_csv("C:/Users/Arpit/mlp_eval_preds_testaug.csv")
plt.plot(range(len(test_y)), test_y['GPP'], label='Observed GPP', marker='o')
plt.plot(range(len(augmlp)), augmlp['test_mlp2'], label='Predicted GPP by MLP ', marker='o')

# Customize the plot
plt.title("GPP plot 2008 Augmented data", fontsize=16)
plt.xlabel("Month", fontsize=14)
plt.ylabel("Values    (tonnes/hectare)", fontsize=14)
plt.legend()















import matplotlib.pyplot as plt
test_y = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/gpptest2008.csv" )
regmlpaug = pd.read_csv("C:/Users/Arpit/regaug_eval_preds_test.csv")

plt.plot(range(len(test_y)), test_y['GPP'], label='Observed GPP', marker='o')
plt.plot(range(len(regmlpaug)), regmlpaug['test_reg2'], label='Predicted GPP by Reg ', marker='o')
plt.title("GPP plot 2008 Augmented data", fontsize=16)
plt.xlabel("Month", fontsize=14)
plt.ylabel("Values    (tonnes/hectare)", fontsize=14)
plt.legend()








