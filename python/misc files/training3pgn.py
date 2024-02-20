import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import KFold
import models3pgn
import random
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
from torch.autograd import Variable
import os

def train_cv(hparams, model_design, X, Y, data_dir, splits, data, reg=None, raw=None, emb=False, res=None, y3pgn=None, exp=None, hp=False):
    print("Hyperparams", hparams)
    nepoch = hparams['epochs']
    batchsize = hparams['batchsize']
    if reg is not None:
        eta = hparams['eta']
    kf = KFold(n_splits=splits, shuffle = False)
    kf.get_n_splits(X)
    
    mse_t = np.empty((splits, nepoch))
    mse_v = np.empty((splits, nepoch))
    mae = nn.L1Loss()
    if hp:
        mse_t = np.empty(nepoch)
        mse_v = np.empty(nepoch)
    mae_v = np.empty(splits)
        
    predstest = {}
    predstrain = {}
    i = 0

    for t_idx, v_idx in kf.split(X):
        
        x_train, x_val = torch.tensor(X[X.index.isin(t_idx)].to_numpy(), dtype=torch.float32), torch.tensor(X[X.index.isin(v_idx)].to_numpy(), dtype=torch.float32)
        y_train, y_val = torch.tensor(Y[Y.index.isin(t_idx)].to_numpy(), dtype=torch.float32), torch.tensor(Y[Y.index.isin(v_idx)].to_numpy(), dtype=torch.float32)
        
        if reg is not None:
            yp_train, yp_val = torch.tensor(reg.loc[t_idx].to_numpy(), dtype=torch.float32), torch.tensor(reg.loc[v_idx].to_numpy(), dtype=torch.float32)

            train_set = TensorDataset(x_train, y_train, yp_train)
            val_set = TensorDataset(x_val, y_val, yp_val)
        elif res == 2:
            yp_train, yp_val = torch.tensor(y3pgn.loc[t_idx].to_numpy(), dtype=torch.float32), torch.tensor(y3pgn.loc[v_idx].to_numpy(), dtype=torch.float32)
            train_set = TensorDataset(x_train, y_train, yp_train)
            val_set = TensorDataset(x_val, y_val, yp_val)
        elif not emb:
            train_set = TensorDataset(x_train, y_train)
            val_set = TensorDataset(x_val, y_val)
            
        train_set_size = len(train_set)
        sample_id = list(range(train_set_size))
        val_set_size = len(val_set)
        vsample_id = list(range(val_set_size))
       
        train_sampler = torch.utils.data.sampler.RandomSampler(sample_id)
        val_sampler = torch.utils.data.sampler.RandomSampler(vsample_id)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, sampler=train_sampler, shuffle=False)
        batch_validation = False
        if not batch_validation:
            batchsize = len(val_set)
 
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batchsize, sampler=val_sampler, shuffle=False)

        
                            
        if res == 2:
            model = models3pgn.RES(X.shape[1], Y.shape[1], model_design['layersizes'])
       
        else:
            model = models3pgn.NMLP(X.shape[1], Y.shape[1], model_design['layersizes'])

        print("INIMODEL", model)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr = hparams['lr'])
        train_loss = []
        val_loss = []
        print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        for ep in range(nepoch):
            model.train()
            batch_diff = []
            batch_loss = []
            for step, train_data in enumerate(train_loader):
                xt = train_data[0]
                yt = train_data[1]
                if reg is not None or res == 2:
                    yp = train_data[2]
                

                optimizer.zero_grad()

                # forward
               
                if res == 1:
                    y_hat = model(xt)
                elif res == 2:
                    y_hat = model(xt, yp)
                else:
                    y_hat = model(xt)
                if reg is not None and not emb:
                    loss = criterion(y_hat, yt) + eta*criterion(y_hat, yp)
               
                else:
                    loss = criterion(y_hat, yt)

                # backward
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            model.eval()
            # results per epoch
            train_loss = sum(batch_loss)/len(batch_loss)

            if exp == 2 and not hp:
                mse_t[i, ep] = train_loss
                
            if hp or exp != 2:
                # test model
                model.eval()
                
                e_bl = []
                # deactivate autograd
                
                for step, val_sample in enumerate(val_loader):
                    x_vall = val_sample[0]
                    y_vall = val_sample[1]
                    if reg is not None or res == 2:
                        yp_vall = val_sample[2]
                        
                       
                        if res == 1:
                            y_hat_val = model(x_vall)
                        elif res == 2:
                            
                            y_hat_val = model(x_vall, yp_vall)
                        else:
                            y_hat_val = model(x_vall)
                        
                    
                    else:
                        y_hat_val = model(x_vall)

                    if reg is not None and not emb:
                        loss = criterion(y_hat_val, y_vall) + eta*criterion(y_hat_val, yp_vall)
                    
                    else:
                        loss = criterion(y_hat_val, y_vall)
                    e_bl.append(loss.item())

            if exp != 2 or hp:
                val_loss = (sum(e_bl) / len(e_bl))
            if not hp and exp !=2:
                mse_t[i, ep] = train_loss
                mse_v[i, ep] = val_loss
            if hp:
                mse_t[ep] = train_loss
                mse_v[ep] = val_loss
            if exp == 2 and not hp:
                
                model.eval()
                
                if res == 1:
                    y_hat_val = model(x_val)
                    y_hat_t = model(x_train)
                elif res == 2:
                    
                    y_hat_val = model(x_val, yp_val)
                    y_hat_t = model(x_train, yp_train)
                else:
                    y_hat_val = model(x_val)
                    y_hat_t = model(x_train)
                mse_v[i, ep] = criterion(y_hat_val, y_val)
                
        if hp:
            
            return {'train_loss': mse_t, 'val_loss':mse_v}
        elif exp == 2 and not hp:
            model.eval()
           
            if res == 1:
                y_hat_val = model(x_val)
                y_hat_t = model(x_train)
            elif res == 2:
                y_hat_val = model(x_val, yp_val)
                y_hat_t = model(x_train, yp_train)
            else:
                y_hat_val = model(x_val)
                y_hat_t = model(x_train)
            
            predstrain = {f"site{i}": y_hat_t.detach().flatten().numpy()}
            predstest = {f"site{i}": y_hat_val.detach().flatten().numpy()}
            pd.DataFrame.from_dict(predstrain).to_csv(f'2{data}train{i}.csv')
            pd.DataFrame.from_dict(predstest).to_csv(f'2{data}test{i}.csv')
        if exp == 2:
            mae_v[i] = mae(y_hat_val, y_val)
            
        i += 1
            
        if exp != 2:
              print("Saving model to: ", os.path.join(data_dir, f"{data}_model{i}.pth"))
              torch.save(model.state_dict(), os.path.join(data_dir, f"{data}_model{i}.pth"))
      
        else: 
            torch.save(model.state_dict(), os.path.join(data_dir, f"2{data}_model{i}.pth"))
        
    if exp == 2 and not hp:
        td = {}
        se = {}
        ae = {}
        for i in range(splits):
            td.update({f"site{i}": mse_t[i, :]})
            se.update({f"site{i}": mse_v[i, :]})
            ae.update({f"site{i}": np.array([mae_v[i]])})
        out =  td, se, ae
    else:
        out = {"train_loss": mse_t, "val_loss": mse_v}
                    
    return out

