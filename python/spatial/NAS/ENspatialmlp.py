
import hp3pgn
import models3pgn
import torch
import pandas as pd
import numpy as np
import argparse



def EN2mlp():
    
    x= pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/multisiteclimS2004.csv")
    y= pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/multisitegpp2004.csv")
    splits = 4
    
    x.index, y.index = np.arange(0, len(x)), np.arange(0, len(y))

    arch_grid, par_grid = hp3pgn.NASSearchSpace(x.shape[1], y.shape[1], 100, 100, 4)
    res = hp3pgn.NASSearch(arch_grid, par_grid, x, y, splits, "2hpmlp", exp=2, hp=True)

    res.to_csv("N2mlpHP.csv")


EN2mlp()
