import hp3pgn
import models3pgn
import torch
import pandas as pd
import numpy as np
import argparse
import torch
import pandas as pd
import numpy as np
import argparse


def EN2reg():
    x= pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/multisiteclimS2004.csv")
    y= pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/multisitegpp2004.csv")
    reg=pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/gpptrainmulti2004.csv")
    splits = 4
    print(x,y, reg)
    
    x.index, y.index, reg.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(reg))
    arch_grid, par_grid = hp3pgn.NASSearchSpace(x.shape[1], y.shape[1], 100, 100, 4, reg=True)
    res = hp3pgn.NASSearch(arch_grid, par_grid, x, y, splits, "NASreg", reg=reg, exp=2, hp=True)
    res.to_csv("N2regHP.csv")


EN2reg()

