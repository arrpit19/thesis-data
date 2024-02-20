

import hp3pgn

import torch
import pandas as pd
import numpy as np
import argparse


def ENres2():
    x= pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/augclim.csv")
    y= pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/auggppNAS.csv")
    
    
    splits = 4
    y3pgn = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/augr3pgnNASpred.csv")

    splits = 4

    #y = y.to_frame()
    x.index, y.index, y3pgn.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(y3pgn))
            
    arch_grid, par_grid = hp3pgn.NASSearchSpace(x.shape[1], y.shape[1], 100, 100, 4)
    res = hp3pgn.NASSearch(arch_grid, par_grid, x, y, splits, "NASpres2", res=2, y3pgn=y3pgn, hp=True)
    res.to_csv("Nres2HPNAS.csv")


ENres2()
