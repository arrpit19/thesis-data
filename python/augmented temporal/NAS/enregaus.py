

import hp3pgn
import training3pgn
import models3pgn
import torch
import pandas as pd
import numpy as np
import argparse


def ENreg():
    
   


    x= pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/augclim.csv")
    y= pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/auggppNAS.csv")
    
    yp = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/augr3pgnNASpred.csv")
    reg = yp
    print(reg)

    splits =4 

    print("INPUTS: \n", x, "Outputs: \n", y, "RAW DATA: \n", reg)
    x.index, y.index, reg.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(reg))

    arch_grid, par_grid = hp3pgn.NASSearchSpace(x.shape[1], y.shape[1], 100, 100, 4, reg=True)
    res = hp3pgn.NASSearch(arch_grid, par_grid, x, y, splits, "NASreg", reg=reg, hp=True)
    res.to_csv("f-regNAS.csv")


ENreg()

