import hp3pgn
import torch
import pandas as pd
import numpy as np
import argparse


def ENres2():
    data_file1 = "C:/Users/Arpit/Documents/ProfoundData/nasdataSSscaled.csv" 
    data_file2 = "C:/Users/Arpit/Documents/ProfoundData/YNAS.csv"


    x = pd.read_csv(data_file1)

    print(x)


    y = pd.read_csv(data_file2)
    
    splits = 2
    y3pgn = pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/ypredNAS.csv")

    splits = 2

    #y = y.to_frame()
    x.index, y.index, y3pgn.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(y3pgn))
            
    arch_grid, par_grid = hp3pgn.NASSearchSpace(x.shape[1], y.shape[1], 100, 100, 4)
    res = hp3pgn.NASSearch(arch_grid, par_grid, x, y, splits, "NASpres2", res=2, y3pgn=y3pgn, hp=True)
    res.to_csv("Nres2HP.csv")


ENres2()


