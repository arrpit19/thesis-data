import torch
import pandas as pd
import numpy as np
import argparse
import random
import models3pgn
import training3pgn
import hp3pgn


x= pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/augclim.csv")
y= pd.read_csv("C:/Users/Arpit/Documents/ProfoundData/auggppNAS.csv")

x.index, y.index = np.arange(0, len(x)), np.arange(0, len(y))
arch_grid, par_grid = hp3pgn.NASSearchSpace(x.shape[1], y.shape[1], 100, 100, 4)

print(len(par_grid))

splits=4

res = hp3pgn.NASSearch(arch_grid, par_grid, x, y, splits, "NASmlp", hp=True)

res.to_csv("faug.csv")