from pyGameWorld import PGWorld, ToolPicker, loadFromDict
from pyGameWorld.viewer import *
import json
import pygame as pg
from gaussian_policy_torch import Gaussian2DPolicy, random_init_policy, plot_policies
from random import choice, randint, random, choices
import numpy as np
from utils import *
import argparse

# Load level in from json file
# For levels used in experiment, check out Level_Definitions/
json_dir = "./Trials/Strategy/"
tnm = "CatapultAlt"
# tnm = "Basic"

with open(json_dir+tnm+'.json','r') as f:
  btr = json.load(f)


tp = ToolPicker(btr)

# View that placement
# demonstrateTPPlacement(tp, 'obj2', (300, 500))
path_dict, success, time_to_success = tp.observeFullPlacementPath('obj2', (240, 550))
print(success, time_to_success)
# path_dict, success, time_to_success = tp.observeFullPath()
if path_dict:
    pg.display.set_mode((10,10))
    sc = drawPathSingleImageWithTools(tp, path_dict, with_tools=True)
    img = sc.convert_alpha()
    pg.image.save(img, 'data/test.png')