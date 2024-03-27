from pyGameWorld import PGWorld, ToolPicker, loadFromDict
from pyGameWorld.viewer import *
from pyGameWorld.jsrun import *
from pyGameWorld.helpers import *
import json
import pygame as pg
import pymunk as pm
from gaussian_policy_torch import Gaussian2DPolicy, random_init_policy, plot_policies
from random import choice, randint, random, choices
import numpy as np
from utils import *
import argparse
from copy import deepcopy
import networkx as nx
from collections import Counter
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import pygraphviz
from datetime import datetime
import os
from strategy import *


experiment_id = generate_experiment_id()
dir_name = 'data/'+experiment_id
os.makedirs(dir_name)

# Load level in from json file
# For levels used in experiment, check out Level_Definitions/
json_dir = "./Trials/Strategy/"
tnm = "CatapultAlt"
# tnm = "Basic"    

with open(json_dir+tnm+'.json','r') as f:
  btr0 = json.load(f)

tp0 = ToolPicker(btr0)
# NOTE - movable_objects: red and blue objects
# movable_objects = [ obj for obj in list(btr['world']['objects'].keys()) if btr['world']['objects'][obj]['color'] in ['blue', 'red']] + list(btr['tools'].keys())

movable_obj_dict = {i:j for i,j in tp0.objects.items() if j.color in [(255, 0, 0, 255), (0, 0, 255, 255)]}
movable_objects = list(movable_obj_dict.keys())
# get inital path with no tool
path_dict0, success, time_to_success = tp0.observeStatePath()

# NOTE - StrategyGraph
strategy_graph = StrategyGraph()

success = False

while not success:
    btr = deepcopy(btr0)
    tp = ToolPicker(btr)
    sample_obj = choices(movable_objects, weights=[1/(strategy_graph.obj_count[m]+1) for m in movable_objects], k=1)[0] # TODO - sample from movable objects and tools
    pos = None
    if sample_obj in tp.toolNames:
        scaled_pos = get_prior_SSUP(obj_dict)
        path_dict, success, time_to_success = tp.observePlacementStatePath(toolname=sample_obj,position=scaled_pos)
    else:
        # sample_ext = choice(['pos', 'vel', 'kick']) # choose what counterfactual extrinsics
        sample_ext = choice(['vel', 'kick']) # choose what counterfactual extrinsics
        rand_rad = random()*2*np.pi
        if sample_ext == 'pos': # fixed pos, smaple from path
            # TODO - should sample from other counterfactual conditions 
            sample_pose = choice(path_dict0[sample_obj]) # sample from intial path
            pos0, rot0 = sample_pose[0:2], sample_pose[2]
            # change the worlddict directly
            btr['world']['objects'][sample_obj]['position'] = (pos0[0]+rand_rad*randint(0,10), pos0[1]+rand_rad*randint(0,10))
            btr['world']['objects'][sample_obj]['rotation'] = (rot0+(random()-0.5)*0.1)
            btr['world']['objects'][sample_obj]['color'] = (0, 0, 0)
        elif sample_ext == 'vel': # set velocity
            random_vel = randint(1,50)*10
            if sample_obj == "KeyBall":
                random_vel = randint(1,50)*2
            if sample_obj == "CataBall":
                rand_rad = 0.5*np.pi + random() * 0.5*np.pi
            velocity = (np.cos(rand_rad)*random_vel,  np.sin(rand_rad)*random_vel)
            btr['world']['objects'][sample_obj]['velocity'] = velocity
        elif sample_ext == 'kick':
            # pos = (randint(0,600), randint(0,600))
            random_imp = randint(1, 50)*10000
            rand_rad = -1.5*np.pi
            impulse = pm.Vec2d(np.cos(rand_rad)*random_imp,  np.sin(rand_rad)*random_imp)

        tp = ToolPicker(btr)

        # NOTE - simulate path
        # TODO - check placement collide
        # print('sample_ext', sample_ext)
        if sample_ext == 'kick': # apply random force
            
            # pos = pm.Vec2d(randint(BB[0][0],BB[1][0]), randint(BB[0][1],BB[1][1]))
            # tp.world.kick(sample_obj, impulse, pos)
            # force_times = {(i*0.1):[[sample_obj, impulse, pos]] for i in range(2)}
            force_times = {0.0:[[sample_obj, impulse, pos]], 0.1:[[sample_obj, impulse, pos]], 0.2:[[sample_obj, impulse, pos]]}
            path_dict, collisions, success, time_to_success = pyGetCollisionsAddForces(tp.world, force_times=force_times)
            init_pose = path_dict[sample_obj][2]
        else:
            path_dict, success, time_to_success = tp.observeStatePath()
            path, collisions, end, t = tp.observeFullCollisionEventsNoTool()
            # for c in col:
            #     print(c)
            init_pose = path_dict[sample_obj][1] # NOTE - second is more stable
        collision_pattern = [c[0:2] for c in  collisions]

        if sample_ext == 'kick':
            draw_path(tp, path_dict, dir_name+'/test_kick.png')

    # TODO - add check desire effect function
    if success and sample_obj in tp.toolNames: # tools
        print("Success!", sample_obj, btr['tools'][sample_obj], sample_ext)
    elif success and sample_obj not in tp.toolNames: # objects
        print("Success! {:>10} {:>28} {:>4}".format(sample_obj, str([int(num) for num in init_pose]), sample_ext))
        change_obj = []
        # NOTE - build strategy
        ext_info = {'pos': init_pose[0:2], 'rot': init_pose[2], 'vel': init_pose[3:5], 'path': path_dict, 'collision': collision_pattern}
        sample_strat_id = strategy_graph.add_strategy_by_ext(tp, sample_obj, ext_info) # update strategy graph

        sample_strat = strategy_graph.get_strtegy(sample_strat_id)

        path_set = sample_strat.get_paths()

        draw_multi_paths(btr['world'], path_set, dir_name+'/'+strategy_graph.graph.nodes[sample_strat]['label']+'.png')

        # NOTE - link succ strategies with all contact objects


        # pred
        strategy_graph.check_strategy_successors(sample_strat, sample_obj, path_dict, path_dict0, collision_pattern)
        strategy_graph.check_strategy_predecessors(sample_strat, sample_obj, path_dict[sample_obj], collision_pattern)


        strategy_graph.merge_strategies(sample_strat)
        
                        
        # movable_objects = [item for item in movable_objects if item != sample_obj]
        success = False
        draw_path(tp, path_dict, dir_name+'/test.png')

        strategy_graph.save_graph(dir_name+'/graph_.png')
        strategy_graph.transitive_reduction()
        strategy_graph.save_graph(dir_name+'/graph.png')

