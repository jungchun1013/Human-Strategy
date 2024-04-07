import argparse
import json
from random import choice, randint, random, choices
import os
from copy import deepcopy
import sys
# import logging
import math
import numpy as np
from pyGameWorld import ToolPicker, objectBoundingBox
import pymunk as pm
from pyGameWorld.jsrun import pyGetCollisionsAddForces, pyGetCollisions, pyGetCollisionsPlacement
from src.strategy import StrategyGraph
from src.utils import *

def sample(args, strategies_graph):
    tp = args.tp
    btr = args.btr
    weights = [ math.e**(-strategies_graph.obj_count[m]*0.2)
        for m in args.available_objects
    ]
    sample_obj = choices(args.available_objects, weights=weights, k=1)[0]
    pos = None


    is_gmm_sample = False
    gmm_sample_pos = None
    if random() < 0.05 and strategies_graph.full_placement_graphs: # exploit
        graph = choice(strategies_graph.full_placement_graphs)
        obj_list = [nd for nd in graph.nodes() if 'gmm' in graph.nodes[nd]]
        obj_weight = [ 1/len(graph.nodes[nd]['ext'])
            for nd in graph.nodes() if 'gmm' in graph.nodes[nd]
        ]
        if obj_list:
            print('Exploiting')
            sample_obj = choices(obj_list, weights=obj_weight)[0]
            gmm_sample = graph.nodes[sample_obj]['gmm'].sample()[0][0]
            is_gmm_sample = True
            if sample_obj in args.tool_objects:
                gmm_sample_pos = list(gmm_sample[0:2])
                print(gmm_sample_pos)
                path_dict, success, time_to_success = tp.observePlacementStatePath(
                    toolname=sample_obj,
                    position=gmm_sample_pos
                )
                path, collisions, end, _ = pyGetCollisionsPlacement(
                    tp.world,
                    tp,
                    sample_obj,
                    gmm_sample_pos)
                init_pose = [gmm_sample_pos[0], gmm_sample_pos[1], 0, 0, 0]
                sample_ext = 'tool'
                path_info = path_dict, collisions, success, time_to_success
            else:
                # btr['world']['objects'][sample_obj]['position'] = gmm_sample[0:2]
                # btr['world']['objects'][sample_obj]['rotation'] = gmm_sample[2]
                btr['world']['objects'][sample_obj]['velocity'] = gmm_sample[3:]
                tp = ToolPicker(btr)
                path_dict, success, time_to_success = tp.observeStatePath()
                path, collisions, end, _ = pyGetCollisions(tp.world)
                init_pose = path_dict[sample_obj][0]
                sample_ext = 'exploit'
                path_info = path_dict, collisions, success, time_to_success
            return sample_obj, sample_ext, init_pose, path_info

    if sample_obj in args.tool_objects:
        scaled_pos = get_prior_SSUP(args.tp0) if not is_gmm_sample else gmm_sample[0:2]
        path_dict, success, time_to_success = tp.observePlacementStatePath(
            toolname=sample_obj,
            position=scaled_pos
        )
        path, collisions, end, _ = pyGetCollisionsPlacement(
            tp.world,
            tp,
            sample_obj,
            scaled_pos
        )

        init_pose = [scaled_pos[0], scaled_pos[1], 0, 0, 0]
        sample_ext = 'tool'
        path_info = path_dict, collisions, success, time_to_success
    else:
        # NOTE - choose what counterfactual extrinsics
        # sample_ext = choice(['pos', 'vel', 'kick'])
        sample_ext = choice(['vel', 'kick'])
        rand_rad = random()*2*np.pi
        # SECTION - sample extrinsics and simulate path
        if sample_ext in ['pos', 'vel']: # fixed pos, smaple from path
            if sample_ext == 'pos':
                btr = args.ext_sampler.sample_pos(sample_obj)
            elif sample_ext == 'vel': # set velocity
                btr = args.ext_sampler.sample_vel(sample_obj)
            tp = ToolPicker(btr)
            path_dict, success, time_to_success = tp.observeStatePath()
            path, collisions, end, t = pyGetCollisions(tp.world)
            init_pose = path_dict[sample_obj][0]
            path_info = path_dict, collisions, success, time_to_success
        elif sample_ext == 'kick':
            random_scale = randint(1,50) * 10000
            # sample_obj = 'Lever'
            rand_rad = -1.5*np.pi # downward
            impulse = pm.Vec2d(np.cos(rand_rad)*random_scale,  np.sin(rand_rad)*random_scale)
            BB = objectBoundingBox(tp.objects[sample_obj])
            pos = pm.Vec2d( \
                randint(BB[0][0],BB[1][0])-(BB[1][0]-BB[0][0])/2-BB[0][0], \
                randint(BB[0][1],BB[1][1])-(BB[1][1]-BB[0][1])/2-BB[0][1])
            tp = ToolPicker(btr)
            force_times = {
                0.0:[[sample_obj, impulse, pos]],
                0.1:[[sample_obj, impulse, pos]],
                0.2:[[sample_obj, impulse, pos]]
            }
            path_info = pyGetCollisionsAddForces(tp.world, force_times=force_times)
            path_dict, collisions, success, time_to_success = path_info
            init_pose = path_dict[sample_obj][2]
    
    return sample_obj, sample_ext, init_pose, path_info

def standardize_collisions(collisions):
    for i, c in enumerate(collisions):
        o1, o2, ts, te, ci = c
        if isinstance(ci[0], pm.Vec2d):
            ci[0] = [ci[0].x, ci[0].y]
        collisions[i] = [o1, o2, ts, te, ci]
    return collisions  

def run(args):
    args.experiment_id = generate_experiment_id()
    args.dir_name = 'data/' + args.experiment_id
    os.makedirs(args.dir_name)

    # Load level in from json file
    # For levels used in experiment, check out Level_Definitions/
    args.json_dir = "./environment/Trials/Strategy/"
    # Basic Table_A
    args.tnm = "CatapultAlt"

    with open(args.json_dir + args.tnm + '.json','r') as f:
        args.btr0 = json.load(f)

    args.tp0 = ToolPicker(args.btr0)

    movable_obj_dict = {i:j for i, j in args.tp0.objects.items()
                        if j.color in [(255, 0, 0, 255), (0, 0, 255, 255)]
    }
    args.movable_objects = list(movable_obj_dict.keys())
    args.tool_objects = list(args.tp0.toolNames)
    args.available_objects = args.movable_objects + args.tool_objects

    path_dict0, success, _ = args.tp0.observeStatePath()
    args.ext_sampler = ExtrinsicSampler(args.btr0, path_dict0)
    counterfactual_run(args)



def counterfactual_run(args):
    # NOTE - MechanismGraph
    strategies_graph = StrategyGraph()
    success = False
    image_num = 0
    sample_num = 0
    while not success:
        args.btr = deepcopy(args.btr0)
        args.tp = ToolPicker(args.btr)

        sample_obj, sample_ext, init_pose, path_info = sample(args, strategies_graph)
        path_dict, collisions, success, _ = path_info
        collisions = standardize_collisions(collisions)

        # !SECTION
        if success:
            strategies_graph.build_graph(args, sample_obj, init_pose, path_info)
            success = False
            image_num += 1
            # sample
            for g in strategies_graph.full_placement_graphs:
                if all('model' in g.nodes[nd] for nd in g.nodes()
                    if nd not in args.tp.toolNames
                ):
                    cur_nd = 'Goal'
                    gmm = g.nodes[cur_nd]['model']
                    sample_pos = gmm.sample(n_samples=2)
                    print('Sampled: Goal', sample_pos[0])
                    sample_pos = sample_pos[0]

                    while True:
                        pred = choice(list(g.predecessors(cur_nd)))
                        cur_nd = pred
                        if cur_nd in  args.tp.toolNames:
                            print('Tool:', cur_nd)
                            break
                        gpr = g.nodes[cur_nd]['model']
                        samples = gpr.sample_y(sample_pos, n_samples=2)
                        arr = np.array(samples)
                        sample_pos = np.transpose(arr, (0, 2, 1))
                        sample_pos = [item for sublist in sample_pos for item in sublist]

                        print('Sampled:', cur_nd, sample_pos)

                    sample_obj = cur_nd
                    for s in sample_pos:
                        scaled_pos = list(s[0:2])
                        path_dict, success, _ =  args.tp.observePlacementStatePath(
                            toolname=sample_obj,
                            position=scaled_pos
                        )
                        print('Success:', cur_nd, scaled_pos, success)
                        if success is False:
                            sample_num += 1
                            print('Sample Num:', sample_num)
                        if success:
                            sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SSUP')

    parser.add_argument('-a', '--algorithm',
                        help='SSUP/ours', default='SSUP')
    parser.add_argument('-t', '--num_trial',
                        help='number of trials', type=int, default=250)
    parser.add_argument('--tnm',
                        help='task name', type=int, default=250)
    parser.add_argument('-v', '--verbose',
                        help='Increase output verbosity', action='store_true')
    parser.add_argument('--eps',
                        help='epsilon', type=int, default=0.2)
    parser.add_argument('--eps-decay-rate',
                        help='epsilon decay rate', type=int, default=0.95)
    parser.add_argument('--max_attempt',
                        help='max number of attempt', type=int, default=200)
    parser.add_argument('--max_sim_attempt',
                        help='max number of simulated attempt', type=int, default=1000)
    args = parser.parse_args()

    run(args)
