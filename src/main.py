import argparse
import json
from random import choice, randint, random, choices
import os
from copy import deepcopy
import sys
# import logging
import math
import numpy as np
from numpy import random as rd
from pyGameWorld import ToolPicker, objectBoundingBox
from pyGameWorld.jsrun import pyGetCollisionsAddForces, pyGetCollisions, pyGetCollisionsPlacement
import pymunk as pm
from src.strategy import StrategyGraph
from src.utils import ExtrinsicSampler, generate_experiment_id, get_prior_SSUP, draw_path

def get_simluation_result_from_model(args, btr, sample_obj, sample_ext, graph):
    '''
        sample from simple gaussian model in graph
    '''
    tp = args.tp
    gmm_sample = graph.nodes[sample_obj]['gmm'].sample()[0][0]
    if sample_obj in args.tool_objects:
        gmm_sample_pos = list(gmm_sample[0:2])
        path_dict, success, _ = tp.observePlacementStatePath(
            toolname=sample_obj,
            position=gmm_sample_pos
        )
        path, collisions, end, _ = pyGetCollisionsPlacement(
            tp.world,
            tp,
            sample_obj,
            gmm_sample_pos)
        init_pose = [gmm_sample_pos[0], gmm_sample_pos[1], 0, 0, 0]
    else:
        # btr['world']['objects'][sample_obj]['position'] = gmm_sample[0:2]
        # btr['world']['objects'][sample_obj]['rotation'] = gmm_sample[2]
        btr['world']['objects'][sample_obj]['velocity'] = gmm_sample[3:]
        tp = ToolPicker(btr)
        path_dict, success, _ = tp.observeStatePath()
        path, collisions, end, _ = pyGetCollisions(tp.world)
        init_pose = path_dict[sample_obj][0]
    path_info = path_dict, collisions, success
    return sample_obj, sample_ext, init_pose, path_info


def simulate(args, btr, sample_obj, sample_ext, init_pose):
    '''
        sample from given object and extrinsics type
    '''
    tp = args.tp
    if sample_ext in ['pos', 'vel', 'ang']: # fixed pos, smaple from path
        btr['world']['objects'][sample_obj]['position'] = gmm_sample[0:2]
        btr['world']['objects'][sample_obj]['rotation'] = gmm_sample[2]
        btr['world']['objects'][sample_obj]['velocity'] = gmm_sample[3:]
        tp = ToolPicker(btr)
        path_dict, success, _ = tp.observeStatePath()
        path, collisions, end, _ = pyGetCollisions(tp.world)
        # path_dict, collisions, success, _ = tp.runStatePath(noisy=noisy)
        init_pose = path_dict[sample_obj][0]
    elif sample_ext == 'tool':
        sample_pos = init_pose[0:2]
        path_dict, success, _ = tp.observePlacementStatePath(
            toolname=sample_obj,
            position=sample_pos
        )
        path, collisions, end, _ = pyGetCollisionsPlacement(
            tp.world,
            tp,
            sample_obj,
            sample_pos,
        )
        # path_dict, collisions, success, _ = tp.runStatePath(
        #     sample_obj,
        #     sample_pos,
        #     noisy=noisy
        # )
        init_pose = [sample_pos[0], sample_pos[1], 0, 0, 0]
    elif sample_ext == 'kick':
        init_pose = {
            0.0:[[sample_obj, impulse, sample_pos]],
            0.1:[[sample_obj, impulse, sample_pos]],
            0.2:[[sample_obj, impulse, sample_pos]]
        }
        path_dict, collisions, success, _ = pyGetCollisionsAddForces(
            tp.world,
            force_times=force_times
        )
        if not noisy:
            init_pose = path_dict[sample_obj][2]
        else:
            init_pose = force_times

    path_info = path_dict, collisions, success
    return sample_obj, sample_ext, init_pose, path_info

def sample_and_simulate(args, btr, sample_obj, sample_ext, noisy=False):
    '''
        sample from given object and extrinsics type
    '''
    tp = args.tp
    if sample_ext in ['pos', 'vel', 'ang']: # fixed pos, smaple from path
        if sample_ext == 'pos':
            btr = args.ext_sampler.sample_pos(sample_obj)
        elif sample_ext == 'vel': # set velocity
            btr = args.ext_sampler.sample_vel(sample_obj)
        elif sample_ext == 'ang':
            btr = args.ext_sampler.sample_ang(sample_obj)
        tp = ToolPicker(btr)
        # path_dict, success, _ = tp.observeStatePath()
        # path, collisions, end, _ = pyGetCollisions(tp.world)
        path_dict, collisions, success, _ = tp.runStatePath(noisy=noisy)
        init_pose = path_dict[sample_obj][0]
    elif sample_ext == 'tool':
        sample_pos = get_prior_SSUP(args.tp0, args.movable_objects)
        # path_dict, success, t = tp.observePlacementStatePath(
        #     toolname=sample_obj,
        #     position=sample_pos
        # )
        # path, collisions, end, _ = pyGetCollisionsPlacement(
        #     tp.world,
        #     tp,
        #     sample_obj,
        #     sample_pos,
        # )
        path_dict, collisions, success, t = tp.runStatePath(
            sample_obj,
            sample_pos,
            noisy=noisy
        )
        init_pose = [sample_pos[0], sample_pos[1], 0, 0, 0]
    elif sample_ext == 'kick':
        rand_rad = random()*2*np.pi
        random_scale = randint(1,50) * 10000
        impulse = pm.Vec2d(np.cos(rand_rad)*random_scale,  np.sin(rand_rad)*random_scale)
        BB = objectBoundingBox(tp.objects[sample_obj])
        sample_pos = pm.Vec2d( \
            randint(BB[0][0],BB[1][0])-(BB[1][0]-BB[0][0])/2-BB[0][0], \
            randint(BB[0][1],BB[1][1])-(BB[1][1]-BB[0][1])/2-BB[0][1])
        tp = ToolPicker(btr)
        force_times = {
            0.0:[[sample_obj, impulse, sample_pos]],
            0.1:[[sample_obj, impulse, sample_pos]],
            0.2:[[sample_obj, impulse, sample_pos]]
        }
        path_dict, collisions, success, _ = pyGetCollisionsAddForces(
            tp.world,
            force_times=force_times
        )
        if not noisy:
            init_pose = path_dict[sample_obj][2]
        else:
            init_pose = force_times

    path_info = path_dict, collisions, success
    return sample_obj, sample_ext, init_pose, path_info

def sample_from_counterfactual(args, strategies_graph):
    '''
        do sampling
    '''
    success = False
    while not success:
        btr = args.btr
        ext_prob = 0.1 if strategies_graph.full_placement_graphs else 0
        obj_weight = sum([math.e**(-strategies_graph.obj_count[m]*0.3) for m in args.movable_objects])
        tool_weight = sum([math.e**(-strategies_graph.obj_count[m]*0.3) for m in args.tool_objects])
        ext_weights = [obj_weight, obj_weight, obj_weight, tool_weight, ext_prob]
        ext_weights = [w/sum(ext_weights) for w in ext_weights]
        sample_exts = rd.choice(
            ['pos', 'vel', 'kick', 'tool', 'exploit'],
            # ['pos', 'vel', 'ang', 'kick', 'tool', 'exploit'],
            p=ext_weights,
            size=2,
            replace=False)
        sample_ext = sample_exts[0]
        if sample_ext == 'exploit':
            graph = choice(strategies_graph.full_placement_graphs)
            obj_list = [nd for nd in graph.nodes() if 'gmm' in graph.nodes[nd]]
            # obj_weight = [ 1/len(graph.nodes[nd]['ext'])
            #     for nd in graph.nodes() if 'gmm' in graph.nodes[nd]
            # ]
            obj_weight = [ math.e**(-strategies_graph.obj_count[nd]*0.3)
                for nd in obj_list]
            if obj_list:
                sample_obj = choices(obj_list, weights=obj_weight)[0]
                return get_simluation_result_from_model(args, btr, sample_obj, sample_ext, graph)

            sample_ext = sample_exts[1]

        if sample_ext in ['pos', 'vel', 'ang']: # fixed pos, smaple from path
            weights = [ math.e**(-strategies_graph.obj_count[m]*0.3)
                for m in args.movable_objects
            ]
            sample_obj = choices(args.movable_objects, weights=weights, k=1)[0]
        elif sample_ext == 'kick':
            weights = [ math.e**(-strategies_graph.obj_count[m]*0.3)
                for m in args.movable_objects
            ]
            sample_obj = choices(args.movable_objects, weights=weights, k=1)[0]
        elif sample_ext == 'tool':
            sample_obj = choice(args.tool_objects)
        else:
            print("No sample", sample_exts)

        sample_obj, sample_ext, init_pose, path_info = sample_and_simulate(args, btr, sample_obj, sample_ext, noisy=True)
        path_dict, collisions, success = path_info
        if success:
            print('Noisy Success:', sample_obj, sample_ext)

    sample_obj, sample_ext, init_pose, path_info = sample_and_simulate(args, btr, sample_obj, sample_ext, noisy=False)
    return sample_obj, sample_ext, init_pose, path_info

def sample_from_strategy_graph(args, strategies_graph):
    '''
        do sample from gaussian process model in graph
    '''
    g = choice(strategies_graph.full_placement_graphs)
    if all('model' in g.nodes[nd] for nd in g.nodes()
        if nd not in args.tp.toolNames
    ):
        cur_nd = 'Goal'
        gmm = g.nodes[cur_nd]['model']
        sample_pos = gmm.sample(n_samples=2)
        # print('Sampled: Goal', sample_pos[0])
        sample_pos = sample_pos[0]

        while True:
            pred = choice(list(g.predecessors(cur_nd)))
            cur_nd = pred
            if cur_nd in args.tp.toolNames:
                break
            gpr = g.nodes[cur_nd]['model']
            samples = gpr.sample_y(sample_pos, n_samples=2)
            arr = np.array(samples)
            sample_pos = np.transpose(arr, (0, 2, 1))
            sample_pos = [item for sublist in sample_pos for item in sublist]
            # print('Sampled:', cur_nd, sample_pos)
        return cur_nd, sample_pos

    return None, None

def standardize_collisions(collisions):
    for i, c in enumerate(collisions):
        o1, o2, ts, te, ci = c
        if isinstance(ci[0], pm.Vec2d):
            ci[0] = [ci[0].x, ci[0].y]
        collisions[i] = [o1, o2, ts, te, ci]
    return collisions

def run(args):
    args.experiment_id = generate_experiment_id()
    args.date = args.experiment_id[:6]
    args.time = args.experiment_id[7:]
    args.dir_name = os.path.join('data', args.date, args.time+'_'+args.tnm)
    os.makedirs(args.dir_name)
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

        sample_obj, sample_ext, init_pose, path_info = (
            sample_from_counterfactual(args, strategies_graph)
        )
        path_dict, collisions, success = path_info
        if collisions:
            collisions = standardize_collisions(collisions)

        # !SECTION
        if success:
            print('Success:', sample_obj, sample_ext)
            img_name = str(image_num) + '_' + sample_obj + '.png'
            img_name = os.path.join(args.dir_name, img_name)
            # draw_path(args.tp0, path_dict, img_name)
            strategies_graph.build_graph(args, sample_obj, init_pose, path_info)
            success = False
            image_num += 1
            # sample
            sample_obj = None
            sample_ext = None
            if strategies_graph.full_placement_graphs:
                sample_obj, sample_poss = sample_from_strategy_graph(args, strategies_graph)
            if sample_obj is not None:
                for s in sample_poss:
                    scaled_pos = list(s[0:2])
                    path_dict, success, _ =  args.tp0.observePlacementStatePath(
                        toolname=sample_obj,
                        position=scaled_pos
                    )
                    if success is not None:
                        print('Sample:', sample_obj, scaled_pos, success)
                        sample_num += 1
                        img_name = os.path.join(args.dir_name, 'sample_from_graph.png')
                        # draw_path(args.tp0, path_dict, img_name, scaled_pos)
                    if success:
                        print('Sample Num:', sample_num)
                        sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SSUP')

    parser.add_argument('-a', '--algorithm',
                        help='SSUP/ours', default='SSUP')
    parser.add_argument('-t', '--num_trial',
                        help='number of trials', type=int, default=250)
    parser.add_argument('--tnm',
                        help='task name', type=str, default='CatapultAlt')
    parser.add_argument('--json_dir',
                        help='json dir name', type=str,
                        default='./environment/Trials/Strategy/')
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
