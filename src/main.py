import argparse
import json
from random import choice, randint, random, choices
import os
from copy import deepcopy
# import sys
import logging
import math
import time
from datetime import datetime
import numpy as np
# from numpy import random as rd
# import pygame as pg
from pyGameWorld import ToolPicker, objectBoundingBox
from pyGameWorld.jsrun import pyGetCollisionsAddForces
import pymunk as pm
from src.strategy import StrategyGraph
from src.utils import ExtrinsicSampler, generate_experiment_id, get_prior_SSUP
from src.utils import draw_path, calculate_reward, set_prior_type, normalize_pos
from src.utils import calculate_weights, scale_pos, standardize_collisions
from src.gaussian_policy_torch import random_init_policy, plot_policies

#############################################################
#                                                           #
# ANCHOR - sample, simulate methods                         #
#                                                           #
#############################################################

def simulate_from_gm(args, sample_obj, gmm_pose, noisy=False):
    '''
        simulate in PE given sample object and extrinsics
    '''
    tp = args.tp
    btr = args.btr
    if sample_obj in args.tool_objects:
        sample_pos = list(gmm_pose[0:2])
        path_dict, collisions, success, _ = tp.runStatePath(
            sample_obj,
            sample_pos,
            noisy=noisy
        )
    else:
        # btr['world']['objects'][sample_obj]['position'] = gmm_pose[0:2]
        # btr['world']['objects'][sample_obj]['rotation'] = gmm_pose[2]
        btr['world']['objects'][sample_obj]['velocity'] = gmm_pose[3:]
        tp = ToolPicker(btr)
        path_dict, collisions, success, _ = tp.runStatePath(noisy=noisy)
    return path_dict, collisions, success

def sample_from_gm(args, strategy_graph):
    graph_list = [ (g, strategy_graph.fpg_gmm_list[i])
        for i, g in enumerate(strategy_graph.full_placement_graphs)
        if strategy_graph.fpg_gmm_list[i]]
    graph, obj_list = choice(graph_list)
    obj_weights = [ math.e**(-strategy_graph.obj_count[nd]*args.eps)
        for nd in obj_list]
    assert obj_list
    sample_obj = choices(obj_list, weights=obj_weights)[0]
    gmm_pose = graph.nodes[sample_obj]['gmm'].sample()[0][0]
    return sample_obj, gmm_pose

def extrinsics_sampler(args, sample_type, strategy):
    weights = [ math.e**(-strategy.obj_count[m]*args.eps)
        for m in args.movable_objects
    ]
    sample_obj = choices(args.movable_objects, weights=weights, k=1)[0]
    if sample_type == 'pos':
        btr = args.ext_sampler.sample_pos(sample_obj)
    elif sample_type == 'vel': # set velocity
        btr = args.ext_sampler.sample_vel(sample_obj)
    elif sample_type == 'ang':
        btr = args.ext_sampler.sample_ang(sample_obj)
    tp = ToolPicker(btr)
    path_dict, collisions, success, _ = tp.runStatePath(noisy=args.noisy)
    sample_ext = path_dict[sample_obj][0]
    ext_info = sample_ext
    path_info = path_dict, collisions, success
    return sample_obj, sample_ext, ext_info, path_info

def tool_sampler(args):
    tp = args.tp
    sample_obj = choice(args.tool_objects)
    sample_pos = get_prior_SSUP(args.tp0, args.movable_objects)
    path_dict, collisions, success, t = tp.runStatePath(
        sample_obj,
        sample_pos,
        noisy=args.noisy
    )
    sample_ext = [sample_pos[0], sample_pos[1], 0, 0, 0]
    ext_info = sample_ext
    path_info = path_dict, collisions, success
    return sample_obj, sample_ext, ext_info, path_info

def kick_sampler(args, strategy):
    weights = [ math.e**(-strategy.obj_count[m]*args.eps)
        for m in args.movable_objects
    ]
    sample_obj = choices(args.movable_objects, weights=weights, k=1)[0]
    rand_rad = random()*2*np.pi
    random_scale = randint(1,50) * 10000
    impulse = pm.Vec2d(np.cos(rand_rad)*random_scale,  np.sin(rand_rad)*random_scale)
    bb = objectBoundingBox(args.tp.objects[sample_obj])
    sample_pos = pm.Vec2d( \
        randint(bb[0][0],bb[1][0])-(bb[1][0]-bb[0][0])/2-bb[0][0], \
        randint(bb[0][1],bb[1][1])-(bb[1][1]-bb[0][1])/2-bb[0][1])
    force_times = {
        0.0:[[sample_obj, impulse, sample_pos]],
        0.1:[[sample_obj, impulse, sample_pos]],
        0.2:[[sample_obj, impulse, sample_pos]]
    }
    # FIXME - consider noisy env
    path_dict, collisions, success, _ = pyGetCollisionsAddForces(
        args.tp.world,
        force_times=force_times
    )
    sample_ext = path_dict[sample_obj][2]
    path_info = path_dict, collisions, success
    return sample_obj, sample_ext, force_times, path_info

def sample_ext_by_type(args, sample_type, strategy):
    if sample_type in ['pos', 'vel', 'ang']: # fixed pos, smaple from path
        result = extrinsics_sampler(args, sample_type, strategy)
    elif sample_type == 'kick':
        result = kick_sampler(args, strategy)
    elif sample_type == 'tool':
        result = tool_sampler(args)
    # result: sample_obj, sample_ext, ext_info, path_info
    return result

def sample_exttype(args, strategy_graph):
    '''
        do sampling
    '''
    success = False
    obj_weight = sum(math.e**(-strategy_graph.obj_count[m]*args.eps) for m in args.movable_objects)
    tool_weight = sum(math.e**(-strategy_graph.obj_count[m]*args.eps) for m in args.tool_objects)
    ext_weights = [0, obj_weight, 0, obj_weight, tool_weight]
    ext_weights = [w/sum(ext_weights) for w in ext_weights]
    sample_ext = choices(
        ['pos', 'vel', 'ang', 'kick', 'tool'],
        weights=ext_weights,
        k=1)[0]
    # sample_ext = 'tool'
    return sample_ext

def simulate_placement_from_strategy_graph(args, sample_obj, sample_pos, noisy=False):
    path_dict, collisions, success, _ = args.tp0.runStatePath(
        toolname=sample_obj,
        position=sample_pos,
        noisy=noisy
    )
    path_info = path_dict, collisions, success
    reward = calculate_reward(args.tp0, path_dict)
    return path_info, reward

def sample_from_strategy_graph(args, strategy_graph):
    '''
        do sample from gaussian process model in graph
    '''
    graph_list = [g for g in strategy_graph.full_placement_graphs
        if all('model' in g.nodes[nd] for nd in g.nodes()
            if nd not in args.tp.toolNames
        )
    ]
    g = choice(graph_list)
    cur_nd = 'Goal'
    gmm = g.nodes[cur_nd]['model']
    sample_poss = gmm.sample(n_samples=2)
    sample_poss = sample_poss[0]
    while True:
        pred = choice(list(g.predecessors(cur_nd)))
        cur_nd = pred
        if cur_nd in args.tp.toolNames:
            break
        gpr = g.nodes[cur_nd]['model']
        samples = gpr.sample_y(sample_poss, n_samples=2)
        arr = np.array(samples)
        sample_poss = [list(item)
            for sublist in np.transpose(arr, (0, 2, 1))
            for item in sublist]
        # filtering invalid positions
        sample_poss = [pos for pos in sample_poss
            if pos[0] > 0 and pos[0] < 600 and pos[1] > 0 and pos[1] < 600]
        if not sample_poss: # failed to sample from GPR
            return None, None
        # NOTE - testing process in backward chaining
        # for sample_pos in sample_poss:
        #     btr = deepcopy(args.btr0)
        #     btr['world']['objects'][cur_nd]['position'] = sample_pos[0:2]
        #     btr['world']['objects'][cur_nd]['rotation'] = sample_pos[2]
        #     btr['world']['objects'][cur_nd]['velocity'] = sample_pos[3:]
        #     tp = ToolPicker(btr)
        #     path_dict, collisions, success, _ =  tp.runStatePath()
        #     if success:
        #         logging.info('GPR test %s %s', cur_nd, str(sample_pos))
    return cur_nd, sample_poss[0] # only return one sample

#############################################################
#                                                           #
# ANCHOR - human strategy approachs                         #
#                                                           #
#############################################################

def random_tool(args):
    action_count = 0
    sim_count = 0
    success = False
    while not success:
        sample_obj = choice(args.tool_objects)
        sample_pos = get_prior_SSUP(args.tp0, args.movable_objects)
        path_dict, collisions, success, _ = args.tp0.runStatePath(
            sample_obj,
            sample_pos,
            noisy=args.noisy
        )
        sim_count += 1
        reward = calculate_reward(args.tp0, path_dict)
        logging.info('Simlation %d %d: %s (%d, %d), %s %f',
            action_count, sim_count, sample_obj, sample_pos[0], sample_pos[1], success, reward)
        if reward > args.attempt_threshold:
            path_dict, collisions, success, _ = args.tp0.runStatePath(
                sample_obj,
                sample_pos
            )
            action_count += 1
            logging.info('Attempt %d %d: %s (%d, %d), %s',
                action_count, sim_count, sample_obj, sample_pos[0], sample_pos[1], success)
            if success:
                logging.info('Success')
                args.trial_stats.append([action_count, sim_count])

def SSUP(args):
    tp = args.tp0
    get_prior = args.get_prior
    gaussian_policies = {tool: random_init_policy(
        *(get_prior(args.tp0, args.movable_objects, normalize=True)))
        for tool in args.tp0.toolNames}
    policy_rewards = {i: [] for i in gaussian_policies}
    epsilon = args.eps
    epsilon_decay_rate = args.eps_decay_rate
    action_count = 0
    sim_count = 0
    sample_count = 0
    try_action = False
    success = False
    while not success:
        logging.info('Count: %d, %d, %d', action_count, sim_count, sample_count)
        # SECTION - Sample
        weights = calculate_weights(policy_rewards, gaussian_policies)
        sample_obj = choices(list(gaussian_policies), weights=weights, k=1)[0]
        valid_sample_action = False
        for _ in range(100):
            # Sample from policy (at most sample 100 times)
            pos = gaussian_policies[sample_obj].action()
            sample_pos = scale_pos(pos)
            sample_count += 1
            if pos[0] > -1 and pos[0] < 1 and pos[1] > -1 and pos[1] < 1:
                valid_sample_action = True
                break
        if random() < epsilon or not valid_sample_action:
            # NOTE - sample from prior
            sample_obj = choices(list(gaussian_policies.keys()), k=1)[0]
            sample_pos = get_prior(args.tp0, args.movable_objects)
            pos = normalize_pos(sample_pos)
        # !SECTION
        # SECTION - Simulate
        path_dict, collisions, success, _ = tp.runStatePath(
            toolname=sample_obj,
            position=sample_pos,
            noisy=args.noisy
        )
        reward = calculate_reward(tp, path_dict)
        sim_count += 1

        try_action = reward > 0.6
        if try_action:
            path_dict, collisions, success, _ = tp.runStatePath(
                toolname=sample_obj,
                position=sample_pos,
            )
            reward = calculate_reward(tp, path_dict)
            epsilon *= epsilon_decay_rate
            action_count += 1
        # !SECTION
        # SECTION - Update
        gaussian_policies[sample_obj].update([pos], [reward], learning_rate=0.5)
        policy_rewards[sample_obj].append(reward)
        plot_policies(gaussian_policies, sample_pos)
        logging.info('Attempt %d: %s (%d, %d), %s, %f',
            action_count, sample_obj, sample_pos[0], sample_pos[1], success, reward)
        # !SECTION
        if reward == 1:
            logging.info("Success! %d %d", action_count, sim_count)
            args.trial_stats.append([action_count, sim_count, sample_count])
            success = True
        elif action_count >= 100:
            logging.info("Out of max attempt! %d %d", action_count, sim_count)
            args.trial_stats.append([action_count, sim_count, sample_count])
            break

        # export image of the path
        if path_dict:
            img_name = os.path.join(args.dir_name,
                args.tnm+'.png'
            )
            draw_path(args.tp0, path_dict, img_name)
    return args.trial_stats

def strategy_graph_method(args):
    strategy_graph = StrategyGraph()
    action_count = 0
    sim_count = 0
    sample_count = 0
    while sim_count < args.num_trial: # loop based on number of sim in PE
        args.btr = deepcopy(args.btr0)
        args.tp = ToolPicker(args.btr0)
        success = False
        # SECTION - sample from noisy PE
        while not success:
            # sample extrinsics from gaussian model in SG
            can_sample_from_gm = any(gmm_list != []
                for gmm_list in strategy_graph.fpg_gmm_list)
            if can_sample_from_gm and random() < 0.2:
                sample_type = 'gm'
                sample_obj, sample_ext = sample_from_gm(args, strategy_graph)
                path_info = simulate_from_gm(
                    args,
                    sample_obj,
                    sample_ext,
                    noisy=args.noisy
                )
            else:
                sample_type = sample_exttype(args, strategy_graph)
                sample_obj, sample_ext, ext_info, path_info = sample_ext_by_type(
                    args,
                    sample_type,
                    strategy_graph,
                )
            path_dict, collisions, success = path_info
            sample_count += 1
            if collisions:
                collisions = standardize_collisions(collisions)
            logging.info('Sample: %s %s', sample_obj, sample_type)
            if success:
                sim_count += 1
                logging.info('Noisy Success: %d %d, %s %s',
                    sim_count, sample_count, sample_obj, sample_type)
                strategy_graph.build_graph(args, sample_obj, sample_ext, path_info)
        # !SECTION
        if  sim_count % 10 == 0:
            strategy_graph.merge_graph(args)
            strategy_graph.train()

    logging.info('Start Testing')
    # SECTION - test GPR
    action_count = 0
    sim_count = 0
    sample_count = 0
    args.trial_stats.append({'GPR': [], 'GM': []})
    while sim_count < args.max_sim_attempt:
        sample_obj, sample_pos = sample_from_strategy_graph(args, strategy_graph)
        if not sample_obj: continue
        sample_pos = list(sample_pos[0:2])
        # NOTE - same as SSUP -> simulate in PE before attempt
        path_info, reward = simulate_placement_from_strategy_graph(
            args,
            sample_obj,
            sample_pos,
            noisy=True
        )
        path_dict, collisions, success = path_info
        sample_count += 1
        if success is not None:
            sim_count += 1
        logging.info('GPR Sample: %s (%d %d)', sample_obj, sample_pos[0], sample_pos[1])
        if success is not None and reward > args.attempt_threshold:
            action_count += 1
            logging.info('GPR Simulate: %d %d %d, %s (%d %d) %s',
                action_count, sim_count, sample_count, sample_obj, sample_pos[0], sample_pos[1], success)
            img_name = os.path.join(args.dir_name,
                'sample_from_GP'+str(sim_count)+'.png'
            )
            success = False
            draw_path(args.tp0, path_dict, img_name, sample_pos)
            path_info, reward = simulate_placement_from_strategy_graph(
                args,
                sample_obj,
                sample_pos,
                noisy=False
            )
            path_dict, collisions, success = path_info
            if success:
                logging.info('GPR Attempt Success: %d %d %d', action_count, sim_count, sample_count)
                args.trial_stats[-1]['GPR'].append([action_count, sim_count, sample_count])
                break
    if not success:
        logging.info('GPR out of max attempt: %d %d %d', action_count, sim_count, sample_count)
    # !SECTION - test GPR
    # SECTION - test GM
    action_count = 0
    sim_count = 0
    sample_count = 0
    while sim_count < args.max_sim_attempt:
        nd_list = [ (nd, g) for g in strategy_graph.full_placement_graphs
            for nd in g.nodes() if 'gmm' in g.nodes[nd] and nd in args.tp.toolNames]
        node, graph = choice(nd_list)
        sample_ext = graph.nodes[node]['gmm'].sample()[0][0]
        logging.info('GM Sample: %s (%d %d) %s', sample_obj, sample_ext[0], sample_ext[1], success)
        path_dict, collisions, success, _ =  args.tp0.runStatePath(
            toolname=sample_obj,
            position=list(sample_ext[0:2]),
            noisy=True
        )
        reward = calculate_reward(args.tp0, path_dict)
        sample_count += 1
        if success is not None:
            sim_count += 1
        if reward > args.attempt_threshold:
            logging.info('GM Simulate: %d %d %d', action_count, sim_count, sample_count)
            path_dict, collisions, success, _ =  args.tp0.runStatePath(
                toolname=sample_obj,
                position=list(sample_ext[0:2]),
                noisy=False
            )
            action_count += 1
            if success:
                logging.info('GM Attempt Success: %d %d %d', action_count, sim_count, sample_count)
                args.trial_stats[-1]["GM"].append([action_count, sim_count, sample_count])
                break
    if not success:
        logging.info('GM out of max attempt: %d %d %d', action_count, sim_count, sample_count)
    # !SECTION - test GM

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SSUP')

    parser.add_argument('-a', '--algorithm',
                        help='SSUP/ours', default='SSUP')
    parser.add_argument('-t', '--num-trial',
                        help='number of trials', type=int, default=50)
    parser.add_argument('--num-experiment',
                        help='number of experiments', type=int, default=1)
    parser.add_argument('--tnm',
                        help='task name', type=str, default='CatapultAlt')
    parser.add_argument('--json_dir',
                        help='json dir name', type=str,
                        default='./environment/Trials/Strategy/')
    parser.add_argument('-v', '--verbose',
                        help='Increase output verbosity', action='store_true')
    parser.add_argument('--eps',
                        help='epsilon', type=float, default=0.2)
    parser.add_argument('-d', '--deterministic',
                        help='whether deterministic or noisy in collecting data',
                        action='store_true')
    parser.add_argument('--eps-decay-rate',
                        help='epsilon decay rate', type=int, default=0.95)
    parser.add_argument('--max_attempt',
                        help='max number of attempt', type=int, default=200)
    parser.add_argument('--max-sim-attempt',
                        help='max number of simulated attempt', type=int, default=100)
    parser.add_argument('--attempt-threshold',
                        help='attempt threshold', type=float, default=0.5)
    args = parser.parse_args()
    with open(args.json_dir + args.tnm + '.json','r') as f:
        args.btr0 = json.load(f)
    args.tp0 = ToolPicker(args.btr0)
    args.movable_obj_dict = {i:j for i, j in args.tp0.objects.items()
                        if j.color in [(255, 0, 0, 255), (0, 0, 255, 255)]
    }
    args.experiment_time = datetime.now()
    args.noisy = not args.deterministic
    args.movable_objects = list(args.movable_obj_dict)
    args.tool_objects = list(args.tp0.toolNames)
    args.available_objects = args.movable_objects + args.tool_objects
    path_dict0, _, _ = args.tp0.observeStatePath()
    args.ext_sampler = ExtrinsicSampler(args.btr0, path_dict0)
    args.get_prior = set_prior_type(args)
    args.trial_stats = []
    for i in range(args.num_experiment):
        # NOTE - generate experiment dir
        args.experiment_id = generate_experiment_id()
        args.date = args.experiment_id[:6]
        args.time = args.experiment_id[7:]
        args.exp_name = '_'.join([args.time, args.tnm, args.algorithm])
        args.dir_name = os.path.join('data', args.date, args.exp_name)
        os.makedirs(args.dir_name)
        # NOTE - set up logging
        logger = logging.getLogger()
        if logger.hasHandlers():
            logger.handlers.clear()
        logfile_path = os.path.join(args.dir_name, 'output.log')
        logging.basicConfig(
            filename=logfile_path,
            format='%(levelname)s:%(message)s',
            level=logging.INFO
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(console_handler)
        for arg in vars(args):
            logging.info('%20s %-s', arg, args.__dict__[arg])
        # NOTE - Start experiment
        if args.algorithm == 'SSUP':
            SSUP(args)
        elif args.algorithm == 'random':
            random_tool(args)
        else: # Our algorithm
            strategy_graph_method(args)
        time.sleep(1) # avoid generating same experiment id
    print('End experiment')
    summary_dir = os.path.join('data', args.date,
        '_'.join([args.time,args.tnm, args.algorithm, 'summary']))
    os.makedirs(summary_dir)
    logfile_path = os.path.join(summary_dir, 'summary.log')
    file_handler = logging.FileHandler(logfile_path)
    file_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(file_handler)
    logging.info('Summary')
    for stats in args.trial_stats:
        logging.info('%s', stats)
    duration = datetime.now() - args.experiment_time
    logging.info('%s', duration)
