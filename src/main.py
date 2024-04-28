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
from random import shuffle
# from numpy import random as rd
# import pygame as pg
from pyGameWorld import ToolPicker, objectBoundingBox
from pyGameWorld.jsrun import pyGetCollisionsAddForces
import pymunk as pm
from pyGameWorld.helpers import centroidForPoly
from src.strategy import StrategyGraph, merge_graphs
from src.utils import ExtrinsicSampler, get_prior_SSUP
from src.utils import draw_path, calculate_reward, set_prior_type, normalize_pos, draw_samples
from src.utils import calculate_weights, standardize_collisions, draw_gp_samples, draw_gradient_samples
from src.gaussian_policy import initialize_policy, plot_policies
import src.config as config
import pickle

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
    graph_list = [(g, strategy_graph.fpg_gmm_list[i])
        for i, g in enumerate(strategy_graph.placement_graphs)
        if strategy_graph.fpg_gmm_list[i]]
    if not graph_list:
        return None, None
    graph, obj_list = choice(graph_list)
    obj_weights = [ math.e**(-strategy_graph.obj_count[nd]*args.eps)
        for nd in obj_list]
    assert obj_list
    sample_obj = choices(obj_list, weights=obj_weights)[0]
    gmm_pose = graph.nodes[sample_obj]['GM'].sample()[0][0]
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
    sample_ext = [*tp.objects[sample_obj].position, tp.objects[sample_obj].rotation, *tp.objects[sample_obj].velocity]  
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
    return sample_ext

def estimate_reward(args, sample_obj, sample_pos, noisy=False):
    path_dict, collisions, success, _ = args.tp0.runStatePath(
        toolname=sample_obj,
        position=sample_pos,
        noisy=noisy
    )
    path_info = path_dict, collisions, success
    reward = calculate_reward(args, args.tp0, path_dict)
    return path_info, reward

def sample_from_strategy_graph(args, strategy_graph, idx=0):
    '''
        do sample from gaussian process model in graph
    '''
    graph_list = [g for g in strategy_graph.placement_graphs
        if all('model' in g.nodes[nd] for nd in g.nodes()
            if nd not in list(args.tp0.toolNames) + ['Goal']
        )
    ]
    g = choice(graph_list)
    cur_nd = 'Goal'
    pre_nd = None
    img_poss = []
    img_poss2 = []
    # gmm = g.nodes[cur_nd]['GM']
    # sample_poss = gmm.sample(n_samples=5)
    # sample_poss = sample_poss[0] # idx 1 is the number of gaussian
    goal_pos = centroidForPoly(args.btr0['world']['objects']['Goal']['points'])
    sample_poss = goal_pos
    sample_poss = [[sample_poss[0], sample_poss[1], 0, random()*10-5, random()*10-5]]
    img_name = os.path.join(args.dir_name,
        'GPR_'+str(cur_nd)+'.png'
    )
    data = [d[1] for d in g.nodes['Goal']['ext']]
    draw_samples(args.tp0, data, 'Goal_GM_sample.png')
    obj_ext = list(np.array(sample_poss).mean(axis=0))
    obj_pos = [obj_ext[0], obj_ext[1]]
    img_name = os.path.join(args.dir_name,
        'GM_test_'+cur_nd+'.png'
    )
    draw_samples(args.tp0, sample_poss, img_name)
    pre_nd = cur_nd
    while True:
        if cur_nd in args.tp0.toolNames:
            break
        gpr = g.nodes[cur_nd]['model']
        # relative extrinsics
        pre_nd = choice(list(g.predecessors(cur_nd)))
        if pre_nd not in args.tp0.toolNames:
            if cur_nd == 'Goal':
                prev_target_ext = list(goal_pos) + [0, 0, 0]
            else:
                prev_target_ext = list(args.tp0.objects[cur_nd].position) + [0] + list(args.tp0.objects[cur_nd].velocity)
        else:
            prev_target_ext = None
        if prev_target_ext is not None:
            samples = gpr.sample_y([list(obj_ext[2:]) + list(prev_target_ext)], n_samples=10, random_state=None)
        else:
            samples = gpr.sample_y([obj_ext[2:]], n_samples=10, random_state=None)
        # samples = gpr.sample_y([obj_ext[2:]], n_samples=10)
            
        sample_poss = [list(item)
            for sublist in np.transpose(np.array(samples), (0, 2, 1))
            for item in sublist]
        # set to real extrinsics
        img_poss += [[obj_pos[0]+s[0],obj_pos[1]+s[1],s[2],s[3],s[4]] for s in sample_poss]
        obj_ext = list(np.array(sample_poss).mean(axis=0))
        obj_pos = [obj_pos[0]+obj_ext[0],obj_pos[1]+obj_ext[1]]
        img_poss2 += [[obj_pos[0], obj_pos[1], obj_ext[2], obj_ext[3], obj_ext[4]]]
        

        
        cur_nd = pre_nd
        
        if not sample_poss: # failed to sample from GPR
            return None, None
    img_name = os.path.join(args.dir_name,
        'GP_test_'+str(idx)+'.png'
    )
    draw_gp_samples(args.tp0, [img_poss, img_poss2], img_name)
    return cur_nd, obj_pos # only return one sample

#############################################################
#                                                           #
# ANCHOR - human strategy approachs                         #
#                                                           #
#############################################################

def generalization_test():
    strategy_graphs = []
    for tnm in ['Launch_v2', 'Catapult_5']:
        args.tnm = tnm
        logging.info('Start experiment %s', tnm)
        setup_task_args(tnm)
        strategy_graph = strategy_graph_method()
        strategy_graphs.append(strategy_graph)
    
    with open('strategy_graph.pkl', 'wb') as f:
        pickle.dump(strategy_graphs, f)
    # with open('strategy_graph.pkl', 'rb') as f:
    #     strategy_graphs = pickle.load(f)
    # test on CatapultAlt obj -> lever -> cataball -> keyball -> goal
    args.tnm = 'CatapultAlt'
    with open(args.json_dir + args.tnm + '.json','r') as f:
        args.btr0 = json.load(f)
    args.tp0 = ToolPicker(args.btr0)
    action_count = 0
    sim_count = 0
    sample_count = 0
    args.trial_stats.append({'GPR': [], 'GM': []})
    success = False
    while True:
        g = choice(strategy_graphs)
        g = g.placement_graphs[0]
        cur_nd = 'Goal'
        # gmm = g.nodes[cur_nd]['GM']
        # sample_poss = gmm.sample(n_samples=3)
        # sample_poss = sample_poss[0]
        # NOTE
        sample_poss = centroidForPoly(args.btr0['world']['objects']['Goal']['points'])
        sample_poss = [[sample_poss[0]+random()*40-20, sample_poss[1]+random()*40-20, 0, random()*100-50, random()*100-50]]
        img_name = os.path.join(args.dir_name,
            'Gen_test_'+cur_nd+'.png'
        )
        draw_samples(args.tp0, sample_poss, img_name)
        print(cur_nd, int(sample_poss[0][0]), int(sample_poss[0][1]))
        cur_nd = 'Ball'
        g = [g for g in strategy_graphs[0].placement_graphs if ('Ball2', 'Ball') in g.edges()][0]
        gpr = g.nodes[cur_nd]['model']
        obj_pos = sample_poss[0]
        # NOTE
        samples = gpr.sample_y([obj_pos[2:]], n_samples=3)
        arr = np.array(samples)
        sample_poss = [list(item)
            for sublist in np.transpose(arr, (0, 2, 1))
            for item in sublist]
        img_poss = [[obj_pos[0]+s[0],obj_pos[1]+s[1],s[2],s[3],s[4]] for s in sample_poss]
        img_name = os.path.join(args.dir_name,
            'Gen_test_'+cur_nd+'1.png'
        )
        draw_gp_samples(args.tp0, [img_poss, [obj_pos]], img_name)
        cur_nd = 'Ball'
        g = [g for g in strategy_graphs[1].placement_graphs if ('Catapult', 'Ball') in g.edges()][0]
        gpr = g.nodes[cur_nd]['model']
        sample_pos = list(np.array(sample_poss).mean(axis=0))
        print(cur_nd, sample_pos)

        obj_pos = [obj_pos[0] + sample_pos[0], obj_pos[1] + sample_pos[1], sample_pos[2], sample_pos[3], sample_pos[4]]
        # NOTE
        samples = gpr.sample_y([obj_pos[2:]], n_samples=3)
        arr = np.array(samples)
        sample_poss = [list(item)
            for sublist in np.transpose(arr, (0, 2, 1))
            for item in sublist]
        img_poss = [[obj_pos[0]+s[0],obj_pos[1]+s[1],s[2],s[3],s[4]] for s in sample_poss]
        img_name = os.path.join(args.dir_name,
            'Gen_test_'+cur_nd+'x.png'
        )
        draw_gp_samples(args.tp0, [img_poss, [obj_pos]], img_name)
        sample_pos = list(np.array(sample_poss).mean(axis=0))
        obj_pos = [obj_pos[0] + sample_pos[0], obj_pos[1] + sample_pos[1], sample_pos[2], sample_pos[3], sample_pos[4]]
        cur_nd = 'Catapult'
        g = [g for g in strategy_graphs[1].placement_graphs if ('obj2', 'Catapult') in g.edges()][0]
        gpr = g.nodes[cur_nd]['model']
        # NOTE
        samples = gpr.sample_y([obj_pos[2:]], n_samples=3)
        arr = np.array(samples)
        sample_poss = [list(item)
            for sublist in np.transpose(arr, (0, 2, 1))
            for item in sublist]
        sample_pos = sample_poss[0]
        print(cur_nd, sample_pos)
        img_poss = [[obj_pos[0]+s[0],obj_pos[1]+s[1],s[2],s[3],s[4]] for s in sample_poss]
        img_name = os.path.join(args.dir_name,
            'Gen_test_'+cur_nd+'.png'
        )
        draw_gp_samples(args.tp0, [img_poss, [sample_poss[0]]], img_name)
        obj_pos = [obj_pos[0] + sample_poss[0][0], obj_pos[1] + sample_poss[0][1]]
        sample_pos = obj_pos
        # sample_obj = choice(list(args.tp0.toolNames))
        sample_obj = 'obj2'
        path_info, reward = estimate_reward(
            args,
            sample_obj,
            obj_pos,
            noisy=True
        )
        path_dict, collisions, success = path_info
        sample_count += 1
        if success is not None:
            sim_count += 1
        logging.info('GPR Sample: %s (%d %d)', sample_obj, sample_pos[0], sample_pos[1])
        if success is not None and reward > args.attempt_threshold:
            action_count += 1
            logging.info('GPR Simulate: %d %d %d, %s (%d %d) %s %f',
                action_count, sim_count, sample_count, sample_obj, sample_pos[0], sample_pos[1], success, reward)
            img_name = os.path.join(args.dir_name,
                'sample_from_GPR_'+str(sim_count)+'_'+args.tnm+'.png'
            )
            draw_path(args.tp0, path_dict, img_name, sample_pos)
            path_info, reward = estimate_reward(
                args,
                sample_obj,
                sample_pos,
                noisy=False
            )
            path_dict, collisions, success = path_info
            logging.info('GPR Attempt: %d %d %d (%d, %d) %s %f', action_count, sim_count, sample_count, sample_pos[0], sample_pos[1], success, reward)

            if success:
                # logging.info('GPR Attempt Success: %d %d %d (%d, %d)', action_count, sim_count, sample_count, sample_pos[0], sample_pos[1])
                args.trial_stats[-1]['GPR'].append([action_count, sim_count, sample_count, [sample_pos[0], sample_pos[1]]])
                break
        if sim_count >= 100:
            logging.info('GPR out of max attempt: %d %d %d', action_count, sim_count, sample_count)
            break

def random_tool(args):
    action_count = 0
    sim_count = 0
    success = False
    while action_count < 40:
        sample_obj = choice(args.tool_objects)
        sample_pos = get_prior_SSUP(args.tp0, args.movable_objects)
        path_dict, collisions, success, _ = args.tp0.runStatePath(
            sample_obj,
            sample_pos,
            noisy=args.noisy
        )
        sim_count += 1
        reward = calculate_reward(args, args.tp0, path_dict)
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

def SSUP():
    tp = args.tp0
    get_prior = args.get_prior
    gaussian_policies = initialize_policy(300, 300, 50)
    epsilon = args.eps
    epsilon_decay_rate = args.eps_decay_rate
    action_count = 0
    sim_count = 0
    sample_count = 0
    # Sample ninit points from prior pi(s) for each tool
    # Simulate actions to get noisy rewards rˆ using internal model
    # initialize policy parameters theta using policy gradient on initial points
    for tool in tp.toolNames:
        rewards = []
        samples = []
        for i in range(args.num_init):
            sample_pos = get_prior(tp, args.movable_objects)
            reward = []
            for i in range(args.num_sim):
                path_dict, collisions, success, _ = tp.runStatePath(
                    toolname=tool,
                    position=sample_pos,
                    noisy=True
                )
                r = calculate_reward(args, tp, path_dict)
                reward.append(r)
            reward = np.mean(reward)
            rewards.append(reward)
            samples.append([sample_pos[0], sample_pos[1], int(tool[3])-1])
        print(samples, rewards)
        for i in range(args.num_init):
            gaussian_policies.update([samples[i]], [rewards[i]], learning_rate=args.lr)
        # gaussian_policies.update(samples, rewards, learning_rate=args.lr)
    print(gaussian_policies)
    
    # plot
    img_name = os.path.join(args.dir_name,'plot_init.png')
    plot_policies(args, gaussian_policies, sample_pos, 0, img_name)
    
    sample_poss = []
    success = False
    while not success:
        # SECTION - Sample action
        acting = False
        sample_type = 'GM'
        best_reward = -1
        best_pos = None
        iter_count = 0
        if random() < epsilon:
            # NOTE - sample from prior
            sample_type = 'prior'
            sample_obj = choice(list(tp.toolNames))
            sample_pos = get_prior(args.tp0, args.movable_objects)
            sample = [sample_pos[0], sample_pos[1], int(sample_obj[3])-1]
            sim_count += 1
            # Estimate noisy reward rˆ from internal model on action a
            path_info, reward = estimate_reward(
                args, sample_obj, sample_pos, noisy=True
            )
        else:
            # Sample from policy (at most sample 100 times)
            sample = gaussian_policies.action()
            sample_obj = 'obj'+str(sample[2]+1)
            sample_pos = list(sample[0:2])
            rewards = []
            for i in range(args.num_sim):
                # Estimate noisy reward rˆ from internal model on action a
                path_info, reward = estimate_reward(
                    args, sample_obj, sample_pos, noisy=True
                )
                rewards.append(reward)
            reward = np.mean(rewards)
            iter_count += 1
        logging.info('Simulation %d %d %d: %s %s (%d, %d), %s, %f',
            action_count, sim_count, sample_count, sample_obj, sample_type, sample_pos[0], sample_pos[1], success, reward)
        sample_poss.append([sample_pos, reward])
        # best action
        if reward > best_reward:
            best_reward = reward
            best_pos = sample_pos
        # try_action = reward > args.attempt_threshold
        if reward > args.attempt_threshold:
            acting = True
        elif iter_count >= args.num_iter:
            acting = True
            reward = best_reward
            sample = best_pos
            # reset
            best_reward = -1
            best_pos = None
            iter_count = 0
        sim_count += 1

        # !SECTION
        success = False
        if acting:
            # Observe r from environment on action a.
            path_info, reward = estimate_reward(
                args, sample_obj, sample_pos, noisy=False
            )
            path_dict, collisions, success = path_info
            # epsilon *= epsilon_decay_rate
            action_count += 1
            # If successful, exit.
            logging.info('Attempt %d %d %d: %s %s (%d, %d), %f',
            action_count, sim_count, sample_count, sample_obj, sample_type, sample_pos[0], sample_pos[1], reward)
            print(gaussian_policies)
            if success:
                logging.info("Success! %d %d", action_count, sim_count)
                img_name = os.path.join(args.dir_name,
                    'plot_final.png'
                )
                plot_policies(args, gaussian_policies, sample_pos, int(sample_obj[3])-1, img_name)

                args.trial_stats.append([action_count, sim_count, sample_count])
                break
            # Simulate rˆ assuming other two tool choices.
            # Update policy based on all three estimates and actions.
            for tool in tp.toolNames: 
                CF_sample = [sample[0], sample[1], int(tool[3])-1]
                if tool == sample_obj: 
                    CF_reward = reward
                else:
                    path_dict, collisions, success_, _ = tp.runStatePath(
                        toolname=tool,
                        position=sample_pos,
                        noisy=True
                    )
                    CF_reward = calculate_reward(args, tp, path_dict)
                gaussian_policies.update([CF_sample], [CF_reward], learning_rate=args.lr)
            img_name = os.path.join(args.dir_name,
                    'plot'+str(sim_count)+'.png'
                )
            plot_policies(args, gaussian_policies, sample_pos, int(sample_obj[3])-1, img_name)
        else:
            # Update policy using policy gradient
            gaussian_policies.update([sample], [reward], learning_rate=args.lr)
        
        if action_count >= args.max_attempts or sim_count >= args.max_simulations:
            logging.info("Out of max attempt! %d %d", action_count, sim_count)
            args.trial_stats.append([action_count, sim_count, sample_count])
            break
    return args.trial_stats

def test_GPR(strategy_graph):
    action_count = 0
    sim_count = 0
    sample_count = 0
    args.trial_stats.append({'GPR': [], 'GM': []})
    while action_count < args.max_attempts:
        sample_obj, sample_pos = sample_from_strategy_graph(args, strategy_graph, action_count)
        if not sample_obj: continue
        sample_pos = list(sample_pos[0:2])
        # NOTE - same as SSUP -> simulate in PE before attempt
        path_info, reward = estimate_reward(
            args,
            sample_obj,
            sample_pos,
            noisy=False
        )
        path_dict, collisions, success = path_info
        action_count += 1
        if success:
            logging.info('GPR Attempt Success: %d %d %d (%d, %d)', action_count, sim_count, sample_count, sample_pos[0], sample_pos[1])
            args.trial_stats[-1]['GPR'].append([action_count, sim_count, sample_count, [sample_pos[0], sample_pos[1]]])
            break
    if not success:
        logging.info('GPR out of max attempt: %d %d %d', action_count, sim_count, sample_count)

def test_GM(strategy_graph):
    action_count = 0
    sim_count = 0
    sample_count = 0
    while action_count < args.max_attempts and sim_count < args.max_simulations:
        nd_list = [ (nd, g) for g in strategy_graph.placement_graphs
            for nd in g.nodes() if 'GM' in g.nodes[nd] and nd in args.tp.toolNames]
        node, graph = choice(nd_list)
        sample_ext = graph.nodes[node]['GM'].sample()[0][0]
        sample_obj = node
        logging.info('GM Sample: %s (%d %d)', sample_obj, sample_ext[0], sample_ext[1])
        path_dict, collisions, success, _ =  args.tp0.runStatePath(
            toolname=sample_obj,
            position=list(sample_ext[0:2]),
            noisy=True
        )
        reward = calculate_reward(args, args.tp0, path_dict)
        sample_count += 1
        if success is not None:
            sim_count += 1
        if reward > args.attempt_threshold:
            logging.info('GM Simulate: %d %d %d', action_count, sim_count, sample_count)
            sample_pos = list(sample_ext[0:2])
            path_dict, collisions, success, _ =  args.tp0.runStatePath(
                toolname=sample_obj,
                position=sample_pos,
                noisy=False
            )
            img_name = os.path.join(args.dir_name,
                'sample_from_GM_'+str(sim_count)+'_'+args.tnm+'.png'
            )
            draw_path(args.tp0, path_dict, img_name, sample_pos)
            action_count += 1
            if success:
                logging.info('GM Attempt Success: %d %d %d (%d %d)', action_count, sim_count, sample_count, sample_ext[0], sample_ext[1])
                args.trial_stats[-1]["GM"].append([action_count, sim_count, sample_count, [sample_ext[0], sample_ext[1]]])
                break
    if not success:
        logging.info('GM out of max attempt: %d %d %d', action_count, sim_count, sample_count)

def test_GPR_SSUP(strategy_graph):
    sample_obj, sample_pos = sample_from_strategy_graph(args, strategy_graph)
    gaussian_policies = initialize_policy(sample_pos[0], sample_pos[1], 50)
    success = False
    action_count = 0
    sim_count = 0
    sample_count = 0
    while not success:
        # SECTION - Sample action
        acting = False
        sample_type = 'GM'
        best_reward = -1
        best_pos = None
        iter_count = 0
        if random() < args.eps:
            # NOTE - sample from prior
            sample_type = 'prior'
            sample_obj = choice(list(args.tp0.toolNames))
            sample_pos = args.get_prior(args.tp0, args.movable_objects)
            sample = [sample_pos[0], sample_pos[1], int(sample_obj[3])-1]
            sim_count += 1
            # Estimate noisy reward rˆ from internal model on action a
            path_info, reward = estimate_reward(
                args, sample_obj, sample_pos, noisy=True
            )
        else:
            # Sample from policy (at most sample 100 times)
            sample = gaussian_policies.action()
            sample_obj = 'obj'+str(sample[2]+1)
            sample_pos = list(sample[0:2])
            rewards = []
            for i in range(args.num_sim):
                # Estimate noisy reward rˆ from internal model on action a
                path_info, reward = estimate_reward(
                    args, sample_obj, sample_pos, noisy=True
                )
                rewards.append(reward)
            reward = np.mean(rewards)
            iter_count += 1
        logging.info('Simulation %d %d %d: %s %s (%d, %d), %s, %f',
            action_count, sim_count, sample_count, sample_obj, sample_type, sample_pos[0], sample_pos[1], success, reward)
        # best action
        if reward > best_reward:
            best_reward = reward
            best_pos = sample_pos
        # try_action = reward > args.attempt_threshold
        if reward > args.attempt_threshold:
            acting = True
        elif iter_count >= args.num_iter:
            acting = True
            reward = best_reward
            sample = best_pos
            # reset
            best_reward = -1
            best_pos = None
            iter_count = 0
        sim_count += 1

        # !SECTION
        success = False
        if acting:
            # Observe r from environment on action a.
            path_info, reward = estimate_reward(
                args, sample_obj, sample_pos, noisy=False
            )
            path_dict, collisions, success = path_info
            # epsilon *= epsilon_decay_rate
            action_count += 1
            # If successful, exit.
            logging.info('Attempt %d %d %d: %s %s (%d, %d), %f',
            action_count, sim_count, sample_count, sample_obj, sample_type, sample_pos[0], sample_pos[1], reward)
            print(gaussian_policies)
            if success:
                logging.info("Success! %d %d", action_count, sim_count)
                img_name = os.path.join(args.dir_name,
                    'plot_final.png'
                )
                plot_policies(args, gaussian_policies, sample_pos, int(sample_obj[3])-1, img_name)

                args.trial_stats.append([action_count, sim_count, sample_count])
                break
            # Simulate rˆ assuming other two tool choices.
            # Update policy based on all three estimates and actions.
            for tool in args.tp0.toolNames: 
                CF_sample = [sample[0], sample[1], int(tool[3])-1]
                if tool == sample_obj: 
                    CF_reward = reward
                else:
                    path_dict, collisions, success_, _ = args.tp0.runStatePath(
                        toolname=tool,
                        position=sample_pos,
                        noisy=True
                    )
                    CF_reward = calculate_reward(args, args.tp0, path_dict)
                gaussian_policies.update([CF_sample], [CF_reward], learning_rate=args.lr)
            img_name = os.path.join(args.dir_name,
                    'plot'+str(sim_count)+'.png'
                )
            plot_policies(args, gaussian_policies, sample_pos, int(sample_obj[3])-1, img_name)
        else:
            # Update policy using policy gradient
            gaussian_policies.update([sample], [reward], learning_rate=args.lr)
        
        if action_count >= args.max_attempts or sim_count >= args.max_simulations:
            logging.info("Out of max attempt! %d %d", action_count, sim_count)
            args.trial_stats.append([action_count, sim_count, sample_count])
            break
    return args.trial_stats

def strategy_graph_method(strat_graph=None):
    if strat_graph:
        strategy_graph = strat_graph
    else:
        strategy_graph = StrategyGraph()
    sim_count = 0
    while sim_count < args.num_trials: # loop based on number of sim in PE
        args.btr = deepcopy(args.btr0)
        args.tp = ToolPicker(args.btr0)
        success = False
        # SECTION - sample from noisy PE
        while not success:
            # sample extrinsics from gaussian model in SG
            can_sample_from_gm = any(i
                for i in strategy_graph.fpg_gmm_list)
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
                # sample_type = sample_exttype(args, strategy_graph)
                sample_type = 'tool'
                sample_obj, sample_ext, ext_info, path_info = sample_ext_by_type(
                    args,
                    sample_type,
                    strategy_graph,
                )
            path_dict, collisions, success = path_info
            if collisions:
                collisions = standardize_collisions(collisions)
            # logging.info('Sample: %s %s', sample_obj, sample_type)
            if success:
                sim_count += 1
                args.sim_count = sim_count
                logging.info('Noisy Success: %d, %s %s',
                    sim_count, sample_obj, sample_type)
                graph = strategy_graph.build_graph(args, sample_obj, sample_ext, path_info)
                strategy_graph.set_placement_graph(args, graph, sample_obj)

        # !SECTION
        if  sim_count % 5 == 0:
            strategy_graph.merge_graph(args)
            strategy_graph.train(args)
    return strategy_graph
    # !SECTION - test GM

def setup_experiment_dir():
    # NOTE - generate experiment dir
    exptime_str = args.experiment_time.strftime("%y%m%d_%H%M%S")
    date = exptime_str[:6]
    time = exptime_str[7:]
    args.exp_name = '_'.join([time, args.tnm, args.algorithm])
    args.main_dir_name = os.path.join('data', date, args.exp_name)
    os.makedirs(args.main_dir_name)

def setup_task_args(tnm=None):
    if not tnm:
        tnm = args.tnm
    with open(args.json_dir + tnm + '.json','r') as f:
        args.btr0 = json.load(f)
    args.tp0 = ToolPicker(args.btr0)
    args.movable_obj_dict = {i:j for i, j in args.tp0.objects.items()
                        if j.color in [(255, 0, 0, 255), (0, 0, 255, 255)]
    }
    args.movable_objects = list(args.movable_obj_dict)
    args.tool_objects = list(args.tp0.toolNames)
    path_dict0, _, _ = args.tp0.observeStatePath()
    args.dist0 = min(args.tp0.world.distanceToGoalContainer((path_dict0[obj][i][:2])) 
        for obj in path_dict0 for i in range(len(path_dict0[obj])) 
        if args.tp0.objects[obj].color==(255, 0, 0, 255))
    args.ext_sampler = ExtrinsicSampler(args.btr0, path_dict0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SSUP')

    parser.add_argument('-a', '--algorithm',
                        help='SSUP/ours', default='SSUP')
    parser.add_argument('-t', '--num-trials',
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
                        help='epsilon', type=float, default=None)
    parser.add_argument('--eps-decay-rate',
                        help='epsilon decay rate', type=int, default=None)
    parser.add_argument('--lr',
                        help='learning rate', type=float, default=None)
    parser.add_argument('--max-attempts',
                        help='max number of attempt', type=int, default=None)
    parser.add_argument('--attempt-threshold',
                        help='attempt threshold', type=float, default=None)
    parser.add_argument('-d', '--deterministic',
                        help='whether deterministic or noisy in collecting data',
                        action='store_true')
    # parse args and update config
    args = parser.parse_args()
    if args.algorithm == 'SSUP' or args.algorithm == 'GPR_SSUP':
        cfg = config.SSUP_config
    else:
        cfg = config.OURS_config
    for c in cfg:
        if c not in args:
            args.__dict__[c] = cfg[c]
    for arg in vars(args):
        if args.__dict__[arg] is None and arg in cfg:
            args.__dict__[arg] = cfg[arg]
    # load task json file
    args.noisy = not args.deterministic
    args.experiment_time = datetime.now()
    args.get_prior = set_prior_type(args.algorithm)
    args.trial_stats = []
    setup_experiment_dir()
    for i in range(args.num_experiment):
        # NOTE - generate experiment dir
        # NOTE - set up logging
        trial_name = str(i).zfill(3)
        args.dir_name = os.path.join(args.main_dir_name, trial_name)
        os.makedirs(args.dir_name)
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
            SSUP()
        elif args.algorithm == 'GPR_SSUP':
            setup_task_args()
            # strategy_graph = strategy_graph_method()
            # with open('strategy_graph.pkl', 'wb') as f:
            #     pickle.dump(strategy_graph, f)
            with open('strategy_graph.pkl', 'rb') as f:
                strategy_graph = pickle.load(f)
            setup_task_args('Catapult_3')
            test_GPR_SSUP(strategy_graph)
        elif args.algorithm == 'random':
            random_tool()
        elif args.algorithm == 'GPR':
            setup_task_args()
            strategy_graph = strategy_graph_method()
            with open('strategy_graph.pkl', 'wb') as f:
                pickle.dump(strategy_graph, f)
            # with open('strategy_graph.pkl', 'rb') as f:
            #     strategy_graph = pickle.load(f)
            logging.info('Start Testing')
            test_GPR(strategy_graph)
            test_GM(strategy_graph)
        elif args.algorithm == 'GPR2':
            setup_task_args()
            # strategy_graph = strategy_graph_method()
            # tnm = 'Catapult_3'
            # setup_task_args(tnm)
            # strategy_graph1 = strategy_graph_method()
            # strategy_graphs = [strategy_graph, strategy_graph1]
            # with open('strategy_graphs03.pkl', 'wb') as f:
            #     pickle.dump(strategy_graphs, f)
            with open('strategy_graphs.pkl', 'rb') as f:
                strategy_graphs = pickle.load(f)
            strategy_graph = merge_graphs(args, strategy_graphs)

            tnm = 'Catapult_2'
            logging.info('Start experiment %s', tnm)
            setup_task_args(tnm)
            test_GPR(strategy_graph)
        elif args.algorithm == 'oursSSUP':
            strategy_graph = strategy_graph_method()
        elif args.algorithm == 'gen':
            generalization_test()
        else:
            raise ValueError(f"Invalid algorithm: {args.algorithm}")
    
    time.sleep(1) # avoid generating same experiment id

    print('End experiment')
    logfile_path = os.path.join(args.main_dir_name, 'summary.log')
    file_handler = logging.FileHandler(logfile_path)
    file_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(file_handler)
    logging.info('Summary')
    for stats in args.trial_stats:
        logging.info('%s', str(stats)[1:-1])
    duration = datetime.now() - args.experiment_time
    logging.info('%s', duration)
