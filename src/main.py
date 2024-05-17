import argparse
import json
from random import choice, random, choices
import os
from copy import deepcopy
# import sys
import logging
import math
import time
from datetime import datetime
import pickle
import numpy as np
import networkx as nx
# from numpy import random as rd
# import pygame as pg
from pyGameWorld import ToolPicker
# from pyGameWorld.jsrun import pyGetCollisionsAddForces
# import pymunk as pm
from pyGameWorld.helpers import centroidForPoly
from src.strategy import StrategyGraph, merge_graphs, merge_mechanisms, get_obj_type, train_mechanism
from src.utils import setup_experiment_dir, setup_task_args
from src.utils import get_prior_SSUP
from src.utils import draw_path, calculate_reward, draw_samples
from src.utils import standardize_collisions
from src.utils import save_strategy_graph, load_strategy_graph
from src.utils import sample_ext_by_type
from src.gaussian_policy import initialize_policy, plot_policies
import src.config as config

#############################################################
#                                                           #
# ANCHOR - sample, simulate methods                         #
#                                                           #
#############################################################

def estimate_reward(sample_obj, sample_pos, noisy=False):
    path_dict, collisions, success, _ = args.tp0.runStatePath(
        toolname=sample_obj,
        position=sample_pos,
        noisy=noisy
    )
    path_info = path_dict, collisions, success
    reward = calculate_reward(args, args.tp0, path_dict)
    return path_info, reward

def sample_from_mechanisms(mechanisms, idx=0):
    '''
        sample from mechanisms in strategy graph
    '''
    cur_nd = 'Goal'
    available_objects = args.movable_objects + args.tool_objects
    goal_pos = centroidForPoly(args.btr0['world']['objects']['Goal']['points'])
    # obj_ext: for sampling
    obj_ext =[goal_pos[0], goal_pos[1], 0, random()*10-5, random()*10-5]
    # for backtracking
    obj_pos = goal_pos
    mech = None
    search_list = ['Goal']
    search_graph = nx.DiGraph()
    search_graph.add_node('Goal', visited=True)
    max_search_iter = 10
    iter_count = 0
    while cur_nd != 'PLACED':
        iter_count += 1
        cur_type = get_obj_type(cur_nd)
        # FIXME - should not has (Tool, Goal) in mechanisms
        pre_nds = [obj for obj in available_objects 
            if (get_obj_type(obj), cur_type) in mechanisms and obj != cur_nd
            and (get_obj_type(obj), cur_type) != ('Tool', 'Goal')]
        for pre_nd in pre_nds:
            search_graph.add_node(pre_nd, visited=False)
            search_graph.add_edge(pre_nd, cur_nd)
        is_found = False
        pre_nd = choice(pre_nds)
        if search_graph.nodes[pre_nd]['visited']:
            continue
        search_graph.nodes[pre_nd]['visited'] = True
        pre_type = get_obj_type(pre_nd)
        if (pre_type, cur_type) in mechanisms:
            mech = mechanisms[(pre_type, cur_type)]
        else:
            continue
        if pre_type != 'Tool' and cur_nd == 'Goal':
            prev_target_ext = list(goal_pos) + [0, 0, 0]
        elif pre_type != 'Tool' and cur_nd != 'Goal':
            obj = args.tp0.objects[cur_nd]
            prev_target_ext = (list(obj.position) + [0] + list(obj.velocity))
        else:
            prev_target_ext = None
        if prev_target_ext is not None:
            x = [list(obj_ext[2:]) + list(prev_target_ext)]
        else:
            x = [obj_ext[2:]]
        if 'model' not in mech:
            continue
        sample_poss = mech['model'].sample_y(x, n_samples=8, random_state=None)
        sample_poss = [list(item)
            for sublist in np.transpose(np.array(sample_poss), (0, 2, 1))
            for item in sublist]
        obj_ext_ = list(np.array(sample_poss).mean(axis=0))
        obj_pos_ = [obj_pos[0] + obj_ext_[0], obj_pos[1] + obj_ext_[1]]
        print(pre_nd, cur_nd, obj_pos_)
        btr = deepcopy(args.btr0)
        if pre_nd not in args.tp0.toolNames:
            btr['world']['objects'][pre_nd]['position'] = obj_pos_
            btr['world']['objects'][pre_nd]['rotation'] = obj_ext_[2]
            btr['world']['objects'][pre_nd]['velocity'] = obj_ext_[3:]
            tp = ToolPicker(btr)
            path_dict, collisions, success, _ = tp.runStatePath(
                noisy=args.noisy
            )
        else:
            tp = args.tp0
            path_dict, collisions, success, _ = tp.runStatePath(
                pre_nd,
                obj_pos_,
                noisy=args.noisy
            )
        img_name = os.path.join(args.dir_name,
            'GPR_gen_test_'+str(idx)+'.png'
        )
        draw_path(tp, path_dict, img_name, None)
        if collisions:
            cols = [c[0:2] for c in collisions]
            pre_nd_name = 'PLACED' if pre_nd in args.tp0.toolNames else pre_nd
            if [pre_nd_name, cur_nd] in cols or [cur_nd, pre_nd_name] in cols:
                obj_ext = obj_ext_
                obj_pos = obj_pos_
                cur_nd = pre_nd
                if pre_nd in args.tp0.toolNames:
                    is_found = True
                    break
        # if is_found and pre_nd == 'Goal':
        #     search_list += [pre_nd]
        #     break

        # if not is_found:
        #     print("Failed to sample from GPR")
        #     return None, None
        if iter_count >= max_search_iter:
            break
    if is_found:
        obj = cur_nd
        return obj, obj_pos # only return one sample
    else:
        print("Failed to sample from GPR")
        return None, None

def sample_for_catapultalt(strat_graph, idx=0):
    cur_nd = 'Goal'
    pre_nd = None
    img_poss = [[],[]]
    goal_pos = centroidForPoly(args.btr0['world']['objects']['Goal']['points'])
    # obj_ext: for sampling
    obj_ext =[goal_pos[0], goal_pos[1], 0, random()*10-5, random()*10-5]
    # for backtracking
    obj_pos = goal_pos
    args.sequence_sample_poss.setdefault('Goal', []).append(list(obj_pos)+[0,0,0])
    print(cur_nd,obj_pos)
    # NOTE - keyball
    pre_nd = 'Ball'
    prev_target_ext = list(goal_pos) + [0, 0, 0]
    target_init_pos = goal_pos
    tool_init_pos = args.tp0.objects['KeyBall'].position
    x = [list(obj_ext[2:]) + list(prev_target_ext) + list(tool_init_pos) + list(target_init_pos)]
    g = [g for g in strat_graph.placement_graphs if g.has_edge(pre_nd, cur_nd) and g.has_node('Ball2')][0]
    sample_poss = g.edges[(pre_nd, cur_nd)]['model'].sample_y(x, n_samples=10, random_state=None)
    sample_poss = [list(item)
        for sublist in np.transpose(np.array(sample_poss), (0, 2, 1))
        for item in sublist]
    obj_ext = list(np.array(sample_poss).mean(axis=0))
    obj_pos = [obj_pos[0] + obj_ext[0], obj_pos[1] + obj_ext[1], obj_ext[2], obj_ext[3], obj_ext[4]]
    print((pre_nd, cur_nd),obj_pos)
    args.sequence_sample_poss.setdefault('KeyBall', []).append(obj_pos)
    # Cataball
    cur_nd = 'Ball'
    pre_nd = 'Ball2'
    obj = args.tp0.objects['KeyBall']
    prev_target_ext = (list(obj.position) + [0] + list(obj.velocity))
    target_init_pos = args.tp0.objects['KeyBall'].position
    tool_init_pos = args.tp0.objects['CataBall'].position
    x = [list(obj_ext[2:]) + list(prev_target_ext) + list(tool_init_pos) + list(target_init_pos)]
    g = [g for g in strat_graph.placement_graphs if g.has_edge(pre_nd, cur_nd) and g.has_node('Ball2')][0]
    sample_poss = g.edges[(pre_nd, cur_nd)]['model'].sample_y(x, n_samples=10, random_state=None)
    sample_poss = [list(item)
        for sublist in np.transpose(np.array(sample_poss), (0, 2, 1))
        for item in sublist]
    obj_ext = list(np.array(sample_poss).mean(axis=0))
    print(obj_ext)
    obj_pos = [obj_pos[0] + obj_ext[0], obj_pos[1] + obj_ext[1], obj_ext[2], obj_ext[3], obj_ext[4]]
    # NOTE - assume cataball pos is known
    obj_pos = [473.32756927516067, 154.87794114858616, -4.393402436807605, -32.519386569688876, 41.51949072226279]
    print((pre_nd, cur_nd),obj_pos)
    args.sequence_sample_poss.setdefault('CataBall', []).append(obj_pos)
    # lever
    cur_nd = 'Ball'
    pre_nd = 'Catapult'
    obj = args.tp0.objects['Lever']
    prev_target_ext = (list(obj.position) + [0] + list(obj.velocity))
    target_init_pos = args.tp0.objects['Lever'].position
    tool_init_pos = args.tp0.objects['CataBall'].position
    x = [list(obj_ext[2:]) + list(prev_target_ext) + list(tool_init_pos) + list(target_init_pos)]
    g = [g for g in strat_graph.placement_graphs if g.has_edge(pre_nd, cur_nd)][0]
    sample_poss = g.edges[(pre_nd, cur_nd)]['model'].sample_y(x, n_samples=10, random_state=None)
    sample_poss = [list(item)
        for sublist in np.transpose(np.array(sample_poss), (0, 2, 1))
        for item in sublist]
    obj_ext = list(np.array(sample_poss).mean(axis=0))
    for s in sample_poss:
        args.sequence_sample_poss.setdefault('Lever', []).append([obj_pos[0] + s[0], obj_pos[1] + s[1], s[2], s[3], s[4]])
    obj_pos = [obj_pos[0] + obj_ext[0], obj_pos[1] + obj_ext[1], obj_ext[2], obj_ext[3], obj_ext[4]]
    print((pre_nd, cur_nd),obj_pos)
    # args.sequence_sample_poss.setdefault('Lever', []).append(obj_pos)

    # tool
    cur_nd = 'Catapult'
    pre_nd = 'PLACED'
    obj = args.tp0.objects['Lever']
    target_init_pos = args.tp0.objects['Lever'].position
    x = [obj_ext[2:] + list(target_init_pos)]
    g = [g for g in strat_graph.placement_graphs if g.has_edge(pre_nd, cur_nd)][0]
    sample_poss = g.edges[(pre_nd, cur_nd)]['model'].sample_y(x, n_samples=10, random_state=None)
    sample_poss = [list(item)
        for sublist in np.transpose(np.array(sample_poss), (0, 2, 1))
        for item in sublist]
    obj_ext = list(np.array(sample_poss).mean(axis=0))
    obj_pos = [obj_pos[0] + obj_ext[0], obj_pos[1] + obj_ext[1], obj_ext[2], obj_ext[3], obj_ext[4]]
    args.sequence_sample_poss.setdefault('PLACED', []).append(obj_pos)
    print((pre_nd, cur_nd),obj_pos)
    img_name = os.path.join(args.dir_name,
        'catapult_seq.png'
    )
    img_poss = [args.sequence_sample_poss[v] for v in ['Goal',  'KeyBall', 'CataBall', 'Lever', 'PLACED']]
    draw_samples(args.tp0, img_poss, '', img_name)
    # tool = choice(list(g.nodes[cur_nd]['tools']))
    tool = choice(list(args.tp0.toolNames))
    return tool, obj_pos


def sample_from_strategy_graph(strat_graph, idx=0):
    '''
        do sample from gaussian process model in graph
    '''
    graph_list = [g for g in strat_graph.placement_graphs
        if all('model' in g.edges[e] for e in g.edges()
            if e not in list(args.tp0.toolNames) + ['Goal']
        )
    ]
    g = choice(graph_list)
    print("Selected graph", g.edges())
    cur_nd = 'Goal'
    pre_nd = None
    img_poss = [[],[]]
    goal_pos = centroidForPoly(args.btr0['world']['objects']['Goal']['points'])
    # obj_ext: for sampling
    obj_ext =[goal_pos[0], goal_pos[1], 0, random()*10-5, random()*10-5]
    # for backtracking
    obj_pos = goal_pos
    args.sequence_sample_poss.setdefault(cur_nd, []).append(list(obj_pos)+[0,0,0])
    while cur_nd != 'PLACED':
        # relative extrinsics
        pre_nd = choice(list(g.predecessors(cur_nd)))
        if pre_nd != 'PLACED' and cur_nd == 'Goal':
            prev_target_ext = list(goal_pos) + [0, 0, 0]
        elif pre_nd != 'PLACED' and cur_nd != 'Goal':
            obj = args.tp0.objects[cur_nd]
            prev_target_ext = (list(obj.position) + [0] + list(obj.velocity))
        else:
            prev_target_ext = None
        print(pre_nd, cur_nd)
        if prev_target_ext is not None:
            if cur_nd == 'Goal':
                target_init_pos = goal_pos
            else:
                target_init_pos = args.tp0.objects[cur_nd].position
            tool_init_pos = args.tp0.objects[pre_nd].position
            x = [list(obj_ext[2:]) + list(prev_target_ext) + list(tool_init_pos) + list(target_init_pos)]
        else:
            target_init_pos = args.tp0.objects[cur_nd].position
            x = [obj_ext[2:] + list(target_init_pos)]
        sample_poss = g.edges[(pre_nd, cur_nd)]['model'].sample_y(x, n_samples=10, random_state=None)
        sample_poss = [list(item)
        for sublist in np.transpose(np.array(sample_poss), (0, 2, 1))
        for item in sublist]
        # pos = []
        # print(x)
        # for est_gp in g.edges[(pre_nd, cur_nd)]['model']:
        #     pos.append(est_gp.predict(x))
        # sample_poss = [pos]
        # set to real extrinsics
        obj_ext = list(np.array(sample_poss).mean(axis=0))
        for s in sample_poss:
            args.sequence_sample_poss.setdefault(cur_nd, []).append([obj_pos[0] + s[0], obj_pos[1] + s[1], s[2], s[3], s[4]])
        obj_pos = [obj_pos[0] + obj_ext[0], obj_pos[1] + obj_ext[1], obj_ext[2], obj_ext[3], obj_ext[4]]
        # args.sequence_sample_poss.setdefault(cur_nd, []).append(obj_pos)

        
        print(pre_nd, cur_nd, obj_pos)
        cur_nd = pre_nd
        if not sample_poss: # failed to sample from GPR
            return None, None
    img_name = os.path.join(args.dir_name,
        'strat_seq.png'
    )
    img_poss = [v for v in args.sequence_sample_poss.values()]
    draw_samples(args.tp0, img_poss, '', img_name)
    tool = choice(list(g.nodes[cur_nd]['tools']))
    return tool, obj_pos # only return one sample

def sample_from_gm(strategy_graph):
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

def simulate_from_gm(sample_obj, gmm_pose, noisy=False):
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

#############################################################
#                                                           #
# ANCHOR - Main building strategy graph method              #
#                                                           #
#############################################################

# FIXME - deprecated method
def build_strategy_graph(strat_graph=None):
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
            if can_sample_from_gm and random() < args.gm_ratio:
                sample_type = 'gm'
                sample_obj, sample_ext = sample_from_gm(strategy_graph)
                path_info = simulate_from_gm(
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
                # collisions = standardize_collisions(collisions)
                # for c in collisions:
                #     print(c[0:4])
                # print()
                img_name = os.path.join(args.dir_name,
                    'collision.png'
                )
                draw_path(args.tp0, path_dict, img_name, sample_ext[0:2])
            if success:
                # sim_count += 1
                args.sim_count = sim_count
                # logging.info('Noisy Success: %d, %s %s (%d, %d)',
                    # sim_count, sample_obj, sample_type, sample_ext[0], sample_ext[1])
                graph = strategy_graph.build_graph(args, sample_obj, sample_ext, path_info)
                # NOTE - eliminate the case that the graph does not have path to goal due to noisy env
                if "Goal" in graph.nodes():
                    sim_count += 1
                    strategy_graph.set_placement_graph(args, graph, sample_obj)

        # !SECTION
    strategy_graph.merge_graph(args)
    strategy_graph.train(args)
    return strategy_graph


#############################################################
#                                                           #
# ANCHOR - human strategy approachs                         #
#                                                           #
#############################################################

def random_tool():
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

def initialize_SSUP_policy(policies):
    tp = args.tp0
    get_prior = args.get_prior
    for tool in tp.toolNames:
        rewards = []
        samples = []
        for i in range(args.num_init):
            sample_pos = get_prior(tp, args.movable_objects)
            reward = []
            for j in range(args.num_sim):
                _, r = estimate_reward(
                    tool, sample_pos, noisy=True
                )
                reward.append(r)
            reward = np.mean(reward)
            rewards.append(reward)
            samples.append([sample_pos[0], sample_pos[1], int(tool[3])-1])
        print(samples, rewards)
        for i in range(args.num_init):
            policies.update([samples[i]], [rewards[i]], learning_rate=args.lr)
    print(policies)
    return policies

def counterfactual_update(sample, sample_obj, sample_pos, reward, policies):
    tp = args.tp0
    for tool in tp.toolNames:
        cf_sample = [sample[0], sample[1], int(tool[3])-1]
        if tool == sample_obj:
            cf_reward = reward
        else:
            path_dict, _, _, _ = tp.runStatePath(
                toolname=tool,
                position=sample_pos,
                noisy=True
            )
            cf_reward = calculate_reward(args, tp, path_dict)
        policies.update([cf_sample], [cf_reward], learning_rate=args.lr)
    img_name = os.path.join(args.dir_name,
            'plot'+str(args.sim_count)+'.png'
        )
    plot_policies(args, policies, sample_pos, int(sample_obj[3])-1, img_name)
    return policies
def SSUP(policies):
    tp = args.tp0
    get_prior = args.get_prior
    epsilon = args.eps
    action_count = 0
    sim_count = 0
    # Sample ninit points from prior pi(s) for each tool
    # Simulate actions to get noisy rewards rˆ using internal model
    # initialize policy parameters theta using policy gradient on initial points
    policies = initialize_SSUP_policy(policies)
    # plot
    img_name = os.path.join(args.dir_name,'plot_init.png')
    plot_policies(args, policies, None, 0, img_name)
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
            # Estimate noisy reward rˆ from internal model on action a
            path_info, reward = estimate_reward(
                sample_obj, sample_pos, noisy=True
            )
            sim_count += 1
        else:
            # Sample from policy (at most sample 100 times)
            sample = policies.action()
            sample_obj = 'obj'+str(sample[2]+1)
            sample_pos = list(sample[0:2])
            rewards = []
            for _ in range(args.num_sim):
                # Estimate noisy reward rˆ from internal model on action a
                path_info, reward = estimate_reward(
                    sample_obj, sample_pos, noisy=True
                )
                rewards.append(reward)
                sim_count += 1
            reward = np.mean(rewards)
            iter_count += 1
        logging.info('Simulation %d %d: %s %s (%d, %d), %s, %f',
            action_count, sim_count, sample_obj, sample_type,
            sample_pos[0], sample_pos[1], success, reward)
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
                sample_obj, sample_pos, noisy=False
            )
            success = path_info[2]
            # epsilon *= args.eps_decay_rate
            action_count += 1
            # If successful, exit.
            logging.info('Attempt %d %d: %s %s (%d, %d), %f',
                action_count, sim_count, sample_obj, sample_type,
                sample_pos[0], sample_pos[1], reward)
            print(policies)
            if success:
                logging.info("Success! %d %d", action_count, sim_count)
                img_name = os.path.join(args.dir_name,
                    'plot_final.png'
                )
                plot_policies(args, policies, sample_pos, int(sample_obj[3])-1, img_name)
                args.trial_stats.setdefault('SSUP', []).append([action_count, sim_count])
                break
            # Simulate rˆ assuming other two tool choices.
            # Update policy based on all three estimates and actions.
            args.sim_count = sim_count
            policies = counterfactual_update(sample, sample_obj, sample_pos, reward, policies)
        else:
            # Update policy using policy gradient
            policies.update([sample], [reward], learning_rate=args.lr)
        if action_count >= args.max_attempts or sim_count >= args.max_simulations:
            logging.info("Out of max attempt! %d %d", action_count, sim_count)
            args.trial_stats.setdefault('SSUP', []).append([action_count, sim_count])
            break
    return args.trial_stats


def test_GPR(strategy_structure, sample_method=sample_from_strategy_graph):
    action_count = 0
    sim_count = 0
    while action_count < args.max_attempts:
        sample_obj, sample_pos = sample_method(strategy_structure, action_count)
        if not sample_obj:
            continue
        sample_pos = list(sample_pos[0:2])
        # NOTE - same as SSUP -> simulate in PE before attempt
        path_info, _ = estimate_reward(
            sample_obj,
            sample_pos,
            noisy=False
        )
        success = path_info[2]
        action_count += 1
        logging.info('GPR Attempt: %d %d (%d, %d)', action_count, sim_count,
            sample_pos[0], sample_pos[1])
        if success:
            logging.info('GPR Attempt Success: %d %d (%d, %d)', action_count, sim_count,
                sample_pos[0], sample_pos[1])
            if sample_method == sample_from_strategy_graph:
                method = 'strat'
            else:
                method = 'mech'
            args.trial_stats[method].append([action_count, sim_count,
                [sample_pos[0], sample_pos[1]]])
            break
    img_name = os.path.join(args.dir_name,
        'GP_seq.png'
    )
    img_poss = [v for v in args.sequence_sample_poss.values()]
    draw_samples(args.tp0, img_poss, '', img_name)
    if not success:
        logging.info('GPR out of max attempt: %d %d', action_count, sim_count)

def test_GM(strat_graph):
    action_count = 0
    sim_count = 0
    while action_count < args.max_attempts and sim_count < args.max_simulations:
        nd_list = [ (nd, g) for g in strat_graph.placement_graphs
            for nd in g.nodes() if 'GM' in g.nodes[nd] and nd in args.tp.toolNames]
        node, graph = choice(nd_list)
        sample_ext = graph.nodes[node]['GM'].sample()[0][0]
        sample_obj = node
        logging.info('GM Sample: %s (%d %d)', sample_obj, sample_ext[0], sample_ext[1])
        path_info, reward = estimate_reward(
            sample_obj,
            list(sample_ext[0:2]),
            noisy=True
        )
        path_dict, _, success = path_info
        if success is not None:
            sim_count += 1
        if reward > args.attempt_threshold:
            logging.info('GM Simulate: %d %d', action_count, sim_count)
            sample_pos = list(sample_ext[0:2])
            path_dict, _, success, _ =  args.tp0.runStatePath(
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
                logging.info('GM Attempt Success: %d %d (%d %d)',
                    action_count, sim_count, sample_ext[0], sample_ext[1])
                args.trial_stats[-1]["GM"].append([action_count, sim_count,
                    [sample_ext[0], sample_ext[1]]])
                break
    if not success:
        logging.info('GM out of max attempt: %d %d', action_count, sim_count)

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
    parser.add_argument('--tnm2',
                        help='task name for GPR', type=str, default=None)
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
    if args.algorithm in ['SSUP']:
        cfg = config.SSUP_config
    elif args.algorithm in ['GPR_SSUP', 'GPR_SSUP_GEN']:
        cfg = config.GPR_SSUP_config
    elif args.algorithm in ['GPR_MECH']:
        cfg = config.OURS_config
    args.__dict__['task'] = config.task_config[args.tnm]
    for c in cfg:
        if c not in args:
            args.__dict__[c] = cfg[c]
    for arg in vars(args):
        if args.__dict__[arg] is None and arg in cfg:
            args.__dict__[arg] = cfg[arg]
    # load task json file
    args.experiment_time = datetime.now()
    setup_experiment_dir(args)
    setup_task_args(args)
    args.trial_stats = {'strat': [], 'mech': []}
    for trial_count in range(args.num_experiment):
        args.experiment_time = datetime.now()
        # SECTION - set up logging configuration
        trial_name = str(trial_count).zfill(3)
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
        # NOTE - log args
        for arg in vars(args):
            logging.info('%20s %-s', arg, args.__dict__[arg])
        # !SECTION
        # SECTION - Start experiments
        if args.algorithm == 'random':
            random_tool()
        if args.algorithm == 'SSUP':
            gaussian_policies = initialize_policy(300, 300, 50)
            SSUP(gaussian_policies)
        elif args.algorithm == 'GPR_SSUP':
            strategy_graph = build_strategy_graph()
            # strategy_graph = load_strategy_graph(None, 'strat0.pkl')
            strategy_graph.train(args)
            logging.info('Start Testing')
            sample_object, sample_position = sample_from_strategy_graph(strategy_graph)
            gaussian_policies = initialize_policy(sample_position[0], sample_position[1], 50)
            SSUP(gaussian_policies)
        elif args.algorithm == 'GPR_SSUP_CF':
        elif args.algorithm == 'GPR_SSUP_GEN':
            strategy_graphs = []
            tasks = args.task['training']
            # for tnm in tasks:
            #     logging.info('Task %s', tnm)
            #     setup_task_args(args, tnm)
            #     strategy_graph = build_strategy_graph()
            #     strategy_graphs.append(strategy_graph)
            file_path = os.path.join('data/strategy', 'strat_'+args.tnm+'.pkl')
            # save_strategy_graph(strategy_graphs, file_path)
            strategy_graphs = load_strategy_graph(None, file_path)
            strategy_graph = merge_graphs(args, strategy_graphs)

            # tnm = 'Catapult'
            # setup_task_args(args, tnm)
            # sample_object, sample_position = sample_from_strategy_graph(strategy_graph)
            # gaussian_policies = initialize_policy(sample_position[0], sample_position[1], 50)
            # SSUP(gaussian_policies)
            # input()

            tnm = args.tnm2 if args.tnm2 else args.tnm
            logging.info('Start experiment %s', tnm)
            setup_task_args(args, tnm)
            # GPR
            # sample_object, sample_position = sample_from_strategy_graph(strategy_graph)
            sample_object, sample_position = sample_for_catapultalt(strategy_graph)
            gaussian_policies = initialize_policy(sample_position[0], sample_position[1], 50)
            SSUP(gaussian_policies)
        elif args.algorithm == 'GPR_MECH':
            setup_task_args(args)
            strategy_graphs = []
            tasks = args.task['training']
            for tmn in tasks:
                logging.info('Training %s', tmn)
                setup_task_args(args, tmn)
                strategy_graph = build_strategy_graph()
                strategy_graphs.append(strategy_graph)
            file_path = os.path.join('data/strategy', 'strat_'+args.tnm+'.pkl')
            save_strategy_graph(strategy_graphs, file_path)
            strategy_graphs = load_strategy_graph(None, file_path)
            strategy_graph = merge_graphs(args, strategy_graphs)
            # mechanisms = merge_mechanisms(args, strategy_graphs)
            # train_mechanism(args, mechanisms)
            tnm = args.task['testing']
            logging.info('Start experiment')
            logging.info('Strategy Graph on %s', tnm)
            setup_task_args(args, tnm)
            sample_object, sample_position = sample_from_strategy_graph(strategy_graph)
            gaussian_policies = initialize_policy(sample_position[0], sample_position[1], 50)
            SSUP(gaussian_policies)
            # test_GPR(strategy_graph, sample_from_strategy_graph)
            # print(mechanisms.keys())
            # test_GPR(mechanisms, sample_from_mechanisms)
        else:
            raise ValueError(f"Invalid algorithm: {args.algorithm}")
        time.sleep(1) # avoid generating same experiment id
        # !SECTION

    print('End experiments')
    # SECTION - log summary
    logfile_path = os.path.join(args.main_dir_name, 'summary.log')
    file_handler = logging.FileHandler(logfile_path)
    file_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(file_handler)
    logging.info('Summary')
    for key in args.trial_stats:
        logging.info('%s', key)
        for item in args.trial_stats[key]:
            logging.info('%s', str(item)[1:-1])
    duration = datetime.now() - args.experiment_time
    logging.info('%s', duration)
    # !SECTION
