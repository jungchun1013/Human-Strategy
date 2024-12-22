from random import choice, randint, random, choices
from datetime import datetime
import os
import json
import pickle
import math
import numpy as np
from copy import deepcopy
import pymunk as pm
import pygame as pg
from pyGameWorld import PGWorld, ToolPicker, loadFromDict
from pyGameWorld.viewer import *
from pyGameWorld.viewer import drawPathSingleImageWithTools2
from pyGameWorld.jsrun import *
from pyGameWorld.helpers import *

##############################################
#
# Experiment setup
#
##############################################

def setup_experiment_dir(args):
    # NOTE - generate experiment dir
    exptime_str = args.experiment_time.strftime("%y%m%d_%H%M%S")
    date = exptime_str[:6]
    time = exptime_str[7:]
    args.exp_name = '_'.join([time, args.tnm, args.algorithm])
    args.main_dir_name = os.path.join('data', date, args.exp_name)
    os.makedirs(args.main_dir_name)

def setup_task_args(args, tnm=None, testing=False):
    if not tnm:
        tnm = args.tnm
    if testing:
        args.noisy = False
    else:
        args.noisy = not args.deterministic
    args.get_prior = set_prior_type(args.algorithm)
    with open(args.json_dir + tnm + '.json','r') as f:
        args.btr0 = json.load(f)
    args.tp0 = ToolPicker(args.btr0)

    args.movable_obj_dict = {i:j for i, j in args.tp0.objects.items()
                        if j.color in [(255, 0, 0, 255), (0, 0, 255, 255)]
    }
    args.sequence_sample_poss = {}
    args.sequence_obj_pos = {}
    args.movable_objects = list(args.movable_obj_dict)
    args.tool_objects = list(args.tp0.toolNames)
    # path_dict0, _, _ = args.tp0.observeStatePath()
    args.path_dict0, args.collisions0, _, _ = args.tp0.runStatePath()
    args.tp0

    args.dist0 = min(args.tp0.world.distanceToGoalContainer((args.path_dict0[obj][i][:2])) 
        for obj in args.path_dict0 for i in range(len(args.path_dict0[obj])) 
        if args.tp0.objects[obj].color==(255, 0, 0, 255))
    args.ext_sampler = ExtrinsicSampler(args.tp0, args.btr0, args.path_dict0)
##############################################
#
# Sampler
#
##############################################

def extrinsics_sampler(args, sample_type, strategy, sample_objs= None):
    if sample_objs:
        weights = [ math.e**(-strategy.obj_count[m]*args.eps)
            for m in sample_objs
        ]
        sample_obj = choices(sample_objs, weights=weights, k=1)[0]
    else:
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
    elif sample_type == 'CF':
        btr = args.ext_sampler.sample_CF(sample_obj)
    tp = ToolPicker(btr)

    sample_ext = [*tp.objects[sample_obj].position, tp.objects[sample_obj].rotation, *tp.objects[sample_obj].velocity]  
    ext_info = sample_ext
    path_dict, collisions, success, t = tp.runStatePath(
        noisy=args.noisy
    )
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

def sample_ext_by_type(args, sample_type, strategy, sample_objs):
    if sample_type in ['pos', 'vel', 'ang', 'CF']: # fixed pos, smaple from path
        result = extrinsics_sampler(args, sample_type, strategy, sample_objs)
    elif sample_type == 'kick':
        result = kick_sampler(args, strategy)
    elif sample_type == 'tool':
        result = tool_sampler(args)
    # result: sample_obj, sample_ext, ext_info, path_info
    return result

##############################################
#
# Algorithm
#
##############################################

##############################################

def draw_policies(tp, policies, img_name):
    pg.display.set_mode((10,10))
    worlddict = tp._worlddict
    sc = drawWorldWithTools(tp, worlddict=worlddict)
    colors = [(255,0,255), (255,255,0), (0,255,255)]
    for i in range(3):
        surface_width = 4 * policies.params[f"obj{i+1}_sigma1"].detach().numpy()
        
        surface_height = 4 * policies.params[f"obj{i+1}_sigma2"].detach().numpy()
        ellipse_surface = get_ellipse_surface(surface_width, surface_height, colors[i])
        mean1 = policies.params[f"obj{i+1}_mu1"]
        mean2 = policies.params[f"obj{i+1}_mu2"]
        sc.blit(ellipse_surface.convert_alpha(), (int(mean1-surface_width/2), int(600-mean2-surface_height/2)))

    img = sc.convert_alpha()
    pg.image.save(img, img_name)

def draw_gradient_samples(tp, samples, img_name):
    pg.display.set_mode((10,10))
    worlddict = tp._worlddict
    sc = drawWorldWithTools(tp, worlddict=worlddict)
    colors = [(255,0,0), (0,255,0), (0,0,255)]
    for s in samples:
        s, r = s
        a = 0.1 if r < 0.1 else r
        x, y = s[0], 600-s[1]
        color = (255, 255-int(a*255), 255-int(a*255))
        # vx, vy = s[3]/5, s[4]/5
        pg.draw.circle(sc, color, [x, y], 5)
        # pg.draw.line(sc, 'red', [x, y], [x+vx, y+vy], 2)
    img = sc.convert_alpha()
    pg.image.save(img, img_name)

def draw_samples(tp, samples, task, img_name):
    pg.display.set_mode((10,10))
    worlddict = tp._worlddict
    sc = drawWorldWithTools(tp, worlddict=worlddict)
    if task == 'compare_tool_target':
        colors = [(255, 128, 128), (128, 255, 128), (255, 0, 0), (0, 255, 0)]
    elif task == 'tool_target':
        colors = [(255, 0, 0), (0, 255, 0)]
    elif task == 'single':
        colors = [(255, 0, 0)]
    else:
        gradient = 255//len(samples)
        colors = [(255-i*gradient, 0, i*gradient) for i in range(len(samples)-1)] + [(0,255,0)]
    for col, sample in zip(colors, samples):
        for s in sample:
            x, y = s[0], 600-s[1]
            vx, vy = s[3]/5, s[4]/5
            pg.draw.circle(sc, col, [x, y], 5)
            pg.draw.line(sc, col, [x, y], [x+vx, y-vy], 2)
        img = sc.convert_alpha()
        pg.image.save(img, img_name)

def draw_data(tp, data_points, img_name, group_colors, group_sizes):
    pg.display.set_mode((10,10))
    worlddict = tp._worlddict
    sc = drawWorldWithTools(tp, worlddict=worlddict)
    for s, color, size in zip(data_points, group_colors, group_sizes):
        x, y = s[0], 600-s[1]
        pg.draw.circle(sc, color, [x, y], size)

        img = sc.convert_alpha()
        pg.image.save(img, img_name)

def get_ellipse_surface(surface_width, surface_height, col):
    ellipse_surface = pg.Surface((surface_width, surface_height), pg.SRCALPHA)
    rect = pg.Rect(0, 0, surface_width, surface_height)

    # ellipse_surface = pg.Surface((rect[2], rect[3]), pg.SRCALPHA)
    for y in range(int(surface_height)):
        for x in range(int(surface_width)):
            # 椭圆方程 (x/a)^2 + (y/b)^2 <= 1
            if ((x - surface_width / 2) ** 2) / (surface_width / 2) ** 2 + ((y - surface_height / 2) ** 2) / (surface_height / 2) ** 2 <= 1:
                # distance_to_center = np.sqrt((x - surface_width / 2) ** 2 + (y - surface_height / 2) ** 2)
                # max_distance = np.sqrt((surface_width / 2) ** 2 + (surface_height / 2) ** 2)
                distance_to_center = np.sqrt((x-surface_width/2)**2/(surface_width/2)**2 + (y-surface_height/2)**2/(surface_height/2)**2)
                max_distance = 1
                factor = distance_to_center / max_distance
                color = (col[0], col[1], col[2], int((1-factor)*232))
                ellipse_surface.set_at((x, y), color)
    return ellipse_surface

def draw_ellipse(tp, samples, task, img_name):
    pg.display.set_mode((10,10))
    worlddict = tp._worlddict
    sc = drawWorldWithTools(tp, worlddict=worlddict)
    if task == 'compare_tool_target':
        colors = [(255, 128, 128), (128, 255, 128), (255, 0, 0), (0, 255, 0)]
    elif task == 'tool_target':
        colors = [(255, 0, 0), (0, 255, 0)]
    elif task == 'single':
        colors = [(255, 0, 0)]
    else:
        gradient = 255//len(samples)
        colors = [(255-i*gradient, 0, i*gradient) for i in range(len(samples)-1)] + [(0,255,0)]
    
    for col, sample in zip(colors, samples):
        mean1 = np.mean([s[0] for s in sample], axis=0)
        # cov1 = np.cov([s[0] for s in sample], rowvar=False)
        var1 = np.std([s[0] for s in sample], axis=0)
        mean2 = np.mean([s[1] for s in sample], axis=0)
        # cov2 = np.cov([s[1] for s in sample], rowvar=False)
        var2 = np.std([s[1] for s in sample], axis=0)
        surface_width = int(var1*4)
        surface_height = int(var2*4)
        ellipse_surface = get_ellipse_surface(surface_width, surface_height, col)

        sc.blit(ellipse_surface.convert_alpha(), (mean1-surface_width/2, 600-mean2-surface_height/2))
        img = sc.convert_alpha()
        pg.image.save(img, img_name)


def draw_path(tp, path_dict, img_name, tool_pos=None):
    if not path_dict: return
    pg.display.set_mode((10,10))
    # sc = drawPathSingleImageWithTools(tp, path_dict)
    sc = drawPathSingleImageWithTools2(tp, path_dict, [0], with_tools=True)
    if tool_pos is not None:
        tool_pos = [tool_pos[0], 600-tool_pos[1]]
        pg.draw.circle(sc, (0,0,255), tool_pos, 5)
    img = sc.convert_alpha()
    pg.image.save(img, img_name)

def draw_multi_paths(world, path_set, img_name):
    if not path_set: return
    pg.display.set_mode((10,10))
    sc = drawMultiPathSingleImage(world, path_set)
    img = sc.convert_alpha()
    pg.image.save(img, img_name)

def set_prior_type(args):
    return get_prior_SSUP

def print_stats(trial_stats):
    print('5: ', len([t for t in trial_stats if t <5])/len(trial_stats))
    for i in range(2,7):
        print(str(5*i)+':', len([str(5*j) for j in trial_stats if j <5*i])/len(trial_stats))

def get_prior_SSUP(tp, movable_objects, normalize=False): # NOTE - add tp
    obj = choice(movable_objects)
    BB = objectBoundingBox(tp.objects[obj])
    x = randint(BB[0][0]-20,BB[1][0]+20)
    range_y = choices([(0,BB[0][1]), (BB[1][1],600)], weights=[BB[0][1]-0, 600-BB[1][1]], k=1)[0]
    y = randint(range_y[0],range_y[1])
    if normalize:
        return normalize_pos((x,y))
    return (x,y)

def get_prior_catapult(obj_dict):
    obj = choice(list(obj_dict.keys()))
    # if obj == 'Catapult':
    # prior of Catapult
    x0 = obj_dict['Catapult'].getPolys()[2][2][0]
    y0 = obj_dict['Catapult'].getPolys()[2][2][1]
    x = randint(x0-20, x0+20) # 
    y = randint(y0+20,600)
    return (x,y)

def calculate_reward(args, path_dict):
    if not path_dict:
        return 0
    dist0 = args.dist0
    dist = 1000
    for obj in path_dict:
        if obj == 'PLACED': continue
        for i in range(len(path_dict[obj])):
            if args.tp0.objects[obj].color == (255, 0, 0, 255):
                dist_tmp = args.tp0.world.distanceToGoalContainer((path_dict[obj][i][:2]))
                if dist > dist_tmp:
                    dist = dist_tmp
                # reward = -(dist/dist0-0.5)*2
    reward = 1-(dist/dist0)
    return reward

def normalize_pos(pos):
    return ((pos[0]-300)/100, (pos[1]-300)/100)

def scale_pos(pos):
    return ((pos[0]*100)+300, (pos[1]*100)+300)


def calculate_weights(policy_rewards, gaussian_policies):
    # weights=[sum(policy_rewards[i])/len(policy_rewards[i]) if policy_rewards[i] else 1 for i in gaussian_policies.keys()]
    # if min(weights) < 0:
    #     weights = [i - min(weights) for i in weights]

    weights = [sorted(policy_rewards).index(x)+1 for x in policy_rewards]
    return weights

def is_sublist(larger_list, sublist):
    # Lengths of the lists
    len_larger = len(larger_list)
    len_sublist = len(sublist)

    # A sublist cannot exist in a smaller list
    if len_sublist > len_larger:
        return False

    # Check each possible starting position in larger_list
    for start in range(len_larger - len_sublist + 1):
        # Check if the sublist matches at this position
        if larger_list[start:start + len_sublist] == sublist:
            return True

    # No match found
    return False

def standardize_collisions(collisions):
    for i, c in enumerate(collisions):
        o1, o2, ts, te, ci = c
        if isinstance(ci[0], pm.Vec2d):
            ci[0] = [ci[0].x, ci[0].y]
        collisions[i] = [o1, o2, ts, te, ci]
    return collisions

class ExtrinsicSampler():
    def __init__(self, tp, btr, path_dict):
        self._tp = tp
        self._btr = btr
        self.path_dict = path_dict
    
    def sample_pos(self, sample_obj):
        # TODO - sample from other counterfactual conditions
        rand_rad = random()*2*np.pi
        sample_pose = choice(self.path_dict[sample_obj]) # sample from intial path
        pos0, rot0 = sample_pose[0:2], sample_pose[2]
        rand_scale = randint(0,10)
        btr = deepcopy(self._btr)
        btr['world']['objects'][sample_obj]['position'] = (pos0[0] + rand_rad * rand_scale, pos0[1] + rand_rad * rand_scale)
        btr['world']['objects'][sample_obj]['rotation'] = (rot0 + (random()-0.5)*0.1)
        btr['world']['objects'][sample_obj]['color'] = (0, 0, 0) # fix pos
        return btr
    def sample_vel(self, sample_obj):
        rand_rad = random()*2*np.pi
        rand_scale = randint(1,50)*10
        velocity = [np.cos(rand_rad) * rand_scale,  np.sin(rand_rad) * rand_scale]
        btr = deepcopy(self._btr)
        btr['world']['objects'][sample_obj]['velocity'] = velocity
        return btr
    def sample_ang(self, sample_obj):
        rand_ang = randint(1,50)*10*2*np.pi
        btr = deepcopy(self._btr)
        btr['world']['objects'][sample_obj]['angular_velocity'] = rand_ang
        return btr

    def sample_CF(self, sample_obj):
        rand_rad = random()*2*np.pi
        rand_pos_rad = random()*2*np.pi
        sample_pose = choice(self.path_dict[sample_obj]) # sample from intial path
        pos0, rot0 = sample_pose[0:2], sample_pose[2]
        btr = deepcopy(self._btr)
        pos0 = self._tp.objects[sample_obj].getPos()
        rand_scale = randint(0,10)
        btr['world']['objects'][sample_obj]['position'] = (pos0[0] + np.cos(rand_pos_rad) * rand_scale, pos0[1] + np.sin(rand_pos_rad) * rand_scale)
        rand_scale = randint(1,60)*10
        velocity = [np.cos(rand_rad) * rand_scale,  np.sin(rand_rad) * rand_scale]
        btr['world']['objects'][sample_obj]['velocity'] = velocity
        btr['world']['objects'][sample_obj]['rotation'] = (rot0 + (random()-0.5)*0.1)
        tp = ToolPicker(btr)
        tp
        return btr

def node_match(node1, node2):
    print(node1, node2)
    return node1 == node2

def load_strategy_graph(strategy_graph, file_name='strategy_graph.pkl'):
    with open(file_name, 'rb') as f:
        strategy_graph = pickle.load(f)
    return strategy_graph
def save_strategy_graph(strategy_graph, file_name='strategy_graph.pkl'):
    with open(file_name, 'wb') as f:
        pickle.dump(strategy_graph, f)
    return strategy_graph

# def generalization_test():
#     strat_graphs = []
#     for tnm in ['Launch_v2', 'Catapult_5', 'Launch_A', 'Catapult']:
#         args.tnm = tnm
#         logging.info('Start experiment %s', tnm)
#         setup_task_args(args, tnm)
#         strategy_graph = build_strategy_graph()
#         strat_graphs.append(strategy_graph)

#     with open('strategy_graph.pkl', 'wb') as f:
#         pickle.dump(strat_graphs, f)
#     # with open('strategy_graph.pkl', 'rb') as f:
#     #     strat_graphs = pickle.load(f)
#     # test on CatapultAlt obj -> lever -> cataball -> keyball -> goal
#     args.tnm = 'CatapultAlt'
#     with open(args.json_dir + args.tnm + '.json','r') as f:
#         args.btr0 = json.load(f)
#     args.tp0 = ToolPicker(args.btr0)
#     action_count = 0
#     sim_count = 0
#     sample_count = 0
#     args.trial_stats.append({'GPR': [], 'GM': []})
#     success = False
#     while True:
#         g = choice(strat_graphs)
#         g = g.placement_graphs[0]
#         cur_nd = 'Goal'
#         # gmm = g.nodes[cur_nd]['GM']
#         # sample_poss = gmm.sample(n_samples=3)
#         # sample_poss = sample_poss[0]
#         pre_nd = None
#         # NOTE
#         sample_poss = centroidForPoly(args.btr0['world']['objects']['Goal']['points'])
#         sample_poss = [[
#             sample_poss[0]+random()*40-20,
#             sample_poss[1]+random()*40-20,
#             0,
#             random()*100-50,
#             random()*100-50]]
#         img_name = os.path.join(args.trial_dir_name,
#             'Gen_test_'+cur_nd+'.png'
#         )
#         print(cur_nd, int(sample_poss[0][0]), int(sample_poss[0][1]))
#         cur_nd = 'Ball'
#         g = [g for g in strat_graphs[0].placement_graphs if ('Ball2', 'Ball') in g.edges()][0]
#         gpr = g.edges[(pre_nd, cur_nd)]['model']
#         obj_pos = sample_poss[0]
#         # NOTE
#         samples = gpr.sample_y([obj_pos[2:]], n_samples=3)
#         arr = np.array(samples)
#         sample_poss = [list(item)
#             for sublist in np.transpose(arr, (0, 2, 1))
#             for item in sublist]
#         img_poss = [[obj_pos[0]+s[0],obj_pos[1]+s[1],s[2],s[3],s[4]] for s in sample_poss]
#         img_name = os.path.join(args.trial_dir_name,
#             'Gen_test_'+cur_nd+'1.png'
#         )
#         draw_gp_samples(args.tp0, [img_poss, [obj_pos]], 'tool_target', img_name)
#         cur_nd = 'Ball'
#         g = [g for g in strat_graphs[1].placement_graphs if ('Catapult', 'Ball') in g.edges()][0]
#         gpr = g.edges[(pre_nd, cur_nd)]['model']
#         sample_pos = list(np.array(sample_poss).mean(axis=0))
#         print(cur_nd, sample_pos)

#         obj_pos = [obj_pos[0] + sample_pos[0], obj_pos[1] + sample_pos[1],
#             sample_pos[2], sample_pos[3], sample_pos[4]]
#         # NOTE
#         samples = gpr.sample_y([obj_pos[2:]], n_samples=3)
#         arr = np.array(samples)
#         sample_poss = [list(item)
#             for sublist in np.transpose(arr, (0, 2, 1))
#             for item in sublist]
#         img_poss = [[obj_pos[0] + s[0], obj_pos[1] + s[1], s[2], s[3], s[4]]
#             for s in sample_poss]
#         img_name = os.path.join(args.trial_dir_name,
#             'Gen_test_'+cur_nd+'x.png'
#         )
#         draw_gp_samples(args.tp0, [img_poss, [obj_pos]], img_name)
#         sample_pos = list(np.array(sample_poss).mean(axis=0))
#         obj_pos = [obj_pos[0] + sample_pos[0], obj_pos[1] + sample_pos[1], 
#             sample_pos[2], sample_pos[3], sample_pos[4]]
#         cur_nd = 'Catapult'
#         g = [g for g in strat_graphs[1].placement_graphs if ('obj2', 'Catapult') in g.edges()][0]
#         gpr = g.edges[(pre_nd, cur_nd)]['model']
#         # NOTE
#         samples = gpr.sample_y([obj_pos[2:]], n_samples=3)
#         arr = np.array(samples)
#         sample_poss = [list(item)
#             for sublist in np.transpose(arr, (0, 2, 1))
#             for item in sublist]
#         sample_pos = sample_poss[0]
#         print(cur_nd, sample_pos)
#         img_poss = [[obj_pos[0]+s[0],obj_pos[1]+s[1],s[2],s[3],s[4]] for s in sample_poss]
#         img_name = os.path.join(args.trial_dir_name,
#             'Gen_test_'+cur_nd+'.png'
#         )
#         draw_gp_samples(args.tp0, [img_poss, [sample_poss[0]]], img_name)
#         obj_pos = [obj_pos[0] + sample_poss[0][0], obj_pos[1] + sample_poss[0][1]]
#         sample_pos = obj_pos
#         # sample_obj = choice(list(args.tp0.toolNames))
#         sample_obj = 'obj2'
#         path_info, reward = estimate_reward(
#             sample_obj,
#             obj_pos,
#             noisy=True
#         )
#         path_dict, _, success = path_info
#         sample_count += 1
#         if success is not None:
#             sim_count += 1
#         logging.info('GPR Sample: %s (%d %d)', sample_obj, sample_pos[0], sample_pos[1])
#         if success is not None and reward > args.attempt_threshold:
#             action_count += 1
#             logging.info('GPR Simulate: %d %d %d, %s (%d %d) %s %f',
#                 action_count, sim_count, sample_count, sample_obj,
#                 sample_pos[0], sample_pos[1], success, reward)
#             img_name = os.path.join(args.trial_dir_name,
#                 'sample_from_GPR_'+str(sim_count)+'_'+args.tnm+'.png'
#             )
#             draw_path(args.tp0, path_dict, img_name, sample_pos)
#             path_info, reward = estimate_reward(
#                 sample_obj,
#                 sample_pos,
#                 noisy=False
#             )
#             path_dict, collisions, success = path_info
#             logging.info('GPR Attempt: %d %d %d (%d, %d) %s %f',
#                 action_count, sim_count, sample_count,
#                 sample_pos[0], sample_pos[1], success, reward)
#             if success:
#                 args.trial_stats[-1]['GPR'].append([action_count, sim_count, sample_count,
#                     [sample_pos[0], sample_pos[1]]])
#                 break
#         if sim_count >= 100:
#             logging.info('GPR out of max attempt: %d %d %d', action_count, sim_count, sample_count)
#             break
