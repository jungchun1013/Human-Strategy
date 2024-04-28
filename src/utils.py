from random import choice, randint, random, choices
from datetime import datetime
import numpy as np
from copy import deepcopy
import pymunk as pm
import pygame as pg
from pyGameWorld import PGWorld, ToolPicker, loadFromDict
from pyGameWorld.viewer import *
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

def setup_task_args(args, tnm=None):
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


##############################################
#
# Algorithm
#
##############################################

##############################################

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

def draw_samples(tp, samples, img_name):
    pg.display.set_mode((10,10))
    worlddict = tp._worlddict
    sc = drawWorldWithTools(tp, worlddict=worlddict)
    colors = [(255,0,0), (0,255,0), (0,0,255)]
    for s in samples:
        x, y = s[0], 600-s[1]
        vx, vy = s[3]/5, s[4]/5
        pg.draw.circle(sc, 'red', [x, y], 5)
        pg.draw.line(sc, 'red', [x, y], [x+vx, y-vy], 2)
    img = sc.convert_alpha()
    pg.image.save(img, img_name)        

def draw_gp_samples(tp, samples, img_name, path_dict=None):
    pg.display.set_mode((10,10))
    worlddict = tp._worlddict
    if path_dict:
        sc = drawPathSingleImageWithTools(tp, path_dict)
    else:
        sc = drawWorldWithTools(tp, worlddict=worlddict)
    for s in samples[0]:
        if s is None: continue
        x, y = s[0], 600-s[1]
        vx, vy = s[3]/5, s[4]/5
        pg.draw.circle(sc, 'red', [x, y], 6)
        pg.draw.line(sc, 'red', [x, y], [x+vx, y-vy], 3)
    for s in samples[1]:
        if s is None: continue
        x, y = s[0], 600-s[1]
        vx, vy = s[3]/5, s[4]/5
        pg.draw.circle(sc, 'green', [x, y], 5)
        pg.draw.line(sc, 'green', [x, y], [x+vx, y-vy], 2)
    img = sc.convert_alpha()
    pg.image.save(img, img_name)    


def draw_4_samples(tp, samples, img_name, path_dict=None):
    pg.display.set_mode((10,10))
    worlddict = tp._worlddict
    if path_dict:
        sc = drawPathSingleImageWithTools(tp, path_dict)
    else:
        sc = drawWorldWithTools(tp, worlddict=worlddict)
    
    color = (255, 128, 128)
    for s in samples[0]:
        if s is None: continue
        x, y = s[0], 600-s[1]
        vx, vy = s[3]/5, s[4]/5
        pg.draw.circle(sc, color, [x, y], 5)
        pg.draw.line(sc, color, [x, y], [x+vx, y-vy], 2)

    color = (128, 255, 128)
    for s in samples[1]:
        if s is None: continue
        x, y = s[0], 600-s[1]
        vx, vy = s[3]/5, s[4]/5
        pg.draw.circle(sc, color, [x, y], 5)
        pg.draw.line(sc, color, [x, y], [x+vx, y-vy], 2)

    color = (255, 0, 0)
    for s in samples[2]:
        if s is None: continue
        x, y = s[0], 600-s[1]
        vx, vy = s[3]/5, s[4]/5
        pg.draw.circle(sc, color, [x, y], 3)
        pg.draw.line(sc, color, [x, y], [x+vx, y-vy], 1)

    color = (0, 255, 0)
    for s in samples[3]:
        if s is None: continue
        x, y = s[0], 600-s[1]
        vx, vy = s[3]/5, s[4]/5
        pg.draw.circle(sc, color, [x, y], 3)
        pg.draw.line(sc, color, [x, y], [x+vx, y-vy], 1)

    img = sc.convert_alpha()
    pg.image.save(img, img_name) 

def draw_path(tp, path_dict, img_name, tool_pos=None):
    if not path_dict: return
    pg.display.set_mode((10,10))
    sc = drawPathSingleImageWithTools(tp, path_dict)
    if tool_pos:
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

def calculate_reward(args, tp, path_dict):
    if not path_dict:
        return 0
    dist0 = args.dist0
    dist = 1000
    for obj in path_dict:
        if obj == 'PLACED': continue
        for i in range(len(path_dict[obj])):
            if tp.objects[obj].color == (255, 0, 0, 255):
                dist_tmp = tp.world.distanceToGoalContainer((path_dict[obj][i][:2]))
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
    def __init__(self, btr, path_dict):
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

def node_match(node1, node2):
    print(node1, node2)
    return node1 == node2

def load_strategy_graph(strategy_graph, file_name='strategy_graph.pkl'):
    with open('strategy_graph.pkl', 'rb') as f:
        strategy_graph = pickle.load(f)
    return strategy_graph
def save_strategy_graph(strategy_graph, file_name='strategy_graph.pkl'):
    with open(file_name, 'wb') as f:
        pickle.dump(strategy_graph, f)
    return strategy_graph