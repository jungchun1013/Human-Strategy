import argparse
import json
from random import choice, randint, random, choices
import numpy as np
from copy import deepcopy
from pyGameWorld import ToolPicker, objectBoundingBox
import pymunk as pm
from pyGameWorld.jsrun import pyGetCollisionsAddForces
from mechanism import MechanismGraph
from utils import *
import os

# Load level in from json file
# For levels used in experiment, check out Level_Definitions/
json_dir = "./Trials/Strategy/"
# Basic Table_A
tnm = "CatapultAlt"

with open(json_dir+tnm+'.json','r') as f:
  btr0 = json.load(f)

tp0 = ToolPicker(btr0)

movable_obj_dict = {i:j for i,j in tp0.objects.items() if j.color in [(255, 0, 0, 255), (0, 0, 255, 255)]}
movable_objects = list(movable_obj_dict.keys())
tools = list(tp0.toolNames)
available_objects = movable_objects + tools

path_dict0, success, time_to_success = tp0.observeStatePath()


    
def run(args):
    args.experiment_id = generate_experiment_id()
    args.dir_name = 'data/' + args.experiment_id
    os.makedirs(args.dir_name)
    CF_run(args)



def CF_run(args):
    # NOTE - MechanismGraph
    mechanism_graph = MechanismGraph()
    success = False
    ext_sampler = ExtrinsicSampler(btr0, path_dict0)
    image_num = 0
    while not success:
        btr = deepcopy(btr0)
        tp = ToolPicker(btr)
        # FIXED - sample from movable objects and tools
        sample_obj = choices(available_objects, weights=[1/(mechanism_graph.obj_count[m]+1) for m in available_objects], k=1)[0]
        pos = None
        if sample_obj in tools:
            scaled_pos = get_prior_SSUP(tp0)
            # scaled_pos = [240, 550]
            path_dict, success, time_to_success = tp.observePlacementStatePath(toolname=sample_obj,position=scaled_pos)
            path, collisions, end, t = tp.observeFullCollisionEvents(toolname=sample_obj,position=scaled_pos)
            init_pose = [scaled_pos[0], scaled_pos[1], 0, 0, 0, 0]
        else:
            # NOTE - choose what counterfactual extrinsics
            # sample_ext = choice(['pos', 'vel', 'kick'])
            sample_ext = choice(['vel', 'kick']) 
            rand_rad = random()*2*np.pi
            # SECTION - sample extrinsics and simulate path
            if sample_ext in ['pos', 'vel']: # fixed pos, smaple from path
                if sample_ext == 'pos':
                    btr = ext_sampler.sample_pos(sample_obj)
                elif sample_ext == 'vel': # set velocity
                    btr = ext_sampler.sample_vel(sample_obj)
                tp = ToolPicker(btr)
                path_dict, success, time_to_success = tp.observeStatePath()
                path, collisions, end, t = tp.observeFullCollisionEventsNoTool()
                init_pose = path_dict[sample_obj][0]
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
                force_times = {0.0:[[sample_obj, impulse, pos]], 0.1:[[sample_obj, impulse, pos]], 0.2:[[sample_obj, impulse, pos]]}
                path_dict, collisions, success, time_to_success = pyGetCollisionsAddForces(tp.world, force_times=force_times)
                init_pose = path_dict[sample_obj][2]

            collision_pattern = [c[0:2] for c in  collisions]
            # !SECTION

        if success and sample_obj in tp.toolNames: # tools
            print("Success! ", sample_obj, btr['tools'][sample_obj], sample_ext)
        elif success and sample_obj not in tp.toolNames: # objects
            print("Extrinsic {:>10} {:>4} {:>28} {:>4}".format(sample_obj, sample_ext, str([int(num) for num in init_pose]), sample_ext))

        # NOTE - build mechanism
        if success:
            ext_info = {'pos': init_pose[0:2], 'rot': init_pose[2], 'vel': init_pose[3:5], 'path': path_dict, 'collision': collision_pattern}

            sample_mech_id = mechanism_graph.add_mechanism_by_ext(tp, sample_obj, ext_info)
            sample_mech = mechanism_graph.get_mechanism(sample_mech_id)

            draw_multi_paths(btr['world'], sample_mech.get_paths(), args.dir_name+'/'+sample_mech_id+'.png')

            # check if sample_mech is a successor mech of existing mechs
            mechanism_graph.check_mechanism_successors(sample_mech, path_dict0)
            if sample_mech.obj_name not in tp.toolNames:
                mechanism_graph.check_mechanism_predecessors(sample_mech, path_dict0)
            mechanism_graph.merge_mechanisms(tp, sample_mech_id)

            mechanism_graph.save_graph(args.dir_name+'/graph_.png')
            mechanism_graph.transitive_reduction()
            mechanism_graph.save_graph(args.dir_name+'/graph_'+str(image_num)+'.png')
            image_num += 1
            mechanism_graph.save_graph(args.dir_name+'/graph.png')
        success = False

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