'''
update as SSUP
the model to update is per object(strategy)
'''
from pyGameWorld import PGWorld, ToolPicker, loadFromDict
from pyGameWorld.viewer import *
import json
import pygame as pg
from gaussian_policy_torch import Gaussian2DPolicy, random_init_policy, plot_policies
from random import choice, randint, random, choices
from utils import *
import argparse

# Load level in from json file
# For levels used in experiment, check out Level_Definitions/
json_dir = "./Trials/Strategy/"
# Basic Table_A
tnm = "Catapult"

with open(json_dir+tnm+'.json','r') as f:
  btr = json.load(f)

tp = ToolPicker(btr)

# extract red object (ball) and blue object (moveable object)
obj_dict = {i:j for i,j in tp.getObjects().items() if j.color in [(255, 0, 0, 255), (0, 0, 255, 255)]}

# run SSUP
def run(args):
    args.get_prior = set_prior_type(args)
    args.trial_stats = []
    for t in range(args.num_trial):
        args.trial_stats = SSUP(args)
        if (t+1) % 10 == 0:
            print('====', t, '====')
            print_stats(args.trial_stats)

    print_stats(args.trial_stats)



def SSUP(args):
    get_prior = args.get_prior
    print(tp.toolNames, tp.objects)
    all_object_keys = tp.toolNames + tp.objects.keys()
    quit()
    gaussian_policies = {k:random_init_policy(*normalize_pos(get_prior(obj_dict))) for k in all_object_keys}
    policy_rewards = {k:[] for k in gaussian_policies.keys()}
    epsilon = args.eps
    epsilon_decay_rate = args.eps_decay_rate
    i = 0
    sim_i = 0
    try_action = False
    success = False
    while not success:
        # SECTION - Sample
        weights = calculate_weights(policy_rewards, gaussian_policies)
        sample_obj = choices(list(gaussian_policies.keys()), weights=weights, k=1)[0]

        valid_sample_action = False
        for _ in range(100):
            pos = gaussian_policies[sample_obj].action()
            scaled_pos = scale_pos(pos)
            if pos[0] > -1 and pos[0] < 1 and pos[1] > -1 and pos[1] < 1:
                valid_sample_action = True
                break

        if random() < epsilon or not valid_sample_action:
            sample_obj = choices(list(gaussian_policies.keys()), k=1)[0]
            scaled_pos = get_prior(obj_dict)
            pos = normalize_pos(pos)
        # !SECTION

        # SECTION - Simulate
        path_dict, success, time_to_success = tp.runNoisyPath(toolname=sample_obj,position=scaled_pos,maxtime=20.)
        reward = calculate_reward(tp, path_dict)

        try_action = True if reward > 0.6 else False
        if try_action:
            path_dict, success, time_to_success = tp.observeFullPath(toolname=sample_obj,position=scaled_pos,maxtime=20.)
            # path_dict, success, time_to_success = tp.observeFullPlacementPath(toolname=sample_obj,position=scaled_pos)
            reward = calculate_reward(tp, path_dict)
            epsilon *= epsilon_decay_rate
            i += 1
        # !SECTION

        # SECTION - Update
        gaussian_policies[sample_obj].update([pos], [reward], learning_rate=0.5)
        policy_rewards[sample_obj].append(reward)
        plot_policies(gaussian_policies, scaled_pos)
        # print(gaussian_policies[sample_obj])
        print(i, sample_obj, pos, reward, success, time_to_success, sim_i)
        sim_i += 1
        # !SECTION
        
        if reward == 1:
            print("Success!", i, sim_i)
            args.trial_stats.append(i)
            success = True
        elif i >= 30:
            print("Fail!", i, sim_i)
            args.trial_stats.append(i)
            break

        # export image of the path
        if path_dict:
            pg.display.set_mode((10,10))
            sc = drawPathSingleImageWithTools(tp, path_dict, with_tools=True)
            img = sc.convert_alpha()
            pg.image.save(img, 'image/Strategy/'+tnm+'.png')

    return args.trial_stats


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