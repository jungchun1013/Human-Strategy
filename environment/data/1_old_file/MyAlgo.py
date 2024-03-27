from pyGameWorld import PGWorld, ToolPicker, loadFromDict
from pyGameWorld.viewer import *
import json
import pygame as pg
from gaussian_policy import Gaussian2DPolicy, random_init_policy, plot_policies
from random import choice, randint, random, choices
import numpy as np

# Load level in from json file
# For levels used in experiment, check out Level_Definitions/
json_dir = "./Trials/Original/"
tnm = "Catapult"
# tnm = "Basic"

with open(json_dir+tnm+'.json','r') as f:
  btr = json.load(f)

tp = ToolPicker(btr)

# SSUP
obj_dict = {i:j for i,j in tp.getObjects().items() if j.color in [(255, 0, 0, 255), (0, 0, 255, 255)]}
def get_prior(obj_dict):
    obj = choice(list(obj_dict.keys()))
    # if obj == 'Catapult':
    # prior of Catapult
    x0 = obj_dict['Catapult'].getPolys()[2][2][0]
    y0 = obj_dict['Catapult'].getPolys()[2][2][1]
    x = randint(x0-20, x0+20) # 
    y = randint(y0+20,600)

    # elif obj == 'Ball':
    #     # x = randint(112-20, 200+20)
    #     x = randint(90-20, 90+20)
    #     y = randint(0,600)

    return (x,y)
trial_stats = []

for t in range(250):
    gaussian_policies = {'obj'+str(i+1):random_init_policy(*get_prior(obj_dict)) for i in range(3)}
    policy_rewards = {i:[] for i in gaussian_policies.keys()}
    i=0
    try_action = False
    while True:
        
        weights=[sum(policy_rewards[i])/len(policy_rewards[i]) if policy_rewards[i] else 1 for i in gaussian_policies.keys()]
        if min(weights) < 0:
            weights = [i - min(weights)+0.1 for i in weights]
        sample_obj = choices(list(gaussian_policies.keys()), weights=weights, k=1)[0]
        # sample_obj = 'obj2'
        # sample_obj = 'obj1'
        while True:
            for _ in range(100):
                pos = gaussian_policies[sample_obj].action()
                if pos[0] > -1 and pos[0] < 1 and pos[1] > -1 and pos[1] < 1:
                    break
                if _ == 99:
                    pos = get_prior(obj_dict)
                    pos = ((pos[0]-300)/300, (pos[1]-300)/300)
        if random() < 5/(i+10):
            scale_pos = get_prior(obj_dict)
            pos = ((pos[0]-300)/300, (pos[1]-300)/300)
        else:
            scale_pos = (int(pos[0]*300+300), int(pos[1]*300+300))

        path_dict, success, time_to_success = tp.runNoisyPath(toolname=sample_obj,position=scale_pos,maxtime=20.)
        reward = calculate_reward(tp, path_dict)
        try_action = True if reward > 0 else False

        if try_action:
            path_dict, success, time_to_success = tp.observePlacementPath(toolname=sample_obj,position=scale_pos,maxtime=20.)
            reward = calculate_reward(tp, path_dict)

        gaussian_policies[sample_obj].update(np.array([pos]), np.array([reward]), learning_rate=0.05)
        policy_rewards[sample_obj].append(reward)
        plot_policies(gaussian_policies)
        gaussian_policies[sample_obj].update(np.array([pos]), np.array([reward]), learning_rate=0.05)
        policy_rewards[sample_obj].append(reward)
        plot_policies(gaussian_policies)

        if reward == 1:
            print("Success!", i)
            trial_stats.append(i)
            break
        i+=1

    if t==50:
        print('5: ', len([t for t in trial_stats if t <5])/len(trial_stats))
        for i in range(2,7):
            print(str(5*i)+':', len([str(5*j) for j in trial_stats if j <5*i])/len(trial_stats))

print(trial_stats)
print('5: ', len([t for t in trial_stats if t <5])/len(trial_stats))
for i in range(2,7):
    print(str(5*i)+':', len([str(5*j) for j in trial_stats if j <5*i])/len(trial_stats))