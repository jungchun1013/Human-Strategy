import networkx as nx
from collections import Counter
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import pygraphviz
import os
import math
from utils import *

class StrategyGraph():
    '''
        A graph of strategies
        node: Strategy
        id: object + id
        obj_name: object name
    '''
    def __init__(self):
        self.graph = nx.DiGraph()
        self.obj_count = Counter()

    def add_strategy(self, obj_name, tp_obj, pos=None, vel=None, path=None, collision=None):
        # add strategy if no identical strategy exists
        # add extrinsics to existing strategy if identical strategy exists
        is_exist = False
        for s in self.graph.nodes(): # NOTE - check if identical strategy exists
            # if s.is_identical_strategy(obj_name, pos, vel, None) and collision == s.collision:
            if s.is_identical_strategy(obj_name, pos, vel, None) or collision == s.collision:
                s.add_extrinsics(pos, vel, path)
                node = s
                is_exist = True
                print('add extrinsic to', self.graph.nodes[s]['label'])
                break
        if not is_exist:
            strategy = Strategy(obj_name, tp_obj, pos, vel, path, collision)
            obj_count = self.obj_count.get(obj_name, 0)
            strat_id = obj_name + str(obj_count)
            is_reach_goal = True if collision and ([obj_name, 'Goal'] in collision or ['Goal', obj_name] in collision) else False
            self.graph.add_node(strategy, obj_name=obj_name, label=strat_id, color='blue' if is_reach_goal else 'black')
            node = strategy
            self.obj_count[obj_name] += 1
        
        return node

    def add_strategy1(self, strategy):
        obj_count = self.obj_count.get(strategy.obj_name, 0)
        strat_id = strategy.obj_name + str(obj_count)
        self.graph.add_node(strategy, obj_name=strategy.obj_name, label=strat_id)
        self.obj_count[strategy.obj_name] += 1
        
        return strategy

    def transitive_reduction(self):
        TR = nx.transitive_reduction(self.graph)
        TR.add_nodes_from(self.graph.nodes(data=True))
        TR.add_edges_from((u, v, self.graph.edges[u, v]) for u, v in TR.edges)
        self.graph = TR

    def get_strategies(self, obj_name):
        return list([nd for nd in self.graph.nodes() if nd.obj_name == obj_name])
    
    def merge_strategies(self, start_strat):
        succ = list(self.graph.successors(start_strat))
        succ_obj_count = Counter([s.obj_name for s in succ])
        for obj_name, count in succ_obj_count.items():
            if count > 1:
                new_strategy = Strategy(obj_name, None, None, None, None, None)
                removed_list = []
                for s in succ:
                    if s.obj_name == obj_name:
                        if not new_strategy.obj: new_strategy.obj = s.obj
                        strat = s
                        new_strategy.collision = strat.collision
                        for ext in strat.get_extrinsics():
                            new_strategy.add_extrinsics(ext[0], ext[1], ext[2])
                        # self.graph.remove_node(s)
                        removed_list.append(s)
                new_strat = self.add_strategy1(new_strategy)
                self.graph.add_edge(start_strat, new_strat)
                for r in removed_list:
                    for p in self.graph.predecessors(r):
                        self.graph.add_edge(p, new_strat)
                        print('r---  ', self.graph.nodes[p]['label'], self.graph.nodes[new_strat]['label'])
                    print('rem', self.graph.nodes[r]['label'])
                    self.graph.remove_node(r)
                print('merge to', self.graph.nodes[new_strat]['label'])
                
                
                # merge strategies
    # check if there is the path of existing strategy which has the subpath,   new_strat -> old_strat   
    # FIXME - consistency   
    def check_strategy_successors(self, strat, sample_obj, path_dict, path_dict0, collision_pattern):
        for moved_obj_name in path_dict: # get moved objects
            if not is_same_paths(path_dict[moved_obj_name], path_dict0[moved_obj_name]) and moved_obj_name != sample_obj:
                strategies = self.get_strategies(moved_obj_name) # get all strategies with object s
                for strat_succ in strategies:
                    # succ
                    # if self.is_strategy_in_path(moved_obj_name, path_dict[moved_obj_name], strat, collision_pattern):
                    for p in path_dict[moved_obj_name][2:-2]:
                        if strat_succ.is_identical_strategy(moved_obj_name, p[0:2], p[3:], p[2]) or is_succ_col(collision_pattern, strat_succ.collision):
                            print('===', self.graph.nodes()[strat]['label'], self.graph.nodes()[strat_succ]['label'])
                            self.graph.add_edge(strat, strat_succ)
                            break
        return False
    # given a existing strategy, check if the path is a subpath of the existing strategy,  old_strat -> new_strat    
    def check_strategy_predecessors(self, strat, obj_name, path, collision_pattern):
        for strat_pred in self.graph.nodes():
            if strat_pred.obj_name != obj_name and strat not in self.graph.successors(strat_pred):
                path_set = strat_pred.get_paths()
                is_in_strat = False
                for path0 in path_set:
                    for p in path0[obj_name][2:-2]:
                        if strat.is_identical_strategy(obj_name, p[0:2], p[3:], p[2]):
                            # for succ in self.graph.successors(strat0):
                            #         self.graph.add_edge(strat, succ)
                            #         print('---', self.graph.nodes()[strat0]['label'], self.graph.nodes()[strat]['label'])
                            self.graph.add_edge(strat_pred, strat)
                            print('///', self.graph.nodes()[strat_pred]['label'], self.graph.nodes()[strat]['label'])
                            is_in_strat = True
                            break
                    if is_in_strat:
                        break
                            # return True
        return False

    def save_graph(self, img_name):
        # Draw the graph
        subgraph = self.graph.subgraph([n for n in self.graph.nodes if self.graph.out_degree(n) > 0 or self.graph.in_degree(n) > 0])
        A = nx.nx_agraph.to_agraph(subgraph)
        A.layout('dot')
        A.draw(img_name)
        # pos=graphviz_layout(self.graph, prog='dot')
        # nx.draw(self.graph, with_labels=True, node_color='lightblue', pos=pos)

        # # Save the graph as an image
        # plt.show()
        # plt.savefig('image/Strategy/graph.png')
        # plt.close()

class Strategy():
    def __init__(self, name, tp_obj, pos=None, vel=None, path=None, collision=None, is_reach_goal=False):
        '''
            A strategy is a set of extrinsics of the object that can be used to achieve a goal
            TODO - extrinsic distribution
        '''
        self.obj_name = name
        self.obj = tp_obj
        self.is_reach_goal = is_reach_goal
        self.collision = collision
        if pos: # TODO - create by add_extrinsics
            self.extrinsics = [[pos, vel, path]]
        else:
            self.extrinsics = []

    def add_extrinsics(self, pos, vel, path):
        self.extrinsics.append([pos, vel, path])

    def get_extrinsics(self):
        return self.extrinsics
    def get_paths(self):
        return [ext[2] for ext in self.extrinsics]
    
    def is_identical_strategy(self, obs_name, pos, vel, rot):
        if self.obj_name != obs_name:
            return False
        is_identical_ext = []
        for extrinsic in self.extrinsics:
            self_pos, self_vel, _ = extrinsic
            dist = ((self_pos[0] - pos[0]) ** 2 + (self_pos[1] - pos[1]) ** 2) ** 0.5
            is_identical_pos = dist <= 50
            vel_dist = ((self_vel[0] - vel[0]) ** 2 + (self_vel[1] - vel[1]) ** 2) ** 0.5
            is_identical_vel = vel_dist <= 50
            rad_diff = abs(math.atan2(self_vel[1], self_vel[0])%(2*math.pi) - math.atan2(vel[1], vel[0])%(2*math.pi))
            is_identical_radian = rad_diff < 0.1
            is_identical_ext.append(is_identical_vel and is_identical_pos and is_identical_radian)
        # TODO - rotation
        return any(is_identical_ext)
# is_sublist(larger_list, sub_list)   
def is_succ_col(col, subcol): # self is full path, check if col is a subpath by collision
    if col:
        return is_sublist(col, subcol)

# TODO - fix
def is_same_paths(path, path1):
    
    path_len = min(len(path), len(path1))
    for i in range(path_len):
        pos, rot, vel = path[i][0:2], path[i][2], path[i][3:]
        pos1, rot1, vel1 = path1[i][0:2], path1[i][2], path1[i][3:]
        distance = ((pos1[0] - pos[0]) ** 2 + (pos1[1] - pos[1]) ** 2) ** 0.5
        is_identical_pos = distance <= 5
        vel_dist = ((vel1[0] - vel[0]) ** 2 + (vel1[1] - vel[1]) ** 2) ** 0.5
        is_identical_vel = vel_dist <= 5
        if not (is_identical_pos and is_identical_vel):
            return False
    
    return True