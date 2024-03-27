'''
    build strtegy graph
    mechanism: specific extrinsics
'''
import networkx as nx
from collections import Counter
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import pygraphviz
import os
import math
from utils import *
import itertools


class MechanismGraph():
    def __init__(self):
        self.graph = nx.DiGraph()
        self.obj_count = Counter()

    def _generate_id(self, obj_name):
        return obj_name +'-'+ str(self.obj_count.get(obj_name, 0))

    def add_mechanism_by_ext(self, tp, obj_name, ext_info: dict):
        '''
            ext_info: {pos, vel, path, collision}
        '''
        is_exist = False
        for nd in self.graph.nodes(): 
            mechanism = self.get_mechanism(nd)
            # if mech.check_ext_match(obj_name, ext_info) or collision == mech.collision:
            if mechanism.check_ext_match(obj_name, ext_info):
                mechanism.add_extrinsics(ext_info)
                print('add extrinsic to', nd)
                is_exist = True
                # break
        if not is_exist:
            mech_id = self._generate_id(obj_name)
            ext_info['id'] = mech_id
            mechanism = Mechanism(tp, obj_name, mech_id, ext_info)
            is_reach_goal = mechanism.is_reach_goal
            self.graph.add_node(mech_id, mechanism=mechanism, obj_name=obj_name, label=mech_id, color='blue' if is_reach_goal else 'black')
            self.obj_count[obj_name] += 1
            print('create', mech_id)

        return mechanism.id

    def add_mechanism(self, tp, mechanism):
        mech_id = self._generate_id(mechanism.obj_name)
        mechanism.id = mech_id
        # mechanism = Mechanism(tp, mechanism.obj_name, mech_id, ext_info)
        self.graph.add_node(mech_id, mechanism=mechanism, obj_name=mechanism.obj_name, label=mech_id, color='blue' if mechanism.is_reach_goal else 'black')
        self.obj_count[mechanism.obj_name] += 1
        return mech_id

    def transitive_reduction(self):
        TR = nx.transitive_reduction(self.graph)
        TR.add_nodes_from(self.graph.nodes(data=True))
        TR.add_edges_from((u, v, self.graph.edges[u, v]) for u, v in TR.edges)
        self.graph = TR

    def get_mechanisms(self, obj_name):
        return list([nd for nd in self.graph.nodes() if nd.obj_name == obj_name])

    def get_mechanism(self, mech_id):
        return self.graph.nodes[mech_id]['mechanism']

    def check_mechanism_successors(self, mech, path_dict0):
        for nd_succ in [nd for nd in self.graph.nodes() if self.get_mechanism(nd).obj]: # NOTE - no obj -> tools
            mech_succ = self.get_mechanism(nd_succ)
            if mech_succ.obj_name != mech.obj_name:
                is_ext_match = False
                for succ_ext_info, ext_info in itertools.product(mech_succ.extrinsics_list, mech.extrinsics_list):
                    path = ext_info['path'][mech_succ.obj_name]
                    succ_path = succ_ext_info['path'][mech_succ.obj_name] # chekck if this node inculde the input mech
                    if not is_same_paths(path_dict0[mech_succ.obj_name], succ_path):
                        # ext = (ext_info['pos'], ext_info['rot'], ext_info['vel'])
                        print('check_ext_match succ', mech.id, '->', nd_succ)
                        for p in path:
                            ext_info_p = {'pos': p[0:2], 'rot': p[2], 'vel': p[3:]}
                            if mech_succ.check_ext_match(mech_succ.obj_name, ext_info_p):
                                print('>>>', mech.id, nd_succ)
                                self.graph.add_edge(mech.id, nd_succ)
                                is_ext_match = True
                                break
                        if is_ext_match:
                            break
    
    def check_mechanism_predecessors(self, mech, path_dict0):
        for nd_pred in self.graph.nodes():
            mech_pred = self.get_mechanism(nd_pred)
            if mech_pred.obj_name != mech.obj_name:
                is_ext_match = False
                for pred_ext_info, ext_info in itertools.product(mech_pred.extrinsics_list, mech.extrinsics_list):
                    path = ext_info['path'][mech.obj_name]
                    pred_path = pred_ext_info['path'][mech.obj_name] # chekck if this node inculde the input mech
                    if not is_same_paths(path_dict0[mech.obj_name], pred_path):
                        # ext = (ext_info['pos'], ext_info['rot'], ext_info['vel'])
                        print('check_ext_match pred', nd_pred, '->', mech.id)
                        for p in path:
                            ext_info_p = {'pos': p[0:2], 'rot': p[2], 'vel': p[3:]}
                            if mech.check_ext_match(mech.obj_name, ext_info_p):
                                print('<<<', nd_pred, mech.id)
                                self.graph.add_edge(nd_pred, mech.id)
                                is_ext_match = True
                                break
                        if is_ext_match:
                            break


    # FIXME
    def merge_mechanisms(self, tp, start_mech):
        nodes = list(self.graph.nodes())
        for start_mech in nodes:
            if start_mech in self.graph.nodes():
                self.merge(tp, start_mech)

    def merge(self, tp, start_mech):
        succ = list(self.graph.successors(start_mech))
        succ_obj_count = Counter([self.get_mechanism(s).obj_name for s in succ])
        for obj_name, count in succ_obj_count.items():
            if count > 1:
                new_mechanism = Mechanism(tp, obj_name, None, None)
                removed_list = []
                for mech_id in succ:
                    mech = self.get_mechanism(mech_id)
                    if mech.obj_name == obj_name:
                        for ext in mech.get_extrinsics():
                            new_mechanism.add_extrinsics(ext)
                        # self.graph.remove_node(s)
                        removed_list.append(mech_id)
                new_mech_id = self.add_mechanism(tp, new_mechanism)
                self.graph.add_edge(start_mech, new_mech_id)
                for r in removed_list:
                    for p in self.graph.predecessors(r):
                        self.graph.add_edge(p, new_mech_id)
                        print('p---  ', p, new_mech_id)
                    for p in self.graph.successors(r):
                        self.graph.add_edge(new_mech_id, p)
                        print('s---  ', new_mech_id, p)
                    print('rem', p)
                    self.graph.remove_node(r)
                print('merge to', new_mech_id)

    def merge1(self, tp, start_mech):
        succ_succ_count = Counter()
        succ_succ_dict = {}
        succ_obj_count = Counter([self.get_mechanism(s).obj_name for s in succ])
        two_step_succ = []
        for s in succ:
            succ_succ = list(self.graph.successors(s))
            for ss in succ_succ:
                two_step_succ.append([s, ss])

        succ_succ_count = Counter([self.get_mechanism(s).obj_name for tss in two_step_succ])


        
        for obj_name, count in succ_obj_count.items():
            if count > 1:
                new_mechanism = Mechanism(tp, obj_name, None, None)
                removed_list = []
                for mech_id in succ:
                    mech = self.get_mechanism(mech_id)
                    if mech.obj_name == obj_name:
                        for ext in mech.get_extrinsics():
                            new_mechanism.add_extrinsics(ext)
                        # self.graph.remove_node(s)
                        removed_list.append(mech_id)
                new_mech_id = self.add_mechanism(tp, new_mechanism)
                self.graph.add_edge(start_mech, new_mech_id)
                for r in removed_list:
                    for p in self.graph.predecessors(r):
                        self.graph.add_edge(p, new_mech_id)
                        print('p---  ', p, new_mech_id)
                    for p in self.graph.successors(r):
                        self.graph.add_edge(new_mech_id, p)
                        print('s---  ', new_mech_id, p)
                    print('rem', p)
                    self.graph.remove_node(r)
                print('merge to', new_mech_id)


    def transitive_reduction(self):
        TR = nx.transitive_reduction(self.graph)
        TR.add_nodes_from(self.graph.nodes(data=True))
        TR.add_edges_from((u, v, self.graph.edges[u, v]) for u, v in TR.edges)
        self.graph = TR
    def save_graph(self, img_name):
        # Draw the graph
        subgraph = self.graph.subgraph([n for n in self.graph.nodes if self.graph.out_degree(n) > 0 or self.graph.in_degree(n) > 0])
        A = nx.nx_agraph.to_agraph(subgraph)
        A.layout('dot')
        A.draw(img_name)



class Mechanism():
    # def __init__(self, tp, name, pos=None, vel=None, path=None, collision=None, is_reach_goal=False):
    def __init__(self, tp, obj_name, mech_id, ext_info: dict):
        self.obj_name = obj_name
        if obj_name in tp.objects:
            self.obj = tp.objects[obj_name]
        else:
            self.obj = None
        self.id = mech_id
        # self.pos = ext_info['pos']
        # self.vel = ext_info['vel']
        # self.path = ext_info['path']
        # self.collision = ext_info['collision']
        self.extrinsics_list = []
        if ext_info:
            self.is_reach_goal = True if ext_info['collision'] and ([obj_name, 'Goal'] in ext_info['collision'] or ['Goal', obj_name] in ext_info['collision']) else False
        else:
            self.is_reach_goal = False
        if ext_info:
            self.add_extrinsics(ext_info)

    def check_ext_match(self, obj_name, ext_info):
        if self.obj_name != obj_name:
            return False
        is_match_ext = []
        for ext_info0 in self.extrinsics_list:
            dist = ((ext_info0['pos'][0] - ext_info['pos'][0]) ** 2 \
                + (ext_info0['pos'][1] - ext_info['pos'][1]) ** 2) ** 0.5
            vel_dist = ((ext_info0['vel'][0] - ext_info['vel'][0]) ** 2 \
                + (ext_info0['vel'][1] - ext_info['vel'][1]) ** 2) ** 0.5
            rad_diff = \
                abs(math.atan2(ext_info0['vel'][1], ext_info0['vel'][0])%(2*math.pi) \
                    - math.atan2(ext_info['vel'][1], ext_info['vel'][0])%(2*math.pi))
            
            is_same_pos = dist <= 50
            is_same_vel = vel_dist <= 50
            is_same_rad = rad_diff < 0.1
            is_match_ext.append(is_same_pos and is_same_vel and is_same_rad)

        return any(is_match_ext)

    def get_extrinsics(self):
        return self.extrinsics_list
        
    def add_extrinsics(self, ext_info: dict):
        self.extrinsics_list.append(ext_info)

    def get_paths(self):
        # return all extrinsic path
        return [ext['path'] for ext in self.extrinsics_list]


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