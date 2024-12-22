# strtegy graph class

from collections import Counter, namedtuple
import networkx as nx
from scipy.stats import norm
from sklearn import mixture
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from src.utils import node_match, draw_multi_paths, draw_samples, draw_samples, draw_samples
import os
from random import choice
from copy import deepcopy
from gplearn.genetic import SymbolicRegressor
from pyGameWorld.helpers import centroidForPoly
import json

class StrategyManager():
    def __init__(self, args):
        self.mechanism_set = MechanismSet(args)
        # self.strategy_graph_set = StrategyGraphSet()
        self.strategy_graphs = {}
        self.path_data = {}
        # self.placement_graphs = []
        self.obj_count = Counter()
    def __str__(self):
        _str = '========== strategy_graphs ==========\n'
        for task_series in self.strategy_graphs:
            for strat_type in self.strategy_graphs[task_series]:
                _str += f'{task_series} - {strat_type}\n'
                graph = self.strategy_graphs[task_series][strat_type]
                _str += 'Nodes: [\'ext\', \'path\', \'GM\']\n'
                for node, data in graph.nodes(data=True):
                    _str += f' {node} '
                _str += 'Edges: [\'label\', \'mech\', \'model\']\n'
                for u, v, data in graph.edges(data=True):
                    _str += f' ({u}, {v}) '
        _str += '========= mechanism_set ===========\n'
        for obj_type in self.mechanism_set:
            _str += f'{obj_type}\n'
            for mech in self.mechanism_set.mechanisms[obj_type]:
                _str += f'  {mech}\n'
        _str += '===================================\n'

        return _str

    def __repr__(self):
        return self.__str__()

    def path2strategy(self, args, graph, start_tool='PLACED'):
        '''
        extract strategy_graph
        '''
        s_graph = deepcopy(graph)
        for node in s_graph.nodes(): # reset for update
            s_graph.nodes[node]['ext'] = []
            s_graph.nodes[node]['path'] = []
        
        # take path graph
        path = nx.shortest_path(s_graph, start_tool, 'Goal')
        # path_graph = nx.Graph(s_graph.subgraph(path))
        path_graph = nx.DiGraph(s_graph.subgraph(path))
        
        return path_graph
        
    def merge_path_info(self, args):
        # FIXME add strategy immediately after path is found
        for task_series in self.strategy_graphs:
            for strat_type in self.strategy_graphs[task_series]:
                graph = self.strategy_graphs[task_series][strat_type]
                
                for path in self.path_data[task_series][strat_type]:
                    graph = self.add_strategy(graph, path)
                for cur_node, succ_node in graph.edges():

                    src_info = {'obj': cur_node, 'type': self._check_object_type(cur_node), 'ext': graph.nodes[cur_node]['ext'], 'path': graph.nodes[cur_node]['path']}
                    tar_info = {'obj': succ_node, 'type': self._check_object_type(succ_node), 'ext': graph.nodes[succ_node]['ext'], 'path': graph.nodes[succ_node]['path']}
                    
                    mech = Mechanism(src_info, tar_info)
                    graph.edges[(cur_node, succ_node)]['mech'] = mech
                    self.mechanism_set.mechanisms.setdefault(self._check_object_type(succ_node), [])
                    if str(mech) not in [str(m) for m in self.mechanism_set.mechanisms[self._check_object_type(succ_node)]]:
                        self.mechanism_set.mechanisms[self._check_object_type(succ_node)].append(mech)

    def add_strategy(self, graph, path):
        for node in path.nodes():
            if node not in graph.nodes():
                continue
            graph.nodes[node]['ext'] += path.nodes[node]['ext']
            graph.nodes[node]['path'] += path.nodes[node]['path']
            if node == 'PLACE':
                graph.nodes[node]['tools'] += path.nodes[node]['tools']
        return graph
    
    def _check_object_type(self, obj_name):
        if 'obj' in obj_name or 'PLACED' in obj_name:
            return 'Tool'
        elif 'Ball' in obj_name:
            return 'Ball'
        elif 'Goal' in obj_name:
            return 'Goal'
        elif 'Catapult' in obj_name or 'Lever' in obj_name or 'Plate' in obj_name or 'Support' in obj_name:
            return 'Stick'
        else:
            return 'Block'

    def _extract_ext_info(self, args, col, path_dict, nodes, graph, list_info, ext_info, prev_col_idx):
        cur_node, succ_node = nodes
        succ_list, ext_col_list, node_list = list_info
        # col, succ node ext_info, prev_col_idx
        col_end_time = col[3] if col[3] else col[2]
        if succ_node != 'Goal':
            col_end_idx = min(
                int((col_end_time+0.1)*10), int((col[2]+0.1)*10)+20
            ) # collision after 0.5s
        else:
            col_end_idx = int((col_end_time+0.1)*10)
        col_end_idx = min(
            col_end_idx, len(path_dict[cur_node])-1
        )
        time_pass = col_end_idx/10 - 0.1
        # ext_info['prev']['src'] = path_dict[cur_node][prev_col_idx]
        ext_info['curr']['src'] = path_dict[cur_node][col_end_idx]
        if succ_node != 'Goal':
            ext_info['prev']['tgt'] = path_dict[succ_node][prev_col_idx]
            ext_info['curr']['tgt'] = path_dict[succ_node][col_end_idx]
            ext_info['init']['tgt'] = path_dict[succ_node][0]
            succ_list.append(succ_node)
        else:
            # use ext_info['curr']['src'] as goal position
            ext_info['prev']['tgt'] = path_dict[cur_node][col_end_idx][0:2] + [0,0,0]
            ext_info['curr']['tgt'] = path_dict[cur_node][col_end_idx][0:2] + [0,0,0]
            ext_info['init']['tgt'] = centroidForPoly(args.btr0['world']['objects']['Goal']['points'])
        graph.add_node(
            succ_node,
            ext=[[ext_info['prev']['src'], ext_info['prev']['tgt'], ext_info['curr']['src'], ext_info['curr']['tgt'], ext_info['init']['src'], ext_info['init']['tgt']]],
            path=[path_dict]
        )
        node_list.append(succ_node)
        # NOTE - set name as cur_nd PLACE and mark tool in attr
        # NOTE - edge record mechanism

        graph.add_edge(cur_node, succ_node, label=ext_col_list)
        ext_col_list = []
        ext_info['prev']['src'] = ext_info['curr']['tgt']
        ext_info['init']['src'] = ext_info['init']['tgt']
        prev_col_idx = col_end_idx

        list_info = succ_list, ext_col_list, node_list
        return list_info, ext_info, prev_col_idx, time_pass

    def _is_valid_collision_and_not_has_path(self, args, cur_node, succ_node, graph):
        is_goal_collision = (
            succ_node == 'Goal' and cur_node != 'PLACED'
            and (args.tp.objects[cur_node].color == (255, 0, 0, 255))
        )
        is_valid_collision = succ_node != 'Goal' or is_goal_collision
        has_path = (
            succ_node in graph and cur_node in graph
            and nx.has_path(graph, succ_node, cur_node)
        )   
        return is_valid_collision and not has_path

    def build_path(self, args, task_series, sample_obj, init_pose, path_info):
        path_dict, collisions, success = path_info
        succ_list = [sample_obj]
        node_list = [sample_obj]
        graph = nx.DiGraph()
        self.obj_count[sample_obj] += 1
        # add visited flag to collisions
        is_col_visited = [False for _ in collisions]
        is_col_visited_CF = [False for _ in args.collisions0]
        ext_info = {
            'prev': {'src': init_pose, 'tgt': None},
            'curr': {'src': None, 'tgt': None},
            'init': {'src': init_pose, 'tgt': init_pose}
        }
        available_objects = args.movable_objects + args.tool_objects + ['Goal']
        poss = {'prev_tool':[], 'prev_target':[], 'curr_tool':[], 'curr_target':[]}
        is_goal_collision = False
        time_pass = 0
        while succ_list:
            cur_nd_name = succ_list.pop(0)
            cur_node = "PLACED" if cur_nd_name in args.tp.toolNames else cur_nd_name
            ext_col_list = []
            # NOTE - add new current node to strategy graph
            if not graph.has_node(cur_nd_name):
                ext_info['curr']['tgt'] = init_pose if cur_node == "PLACED" else path_dict[cur_node][0]
                graph.add_node(
                    cur_node,
                    ext=[[None, None,
                          None, ext_info['curr']['tgt'],
                          None, ext_info['init']['tgt']
                    ]],
                    path=[path_dict]
                )
                ext_info['prev']['src'] = ext_info['curr']['tgt']
                graph.nodes[cur_node]['tools'] = [cur_nd_name]

            prev_col_idx = 0
            # NOTE - find collision without tool but not occurred when tool is placed
            # FIXED - consider all CF collision given curr_node
            col_CF_targets = []
            CF_nodes = []
            # print('args.collisions0')
            for idx, col in enumerate(args.collisions0):
                # print(col[0], col[1], col[2], col[3])
                if col[2] > time_pass  and not is_col_visited_CF[idx]:
                    if col[0] == cur_node and col[1] in available_objects:
                        tar = col[1]
                    elif col[1] == cur_node and col[0] in available_objects:
                        tar = col[0]
                    else:
                        continue
                    is_col_visited_CF[idx] = True
                    CF_nodes.append(tar)
                    # if tar not in col_CF_target:
                    col_CF_targets.append(col)
                    break
            # NOTE - find collision related to current strategy graph
            # print('collisions')
            for idx, col in enumerate(collisions):
                # print(col[0], col[1], col[2], col[3])
                
                if col[0] == cur_node or col[1] == cur_node:
                    is_col_visited[idx] = True
                    succ_node = col[1] if col[0] == cur_node else col[0]
                else:
                    # The collision does not relate to the current node
                    continue
                # FIXME - catapult tmp setting
                # if cur_node == 'CataBall' and succ_node == 'Lever':
                #     continue
                if succ_node in available_objects:
                    if self._is_valid_collision_and_not_has_path(args, cur_node, succ_node, graph):
                        nodes = cur_node, succ_node
                        list_info = succ_list, ext_col_list, node_list
                        list_info, ext_info, prev_col_idx, time_pass = self._extract_ext_info(args, col, path_dict, nodes, graph, list_info, ext_info, prev_col_idx)
                        break 
                else:
                    ext_col_list.append(succ_node)

            # NOTE - counterfactual
            if not succ_list and col_CF_targets:
                # col_CF_target: if CF, col_CF_target != none
                for CF_node, col_CF_target in zip(CF_nodes, col_CF_targets):
                    if self._is_valid_collision_and_not_has_path(args, cur_node, CF_node, graph):
                        nodes = cur_node, CF_node
                        list_info = succ_list, ext_col_list, node_list
                    list_info, ext_info, prev_col_idx, time_pass = self._extract_ext_info(args, col_CF_target, path_dict, nodes, graph, list_info, ext_info, prev_col_idx)
                    break
        print('-', graph.edges(), path_info[2])
        # NOTE - find goal -> record path
        if 'Goal' in graph.nodes():
            start_tool = 'PLACED' if 'obj' in sample_obj else sample_obj
            path_graph = self.path2strategy(args, graph, start_tool)
            
            G_str = str(path_graph.edges())
            self.path_data.setdefault(task_series, {})
            self.path_data[task_series].setdefault(G_str, [])
            self.path_data[task_series][G_str].append(graph)
            
            self.strategy_graphs.setdefault(task_series, {})
            self.strategy_graphs[task_series].setdefault(G_str, [])
            self.strategy_graphs.setdefault(task_series, {})[G_str] = path_graph
            # self.strategy_graphs.setdefault(task_series, {})[G_str] = graph
            return graph
        else:
            return None

    def train(self, args):
        print('Training')
        for task_series in self.strategy_graphs:
            for strat_type in self.strategy_graphs[task_series]:
                graph = self.strategy_graphs[task_series][strat_type]
                print(graph.edges())
                for nd in graph.nodes():
                    data = [d[3] for d in graph.nodes[nd]['ext'] if d[3] is not None]
                    if len(data) < 2:
                        continue
                    img_name = os.path.join(args.trial_dir_name,
                        'train_GM_sample_'+task_series+nd+'.png'
                    )
                    draw_samples(args.tp0, [data], 'single', img_name)
                    gmm = mixture.GaussianMixture(
                        n_components=1, 
                        covariance_type='full'
                    )
                    gmm.fit(data)
                    graph.nodes[nd]['GM'] = gmm
                for nd_i, nd_j in graph.edges():
                    if nd_j not in args.tp0.toolNames: # the node that exclude tool object
                        data = graph.nodes[nd_j]['ext']
                        tool_data, target_data = zip(*[[d[0], d[3]+d[1]+list(d[4][0:2])+list(d[5][0:2])] for d in data if d[0] is not None])

                        img_name = os.path.join(args.trial_dir_name,
                            'train_GP_sample_'+task_series+nd_i+nd_j+'.png'
                        )
                        img_data = zip(*[d for d in data if d[0] is not None])
                        draw_samples(args.tp0, [tool_data, target_data], 'tool_target', img_name)

                        y = deepcopy(list(tool_data))
                        x = deepcopy(list(target_data))
                        if nd_i in args.tp0.toolNames:
                            feature_names = ['cur_targ_r', 'cur_targ_vx', 'cur_targ_vy']
                        else:
                            feature_names = ['cur_targ_r', 'cur_targ_vx', 'cur_targ_vy', 'prev_target_x', 'prev_target_y', 'prev_target_r', 'prev_target_vx', 'prev_target_vy']

                        draw_samples(args.tp0, list(img_data), 'compare_tool_target', os.path.join(args.trial_dir_name, 'train4_'+nd_i+nd_j+'.png'))
                        for i in range(len(tool_data)):
                            y[i][0] -= x[i][0] # prev_tool - cur_target
                            y[i][1] -= x[i][1]
                            # if any(pred in args.tp0.toolNames for pred in graph.predecessors(nd)):
                            if nd_i == 'PLACED':
                                # x[i][12:] target init pos
                                x[i] = x[i][2:5]+x[i][12:]
                            else:
                                x[i] = x[i][2:]
                        if len(x) < 2:
                            continue
                        # print('GPR', len(x), len(x[0]), end=' ')
                        kernel = DotProduct() + WhiteKernel()
                        gpr = GaussianProcessRegressor(kernel=kernel, random_state=None).fit(x, y)
                        graph.edges[(nd_i, nd_j)]['model'] = gpr
                        graph.edges[(nd_i, nd_j)]['mech'].model = gpr

class Mechanism():
    def __init__(self, source_info, target_info, model=None):
        if source_info:
            self.source_info = source_info
        else:
            self.source_info = {'obj': None, 'type': None, 'ext': None, 'path': None}
        if target_info:
            self.target_info = target_info
        else:
            self.target_info = {'obj': None, 'type': None, 'ext': None, 'path': None}
        self.model = model
    
    def __str__(self):
        # return f'{self.source_info["obj"]}({self.source_info["type"]}) <- {self.target_info["obj"]}({self.target_info["type"]}) {hex(id(self))}'
        return f'{self.source_info["obj"]}({self.source_info["type"]}) <- {self.target_info["obj"]}({self.target_info["type"]})'
    def __repr__(self):
        return self.__str__()
    
    def set_target_info(self, target_info):
        self.target_info = target_info



dummyMech = Mechanism(None, None, None)
# object type: Ball, Tool, Goal, Stick, Block

class MechanismSet():
    '''
    Each mechanism is unique (task, graph, soure, target)
    
    '''
    def __init__(self, args, strat_graph=None):
        # search by mechanism target for source (backward chaining)
        self.mechanisms = {}
        # if strat_graph:
        #     self.decompose_mechanisms(args, strat_graph)

    def __iter__(self):
        # This method makes the class iterable
        return iter(self.mechanisms)
    

    

    def _check_object_type(self, obj_name):
        if 'obj' in obj_name or 'PLACED' in obj_name:
            return 'Tool'
        elif 'Ball' in obj_name:
            return 'Ball'
        elif 'Goal' in obj_name:
            return 'Goal'
        elif 'Catapult' in obj_name or 'Lever' in obj_name or 'Plate' in obj_name or 'Support' in obj_name:
            return 'Stick'
        else:
            return 'Block'


def get_obj_type(obj_name):
    # TODO - complete related object type
    if 'Ball' in obj_name:
        u_type = 'Ball'
    elif 'PLACED' in obj_name:
        u_type = 'Tool'
    elif 'obj' in obj_name:
        u_type = 'Tool'
    elif 'Goal' in obj_name:
        u_type = 'Goal'
    else:
        u_type = 'Poly'
    return u_type

def merge_mechanisms(args, strategy_graphs):
    mechanism_list = {}
    for SG in strategy_graphs:
        for graph in SG.placement_graphs:
            for u, v in graph.edges():
                u_type = get_obj_type(u)
                v_type = get_obj_type(v)
                mechanism = {
                    'info': [[args.tnm, u, v]],
                    'tool_ext': graph.nodes[u]['ext'],
                    'target_ext': graph.nodes[v]['ext'],
                    'tool_path': graph.nodes[u]['path'],
                    'target_path': graph.nodes[v]['path'],
                }
                if u == 'Place':
                    mechanism['tools'] = graph.nodes[u]['tools']

                if (u_type, v_type) not in mechanism_list:
                    mechanism_list[(u_type, v_type)] = mechanism
                else:
                    for key in mechanism:
                        mechanism_list[(u_type, v_type)][key] += mechanism[key]
    return mechanism_list

def train_mechanism(args, mechanism_list):
    print('Training mechanism')
    for mech, mech_info in mechanism_list.items():
        u, v = mech
        data = mech_info['target_ext']
        tool_data, target_data = zip(*[[d[0], d[3]+d[1]] for d in data if d[0] is not None])
        y = deepcopy(list(tool_data))
        x = deepcopy(list(target_data))
        for i in range(len(tool_data)):
            y[i][0] -= x[i][0] # prev_tool - cur_target
            y[i][1] -= x[i][1]
            x[i] = x[i][2:5] if u == 'Tool' else x[i][2:]
        if len(x) < 2:
            continue
        # print('GPR', len(x), len(x[0]), end=' ')
        kernel = DotProduct() + WhiteKernel()
        gpr = GaussianProcessRegressor(kernel=kernel).fit(x, y)
        mechanism_list[mech]['model'] = gpr
    return mechanism_list

def build_graph_from_mechanism_seq(mechanism_list, task_series, start_tool):
    strategy_graph = StrategyGraph(task_series, start_tool)
    graph = nx.DiGraph()
    for mech_info in mechanism_list:
        mech, src_obj, tar_obj = mech_info['mech'], mech_info['src'], mech_info['tar']
        if not graph.has_node(src_obj):
            graph.add_node(
                src_obj,
                ext=mech.source_info['ext'],
                path=mech.source_info['path']
            )
        if not graph.has_node(tar_obj):
            graph.add_node(
                tar_obj,
                ext=mech.target_info['ext'],
                path=mech.target_info['path']
            )
        if not graph.has_edge(src_obj, tar_obj):
            graph.add_edge(src_obj, tar_obj, model=mech.model)
    
    strategy_graph.placement_graphs.append(graph)
    return strategy_graph
