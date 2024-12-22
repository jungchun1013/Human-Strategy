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


class Mechanism():
    def __init__(self, source_info, target_info, model):
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
        return f'{self.source_info["type"]} -> {self.target_info["type"]}'
    def __repr__(self):
        return f'{self.source_info["obj"]} -> {self.target_info["obj"]}'
    def set_target_info(self, target_info):
        self.target_info = target_info

dummyMech = Mechanism(None, None, None)
# object type: Ball, Tool, Goal, Stick, Block

class MechanismSet():
    def __init__(self, args, strat_graph=None):
        # search by mechanism target for source (backward chaining)
        self.mechanisms = {}
        if strat_graph:
            self.decompose_mechanisms(args, strat_graph)
    def decompose_mechanisms(self, args, strat_graph_set):
        for tnm, strategy_graph in strat_graph_set.strategy_graphs.items():
            for SG in strategy_graph.placement_graphs:
                for edge, edge_info in SG.edges().items():
                    src, tar = edge
                    self.mechanisms.setdefault(self._check_object_type(tar), [])
                    src_info = {'obj': src, 'type': self._check_object_type(src), 'ext': SG.nodes[src]['ext'], 'path': SG.nodes[src]['path']}
                    tar_info = {'obj': tar, 'type': self._check_object_type(tar), 'ext': SG.nodes[tar]['ext'], 'path': SG.nodes[tar]['path']}
                    if 'model' in edge_info:
                        self.mechanisms[self._check_object_type(tar)].append(
                        Mechanism(src_info, tar_info, edge_info['model']))
        # for mech in self.mechanisms:
        #     for m in self.mechanisms[mech]:
        #         print(m.source_info['obj'],m.target_info['obj'], m.model)
            
    
    def sample_strategy(self, args):
        # sample a strategy from mechanism
        available_obj = args.movable_objects
        curr_obj = 'Goal'
        used_obj = []
        # curr_mech = choice(self.mechanisms['Goal'])
        # curr_mech = {'mech': curr_mech, 'src':curr_mech.source, 'tar':'Goal'}
        strategy = []
        # dummy for while condition
        src_info = {'obj': 'Goal', 'type': 'Goal', 'ext': None, 'path': None}
        # curr_mech_pair = {'mech': Mechanism(src_info, None, None), 'src':'PLACED', 'tar':curr_obj}
        curr_mech_pair = {'mech': Mechanism(src_info, None, None), 'src':curr_obj, 'tar': None}
        while curr_mech_pair['mech'].source_info['type'] != 'Tool':
            # get
            curr_obj_type = self._check_object_type(curr_obj)
            used_obj.append(curr_obj)
            available_mech = self.mechanisms[curr_obj_type]
            mech_choices = []
            for prev_mech in available_mech:
                for obj in available_obj:
                    if self._check_object_type(obj) == prev_mech.source_info['type'] and obj not in used_obj:  
                        mech_choices.append({'mech': prev_mech, 'src':obj, 'tar':curr_obj})
                if prev_mech.source_info['type'] == 'Tool':
                    mech_choices.append({'mech': prev_mech, 'src':'PLACED', 'tar':curr_obj})
            # TODO - assert mech choice is not empty
            curr_mech_pair = choice(mech_choices)
            curr_obj = curr_mech_pair['src']
            strategy.append(curr_mech_pair)
            if curr_mech_pair['mech'].source_info['type'] == 'Tool':
                break
        return strategy

    

    def _check_object_type(self, obj_name):
        if 'obj' in obj_name or 'PLACED' in obj_name:
            return 'Tool'
        elif 'Ball' in obj_name:
            return 'Ball'
        elif 'Goal' in obj_name:
            return 'Goal'
        elif 'Catapult' in obj_name or 'Lever' in obj_name:
            return 'Stick'
        else:
            return 'Block'


class StrategyGraphSet():
    def __init__(self):
        self.strategy_graphs = {}
        self.new_demos = []

    def build_data_for_catapultAlt(self, args, sample_obj, sample_pos, path_info):
        print('build_data_for_catapultAlt')
        temp_SG = StrategyGraph('')
        # FIXME - 84 cur_node = "PLACED" if cur_nd_name in args.tp.toolNames else cur_nd_name
        args.tp = args.tp0
        graph = temp_SG.build_graph(args, sample_obj, sample_pos, path_info)
        if 'Goal' in graph.nodes():
            self.new_demos.append(graph)
        data = {}
        data['Catapult'] = {}
        data['Funnel'] = {}
        data['Launch_v2'] = {}
        data['Catapult']['Ball'] = []
        data['Catapult']['Catapult'] = []
        data['Funnel']['Ball'] = []
        data['Launch_v2']['Goal'] = []
        for SG in self.strategy_graphs:
            for g in self.new_demos:
                if SG == 'Catapult' and 'Lever' in g.nodes():
                    data['Catapult']['Ball'].extend(g.nodes['CataBall']['ext'])
                    data['Catapult']['Catapult'].extend(g.nodes['Lever']['ext'])
                elif SG == 'Funnel' and 'KeyBall' in g.nodes():
                    data['Funnel']['Ball'].extend(g.nodes['KeyBall']['ext'])
                elif SG == 'Launch_v2' and 'Goal' in g.nodes():
                    data['Launch_v2']['Goal'].extend(g.nodes['Goal']['ext'])
                    # data['Launch_v2']['Ball'].extend(graph.nodes['Goal']['ext'])
        for SG in self.strategy_graphs:
            self.strategy_graphs[SG].train_with_new_demos(args, data)


    # def build_data_for_catapultAlt(self, args, sample_obj, sample_pos, path_info):
    #     print('build_data_for_catapultAlt')
    #     temp_SG = StrategyGraph('')
    #     # FIXME - 84 cur_node = "PLACED" if cur_nd_name in args.tp.toolNames else cur_nd_name
    #     args.tp = args.tp0
    #     graph = temp_SG.build_graph(args, sample_obj, sample_pos, path_info)
    #     if 'Goal' in graph.nodes():
    #         self.new_demos.append(graph)
    #     data = {}
    #     data['Catapult'] = {}
    #     data['Funnel'] = {}
    #     data['Launch_v2'] = {}
    #     data['Catapult']['Ball'] = []
    #     data['Catapult']['Catapult'] = []
    #     data['Funnel']['Ball'] = []
    #     data['Launch_v2']['Goal'] = []
    #     for SG in self.mechanisms:
    #         for g in self.new_demos:
    #             if SG == 'Catapult' and 'Lever' in g.nodes():
    #                 data['Catapult']['Ball'].extend(g.nodes['CataBall']['ext'])
    #                 data['Catapult']['Catapult'].extend(g.nodes['Lever']['ext'])
    #             elif SG == 'Funnel' and 'KeyBall' in g.nodes():
    #                 data['Funnel']['Ball'].extend(g.nodes['KeyBall']['ext'])
    #             elif SG == 'Launch_v2' and 'Goal' in g.nodes():
    #                 data['Launch_v2']['Goal'].extend(g.nodes['Goal']['ext'])
    #                 # data['Launch_v2']['Ball'].extend(graph.nodes['Goal']['ext'])
    #     for SG in self.mechanisms:
    #         self.mechanisms[SG].train_with_new_demos(args, data)


class StrategyGraph():
    '''
    Graph structure for strategy
    ext: list of extrinsic data (tool, target)
    '''
    def __init__(self, task_series, start_tool='PLACED'):
        self.path_graphs = []
        self.task_series = task_series
        self.placement_graphs = []
        self.full_placement_idx = []
        self.strategy = []
        self.obj_count = Counter()
        self.main_graph = nx.DiGraph()
        self.fpg_gmm_list = []
        self.start_tool = start_tool

    def build_graph(self, args, sample_obj, init_pose, path_info):
        path_dict, collisions, success = path_info
        succ_list = [sample_obj]
        node_list = [sample_obj]
        graph = nx.DiGraph()
        self.obj_count[sample_obj] += 1
        # add visited flag to collisions
        is_col_visited = [False for _ in collisions]
        is_col_visited0 = [False for _ in args.collisions0]
        prev_tool_ext = init_pose
        orig_tool_ext = init_pose
        orig_target_ext = init_pose
        prev_target_ext = None
        poss = {'prev_tool':[], 'prev_target':[], 'curr_tool':[], 'curr_target':[]}
        is_goal_end = False
        time_pass = []
        while succ_list:
            cur_nd_name = succ_list[0]
            succ_list.pop(0)
            cur_node = "PLACED" if cur_nd_name in args.tp.toolNames else cur_nd_name
            ext_col_list = []
            if not graph.has_node(cur_nd_name):
                curr_target_ext = init_pose if cur_node == "PLACED" else path_dict[cur_node][0]
                graph.add_node(
                    cur_node,
                    ext=[[None, None, None, curr_target_ext, None, orig_target_ext]],
                    path=[path_dict]
                )
                prev_tool_ext = curr_target_ext
                graph.nodes[cur_node]['tools'] = [cur_nd_name]

            prev_col_idx = 0
            # find collision without tool but not occurred when tool is placed
            col_target_without_tool = []
            CF_node = None
            for idx, col in enumerate(args.collisions0):
                if col[0] == cur_node and col[2] > time_pass and not is_col_visited0[idx] and col[1] in available_objects:
                    is_col_visited0[idx] = True
                    CF_node = col[1]
                    if col[1] not in col_target_without_tool:
                        col_target_without_tool = col
                    break
                elif col[1] == cur_node and col[2] > time_pass and not is_col_visited0[idx] and col[0] in available_objects:
                    is_col_visited0[idx] = True
                    CF_node = col[0]
                    if col[0] not in col_target_without_tool:
                        col_target_without_tool = col
                    break
                else:
                    # the collision does not related to current node
                    continue
            # print(cur_node, time_pass, 'col_target_without_tool', col_target_without_tool[0:4])
            for idx, col in enumerate(collisions):
                if col[0] == cur_node:
                    is_col_visited[idx] = True
                    succ_node = col[1]

                elif col[1] == cur_node:
                    is_col_visited[idx] = True
                    succ_node = col[0]
                else:
                    # the collision does not related to current node
                    continue
                # FIXME - catapult tmp setting
                if cur_node == 'CataBall' and succ_node == 'Lever':
                    continue
                available_objects = args.movable_objects + args.tool_objects + ['Goal']
                is_goal_end = (succ_node == 'Goal'
                    and cur_node != 'PLACED'
                    and (args.tp.objects[cur_node].color == (255, 0, 0, 255)))
                has_path = succ_node in graph and cur_node in graph and nx.has_path(graph, succ_node, cur_node)
                if succ_node not in available_objects:
                    # collision with stable objects
                    ext_col_list.append(succ_node)
                elif succ_node in available_objects and not has_path and (succ_node != 'Goal' or is_goal_end):

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
                    time_pass = col_end_idx/10 -0.1
                    # prev_tool_ext = path_dict[cur_node][prev_col_idx]
                    curr_tool_ext = path_dict[cur_node][col_end_idx]
                    if succ_node != 'Goal':
                        prev_target_ext = path_dict[succ_node][prev_col_idx]
                        curr_target_ext = path_dict[succ_node][col_end_idx]
                        orig_target_ext = path_dict[succ_node][0]
                        succ_list.append(succ_node)
                    else:
                        # use curr_tool_ext as goal position
                        prev_target_ext = path_dict[cur_node][col_end_idx][0:2] + [0,0,0]
                        curr_target_ext = path_dict[cur_node][col_end_idx][0:2] + [0,0,0]
                        orig_target_ext = centroidForPoly(args.btr0['world']['objects']['Goal']['points'])
                    graph.add_node(
                        succ_node,
                        ext=[[prev_tool_ext, prev_target_ext, curr_tool_ext, curr_target_ext, orig_tool_ext, orig_target_ext]],
                        path=[path_dict]
                    )
                    node_list.append(succ_node)
                    # NOTE - set name as cur_nd PLACE and mark tool in attr
                    graph.add_edge(cur_node, succ_node, label=ext_col_list)
                    ext_col_list = []
                    # only add one edge
                    # print(cur_nd_name, succ_node)
                    # for i in ['prev_tool', 'prev_target', 'curr_tool', 'curr_target']:
                    #     if poss[i][-1]:
                    #         print(i, [int(j) for j in poss[i][-1]], end=' ')
                    #     else:
                    #         print('None', end=' ')
                    # print()
                    prev_tool_ext = curr_target_ext
                    prev_col_idx = col_end_idx
                    orig_tool_ext = orig_target_ext
                    break
            # NOTE
            has_path = CF_node in graph and cur_node in graph and nx.has_path(graph, CF_node, cur_node)
            if not succ_list and col_target_without_tool and not has_path and (CF_node != 'Goal' or is_goal_end):
                col_wot = col_target_without_tool
                col_end_time = col_wot[3] if col_wot[3] else col_wot[2]
                if CF_node != 'Goal':
                    col_end_idx = min(
                        int((col_end_time+0.1)*10), int((col_wot[2]+0.1)*10)+20
                    ) # collision after 0.5s
                else:
                    col_end_idx = int((col_end_time+0.1)*10)
                col_end_idx = min(
                    col_end_idx, len(path_dict[cur_node])-1
                )
                time_pass = col_end_idx/10 -0.1
                # prev_tool_ext = path_dict[cur_node][prev_col_idx]
                curr_tool_ext = path_dict[cur_node][col_end_idx]
                if CF_node != 'Goal':
                    prev_target_ext = path_dict[CF_node][prev_col_idx]
                    curr_target_ext = path_dict[CF_node][col_end_idx]
                    orig_target_ext = path_dict[CF_node][0]
                    succ_list.append(CF_node)
                else:
                    # use curr_tool_ext as goal position
                    prev_target_ext = path_dict[cur_node][col_end_idx][0:2] + [0,0,0]
                    curr_target_ext = path_dict[cur_node][col_end_idx][0:2] + [0,0,0]
                    orig_target_ext = centroidForPoly(args.btr0['world']['objects']['Goal']['points'])
                graph.add_node(
                    CF_node,
                    ext=[[prev_tool_ext, prev_target_ext, curr_tool_ext, curr_target_ext, orig_tool_ext, orig_target_ext]],
                    path=[path_dict]
                )
                node_list.append(CF_node)
                # NOTE - set name as cur_nd PLACE and mark tool in attr
                graph.add_edge(cur_node, CF_node, label=ext_col_list)
                ext_col_list = []
                # only add one edge
                # print(cur_nd_name, succ_node)
                # for i in ['prev_tool', 'prev_target', 'curr_tool', 'curr_target']:
                #     if poss[i][-1]:
                #         print(i, [int(j) for j in poss[i][-1]], end=' ')
                #     else:
                #         print('None', end=' ')
                # print()
                prev_tool_ext = curr_target_ext
                prev_col_idx = col_end_idx
                orig_tool_ext = orig_target_ext
                break
            # print('+', graph.edges())
        print('-', graph.edges())
        return graph



    def set_placement_graph(self, args, graph, sample_obj):
        self.path_graphs.append(graph)
        start_node = [nd for nd in graph.nodes() if graph.in_degree(nd) == 0][0]
        # condiser counterfactual
        if start_node == sample_obj:
            is_isomorphic = False
            for tg in self.placement_graphs:
                if set(graph.edges()) == set(tg.edges()):
                    is_isomorphic = True
                    # for node in graph.nodes():
                    #     tg.nodes[node]['ext'] += graph.nodes[node]['ext']
                    break
            if not is_isomorphic:
                p_graph = deepcopy(graph)
                # copy for separate placement graph and path graph
                for node in p_graph.nodes(): # reset for update
                    p_graph.nodes[node]['ext'] = []
                self.placement_graphs.append(p_graph)
                self.full_placement_idx.append(0)
                # Find strategy
                node = 'Goal'
                strategy = ['Goal']
                while True:
                    succ_nodes = list(graph.successors(node))
                    if not list(succ_nodes):
                        break
                    strategy = succ_nodes[0]
                    node = succ_nodes[0]

                self.fpg_gmm_list.append([])
                
    def merge_graph(self, args):
        for idx, graph in (zip(self.full_placement_idx, self.placement_graphs)):
            c = 0
            for path in self.path_graphs[idx:]:
                if set(path.edges()).issubset(set(graph.edges())):
                    graph = self.add_strategy(graph, path)
                    c += 1
            graph_idx = self.placement_graphs.index(graph)
            self.full_placement_idx[graph_idx] = len(self.path_graphs)

    def add_strategy(self, graph, path):
        for node in path.nodes():
            graph.nodes[node]['ext'] += path.nodes[node]['ext']
            graph.nodes[node]['path'] += path.nodes[node]['path']
            if node == 'PLACE':
                graph.nodes[node]['tools'] += path.nodes[node]['tools']
        return graph
    
    def train(self, args):
        print('Training')
        for graph in self.placement_graphs:
            # Train tools (no pred) gaussian model
            print(graph.edges())
            for nd in graph.nodes():
                data = [d[3] for d in graph.nodes[nd]['ext'] if d[3] is not None]
                if len(data) < 2:
                    continue
                img_name = os.path.join(args.dir_name,
                    'train_GM_sample_'+self.task_series+nd+'.png'
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

                    img_name = os.path.join(args.dir_name,
                        'train_GP_sample_'+self.task_series+nd_i+nd_j+'.png'
                    )
                    img_data = zip(*[d for d in data if d[0] is not None])
                    draw_samples(args.tp0, [tool_data, target_data], 'tool_target', img_name)

                    y = deepcopy(list(tool_data))
                    x = deepcopy(list(target_data))
                    if nd_i in args.tp0.toolNames:
                        feature_names = ['cur_targ_r', 'cur_targ_vx', 'cur_targ_vy']
                    else:
                        feature_names = ['cur_targ_r', 'cur_targ_vx', 'cur_targ_vy', 'prev_target_x', 'prev_target_y', 'prev_target_r', 'prev_target_vx', 'prev_target_vy']

                    draw_samples(args.tp0, list(img_data), 'compare_tool_target', os.path.join(args.dir_name, 'train4_'+nd_i+nd_j+'.png'))
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
                # print()

        for i, g in enumerate(self.placement_graphs):
            self.fpg_gmm_list[i] = [nd for nd in g.nodes() if 'GM' in g.nodes[nd] and nd != 'Goal']
    
    def train_with_new_demos(self, args, new_data):
        print('train_with_new_demo')
        for graph in self.placement_graphs:
            # Train tools (no pred) gaussian model
            print(graph.edges())
            for nd in graph.nodes():
                data = [d[3] for d in graph.nodes[nd]['ext'] if d[3] is not None]
                if len(data) < 2:
                    continue
                img_name = os.path.join(args.dir_name,
                    'train_GM_sample_'+self.task_series+nd+'.png'
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
                    data = deepcopy(graph.nodes[nd_j]['ext'])
                    print(self.task_series, nd_j)
                    if self.task_series in new_data and nd_j in new_data[self.task_series]:
                        
                        data += new_data[self.task_series][nd_j]

                    tool_data, target_data = zip(*[[d[0], d[3]+d[1]+list(d[4][0:2])+list(d[5][0:2])] for d in data if d[0] is not None])
                    # NOTE - data
                    img_name = os.path.join(args.dir_name,
                        'train_GP_sample_n_'+self.task_series+nd_i+nd_j+'.png'
                    )
                    img_data = zip(*[d for d in data if d[0] is not None])
                    draw_samples(args.tp0, [tool_data, target_data], 'tool_target', img_name)

                    y = deepcopy(list(tool_data))
                    x = deepcopy(list(target_data))
                    if nd_i in args.tp0.toolNames:
                        feature_names = ['cur_targ_r', 'cur_targ_vx', 'cur_targ_vy']
                    else:
                        feature_names = ['cur_targ_r', 'cur_targ_vx', 'cur_targ_vy', 'prev_target_x', 'prev_target_y', 'prev_target_r', 'prev_target_vx', 'prev_target_vy']
                    draw_samples(args.tp0, list(img_data), 'compare_tool_target', os.path.join(args.dir_name, 'train4_'+nd_i+nd_j+'.png'))
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
                # print()

        for i, g in enumerate(self.placement_graphs):
            self.fpg_gmm_list[i] = [nd for nd in g.nodes() if 'GM' in g.nodes[nd] and nd != 'Goal']
def merge_graphs(args, strategy_graphs):
    merge_graph = StrategyGraphSet()
    for SG in strategy_graphs:
        task_series = SG.task_series
        if task_series not in merge_graph.strategy_graphs:
            merge_graph.strategy_graphs[task_series] = StrategyGraph(task_series)
        graph = merge_graph.strategy_graphs[task_series]
        for path in SG.path_graphs:
            start_nd = [nd for nd in path.nodes() if not list(path.predecessors(nd))][0]
            graph.set_placement_graph(args, path, start_nd)
    
    for SG in merge_graph.strategy_graphs:
        merge_graph.strategy_graphs[SG].merge_graph(args)
        merge_graph.strategy_graphs[SG].train(args)

    return merge_graph


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
