# strtegy graph class

from collections import Counter
import networkx as nx
from scipy.stats import norm
from sklearn import mixture
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from src.utils import node_match, draw_multi_paths, draw_samples, draw_samples, draw_samples
import os
from copy import deepcopy
from gplearn.genetic import SymbolicRegressor
from pyGameWorld.helpers import centroidForPoly



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

    # def build_graph(self, args, sample_obj, init_pose, path_info):
    #     path_dict, collisions, success = path_info
    #     succ_list = [sample_obj]
    #     graph = nx.DiGraph()
    #     self.obj_count[sample_obj] += 1
    #     # add visited flag to collisions
    #     is_col_visited = [False for _ in collisions]
    #     prev_tool_ext = init_pose
    #     orig_tool_ext = init_pose
    #     orig_target_ext = init_pose
    #     prev_target_ext = None
    #     poss = {'prev_tool':[], 'prev_target':[], 'curr_tool':[], 'curr_target':[]}
    #     is_goal_end = False
    #     while succ_list:
    #         cur_nd_name = succ_list[0]
    #         succ_list.pop(0)
    #         cur_node = "PLACED" if cur_nd_name in args.tp.toolNames else cur_nd_name
    #         ext_col_list = []
    #         if not graph.has_node(cur_nd_name):
    #             curr_target_ext = init_pose if cur_node == "PLACED" else path_dict[cur_node][0]
    #             graph.add_node(
    #                 cur_node,
    #                 ext=[[None, None, None, curr_target_ext, None, orig_target_ext]],
    #                 path=[path_dict]
    #             )
    #             prev_tool_ext = curr_target_ext
    #             graph.nodes[cur_node]['tools'] = [cur_nd_name]

    #         prev_col_idx = 0

    #         for idx, col in enumerate(collisions):
    #             is_reverse = False
    #             if col[0] == cur_node:
    #                 is_col_visited[idx] = True
    #                 succ_node = col[1]
    #                 is_reverse = False
    #             elif col[1] == cur_node:
    #                 is_col_visited[idx] = True
    #                 succ_node = col[0]
    #                 is_reverse = True
    #             else:
    #                 # the collision does not related to current node
    #                 continue
    #             available_objects = args.movable_objects + args.tool_objects + ['Goal']
    #             is_goal_end = (succ_node == 'Goal'
    #                 and cur_node != 'PLACED'
    #                 and (args.tp.objects[cur_node].color == (255, 0, 0, 255)))
    #             has_path = succ_node in graph and cur_node in graph and nx.has_path(graph, succ_node, cur_node)
    #             if succ_node not in available_objects:
    #                 # collision with stable objects
    #                 ext_col_list.append(succ_node)
    #             elif succ_node in available_objects and not has_path and (succ_node != 'Goal' or is_goal_end):

    #                 col_end_time = col[3] if col[3] else col[2]
    #                 if succ_node != 'Goal':
    #                     col_end_idx = min(
    #                         int((col_end_time+0.1)*10), int((col[2]+0.1)*10)+20
    #                     ) # collision after 0.5s
    #                 else:
    #                     col_end_idx = int((col_end_time+0.1)*10)
    #                 col_end_idx = min(
    #                     col_end_idx, len(path_dict[cur_node])-1
    #                 )
    #                 # prev_tool_ext = path_dict[cur_node][prev_col_idx]
    #                 curr_tool_ext = path_dict[cur_node][col_end_idx]
    #                 if succ_node != 'Goal':
    #                     prev_target_ext = path_dict[succ_node][prev_col_idx]
    #                     curr_target_ext = path_dict[succ_node][col_end_idx]
    #                     orig_target_ext = path_dict[succ_node][0]
    #                     succ_list.append(succ_node)
    #                 else:
    #                     # use curr_tool_ext as goal position
    #                     prev_target_ext = path_dict[cur_node][col_end_idx][0:2] + [0,0,0]
    #                     curr_target_ext = path_dict[cur_node][col_end_idx][0:2] + [0,0,0]
    #                     orig_target_ext = centroidForPoly(args.btr0['world']['objects']['Goal']['points'])
    #                 graph.add_node(
    #                     succ_node,
    #                     ext=[[prev_tool_ext, prev_target_ext, curr_tool_ext, curr_target_ext, orig_tool_ext, orig_target_ext]],
    #                     path=[path_dict]
    #                 )
    #                 # NOTE - set name as cur_nd PLACE and mark tool in attr
    #                 graph.add_edge(cur_node, succ_node, label=ext_col_list)
    #                 ext_col_list = []
    #                 # only add one edge
    #                 # print(cur_nd_name, succ_node)
    #                 # for i in ['prev_tool', 'prev_target', 'curr_tool', 'curr_target']:
    #                 #     if poss[i][-1]:
    #                 #         print(i, [int(j) for j in poss[i][-1]], end=' ')
    #                 #     else:
    #                 #         print('None', end=' ')
    #                 # print()
    #                 prev_tool_ext = curr_target_ext
    #                 prev_col_idx = col_end_idx
    #                 orig_tool_ext = orig_target_ext
    #                 break
    #     print('-', graph.edges())
    #     return graph


    def build_graph(self, args, sample_obj, init_pose, path_info):
        path_dict, collisions, success = path_info
        succ_list = [sample_obj]
        node_list = [sample_obj]
        graph = nx.DiGraph()
        self.obj_count[sample_obj] += 1
        # add visited flag to collisions
        is_col_visited = [False for _ in collisions]
        prev_tool_ext = init_pose
        orig_tool_ext = init_pose
        orig_target_ext = init_pose
        prev_target_ext = None
        poss = {'prev_tool':[], 'prev_target':[], 'curr_tool':[], 'curr_target':[]}
        is_goal_end = False
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

            for idx, col in enumerate(collisions):
                is_reverse = False
                if col[0] == cur_node:
                    is_col_visited[idx] = True
                    succ_node = col[1]
                    is_reverse = False
                elif col[1] == cur_node:
                    is_col_visited[idx] = True
                    succ_node = col[0]
                    is_reverse = True
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
                    # est_gp = SymbolicRegressor(population_size=5000,
                    #        generations=20, stopping_criteria=-1,
                    #        p_crossover=0.7, p_subtree_mutation=0.1,
                    #        p_hoist_mutation=0.05, p_point_mutation=0.1,
                    #        max_samples=0.9, verbose=1,
                    #        parsimony_coefficient=0.05,
                    #        feature_names = feature_names,
                    #        random_state=0)
                    
                    # est_gps = []
                    # for y_ in zip(*y):
                    #     est_gp.fit(x, y_)
                    #     print(est_gp._program)
                    #     est_gps.append(est_gp)
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
    # merge_SG = StrategyGraph()
    # all_path = [sg.path_graphs for sg in strategy_graphs]
    # all_path = [path for paths in all_path for path in paths]
    # for graph in all_path:
    #     start_nd = [nd for nd in graph.nodes() if not list(graph.predecessors(nd))][0]
    #     merge_SG.set_placement_graph(args, graph, start_nd)
    # merge_SG.merge_graph(args)
    # merge_SG.train(args)
    # return merge_SG
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