# strtegy graph class

from collections import Counter
import networkx as nx
from scipy.stats import norm
from sklearn import mixture
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from src.utils import node_match, draw_multi_paths, draw_samples, draw_gp_samples, draw_4_samples
import os
from copy import deepcopy





class StrategyGraph():
    '''
    Graph structure for strategy
    ext: list of extrinsic data (tool, target)
    '''
    def __init__(self):
        self.path_graphs = []
        self.placement_graphs = []
        self.full_placement_idx = []
        self.obj_count = Counter()
        self.main_graph = nx.DiGraph()
        self.fpg_gmm_list = []

    def build_graph(self, args, sample_obj, init_pose, path_info):
        path_dict, collisions, success = path_info
        succ_list = [sample_obj]
        graph = nx.DiGraph()
        self.obj_count[sample_obj] += 1
        # add visited flag to collisions
        is_col_visited = [False for _ in collisions]
        prev_tool_ext = init_pose
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
                    cur_nd_name,
                    ext=[[None, None, None, curr_target_ext]],
                    path=[path_dict]
                )
                prev_tool_ext = curr_target_ext

            # print(cur_nd_name, [int(j) for j in prev_tool_ext], '-')
            # prev_tool_ext = graph.nodes[cur_nd_name]['ext'][-1][1]
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
                available_objects = args.movable_objects + args.tool_objects + ['Goal']
                is_goal_end = (succ_node == 'Goal'
                    and cur_nd_name not in args.tp.toolNames
                    and (args.tp.objects[cur_node].color == (255, 0, 0, 255)))
                if succ_node not in available_objects:
                    # collision with stable objects
                    ext_col_list.append(succ_node)
                elif (succ_node in available_objects and (succ_node, cur_node) not in graph.edges()) or is_goal_end:

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
                        succ_list.append(succ_node)
                    else:
                        # use curr_tool_ext as goal position
                        prev_target_ext = path_dict[cur_node][col_end_idx][0:2] + [0,0,0]
                        curr_target_ext = path_dict[cur_node][col_end_idx][0:2] + [0,0,0]
                    graph.add_node(
                        succ_node,
                        ext=[[prev_tool_ext, prev_target_ext, curr_tool_ext, curr_target_ext]],
                        path=[path_dict]
                    )
                    graph.add_edge(cur_nd_name, succ_node, label=ext_col_list)
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
                    break
        return graph

    def set_placement_graph(self, args, graph, sample_obj):
        if "Goal" in graph.nodes():
            self.path_graphs.append(graph)
            if sample_obj in args.tp0.toolNames:
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
        return graph
    

    def train(self, args):
        print('Training')
        for graph in self.placement_graphs:
            # Train tools (no pred) gaussian model
            # print(graph.edges())
            for nd in graph.nodes():
                data = [d[3] for d in graph.nodes[nd]['ext'] if d[3] is not None]
                # print('==', nd, end=' ')
                if len(data) < 2:
                    print()
                    continue
                # print('GM', len(data), end=' ')
                img_name = os.path.join(args.dir_name,
                    'GM_sample_'+nd+'.png'
                )
                draw_samples(args.tp0, data, img_name)
                gmm = mixture.GaussianMixture(
                    n_components=1, 
                    covariance_type='full'
                )
                gmm.fit(data)
                graph.nodes[nd]['GM'] = gmm
                if nd not in args.tp0.toolNames: # the node that exclude tool object
                    data = graph.nodes[nd]['ext']
                    tool_data, target_data = zip(*[[d[0], d[3]+d[1]] for d in data if d[0] is not None])

                    y = deepcopy(list(tool_data))
                    x = deepcopy(list(target_data))
                    img_name = os.path.join(args.dir_name,
                        'GP_sample_'+nd+'.png'
                    )
                    data_ = zip(*[d for d in data if d[0] is not None])

                    draw_4_samples(args.tp0, list(data_), os.path.join(args.dir_name, 'sample4_'+nd+'.png'))
                    for i in range(len(tool_data)):
                        y[i][0] -= x[i][0] # prev_tool - cur_target
                        y[i][1] -= x[i][1]
                        # x[i] = x[i][2:5]
                        if any(pred in args.tp0.toolNames for pred in graph.predecessors(nd)):
                            x[i] = x[i][2:5]
                        else:
                            x[i] = x[i][2:]
                    if len(x) < 2:
                        print()
                        continue
                    # print('GPR', len(x), len(x[0]), end=' ')
                    kernel = DotProduct() + WhiteKernel()
                    gpr = GaussianProcessRegressor(kernel=kernel).fit(x, y)
                    graph.nodes[nd]['model'] = gpr
                # print()

                
        for i, g in enumerate(self.placement_graphs):
            self.fpg_gmm_list[i] = [nd for nd in g.nodes() if 'GM' in g.nodes[nd] and nd != 'Goal']

                
def merge_graphs(args, strategy_graphs):
    merge_SG = StrategyGraph()
    all_path = [sg.path_graphs for sg in strategy_graphs]
    all_path = [path for paths in all_path for path in paths]
    for graph in all_path:
        start_nd = [nd for nd in graph.nodes() if not list(graph.predecessors(nd))][0]
        merge_SG.set_placement_graph(args, graph, start_nd)
        # if start_nd not in args.tp.toolNames:
        #     merge_SG.path_graphs.append(graph)
        # else:
        #     is_isomorphic = False
        #     for tg in merge_SG.placement_graphs:
        #         if set(graph.edges()) == set(tg.edges()):
        #             is_isomorphic = True
        #             for node in graph.nodes():
        #                 tg.nodes[node]['ext'] += graph.nodes[node]['ext']
        #             break
        #     if not is_isomorphic:
        #         merge_SG.placement_graphs.append(graph)
        #         merge_SG.full_placement_idx.append(0)
        #         merge_SG.fpg_gmm_list.append([])
    merge_SG.merge_graph(args)
    merge_SG.train(args)
    return merge_SG