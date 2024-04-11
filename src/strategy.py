# strtegy graph class

from collections import Counter
import networkx as nx
from scipy.stats import norm
from sklearn import mixture
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from src.utils import node_match, draw_multi_paths
import os





class StrategyGraph():
    '''
    Graph structure for strategy
    ext: list of extrinsic data (tool, target)
    '''
    def __init__(self):
        self.path_graphs = []
        self.full_placement_graphs = []
        self.full_placement_idx = []
        self.obj_count = Counter()
        self.main_graph = nx.DiGraph()
        self.is_fpg_gmm_built = []
        self.is_fpg_gpr_built = []
        self.fpg_gmm_list = []

    def build_graph(self, args, sample_obj, init_pose, path_info):
        path_dict, collisions, success = path_info
        succ_list = []
        succ_list.append(sample_obj)
        graph = nx.DiGraph()
        self.obj_count[sample_obj] += 1
        # add visited flag to collisions
        collisions = [c + [False] for c in collisions]

        while succ_list:
            cur_nd_name = succ_list[0]
            cur_node = "PLACED" if cur_nd_name in args.tp.toolNames else cur_nd_name
            succ_list.pop(0)
            ext_col_list = []
            if not graph.has_node(cur_nd_name):
                init_ext = init_pose if cur_node == "PLACED" else path_dict[cur_node][0]
                graph.add_node(
                    cur_nd_name,
                    ext=[[None, init_ext]],
                    path=[path_dict]
                )
                prev_ext = init_pose
            else:
                prev_ext = graph.nodes[cur_nd_name]['ext'][0][0]

            # print(list(zip(visited, collisions)))
            for col in [c for c in collisions if not c[-1]]:
                is_reverse = False
                is_succ = False

                if col[0] == cur_node:
                    succ_node = col[1]
                    col[-1] = True
                    is_reverse = False
                    is_succ = True
                elif col[1] == cur_node:
                    col[-1] = True
                    succ_node = col[0]
                    is_reverse = True
                    is_succ = True
                else:
                    succ_node = None
                    is_succ = False
                if (is_succ and succ_node in args.available_objects
                    and (succ_node, cur_node) not in graph.edges()):
                    col_time = col[3] if col[3] else col[2]
                    col_end_idx = min(
                        int((col_time+0.1)*10), len(path_dict[succ_node])-1
                    )
                    graph.add_node(
                        succ_node,
                        ext=[[prev_ext, path_dict[succ_node][col_end_idx]]],
                        path=[path_dict]
                    )
                    graph.add_edge(cur_nd_name, succ_node, label=ext_col_list)
                    ext_col_list = []
                    prev_ext = path_dict[succ_node][col_end_idx]
                    succ_list.append(succ_node)
                elif (is_succ and succ_node == 'Goal'
                    and (cur_nd_name not in args.tp.toolNames
                    and args.tp.objects[cur_nd_name].color == (255, 0, 0, 255))): # red
                    col_end_idx = min(
                        int((col[2]+0.1)*10),
                        len(path_dict[cur_node])-1
                    )
                    graph.add_node(
                        succ_node,
                        ext=[[path_dict[cur_node][col_end_idx], None]],
                        path=[path_dict]
                    )
                    graph.add_edge(cur_nd_name, succ_node, label=ext_col_list)
                    ext_col_list=[]
                    break
                elif is_succ and succ_node not in args.available_objects:
                    ext_col_list.append(succ_node)

            if "Goal" not in graph.nodes() and not succ_list:
                if cur_nd_name == sample_obj:
                    break
        if "Goal" in graph.nodes():
            if sample_obj not in args.tp.toolNames:
                self.path_graphs.append(graph)
            else:
                is_isomorphic = False
                for tg in self.full_placement_graphs:
                    if set(graph.edges()) == set(tg.edges()):
                        is_isomorphic = True
                        for node in graph.nodes():
                            tg.nodes[node]['ext'] += graph.nodes[node]['ext']
                        break
                if not is_isomorphic:
                    self.full_placement_graphs.append(graph)
                    self.full_placement_idx.append(0)
                    self.is_fpg_gmm_built.append(False)
                    self.is_fpg_gpr_built.append(False)
                    self.fpg_gmm_list.append([])

        return graph

    def merge_graph(self, args):
        for idx, graph in (zip(self.full_placement_idx, self.full_placement_graphs)):
            for path in self.path_graphs[idx:]:
                if all(edge in graph.edges() for edge in path.edges()):
                    graph = self.add_strategy(graph, path)
            graph_idx = self.full_placement_graphs.index(graph)
            self.full_placement_idx[graph_idx] = len(self.path_graphs)
            # log
            for node in graph.nodes():
                if len([d for d in graph.nodes[node]['ext'] if d[0] is not None]) > 0:
                    data = graph.nodes[node]['ext']
                    tool_data, _ = zip(*[d for d in data if d[0] is not None])
                    print(node, len(graph.nodes[node]['ext']), len(tool_data))
                else:
                    print(node, len(graph.nodes[node]['ext']), '*')
            print('-')
            path_set = [path for node in graph.nodes() 
                for path in graph.nodes[node]['path']
                if node != 'Goal'
            ]
            node_str = '-'.join(graph.nodes())+'.png'
            img_name = os.path.join(args.dir_name, node_str)
            draw_multi_paths(args.btr0['world'], path_set, img_name)
        self.path_graphs = []

    def add_strategy(self, graph, path):
        for node in path.nodes():
            graph.nodes[node]['ext'] += path.nodes[node]['ext']
            graph.nodes[node]['path'] += path.nodes[node]['path']

        return graph
    

    def train(self):
        for graph in self.full_placement_graphs:
            for nd in [nd for nd in graph.nodes() if not list(graph.predecessors(nd))]:
                print("======",nd,"======")
                data = [d[1] for d in graph.nodes[nd]['ext']]
                if len(data) < 2:
                    continue
                gmm = mixture.GaussianMixture(
                    n_components=1, 
                    covariance_type='full'
                )
                gmm.fit(data)
                graph.nodes[nd]['gmm'] = gmm
            for nd_i, nd_j in graph.edges():
                if nd_j == 'Goal':
                    print("==",nd_i, nd_j, "==")
                    data = [d[0] for d in graph.nodes[nd_j]['ext']]
                    if len(data) < 2:
                        continue
                    gmm = mixture.GaussianMixture(
                        n_components=1,
                        covariance_type='full'
                    )
                    gmm.fit(data)
                    graph.nodes[nd_j]['model'] = gmm
                    print('GMM')
                else:
                    print("==",nd_i, nd_j)
                    data = graph.nodes[nd_j]['ext']

                    tool_data, target_data = zip(*[d for d in data if d[0] is not None])
                    tool_data = list(tool_data)
                    target_data = list(target_data)
                    if len(tool_data) < 2:
                        continue
                    x = target_data
                    y = tool_data
    
                    kernel = DotProduct() + WhiteKernel()

                    # Create the GPR model
                    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(x, y)
                    graph.nodes[nd_j]['model'] = gpr
                    print('GPR', end=' ')
                    data = [d[1] for d in graph.nodes[nd_j]['ext']]
                    if len(data) < 2:
                        continue
                    gmm = mixture.GaussianMixture(
                        n_components=1, 
                        covariance_type='full'
                    )
                    gmm.fit(data)
                    graph.nodes[nd_j]['gmm'] = gmm
                    print('GMM')
                    # gmm = mixture.GaussianMixture(n_components=1, covariance_type='full')
                    # gmm.fit(data)
                    # print(gmm.means_, gmm.covariances_)

                    # data = [d[1] for d in graph.nodes[nd_j]['ext']]
                    # data = zip(*data)
                    # print(data)
                    # for data_points in data:
                    #     # Calculate the mean and standard deviation of the data points
                    #     mu, std = norm.fit(data_points)
                    #     print(mu, std)
                    #     graph.nodes[nd_j]['ext'] = (mu, std)
        for i, g in enumerate(self.full_placement_graphs):
            self.fpg_gmm_list[i] = [nd for nd in g.nodes() if 'gmm' in g.nodes[nd]]
            if not self.is_fpg_gmm_built[i] and self.fpg_gmm_list:
                self.is_fpg_gmm_built[i] = True
            self.is_fpg_gpr_built[i] = all(
                'model' in g.nodes[nd] for nd in g.nodes() 
                if not list(g.predecessors(nd))
            )

                
