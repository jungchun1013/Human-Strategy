

from collections import Counter
import networkx as nx
from copy import deepcopy
from scipy.stats import norm
from sklearn import mixture
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from src.utils import node_match





class StrategyGraph():
    def __init__(self):
        self.path_graphs = []
        self.full_placement_graphs = []
        self.full_placement_idx = []
        self.obj_count = Counter()
        self.main_graph = nx.DiGraph()

    def build_graph(self, collisions, sample_obj, init_pose, tp, path_dict, available_obj):
        succ_list = []
        succ_list.append(sample_obj)
        graph = nx.DiGraph()
        self.obj_count[sample_obj] += 1
        # add visited flag to collisions
        collisions = [ c + [False] for c in collisions]
        
        while succ_list:
            cur_nd_name = succ_list[0]
            cur_node = "PLACED" if cur_nd_name in tp.toolNames else cur_nd_name
            succ_list.pop(0)
            ext_col_list = []
            pre_ext = None
            if not graph.has_node(cur_nd_name):
                init_ext = init_pose if cur_node == "PLACED" else path_dict[cur_node][0]
                graph.add_node(cur_nd_name, ext=[[None, init_ext]])
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
                if (is_succ and succ_node in available_obj 
                    and (succ_node, cur_node) not in graph.edges()):
                    col_time = col[3] if col[3] else col[2]
                    col_end_idx = min(
                        int((col_time+0.1)*10), len(path_dict[succ_node])-1
                    )
                    graph.add_node(
                        succ_node, 
                        ext=[[prev_ext, path_dict[succ_node][col_end_idx]]]
                    )
                    graph.add_edge(cur_nd_name, succ_node, label=ext_col_list)
                    ext_col_list = []
                    prev_ext = path_dict[succ_node][col_end_idx]
                    succ_list.append(succ_node)
                elif (is_succ and succ_node == 'Goal'
                    and (cur_nd_name not in tp.toolNames
                    and tp.objects[cur_nd_name].color == (255, 0, 0, 255))): # red
                    col_end_idx = min(
                        int((col[2]+0.1)*10), 
                        len(path_dict[cur_node])-1
                    )
                    graph.add_node(
                        succ_node, 
                        ext=[[path_dict[cur_node][col_end_idx], None]]
                    )
                    graph.add_edge(cur_nd_name, succ_node, label=ext_col_list)
                    ext_col_list=[]
                    break
                elif is_succ and succ_node not in available_obj:
                    ext_col_list.append(succ_node)
            
            if "Goal" not in graph.nodes() and not succ_list:
                if cur_nd_name == sample_obj:
                    break

        # for c in collisions:
        #     print(c)

        if "Goal" in graph.nodes():
            print(graph.edges())
            if sample_obj not in tp.toolNames:
                self.path_graphs.append(graph)
            else:
                is_isomorphic = False
                for tg in self.full_placement_graphs:
                    if set(graph.edges()) == set(tg.edges()):
                    # if nx.is_isomorphic(tg, graph, node_match=node_match):
                        is_isomorphic = True
                        for node in graph.nodes():
                            tg.nodes[node]['ext'] += graph.nodes[node]['ext']
                        break
                if not is_isomorphic:
                    self.full_placement_graphs.append(graph)
                    self.full_placement_idx.append(0)
                self.merge_graph()
                self.train()

        return graph

    def merge_graph(self):
        for idx, graph in (zip(self.full_placement_idx, self.full_placement_graphs)):
            for path in self.path_graphs[idx:]:
                if all(edge in graph.edges() for edge in path.edges()):
                    graph = self.add_strategy(graph, path)
            graph_idx = self.full_placement_graphs.index(graph)
            self.full_placement_idx[graph_idx] = len(self.path_graphs)
            print("full_placement_graphs")
            print(graph.edges())
            # log
            for node in graph.nodes():
                if len([d for d in graph.nodes[node]['ext'] if d[0] != None]) > 0:
                    data = graph.nodes[node]['ext']
                    tool_data, target_data = zip(*[d for d in data if d[0] != None])
                    print(node, len(graph.nodes[node]['ext']), len(tool_data))
                else:
                    print(node, len(graph.nodes[node]['ext']), '*')

        self.path_graphs = []
        


    def add_strategy(self, graph, path):
        for node in path.nodes():
            graph.nodes[node]['ext'] += path.nodes[node]['ext']

        return graph
    

    def train(self):
        for graph in self.full_placement_graphs:
            for nd in [nd for nd in graph.nodes() if not list(graph.predecessors(nd))]:
                print("===",nd,"===")
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
                    print("==",nd_i, nd_j, "==")
                    data = graph.nodes[nd_j]['ext']

                    tool_data, target_data = zip(*[d for d in data if d[0] != None])
                    tool_data = list(tool_data)
                    target_data = list(target_data)
                    if len(tool_data) < 2:
                        continue
                    X = target_data
                    y = tool_data
    

                    kernel = DotProduct() + WhiteKernel()

                    # Create the GPR model
                    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X, y)
                    graph.nodes[nd_j]['model'] = gpr
                    print('GPR')
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
            


