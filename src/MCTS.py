from src.strategy import Mechanism
from random import shuffle
import networkx as nx
import json
import hashlib
import math

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.used_obj = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_actions())

    def best_child(self, c_param=0.4):

        # UCB
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(child.visits) / child.visits))
            if child.visits > 0 else 0 for child in self.children
        ]
        if choices_weights:
            return self.children[choices_weights.index(max(choices_weights))]
        return None

    def add_child(self, child_node):
        self.children.append(child_node)

    def update(self, value):
        # average
        self.visits += 1
        self.value += value
        
    def estimate_value(self):
        if self.visits == 0:
            return 0
        # return self.value / self.visits
        ucb_value = self.value / self.visits + np.sqrt(2 * np.log(np.sum(self.visits)) / self.visits)
        return ucb_value

class StrategyGraphState:
    '''
    record the path (state) of the strategy graph at that node'''
    mechanisms = None
    def __init__(self, graph=None, available_objects=None, used_objects=None, curr_mech=None):
        self.graph = graph or nx.DiGraph()
        self.full_graph = graph or nx.DiGraph()
        self.available_objects = available_objects or []
        self.used_objects = used_objects or []
        self.curr_mech_pair = curr_mech
        self.legal_actions = None

    @classmethod
    def set_mechanisms(cls, mechanisms):
        cls.mechanisms = mechanisms

    def is_terminal(self):
        # return bool([edge for edge in self.graph.edges() if self.graph.nodes[edge[1]]['tuple']['mech'].source_info['type'] == 'Tool'])
        return self.curr_mech_pair.get('src') == 'PLACED'

    def get_legal_actions(self):
        curr_obj = self.curr_mech_pair.get('src')
        # if self.legal_actions is not None:
        #     return self.legal_actions
        curr_obj_type = StrategyGraphState.mechanisms._check_object_type(curr_obj)
        if curr_obj_type == 'Tool':
            return []
        available_mech = StrategyGraphState.mechanisms.mechanisms[curr_obj_type]
        mech_choices = []
        for prev_mech in available_mech: 
            for obj in self.available_objects:
                if StrategyGraphState.mechanisms._check_object_type(obj) == prev_mech.source_info['type'] and obj not in self.used_objects:  
                    mech_choices.append({'mech': prev_mech, 'src':obj, 'tar':curr_obj})
            if prev_mech.source_info['type'] == 'Tool':
                mech_choices.append({'mech': prev_mech, 'src':'PLACED', 'tar':curr_obj})
        self.legal_actions = mech_choices
        return mech_choices

    def move(self, curr_node, action):
        # 執行行動並返回新狀態
        # print('move', curr_node.state.curr_mech_pair, action)
        new_graph = self.graph.copy()
        src = action['src']
        tar = action['tar']

        new_graph.add_node(src)
        # new_graph.add_edge(tar, src, mech=action['mech'])
        new_graph.add_edge(src, tar, mech=action['mech'])
        new_state = StrategyGraphState(new_graph, self.available_objects, self.used_objects + [action['tar']], action)


        new_node = Node(new_state)
        return new_state

    def get_reward(self):
        # 獲取當前狀態的獎勵
        if 'PLACED' in self.graph.nodes():
            # return 1
            # return random.random()
            reward = 0
            if len(self.graph.nodes()) == 5:
                reward += 1
            for node in self.graph.nodes():
                if node in ['PLACED', 'Goal', 'CataBall', 'Lever', 'KeyBall']:
                    reward += len(node)
            return reward
        return 0


class StrategyGraphMCTS:
    def __init__(self, simulation_no=1000, args=None, strategy_manager=None):
        self.simulation_no = simulation_no
        self.args = args
        # self.strategy_manager = strategy_manager
        self.root = None

    def sample(self):
        # root = Node(initial_state)
        available_obj = self.args.movable_objects
        curr_obj = 'Goal'
        used_obj = []
        strategy = []
        src_info = {'obj': 'Goal', 'type': 'Goal', 'ext': None, 'path': None}
        curr_mech_pair = {'mech': Mechanism(src_info, None, None), 'src': 'Goal', 'tar': None}
        init_state = StrategyGraphState(None, available_obj, [], curr_mech_pair)

        if not self.root:
            self.root = Node(init_state)

        node = self._tree_policy(self.root)
        # reward = self._default_policy(node)
        self._backup(node, 0)

        return self._best_action(self.root), node

    def search(self):
        available_obj = self.args.movable_objects
        curr_obj = 'Goal'
        used_obj = []
        strategy = []
        src_info = {'obj': 'Goal', 'type': 'Goal', 'ext': None, 'path': None}
        curr_mech_pair = {'mech': Mechanism(src_info, None, None), 'src': 'Goal', 'tar': None}
        init_state = StrategyGraphState(None, available_obj, [], curr_mech_pair)

        root = Node(init_state)
        ####
        for _ in range(self.simulation_no):
            node = self._tree_policy(root)
            reward = self._default_policy(node)
            self._backup(node, reward)
        
        self.print_graph(root)

        return self._best_action(root), node

    def update_reward(self, node, reward):
        # get reward from simulation
        self._backup(node, reward)

    def print_graph(self, node=None, reward_only=False):
        # Create a set to store visited nodes
        visited = set()
        if not node:
            node = self.root

        # Define the DFS function
        print('---- graph -----')
        def dfs(node, depth=0):
            indent = '  ' * depth  # Create an indent based on the current depth
            if not reward_only or node.value > 0:
                print(indent+str(node.state.curr_mech_pair['tar'])+'-'+str(node.state.curr_mech_pair['src'])+' '+str(node.state.curr_mech_pair['mech']), str(node.state.graph.edges()), node.estimate_value(), ('t' if node.state.is_terminal() else''))  # Print the node
            visited.add(node)  # Mark the node as visited
            # Visit all the node's children
            for child in node.children:
                if child not in visited:
                    dfs(child, depth=depth + 1)
        # Start the DFS at the root node
        dfs(node)
        print('----------------')


    def _expand(self, node):
        # print('_expand', node)
        tried_actions = [child.state.curr_mech_pair for child in node.children]
        legal_actions = node.state.get_legal_actions()
        shuffle(legal_actions)
        # print('legal_actions', [ (a['src'], a['tar'], a['mech']) for a in legal_actions])
        for action in legal_actions:
            if action not in tried_actions:
                next_state = node.state.move(node, action)
                child_node = Node(next_state, node)
                node.add_child(child_node)
                return child_node
        raise Exception("Should never reach here")

    def _tree_policy(self, node):
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                # return self._expand(node)
                node = self._expand(node)
            else:
                node = node.best_child()
        return node

    def _default_policy(self, node):
        current_state = node.state
        while not current_state.is_terminal():
            legal_actions = current_state.get_legal_actions()
            if not legal_actions:
                break
            action = random.choice(legal_actions)
            current_state = current_state.move(node, action)
        return current_state.get_reward()

    def _backup(self, node, reward):
        state = node.state
        state.full_graph = state.graph
        while node is not None:
            node.update(reward)
            if node.parent:
                best_child = node.parent.best_child()
                if node.value/node.visits >= best_child.value/best_child.visits:
                    node.state.full_graph = state.full_graph
            node = node.parent

    def _best_action(self, root):
        # return root.state.full_graph
        return root.best_child().state.full_graph