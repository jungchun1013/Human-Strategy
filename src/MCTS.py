from src.strategy import StrategyGraph, Mechanism
import random
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

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        if choices_weights:
            return self.children[choices_weights.index(max(choices_weights))]
        return None

    def add_child(self, child_node):
        self.children.append(child_node)

    def update(self, value):
        self.visits += 1
        self.value += value

class StrategyGraphState:
    '''
    record the path (state) of the strategy grpah at that node'''
    mechanisms = None
    def __init__(self, graph=None, available_objects=None, used_objects=None, curr_mech=None):
        self.graph = graph or nx.DiGraph()
        self.available_objects = available_objects or []
        self.used_objects = used_objects or []
        self.curr_mech_pair = curr_mech
        self.legal_actions = None

    @classmethod
    def set_mechanisms(cls, mechanisms):
        cls.mechanisms = mechanisms

    def is_terminal(self):
        return bool([edge for edge in self.graph.edges() if self.graph.nodes[edge[1]]['tuple']['mech'].source_info['type'] == 'Tool'])

    def get_legal_actions(self):
        if self.legal_actions is not None:
            return self.legal_actions
        curr_obj = self.curr_mech_pair.get('src')
        print('curr_obj', curr_obj)
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
        new_graph = self.graph.copy()
        src = action['src']
        tar = action['tar']
        new_graph.add_node(src, tuple=action)
        new_graph.add_edge(tar, src)
        new_state = StrategyGraphState(new_graph, self.available_objects, self.used_objects + [action['tar']], action)

        new_node = Node(new_state)
        return new_state

    def get_reward(self):
        # 獲取當前狀態的獎勵
        if 'PLACED' in self.graph.nodes():
            # return 1
            return random.random()
        return 0


class StrategyGraphMCTS:
    def __init__(self, simulation_no=1000, args=None):
        self.simulation_no = simulation_no
        self.args = args

    def sample(self):
        # root = Node(initial_state)
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
    def search(self):
        # root = Node(initial_state)
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

        return self._best_action(root)

    def _tree_policy(self, node):
        while not node.state.is_terminal():
            print('+', node.state.graph.edges())
            if not node.is_fully_expanded():
                print('not fully expanded')
                return self._expand(node)
            else:
                print('expanded')
                node = node.best_child()
        return node

    def _expand(self, node):
        tried_actions = [child.state.curr_mech_pair for child in node.children]
        legal_actions = node.state.get_legal_actions()
        print('legal_actions', legal_actions)
        print('tried_actions', tried_actions)
        for action in legal_actions:
            if action not in tried_actions:
                print('action', action)
                next_state = node.state.move(node, action)
                child_node = Node(next_state, node)
                node.add_child(child_node)
                return child_node
        raise Exception("Should never reach here")

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
        while node is not None:
            node.update(reward)
            node.state = state
            node = node.parent

    def _best_action(self, root):
        return root.best_child(c_param=0)