from src.strategy_v2 import Mechanism, check_object_type
from random import shuffle, choice
import networkx as nx
import json
import hashlib
import math
import numpy as np

class Node:
    total_visits = 0
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.used_obj = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_actions())

    def best_child(self, c_param=0.2):

        # UCB
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(Node.total_visits) / child.visits))
            if child.visits > 0 else 0 for child in self.children
        ]
        if choices_weights:
            return self.children[choices_weights.index(max(choices_weights))]


    def add_child(self, child_node):
        self.children.append(child_node)

    def update(self, value):
        # average
        self.visits += 1
        self.value += value
        
    def estimate_value(self, c_param=0.2):
        if self.visits == 0:
            return 0
        # return self.value / self.visits
        ucb_value = self.value / self.visits + c_param * np.sqrt(np.log(Node.total_visits) / self.visits)
        return ucb_value

class StrategySchemaState:
    '''
    record the path (state) of the strategy graph at that node'''
    mechanisms = None
    def __init__(self, graph=None, available_objects=None, used_schema=None, curr_schema=None):
        self.graph = graph or nx.DiGraph()
        self.full_graph = graph or nx.DiGraph()
        self.available_objects = available_objects or []
        self.used_schema = used_schema or []
        self.curr_schema = curr_schema
        self.curr_object = None
        self.legal_actions = None
        self._is_terminal = False # TODO - determine terminal with a prob

    @classmethod
    def set_schema(cls, schema):
        cls.schema = schema

    def is_terminal(self):
        return self._is_terminal

    def get_legal_actions(self):
        # TODO - link previous object in last schema
        curr_schema = self.curr_schema.get('schema')
        available_schema = StrategySchemaState.schema

        schema_choices = []
        for task_type, strat_types in available_schema.items():
            for strat_type, models in strat_types.items():
                gpr, graph, object_types = models
                sample_objs = []
                for obj_type in object_types:
                    same_type_objs = [obj for obj in self.available_objects + ['Goal'] if check_object_type(obj) == obj_type and obj not in sample_objs]
                    sampled_obj = choice(same_type_objs)
                    sample_objs.append(sampled_obj)
                if ';'.join([task_type,strat_type,str(sampled_obj)]) not in self.used_schema:
                    schema_choices.append({'schema': ';'.join([task_type,strat_type,str(sampled_obj)]), 'objs': sample_objs})

        if curr_schema != 'Goal':      
            schema_choices.append({'schema': 'Terminal', 'objs':[]})
        self.legal_actions = schema_choices
        print(schema_choices)
        return schema_choices

    def move(self, curr_node, action):
        # 執行行動並返回新狀態
        new_graph = self.graph.copy()
        if action['schema'] != 'Terminal':
            src = action['objs'][0]
            tar = action['objs'][-1] 

            new_graph.add_node(src)
            # new_graph.add_edge(tar, src, mech=action['mech'])
            new_graph.add_edge(src, tar, schema=action['schema'], objs=action['objs'])
            new_state = StrategySchemaState(new_graph, self.available_objects, self.used_schema + [action['schema']], action)
            new_node = Node(new_state)
        else:
            curr_node.state._is_terminal = True
            new_node = curr_node
            new_state = curr_node.state
            

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


class StrategySchemaMCTS:
    def __init__(self, simulation_no=1000, args=None, strategy_manager=None):
        self.simulation_no = simulation_no
        self.args = args
        # self.strategy_manager = strategy_manager
        self.root = None
        self.visits = 0

    def sample(self):
        # root = Node(initial_state)
        available_obj = self.args.movable_objects
        curr_obj = 'Goal'
        used_obj = []
        strategy = []
        src_info = {'obj': 'Goal', 'type': 'Goal', 'ext': None, 'path': None}
        # FIXME - never deliver model to MCTS
        curr_schema = {'schema': 'Goal', 'objs': []}
        init_state = StrategySchemaState(None, available_obj, [], curr_schema)

        if not self.root:
            self.root = Node(init_state)
        node = self._tree_policy(self.root)
        # reward = self._default_policy(node)
        # self._backup(node, None)

        return node.state.full_graph, node

    def update_reward(self, node, reward):
        # get reward from simulation
        self._backup(node, reward)
    def print_simple_graph(self, node=None, reward_only=False):
        visited = set()
        if not node:
            node = self.root

        # Define the DFS function
        print('---- graph -----')
        def dfs(node, depth=0):
            indent = '  ' * depth  # Create an indent based on the current depth
            if not reward_only or node.value > 0:
                print(indent+str(node.state.curr_schema['schema'])+' '+str(node.state.curr_schema['objs']), Node.total_visits, node.visits, node.estimate_value(), ('t' if node.state.is_terminal() else''))  # Print the node
            visited.add(node)  # Mark the node as visited
            # Visit all the node's children
            for child in node.children:
                if child not in visited:
                    dfs(child, depth=depth + 1)
        # Start the DFS at the root node
        dfs(node)
        print('----------------')
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
                print(indent+str(node.state.curr_schema['schema'])+' '+str(node.state.curr_schema['objs']), str(node.state.graph.edges()), Node.total_visits, node.visits, node.estimate_value(), ('t' if node.state.is_terminal() else''))  # Print the node
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
        tried_actions = [child.state.curr_schema for child in node.children]
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
            # print(node.state.get_legal_actions())

            if not node.is_fully_expanded():
                # return self._expand(node)
                node = self._expand(node)
            else:
                # FIXME - return none
                node = node.best_child()
                
        Node.total_visits += 1
            
                    
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
            if reward is not None:
                node.update(reward)
            if node.parent:
                best_child = node.parent.best_child()
                if node.value/node.visits >= best_child.value/best_child.visits:
                    node.state.full_graph = state.full_graph
            node = node.parent

    def _best_action(self, root):
        # return root.state.full_graph
        return root.best_child().state.full_graph

    def _best_node(self, root):
        return root.best_child()
