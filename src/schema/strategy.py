
class StrategyManager:
    def __init__(self, args):
        self.args = args
        self.strategy_graph = None

    def _extract_kinematic_info(self, obj_name):
        pass

    def build_schema(self, args, task_name, sampled_obj, sampled_ext, path_info):
        path_dict, collisions, success = path_info
        graph = nx.DiGraph()
        is_col_visited = [False for _ in collisions]
        is_col_visited_CF = [False for _ in args.collisions0]
        # ext_info = {
        #     'prev': {'src': sampled_ext, 'tgt': None},
        #     'curr': {'src': None, 'tgt': None},
        #     'init': {'src': sampled_ext, 'tgt': sampled_ext}
        # }
        # available_objects = args.movable_objects + args.tool_objects + ['Goal']
        # poss = {'prev_tool':[], 'prev_target':[], 'curr_tool':[], 'curr_target':[]}
        
        while succ_list:
            cur_nd_name = succ_list.pop(0)
            cur_node = "PLACED" if cur_nd_name in args.tp.toolNames else cur_nd_name

    def build_model(self, args):
        pass