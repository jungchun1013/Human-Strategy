import random
import numpy as np

from src.schema.strategy import StrategyManager

###########################
# Utilities
###########################

def print_args(args):
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

def set_exp_dir(tnm, algorithm):
    exptime_str = datetime.now().strftime("%y%m%d_%H%M%S")
    date = exptime_str[:6]
    time = exptime_str[7:]
    exp_name = '_'.join([time, tnm, algorithm])
    main_dir_name = os.path.join('data', date, exp_name)
    os.makedirs(main_dir_name)
    return main_dir_name

def setup_trial_dir(main_dir_name, trial_count): 
    trial_dir_name = os.path.join(main_dir_name, str(trial_count).zfill(3))
    os.makedirs(trial_dir_name)
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    logging.basicConfig(
        filename=os.path.join(trial_dir_name, 'output.log'),
        format='%(levelname)s:%(message)s',
        level=logging.INFO
    )
    # add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)      

    return trial_dir_name

###########################
# Main
###########################

def sample_and_update_strategy(args, strategy):
    args.btr = deepcopy(args.btr0)
    args.tp = ToolPicker(args.btr0)
    success = False
    sample_type = 'tool'
    sampled_obj, sampled_ext, ext_info, path_info = sample_ext_by_type(
            args,
            sample_type,
            strategy_graph,
            sample_objs = None
        )
    path_dict, collisions, success = path_info
    if success:
        graph = strategy.build_schema(args, task_name, sampled_obj, sampled_ext, path_info)
    return strategy

def learn_strategy(args, strategy):
    attempt_count = 0
    while attempt_count < args.max_attempts:
        strategy = sample_and_update_strategy(args, strategy)
        attempt_count += 1
    return strategy


def schema_learning(args):
    """
    schema learning algorithm
    1. sample from PE
    2. run virtually or physically
    3. collect data
    4. update strategy graph
    """
    strategy_manager = StrategyManager(args)
    training_levels = args.training_tasks
    if not args.use_saved:
        for level in training_levels:
            task_name = config.task2series[level]
            args.task_name = task_name
            strategy_manager = learn_strategy(args, strategy_manager)
            strategy_manager.build_model(args)
    else:
        strategy_manager = load_strategy(args)
    
    

    

def run(args):
    setup_trial_dir(args.main_dir_name, args.trial_count)
    # NOTE - now dont care about the algorithm do schema
    schema_learning(args)
    
    

def main(args):
    print_args(args)
    dir_name = set_exp_dir(args.task_name, args.algorithm)
    print(f"Experiment directory: {dir_name}")
    args.trial_count = 0
    for trial_count in range(args.num_trials):
        run(args)   
        args.trial_count += 1
        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SSUP')
    parser.add_argument('-a', '--algorithm',
                        help='SSUP/ours', default='SSUP')
    parser.add_argument('-t', '--num-trials',
                        help='number of trials/experiments', type=int, default=10)
    parser.add_argument('--max-attempts',
                        help='number of/maximum attempts for each level in training', type=int, default=10)
    parser.add_argument('--task-name',
                        help='task name for loading specific levels', type=str, default='CatapultAlt')
    parser.add_argument('--json_dir',
                        help='json dir with levels', type=str,
                        default='./environment/Trials/Strategy/')
    parser.add_argument('-v', '--verbose',
                        help='Increase output verbosity', action='store_true')
    parser.add_argument('--eps',
                        help='epsilon', type=float, default=None)
    parser.add_argument('--eps-decay-rate',
                        help='epsilon decay rate', type=int, default=None)
    parser.add_argument('--lr',
                        help='learning rate', type=float, default=None)
    parser.add_argument('--attempt-threshold',
                        help='attempt threshold or reward', type=float, default=None)
    parser.add_argument('-d', '--deterministic',
                        help='whether deterministic or noisy in collecting data',
                        action='store_true')
    parser.add_argument('--use-saved',
                        help='whether to use saved strategy graph',
                        action='store_true')
    # parse args and update config
    args = parser.parse_args()
    args = load_args(args)
    
    # Set seed for reproducibility
    seed_value = 42  # You can change this value for different seeds
    random.seed(seed_value)
    np.random.seed(seed_value)
    main(args)