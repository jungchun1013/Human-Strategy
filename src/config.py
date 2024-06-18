OURS_config = dict(
    eps = 0.2,
    max_attempts = 50,
    max_simulations = 4000,
    num_demos = 100,
    attempt_threshold = 0.7,
    gm_ratio = 0
)

SSUP_config = dict(
    num_init = 3,
    num_iter = 10,
    num_sim = 4,
    eps = 0.2,
    eps_decay_rate = 0.95,
    lr =1000,
    max_attempts = 50,
    max_simulations = 4000,
    attempt_threshold = 0.7,
)

GPR_SSUP_config = dict(
    num_init = 3,
    num_iter = 10,
    num_sim = 4,
    eps = 0.2,
    eps_decay_rate = 0.95,
    lr =1000,
    max_attempts = 50,
    max_simulations = 4000,
    attempt_threshold = 0.7,
    num_demos = 10,
    gm_ratio = 0
)


task_config = dict(
    Catapult = {'training':  ['Catapult_1', 'Catapult_2', 'Catapult_3'], 'testing': 'Catapult', 'start_tool': 'PLACED'},
    Unbox = {'training':  ['Unbox_1', 'Unbox_2', 'Unbox_3'], 'testing': 'Unbox', 'start_tool': 'PLACED'},
    Prevention_A = {'training':  ['Prevention_A_1', 'Prevention_A_2', 'Prevention_A_3'], 'testing': 'Prevention_A', 'start_tool': 'PLACED'},
    Launch_v2 = {'training': ['Launch_v2_1', 'Launch_v2_2', 'Launch_v2_3'], 'testing': 'Launch_v2', 'start_tool': 'PLACED'},
    Funnel = {'training': ['Funnel_1', 'Funnel_2', 'Funnel_3'], 'testing': 'Funnel', 'start_tool': 'Ball', 'start_tool': 'CataBall'},
    # CatapultAlt = {'training':  ['Catapult', 'Catapult_2', 'Catapult_3', 'Launch_v2', 'Launch_v2_1', 'Launch_v2_2', 'Funnel', 'Funnel_1'], 'testing': 'CatapultAlt'},
    CatapultAlt = {'training':  ['Funnel_3', 'Funnel', 'Catapult_4', 'Catapult_5', 'Launch_v2_1', 'Launch_v2_3'], 'testing': 'CatapultAlt', 'start_tool': 'PLACED'},
    CatapultAltMod = {'training':  ['CatapultAlt', 'CatapultAlt_2', 'Launch_v2_1', 'Launch_v2_3', 'SlopeLaunch_1', 'SlopeLaunch_1'], 'testing': 'CatapultAlt', 'start_tool': 'PLACED'},
    # CatapultAlt = {'training':  ['CatapultAlt_1', 'CatapultAlt_2'], 'testing': 'CatapultAlt', 'start_tool': 'PLACED'},
    Slope = {'training':  ['Slope_1', 'Slope_2'], 'testing': 'Slope', 'start_tool': 'PLACED'},
    SlopeR = {'training':  ['SlopeR_1', 'SlopeR_2'], 'testing': 'SlopeR', 'start_tool': 'PLACED'},
    SlopeR_v2 = {'training':  ['SlopeR_v2_1', 'SlopeR_v2_2'], 'testing': 'SlopeR_v2', 'start_tool': 'PLACED'},
    SlopeLaunch_1 = {'training':  ['SlopeLaunch_1', 'SlopeLaunch_2'], 'testing': 'SlopeLaunch_1', 'start_tool': 'PLACED'},
    SlopeLaunch_2 = {'training':  ['SlopeLaunch_1', 'SlopeLaunch_2'], 'testing': 'SlopeLaunch_2', 'start_tool': 'PLACED'},
    ChainingUnit = {'training':  ['ChainingUnit_1', 'ChainingUnit_2'], 'testing': 'ChainingUnit', 'start_tool': 'PLACED'},
    Chaining = {'training':  ['ChainingUnit', 'ChainingUnit_1', 'ChainingUnit_2', 'Launch_v2_1', 'Launch_v2_2'], 'testing': 'CatapultAlt', 'start_tool': 'PLACED'},
    SmallSlope = {'training':  ['SmallSlope_1', 'SmallSlope_2'], 'testing': 'SmallSlope', 'start_tool': 'PLACED'},
    Slope_v2 = {'training':  ['Slope_v2_1', 'Slope_v2_2'], 'testing': 'Slope_v2', 'start_tool': 'PLACED'},
    MultiSlope_v3 = {'training':  ['SlopeR_v2', 'SlopeR_v2_1', 'SmallSlope', 'SmallSlope'], 'testing': 'MultiSlope_v3', 'start_tool': 'PLACED'},
)

task2series = dict()

for task_series, config in task_config.items():
    for i in range(1, 6):
        task = f"{task_series}_{i}"
        task2series[task] = task_series
    task2series[task_series] = task_series
# FIXME
additional_task_config = dict(
    CatapultAlt_1 = {'training':  ['CatapultAlt', 'CatapultAlt_2'], 'testing': 'CatapultAlt', 'start_tool': 'PLACED'},
    CatapultAlt_2 = {'training':  ['CatapultAlt_1', 'CatapultAlt_2'], 'testing': 'CatapultAlt', 'start_tool': 'PLACED'}
)

task_config.update(additional_task_config)

