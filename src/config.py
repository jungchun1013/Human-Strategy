OURS_config = dict(
    eps = 0.2,
    max_attempts = 100,
    max_simulations = 1000,
    num_trials = 100,
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
    max_attempts = 100,
    max_simulations = 1000,
    attempt_threshold = 0.7,
)

GPR_SSUP_config = dict(
    num_init = 3,
    num_iter = 10,
    num_sim = 4,
    eps = 0.2,
    eps_decay_rate = 0.95,
    lr =1000,
    max_attempts = 100,
    max_simulations = 1000,
    attempt_threshold = 0.7,
    num_trials = 10,
    gm_ratio = 0
)  

task_config = dict(
    Catapult = {'training':  ['Catapult_1', 'Catapult_2', 'Catapult_3'], 'testing': 'Catapult'},
    Unbox = {'training':  ['Unbox_1', 'Unbox_2', 'Unbox_3'], 'testing': 'Unbox'},
    Launch_v2 = {'training': ['Launch_v2_1', 'Launch_v2_2', 'Launch_v2_3'], 'testing': 'Launch_v2'},
    CatapultAlt = {'training':  ['Launch_v2', 'Launch_v2_1', 'Launch_v2', 'Catapult', 'Catapult_2', 'Catapult_3'], 'testing': 'CatapultAlt'},
)

