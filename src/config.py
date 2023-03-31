class Config():
    
    # mapping function
    COL2STR = {
        '\\q0' : 'q0',
        '\\q95' : 'q95', 
        '\\ipmhd' : 'Ip', 
        '\\kappa' : 'K', 
        '\\tritop' : 'tri-top', 
        '\\tribot': 'tri-bot',
        '\\betap': 'betap',
        '\\betan': 'betan',
        '\\li': 'li', 
        '\\WTOT_DLM03' : 'W-tot', 
        '\\ne_inter01' : 'Ne',
        '\\TS_NE_CORE_AVG' : 'Ne-core', 
        '\\TS_TE_CORE_AVG' : 'Te-core',
        '\\nb11_pnb' : 'NB11-pnb',
        '\\nb12_pnb': 'NB12-pnb',
        '\\nb13_pnb': 'NB13-pnb',
        '\\RC01': 'RC01', 
        '\\RC02': 'RC02', 
        '\\RC03': 'RC03',
        '\\VCM01' : 'VCM01', 
        '\\VCM02' : 'VCM02', 
        '\\VCM03' : 'VCM03',
        '\\EC2_PWR': 'EC2_PWR', 
        '\\EC3_PWR' : 'EC3_PWR', 
        '\\ECSEC2TZRTN' : 'ECS-EC2-TZRTN', 
        '\\ECSEC3TZRTN' : 'ECS-EC3-TZRTN',
        '\\EC1_PWR': 'EC1_PWR', 
        '\\EC4_PWR' : 'EC4_PWR', 
        '\\ECSEC1TZRTN' : 'ECS-EC1-TZRTN', 
        '\\ECSEC4TZRTN' : 'ECS-EC4-TZRTN',
        '\\LV01' : 'LV01',
        '\\bcentr' : 'Bc',
        '\\rsurf':'R',
        '\\aminor':'a',
    }
    
    ## columns for use
    # 0D parameters
    DEFAULT_COLS_0D = ['\\q95','\\betan','\\li', '\\q0']
    
    # control parameter
    DEFAULT_COLS_CTRL = [
        '\\nb11_pnb',
        '\\nb12_pnb',
        '\\nb13_pnb',
        # '\\EC1_PWR', 
        '\\EC2_PWR', 
        '\\EC3_PWR',
        '\\EC4_PWR', 
        # '\\ECSEC1TZRTN', 
        '\\ECSEC2TZRTN', 
        '\\ECSEC3TZRTN',
        '\\ECSEC4TZRTN',
        '\\ipmhd', 
        '\\kappa', 
        '\\tritop', 
        '\\tribot', 
        '\\bcentr',
        '\\rsurf',
        '\\aminor',
    ]
    
    # predictor : configuration for nn_env
    TRANSFORMER_CONF = {
        "n_layers": 4, 
        "n_heads":8, 
        "dim_feedforward" : 1024, 
        "dropout" : 0.2,        
        "RIN" : False,
        "feature_0D_dim" : 128,
        "feature_ctrl_dim": 128,
        "noise_mean" : 0,
        "noise_std" : 1.96
    }
    
    SCINET_CONF = {
        "hid_size" : 1,
        "num_levels" : 2,
        "num_decoder_layer" : 1,
        "concat_len" : 0,
        "groups" : 1,
        "kernel" : 3,
        "dropout" : 0.1,
        "single_step_output_One" : 0,
        "positionalE" : False,
        "modified" : True,
        "RIN" : False,
        "noise_mean" : 0,
        "noise_std" : 1.96
    }
    
    # DDPG configuration
    DDPG_CONF = {
        "mlp_dim" : 128
    }
    
    # SAC configuration
    SAC_CONF = {
        "mlp_dim" : 128
    }
    
    # information of range for action input difference 
    CTRL_DIFF_RANGE = {
        '\\nb11_pnb':[-0.05, 0.05],
        '\\nb12_pnb':[-0.05, 0.05],
        '\\nb13_pnb':[-0.05, 0.05],
        '\\RC01':[-0.1, 0.1], 
        '\\RC02':[-0.1, 0.1], 
        '\\RC03':[-0.1, 0.1],
        '\\VCM01':[-0.1, 0.1], 
        '\\VCM02':[-0.1, 0.1], 
        '\\VCM03':[-0.1, 0.1],
        '\\EC2_PWR':[-0.1, 0.1], 
        '\\EC3_PWR':[-0.1, 0.1], 
        '\\ECSEC2TZRTN':[-0.1, 0.1], 
        '\\ECSEC3TZRTN':[-0.1, 0.1],
        '\\LV01':[-0.5, 0.5], 
        '\\ipmhd':[-0.05, 0.05], 
        '\\kappa':[-0.1, 0.1], 
        '\\tritop':[-0.1, 0.1], 
        '\\tribot':[-0.1, 0.1], 
        '\\bcentr':[-0.05, 0.05]
    }
    
    # control target value
    DEFAULT_TARGETS = {
        "\\betan" : 2.75,
    }