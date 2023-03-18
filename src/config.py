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
        '\\LV01' : 'LV01'
    }
    
    ## columns for use
    # 0D parameters
    '''
    DEFAULT_COLS_0D = [
        '\\q95', '\\ipmhd', '\\kappa', 
        '\\tritop', '\\tribot','\\betap','\\betan',
        '\\li', '\\WTOT_DLM03', '\\ne_inter01',
        '\\TS_NE_CORE_AVG', '\\TS_TE_CORE_AVG'
    ]
    '''
    
    DEFAULT_COLS_0D = ['\\q95','\\betap','\\betan','\\li', '\\q0']
    
    # control parameter
    '''
    DEFAULT_COLS_CTRL = [
        '\\nb11_pnb','\\nb12_pnb','\\nb13_pnb',
        '\\RC01', '\\RC02', '\\RC03',
        '\\VCM01', '\\VCM02', '\\VCM03',
        '\\EC2_PWR', '\\EC3_PWR', 
        '\\ECSEC2TZRTN', '\\ECSEC3TZRTN',
        '\\LV01'
    ]
    '''
    
    DEFAULT_COLS_CTRL = [
        '\\nb11_pnb','\\nb12_pnb','\\nb13_pnb',
        '\\RC01', '\\RC02', '\\RC03',
        '\\VCM01', '\\VCM02', '\\VCM03',
        '\\EC2_PWR', '\\EC3_PWR', 
        '\\ECSEC2TZRTN', '\\ECSEC3TZRTN',
        '\\LV01', '\\ipmhd', '\\kappa', 
        '\\tritop', '\\tribot', '\\bcentr'
    ]
    
    # predictor : configuration for nn_env
    TRANSFORMER_CONF = {
        "n_layers": 2, 
        "n_heads":8, 
        "dim_feedforward" : 1024, 
        "dropout" : 0.1,        
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
    
    # controller : configuration for DDPG
    DDPG_CONF = {
        "mlp_dim" : 128
    }
    
    SAC_CONF = {
        "mlp_dim" : 128
    }
    
    # control target value
    DEFAULT_TARGETS = {
        "\\betan" : 3.0,
        # "\\q95" : 4.0,
    }