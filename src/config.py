class Config():
    ## columns for use
    # 0D parameters
    DEFAULT_COLS_0D = [
        '\\q95', '\\ipmhd', '\\kappa', 
        '\\tritop', '\\tribot','\\betap','\\betan',
        '\\li', '\\WTOT_DLM03', '\\ne_inter01',
        '\\TS_NE_CORE_AVG', '\\TS_TE_CORE_AVG'
    ]

    # diagnose parameter
    DEFAULT_COLS_DIAG = [
        '\\ne_tci01', '\\ne_tci02', '\\ne_tci03', '\\ne_tci04', '\\ne_tci05',
    ]
    
    # control parameter
    DEFAULT_COLS_CTRL = [
        '\\nb11_pnb','\\nb12_pnb','\\nb13_pnb',
        '\\RC01', '\\RC02', '\\RC03',
        '\\VCM01', '\\VCM02', '\\VCM03',
        '\\EC2_PWR', '\\EC3_PWR', 
        '\\ECSEC2TZRTN', '\\ECSEC3TZRTN',
        '\\LV01'
    ]
    
    # predictor : configuration for nn_env
    TRANSFORMER_CONF = {
        "n_layers": 2, 
        "n_heads":8, 
        "dim_feedforward" : 1024, 
        "dropout" : 0.1,        
        "RIN" : True,
        "feature_0D_dim" : 128,
        "feature_ctrl_dim": 128,
    }
    
    # controller : configuration for DDPG
    DDPG_CONF = {
        "mlp_dim" : 128
    }
    
    # control target value
    DEFAULT_TARGETS = {
        "\\betap" : 3.0,
        "\\q95" : 4.0,
    }