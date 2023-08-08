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
        '\\PCPF1U':'PCPF1U', 
        '\\PCPF2U':'PCPF2U', 
        '\\PCPF3U':'PCPF3U', 
        '\\PCPF3L':'PCPF3L',
        '\\PCPF4U':'PCPF4U', 
        '\\PCPF4L':'PCPF4L', 
        '\\PCPF5U':'PCPF5U', 
        '\\PCPF5L':'PCPF5L', 
        '\\PCPF6U':'PCPF6U', 
        '\\PCPF6L':'PCPF6L',
        '\\PCPF7U':'PCPF7U'
    }
    
    ## columns for use
    # We have to consider several cases: (1) 0D parameters control (2) shape parameters control (3) multi-objects control(0D + shape for example) (4) GS-solver
    input_params = {
        "params-control":{
            "state":['\\q95','\\betan','\\betap','\\li'],
            "control":[
                # NBI
                '\\nb11_pnb',
                '\\nb12_pnb',
                '\\nb13_pnb',
                
                # EC heating
                '\\EC2_PWR', 
                '\\EC3_PWR',
                '\\EC4_PWR', 
                '\\ECSEC2TZRTN', 
                '\\ECSEC3TZRTN',
                '\\ECSEC4TZRTN',
                
                # plasma current and magnetic field
                '\\ipmhd', 
                '\\bcentr',
                
                # shape parameter
                '\\kappa', 
                '\\tritop', 
                '\\tribot', 
                '\\rsurf',
                '\\aminor',
            ],
        },
        "shape-control":{
            "state":['\\q95','\\betan','\\betap','\\li', '\\kappa', '\\tritop', '\\tribot', '\\rsurf','\\aminor'],
            "control":[
                # NBI
                '\\nb11_pnb',
                '\\nb12_pnb',
                '\\nb13_pnb',
                
                # EC heating
                '\\EC2_PWR', 
                '\\EC3_PWR',
                '\\EC4_PWR', 
                '\\ECSEC2TZRTN', 
                '\\ECSEC3TZRTN',
                '\\ECSEC4TZRTN',
                
                # plasma current and magnetic field
                '\\ipmhd', 
                '\\bcentr',
                
                # PFPC
                '\\PCPF1U', 
                '\\PCPF2U', 
                '\\PCPF3U', 
                '\\PCPF3L',
                '\\PCPF4U', 
                '\\PCPF4L', 
                '\\PCPF5U', 
                '\\PCPF5L', 
                '\\PCPF6U', 
                '\\PCPF6L',
                '\\PCPF7U'
            ]
        },
        "multi-objective":{
            "state":['\\q95','\\betan','\\betap','\\li', '\\kappa', '\\tritop', '\\tribot', '\\rsurf','\\aminor'],
            "control":[
                # NBI
                '\\nb11_pnb',
                '\\nb12_pnb',
                '\\nb13_pnb',
                
                # EC heating
                '\\EC2_PWR', 
                '\\EC3_PWR',
                '\\EC4_PWR', 
                '\\ECSEC2TZRTN', 
                '\\ECSEC3TZRTN',
                '\\ECSEC4TZRTN',
                
                # plasma current and magnetic field
                '\\ipmhd', 
                '\\bcentr',
                
                # PFPC
                '\\PCPF1U', 
                '\\PCPF2U', 
                '\\PCPF3U', 
                '\\PCPF3L',
                '\\PCPF4U', 
                '\\PCPF4L', 
                '\\PCPF5U', 
                '\\PCPF5L', 
                '\\PCPF6U', 
                '\\PCPF6L',
                '\\PCPF7U'
            ]
        },
        "GS-solver":{
            "state":['\\ipmhd', '\\q95','\\betap', '\li'],
            "control":['\PCPF1U', '\PCPF2U', '\PCPF3U', '\PCPF3L', '\PCPF4U','\PCPF4L', '\PCPF5U', '\PCPF5L', '\PCPF6U', '\PCPF6L', '\PCPF7U'],
        },
        "GS-solver-params-control":{
            "state":['\\ipmhd', '\\q95','\\betap', '\li'],
            "control":['\\kappa', '\\tritop', '\\tribot', '\\rsurf','\\aminor'],
        },
        "visualization":{
            "params-control":['\\q95','\\betan','\\betap','\\li'],
            "shape-control":['\\q95','\\betan','\\betap','\\li', '\\kappa'],
            "multi-objective":['\\q95','\\betan','\\betap','\\li', '\\kappa']
        }   
    }
    
    # model configuration
    model_config = {
        "Transformer":{
            "n_layers": 4, 
            "n_heads":8, 
            "dim_feedforward" : 1024, 
            "dropout" : 0.1,        
            "RIN" : False,
            "feature_0D_dim" : 128,
            "feature_ctrl_dim": 128,
            "noise_mean" : 0,
            "noise_std" : 1.96,
            "kernel_size" : 3,
        },
        "SCINet":{
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
        },
        "NStransformer":{
            "n_layers": 4, 
            "n_heads":8, 
            "dim_feedforward" : 1024, 
            "dropout" : 0.1,        
            "RIN" : False,
            "feature_0D_dim" : 128,
            "feature_ctrl_dim": 128,
            "noise_mean" : 0,
            "noise_std" : 1.96,
            "kernel_size" : 3,
        },
        "GS-solver":{
            "n_PFCs" : 11,
            "alpha_m" : 2,
            "alpha_n" : 1,
            "beta_m" : 2,
            "beta_n" : 1,
            "beta" : 0.102,
            "lamda" : 0.1,
            "Rc" : 1.8,
            "nx" : 65,
            "ny" : 65,
            "hidden_dim" : 128,
            "params_dim" : 4,
        },
        "GS-solver-params-control":{
            "n_PFCs" : 5,
            "alpha_m" : 2,
            "alpha_n" : 1,
            "beta_m" : 2,
            "beta_n" : 1,
            "beta" : 0.102,
            "lamda" : 0.1,
            "Rc" : 1.8,
            "nx" : 65,
            "ny" : 65,
            "hidden_dim" : 128,
            "params_dim" : 4,
        }
    }
    
    # RL algorithm configuration
    control_config = {
        "DDPG":{
            "mlp_dim" : 128
        },
        "SAC":{
            "mlp_dim" : 128
        },
        "target":{
            'params-control' : {
                "\\betan" : 2.5
            }, 
            'shape-control' : {
                "\\betan" : 2.5,
                '\\kappa' : 1.7
            }, 
            'multi-objective' : {
                "\\betan" : 2.5,
                # '\\kappa' : 1.7,
                '\\q95' : 5.2
            }
        }
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