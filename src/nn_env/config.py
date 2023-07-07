class Config():
    
    ## columns for use
    # 0D parameters
    cols_efit = [
        '\\q0',
        '\\q95', 
        '\\ipmhd', 
        '\\kappa', 
        '\\tritop', 
        '\\tribot',
        '\\betap',
        '\\betan',
        '\\li', 
        '\\bcentr',
        '\\rsurf',
        '\\aminor',
    ]
    
    # control parameter
    cols_heating = [
        '\\nb11_pnb',
        '\\nb12_pnb',
        '\\nb13_pnb',
        '\\EC1_PWR', 
        '\\EC2_PWR', 
        '\\EC3_PWR',
        '\\EC4_PWR', 
        '\\ECSEC1TZRTN', 
        '\\ECSEC2TZRTN', 
        '\\ECSEC3TZRTN',
        '\\ECSEC4TZRTN',
    ]

    cols_diagnose = [
        '\\ne_inter01', 
        '\\ne_tci01', 
        '\\ne_tci02', 
        '\\ne_tci03', 
        '\\ne_tci04', 
        '\\ne_tci05',
    ]
    
    cols_control = [
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
    
    TS_TE_CORE_COLS = ['\\TS_CORE1:CORE1_TE', '\\TS_CORE2:CORE2_TE', '\\TS_CORE3:CORE3_TE', '\\TS_CORE4:CORE4_TE', '\\TS_CORE5:CORE5_TE', '\\TS_CORE6:CORE6_TE', '\\TS_CORE7:CORE7_TE', '\\TS_CORE8:CORE8_TE', '\\TS_CORE9:CORE9_TE', '\\TS_CORE10:CORE10_TE', '\\TS_CORE11:CORE11_TE', '\\TS_CORE12:CORE12_TE', '\\TS_CORE13:CORE13_TE', '\\TS_CORE14:CORE14_TE']
    TS_TE_EDGE_COLS = ['\\TS_EDGE1:EDGE1_TE', '\\TS_EDGE2:EDGE2_TE', '\\TS_EDGE3:EDGE3_TE', '\\TS_EDGE4:EDGE4_TE', '\\TS_EDGE5:EDGE5_TE', '\\TS_EDGE6:EDGE6_TE', '\\TS_EDGE7:EDGE7_TE', '\\TS_EDGE8:EDGE8_TE', '\\TS_EDGE9:EDGE9_TE', '\\TS_EDGE10:EDGE10_TE', '\\TS_EDGE11:EDGE11_TE', '\\TS_EDGE12:EDGE12_TE', '\\TS_EDGE13:EDGE13_TE', '\\TS_EDGE14:EDGE14_TE', '\\TS_EDGE15:EDGE15_TE']

    TS_NE_CORE_COLS = ['\\TS_CORE1:CORE1_NE', '\\TS_CORE2:CORE2_NE', '\\TS_CORE3:CORE3_NE', '\\TS_CORE4:CORE4_NE', '\\TS_CORE5:CORE5_NE', '\\TS_CORE6:CORE6_NE', '\\TS_CORE7:CORE7_NE', '\\TS_CORE8:CORE8_NE', '\\TS_CORE9:CORE9_NE', '\\TS_CORE10:CORE10_NE', '\\TS_CORE11:CORE11_NE', '\\TS_CORE12:CORE12_NE', '\\TS_CORE13:CORE13_NE', '\\TS_CORE14:CORE14_NE']
    TS_NE_EDGE_COLS = ['\\TS_EDGE1:EDGE1_NE', '\\TS_EDGE2:EDGE2_NE', '\\TS_EDGE3:EDGE3_NE', '\\TS_EDGE4:EDGE4_NE', '\\TS_EDGE5:EDGE5_NE', '\\TS_EDGE6:EDGE6_NE', '\\TS_EDGE7:EDGE7_NE', '\\TS_EDGE8:EDGE8_NE', '\\TS_EDGE9:EDGE9_NE', '\\TS_EDGE10:EDGE10_NE', '\\TS_EDGE11:EDGE11_NE', '\\TS_EDGE12:EDGE12_NE', '\\TS_EDGE13:EDGE13_NE', '\\TS_EDGE14:EDGE14_NE', '\\TS_EDGE15:EDGE15_NE']

    TS_AVG_COLS = ['\\TS_NE_CORE_AVG', '\\TS_NE_EDGE_AVG', '\\TS_TE_CORE_AVG','\\TS_TE_EDGE_AVG']