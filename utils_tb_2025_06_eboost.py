# config_file = r"./config.json"
# print('Running Utilis')
import json
Swan = None
Year = None
Material = None
#print('S: ',Swan,Year)
glob_cmap = 'jet'

def get_theta_angles(x1, y1, x2, y2, d):
    import numpy as np
    thetaX = np.arctan((x2-x1)/d) #rad
    thetaY = np.arctan((y2-y1)/d) #rad
    return (thetaX, thetaY)

def projectDistZ(x1,x2,y1,y2,z):
    config_file = "./config_"+ str(Material) + '_' + str(Year)[2:] +".json" 

    with open(config_file, "r") as f:
            dizi = json.load(f)
        
    dizi
    mx = (x2-x1)/dizi['d_12']
    xProj = x1 + mx * z
    
    my = (y2-y1)/dizi['d_12']
    yProj = y1 + my * z
    
    return (xProj, yProj)
def configurator(year,material,Swan_h):
    import utils_tb_2025_06_eboost
    import json

    utils_tb_2025_06_eboost.Swan = Swan_h
    utils_tb_2025_06_eboost.Year = year
    utils_tb_2025_06_eboost.Material = material
    # print('SWAN ',utils_tb_2025_06_eboost.Swan,'YEAR ',utils_tb_2025_06_eboost.Year, material,'\n')

    ## Open config and correct offset dizionary ##

    config_file = "./config_"+ material + '_' + str(year)[2:] +".json" 
    # print(config_file)

    with open(config_file, "r") as f:
                dizi = json.load(f)   
    dizi
    if (dizi['offset_y2'] == 0 or dizi['offset_x2'] == 0) & year != 00:
        input("CHECK THE ALIGNMENT, OFFSETS = 0")
    else:
        # print('\noffset_x2 ', dizi['offset_x2'])
        # print('offset_y2 ', dizi['offset_y2'])
        pass
    mycmap = 'jet'
    return dizi, mycmap

def file_corrector(runs):
    import numpy as np
    import h5py
    from collections.abc import Iterable
    # print("Utils_v2 config file ./config_"+ str(Material) + '_' + str(Year)[2:] +".json")
    config_file = "./config_"+ str(Material) + '_' + str(Year)[2:] +".json"
    # print ('Swan ', Swan, config_file)

    with open(config_file, "r") as f:
            dizi = json.load(f)
        
    dizi
    if Swan:
        data_dir = dizi['data_path_Swan']
    else:
        data_dir = dizi['data_path_local']

    if not isinstance(runs, Iterable):
        runs = [runs]
    pos = []
    infos = []
    phs = []
    tmis = []
    qtot =[]
    nclus = []
    info_pluss =[]
    bases = []
    for run in runs:
        data_path = f'{data_dir}/run{run}.h5'
        # print('opening ', data_path) 
        with h5py.File(data_path, 'r', libver='latest', swmr=True) as hf:
            #print(hf.keys())
            # hf["xpos"].shape
            keys = list(hf.keys())
            #for k in hf.keys():
            #    comand = f'{k} = np.array(hf["{k}"])'
                # print(comand)
            #  exec(comand)
            pos.append(np.array(hf['xpos']))
            infos.append(np.array(hf['xinfo']))
            info_pluss.append(np.array(hf['info_plus']))
            phs.append(np.array(hf['digiPH'])) # from 24
            bases.append(np.array(hf['digiBase'])) # from 24
            tmis.append(np.array(hf['digiTime'])) # from 24
            qtot.append(np.array(hf['qtot'])) # from 24
            nclus.append(np.array(hf['nclu'])) # from 24
                
    # print(np.shape(pos))
    # print(np.shape(infos))
            
    xpos = np.concatenate(pos,axis=0)
    xinfo = np.concatenate(infos,axis=0)
    ph = np.concatenate(phs,axis=0)
    tm = np.concatenate(tmis,axis=0)
    qtot = np.concatenate(qtot,axis=0)
    info_plus = np.concatenate(info_pluss,axis=0)
    base = np.concatenate(bases,axis=0)
    
    if Year !=2024:
        nclu = np.concatenate(nclus,axis=0)

    # print('pre purge ',np.shape(xpos))
    # print(np.shape(xinfo))

##purge errors
    logic_pos = (xpos > -1) & (xpos < 15)
    logic_clu = nclu == 1
    logic = logic_pos & logic_clu
    logic2 = logic.all(axis = 1)
    
    xpos = xpos[logic2]   
    info_plus = info_plus[logic2]
    xinfo = xinfo[logic2]
    ph = ph[logic2]
    tm = tm[logic2]
    evi = evi[logic2]
    # print('post purge ', np.shape(xpos))
    
    xpos[:,2] -= dizi['offset_y2']
    xpos[:,3] -= dizi['offset_x2']
    ph_cherry1 = ph[:,0] 
    ph_cherry2 = ph[:,1] 
    ph_calo_photon = ph[:,2]
    ph_calo_desy = ph[:,3]
    ph_scinti= ph[:,4]
    y1 = xpos[:,0]   
    x1 = xpos[:,1]  
    y2 = xpos[:,2]  
    x2 = xpos[:,3]  
    x3 = xpos[:,4]  
    y3 = xpos[:,5]  
    x_cry, y_cry = projectDistZ(x1,x2,y1,y2,dizi['d_1c'])
    theta_in_x = (np.arctan((x_cry-x1)/dizi['d_1c']) * 1e6) # urad
    theta_in_y = (np.arctan((y_cry-y1)/dizi['d_1c']) * 1e6) # urad
    theta_out_x = (np.arctan((x_cry-x3)/dizi['d_c3']) * 1e6) # urad
    theta_out_y = (np.arctan((y_cry-y3)/dizi['d_c3']) * 1e6) # urad

    zeros = np.zeros_like(ph_cherry1)
    return xpos,xinfo,ph,tm,evi,info_plus,\
    ph_cherry1,ph_cherry2,ph_calo_photon,ph_calo_desy,ph_scinti,\
    x1,y1,x2,y2,x3,y3,x_cry,y_cry,theta_in_x,theta_in_y,theta_out_x,theta_out_y
