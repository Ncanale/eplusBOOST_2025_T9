# config_file = r"./config.json"
# print('Running Utilis')
import json
Swan = None
Year = None
Material = None
#print('S: ',Swan,Year)
glob_cmap = 'jet'

def update_config(run,year,material,Swan_h):
    import Utils_v2
    import json

    Utils_v2.Swan = Swan_h
    Utils_v2.Year = year
    Utils_v2.Material = material
    print('SWAN ',Utils_v2.Swan,'YEAR ',Utils_v2.Year, material,'\n')
    config_file = "./config_"+ material + '_' + str(year)[2:] +".json" 
    with open(config_file, "r") as file:
        config_data = json.load(file)
    
    config_data['RunAllignment'] = run
    
    with open(config_file, "w") as file:
        json.dump(config_data, file, indent=4)
        
def configurator(year,material,Swan_h):
    import Utils_v2
    import json

    Utils_v2.Swan = Swan_h
    Utils_v2.Year = year
    Utils_v2.Material = material
    
    ## vvvvvv Open data config and check the offset from dizionary vvvvvv ##
    config_file = "./config_"+ material + '_' + str(year)[2:] +".json" 
    print(config_file)
    with open(config_file, "r") as f:
                dizi = json.load(f)   
    dizi
    if (dizi['offset_y2'] == 0 or dizi['offset_x2'] == 0) & year != 00:
        input("CHECK THE ALIGNMENT, OFFSETS = 0")
    else:
        # print('\noffset_x2 ', dizi['offset_x2'])
        # print('offset_y2 ', dizi['offset_y2'])
        pass
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ##
    ## vvvvvv Open Setup config and check the offset from dizionary vvvvvv ##
    setup_config_file = "./setup_config_"+ str(dizi["Facility"]) + '_' + str(year)[2:] +".json"
    with open(setup_config_file, "r") as fs:
            dizi_setup = json.load(fs)     

    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ##
    mycmap = glob_cmap
    return dizi, dizi_setup, mycmap
def analysis_function(type_analysis,r_random, r_axial,r_scan,scan_labels):
    if type_analysis == 'axial-random':
        n_runs = [r_random, r_axial]
        ana_labels = scan_labels[:2]
    elif type_analysis == 'transition':
        n_runs = r_scan
        ana_labels = scan_labels
    elif type_analysis == 'single':
        single_type = input('which kind of single analysis? (random/axial/other)')
        if single_type == 'random':
            n_runs = [r_random]
            ana_labels = single_type
        elif single_type == 'axial':
            n_runs = [r_axial]
            ana_labels = single_type
        elif single_type == 'other':
            n_runs = [int(input('insert the run number'))]
            ana_labels = 'single daq'
            print('opening run:', n_runs)
    else:
        print('Invalid type_analysis value.')
    return n_runs, ana_labels

def myGauss(x, a, mu, sigma):
    import numpy as np
    return a * np.exp(- (x-mu)**2 / (2*sigma**2))

def myGauss_line(x, a, mu, sigma,m,q):
    import numpy as np
    return a * np.exp(- (x-mu)**2 / (2*sigma**2)) + (m*x + q)

def my_landau(x, A, mu, c):
    import numpy as np
    t = (x - mu) / c
    result = np.where(t < -5.0, 0.0, A * (1.0 / (2 * np.pi)) * np.exp(-0.5 * (t + np.exp(-t))))
    return result

def projectDistZ3(x1,y1,z1,x2,y2,z2,z3):
    mx = (x2-x1)/(z2-z1)
    xProj = x1 + mx * z3
    
    my = (y2-y1)/(z2-z1)
    yProj = y1 + my * z3
    
    return (xProj, yProj)

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

def smooth(y, box_pts):
    """
    Function to smooth data
    """
    import numpy as np
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def Average(lst):
    return sum(lst) / len(lst)

def has_not_colorbar(ax):
    from matplotlib import pyplot as plt
    from matplotlib.colorbar import Colorbar
    for artist in ax.get_children():
        if isinstance(artist, Colorbar):
            return False
    return True

def plottaStereo(rot,crad,currMean,fig,ax, labels,stereotype= " ", Title= '',):
    opts = {"facecolor":'none', "lw":3}
    import numpy as np
    import sys
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle, Ellipse
    from matplotlib.colorbar import Colorbar
    # fig, ax = plt.subplots()
    # fig.set_size_inches(8, 8)
    
    # Plotto lo stereogramma
    # print(rot.shape, rot)
    # print(crad.shape, crad)
    # print(currMean.shape, currMean)
    gonioimg = ax.scatter(rot,crad,c=currMean, cmap="jet")
    
    ax.set_title(Title)
    ax.set_xlabel(r"Rot [$\mu$rad]", fontsize = 14)
    ax.set_ylabel(r"Crad [$\mu$rad]", fontsize = 14)
    
    for i, lab in enumerate(labels):
        ax.text(rot[i],crad[i],s = lab, fontsize = 10)
    ##Ottengo il massimo (o il minimo) dello scan e lo cerchio
    if (stereotype == "MAX" ):
        myXcenter = rot[np.argmax(currMean)]
        myYcenter = crad[np.argmax(currMean)]
        col = "darkred"
    elif (stereotype == "MIN"):
        myXcenter = rot[np.argmin(currMean)]
        myYcenter = crad[np.argmin(currMean)]
        col = "darkblue"
    elif (stereotype == None):
        print("I'lls show only the stereo")
    else:
        input('ERROR MALE MALE')
        sys.exit(1)
    
    if (stereotype != None):
        myXmin, myXmax = ax.get_xlim()
        myYmin, myYmax = ax.get_ylim()
        myPercent = .4
        # print(myXcenter, myYcenter)
        ##Per avere il cerchio che è veramente rotondo, indipendentemente dall'aspect ratio della figura
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        ax.add_patch(Ellipse(xy = (myXcenter, myYcenter), width = (myXmax - myXmin) * myPercent / width, height =  (myYmax - myYmin) * myPercent / height, angle=0,edgecolor = col,  **opts))
        ax.grid()

    # if has_not_colorbar(ax):
    if has_not_colorbar(ax):
        fig.colorbar(gonioimg, ax=ax)
    # fig.colorbar(gonioimg, ax = ax)
    # plt.show()

# Define a quadratic function
def quadratic(x, a, b, c):
    return a*x**2 + b*x + c
    
def quadratic_fit(x,y,err_y,fig,ax):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    # Fit the quadratic model to the data
    params, _ = curve_fit(quadratic, x, y)
    # Create fitted data points
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = quadratic(x_fit, *params)

    ax.errorbar(x,y, yerr = err_y, ls = " ", c = "k", marker = ".")
    ax.plot(x_fit, y_fit, color='red')
    return params

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
    evis =[]
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
            if Year == 00: #Simulations
                phs.append(np.array(hf['ph'])) #23
                tmis.append(np.zeros_like(hf['ph'])) #23
                evis.append(np.zeros_like(hf['ph'])) #23
                nclus.append(np.array(hf['nclu'])) 
                infos.append(np.array(hf['info']))
            else:
                infos.append(np.array(hf['xinfo']))
                info_pluss.append(np.array(hf['info_plus']))
            if Year == 2022:
                if Material == 'Diamond':
                    phs.append(np.array(hf['digiPH'])) #23
                    tmis.append(np.array(hf['digiTime'])) #23
                    evis.append(np.array(hf['ievent'])) #23
                    nclus.append(np.array(hf['nclu'])) 
                else:
                    phs.append(np.array(hf['digi_ph'])) #22
                    tmis.append(np.array(hf['digi_time'])) #22
                    evis.append(np.array(hf['Ievent'])) #22
                    nclus.append(np.array(hf['nclu'])) 
                    bases.append(np.array(hf['digi_base'])) #22
                    
            elif Year == 2023:
                phs.append(np.array(hf['digiPH'])) #23
                tmis.append(np.array(hf['digiTime'])) #23
                evis.append(np.array(hf['ievent'])) #23
                nclus.append(np.array(hf['nclu'])) 
                bases.append(np.array(hf['digiBase'])) #23
            elif Year == 2024:
                phs.append(np.array(hf['digiPH'])) #24
                bases.append(np.array(hf['digiBase'])) #24
                tmis.append(np.array(hf['digiTime'])) #24
                evis.append(np.array(hf['ievent'])) #24
                
                
    # print(np.shape(pos))
    # print(np.shape(infos))
            
    xpos = np.concatenate(pos,axis=0)
    xinfo = np.concatenate(infos,axis=0)
    ph = np.concatenate(phs,axis=0)
    tm = np.concatenate(tmis,axis=0)
    evi = np.concatenate(evis,axis=0)
    info_plus = np.concatenate(info_pluss,axis=0)
    base = np.concatenate(bases,axis=0)
    
    if Year !=2024:
        nclu = np.concatenate(nclus,axis=0)

    # print('pre purge ',np.shape(xpos))
    # print(np.shape(xinfo))

    if Year == 2022:
        if Material == "Diamond":
            switch = False 
            logic_pos = (xpos[:,:4] > -1) & (xpos[:,:4] < 15)
            logic_clu = nclu[:,:4] == 1
            logic = logic_pos & logic_clu
            logic_fin = logic.all(axis = 1)
            xpos  = xpos[logic_fin] 
            xinfo = xinfo[logic_fin]
            ph    = ph[logic_fin] 
            tm    = tm[logic_fin] 
            evi   = evi[logic_fin] 
            nclu  = nclu[logic_fin] 
            # print('post purge ', np.shape(xpos))
            if switch:
                xpos[:,2] -= dizi['offset_y2']
                xpos[:,3] -= dizi['offset_x2']
                # xpos[:,4] -= dizi['offset_y3']
                # xpos[:,5] -= dizi['offset_x3']
                # xpos[:,6] -= dizi['offset_y4']
                # xpos[:,7] -= dizi['offset_x4']
                y1 = xpos[:,0]   
                x1 = xpos[:,1]  
                y2 = xpos[:,2]  
                x2 = xpos[:,3]  
                y3 = xpos[:,4]  
                x3 = xpos[:,5]  
                y4 = -xpos[:,6]  
                x4 = xpos[:,7] 
            else:
                xpos[:,2] -= dizi['offset_x2']
                xpos[:,3] -= dizi['offset_y2']
                # xpos[:,4] -= dizi['offset_y3']
                # xpos[:,5] -= dizi['offset_x3']
                # xpos[:,6] -= dizi['offset_y4']
                # xpos[:,7] -= dizi['offset_x4']
                x1 = xpos[:,0]   
                y1 = xpos[:,1]  
                x2 = xpos[:,2]  
                y2 = xpos[:,3]  
                x3 = xpos[:,4]  
                y3 = xpos[:,5]  
                x4 = xpos[:,6]  
                y4 = -xpos[:,7] 
            Apc1 = ph[:,0]   ##Veto
            Apc2 = ph[:,1]   ##APC
            listChans0 = [2,3,4]
            listChans1 = [5,6,7]
            listChans2 = [8,9,11]
            listChans  = [2,3,4,5,6,7,8,9,11]
            Calo = np.sum(ph[:, listChans],axis = 1)
            Calo0 = np.sum(ph[:, listChans0], axis = 1) # Ch2 to 4 → Genny 1 to 3
            Calo1 = np.sum(ph[:, listChans1], axis = 1) # Ch5 to 7 → Genny 4 to 6
            Calo2 = np.sum(ph[:, listChans2], axis = 1) # Ch8 to 11 → Genny 7 to 9
            Cherry = np.zeros_like(Calo0)
            x_cry, y_cry = projectDistZ(x1,x2,y1,y2,dizi['d_1c'])
            return xpos,xinfo,ph,tm,evi,Cherry,Calo,Calo0,Calo1,Calo2,Apc1,Apc2,x1,y1,x2,y2,x3,y3,x4,y4,x_cry,y_cry 

        else: 
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
            ph_cherry1 = ph[:,6] ##22
            ph_cherry2 = ph[:,7] ##22
            listChans = [1,2]
            ph_calo_photon = np.sum(ph[:, listChans],axis = 1) # Ch1 2 → Rino 1 2
            ph_apc1 = ph[:,3]##22
            ph_apc2 = ph[:,4]##22
            y1 = xpos[:,0]  ##22 
            x1 = xpos[:,1]  ##22
            y2 = xpos[:,2]  ##22
            x2 = xpos[:,3]  ##22
            x_cry, y_cry = projectDistZ(x1,x2,y1,y2,dizi['d_1c'])
            theta_in_x = (np.arctan((x_cry-x1)/dizi['d_1c']) * 1e6) # urad
            theta_in_y = (np.arctan((y_cry-y1)/dizi['d_1c']) * 1e6) # urad
            zeros = np.zeros_like(ph_cherry1)
            return xpos,xinfo,ph,tm,evi,info_plus,\
            ph_cherry1,ph_cherry2,zeros,zeros,ph_calo_photon,zeros,zeros,ph_apc1,ph_apc2,\
            x1,y1,x2,y2,x_cry,y_cry,theta_in_x,theta_in_y
    elif Year ==2023:
##purge errors
        logic = (xpos > -1) & (xpos < 15)
        logic2 = logic.all(axis = 1)  & (info_plus[:,1]>=0)
        xpos = xpos[logic2]   

        xinfo = xinfo[logic2]
        ph = ph[logic2]
        tm = tm[logic2]
        evi = evi[logic2]
        info_plus =info_plus[logic2]
        
        # print('post purge ', np.shape(xpos))
        
        xpos[:,2] -= dizi['offset_x2']
        xpos[:,3] -= dizi['offset_y2']
        ph_cherry1 = ph[:,0] ##23
        ph_calo_photon = ph[:,1]   ##23
        ph_apc1 = ph[:,2]   ##23
        ph_apc2 = ph[:,3]   ##23
        x1 = xpos[:,0]   ##23
        y1 = xpos[:,1]   ##23
        x2 = xpos[:,2]   ##23 
        y2 = xpos[:,3]   ##23 
        x_cry, y_cry = projectDistZ(x1,x2,y1,y2,dizi['d_1c'])
        theta_in_x = (np.arctan((x_cry-x1)/dizi['d_1c']) * 1e6) # urad
        theta_in_y = (np.arctan((y_cry-y1)/dizi['d_1c']) * 1e6) # urad
        zeros = np.zeros_like(ph_cherry1)
        return xpos,xinfo,ph,tm,evi,info_plus,\
            ph_cherry1,zeros,zeros,zeros,ph_calo_photon,zeros,zeros,ph_apc1,ph_apc2,\
            x1,y1,x2,y2,x_cry,y_cry,theta_in_x,theta_in_y 
    
    elif Year == 00:     #Simulations
        logic = (xpos > -1) & (xpos < 15)
        logic2 = logic.all(axis = 1)
        xpos = xpos[logic2]   

        xinfo = xinfo[logic2]
        ph = ph[logic2]
        tm = tm[logic2]
        evi = evi[logic2]
        Apc1 = ph[:,0]   ##Simulations
        Apc2 = ph[:,1]   ##Simulations
        Calo = ph[:,2]   ##Simulations
        x1 = xpos[:,0]   ##Simulations
        y1 = xpos[:,1]   ##Simulations
        x2 = xpos[:,2]   ##Simulations 
        y2 = xpos[:,3]   ##Simulations 
        Cherry = np.zeros_like(Calo)
        x_cry, y_cry = projectDistZ(x1,x2,y1,y2,dizi['d_1c'])
        return xpos,xinfo,ph,tm,evi,Cherry,Calo,Apc1,Apc2,x1,y1,x2,y2,x_cry,y_cry 
    
    elif Year ==2024:
##purge errors
        logic = (xpos > -1) & (xpos < (0.0242*384)) 
        logic2 = logic.all(axis = 1) & (info_plus[:,1]>=0)
        xpos = xpos[logic2]   

        xinfo = xinfo[logic2]
        ph = ph[logic2]
        tm = tm[logic2]
        evi = evi[logic2]
        info_plus = info_plus[logic2]
        base = base[logic2]
    
        # print('post purge ', np.shape(xpos))
        
        xpos[:,2] -= dizi['offset_x2']
        xpos[:,3] -= dizi['offset_y2']
        
        if run> 720616:
            ph_cherry1 = ph[:,6] ##24
            ph_cherry2 = np.zeros_like(ph_cherry1) ##24
            ph_scinti_desy = ph[:,7] ##24
            ph_scinti_after_magnet= ph[:,5] ##24
            ph_apc1 = ph[:,0]   ##24
            ph_apc2 = ph[:,1]   ##24
            ph_calo_photon = ph[:,2]   ##24
            ph_calo_elect1 = ph[:,3]   ##24
            ph_calo_elect2 = ph[:,4]   ##24
        else:  
            ph_cherry1 = ph[:,1] ##24
            ph_cherry2 = ph[:,2] ##24
            ph_scinti_desy = ph[:,3] ##24
            ph_scinti_after_magnet= ph[:,4] ##24
            ph_apc1 = ph[:,8]   ##24
            ph_apc2 = ph[:,9]   ##24
            ph_calo_photon = ph[:,10]   ##24
            ph_calo_elect1 = ph[:,11]   ##24
            ph_calo_elect2 = ph[:,13]   ##24
        y1 = xpos[:,0]  ##24
        x1 = xpos[:,1]  ##24
        y2 = xpos[:,2]  ##24
        x2 = xpos[:,3]  ##24
        x_cry, y_cry = projectDistZ(x1,x2,y1,y2,dizi['d_1c'])
        theta_in_x = (np.arctan((x_cry-x1)/dizi['d_1c']) * 1e6) # urad
        theta_in_y = (np.arctan((y_cry-y1)/dizi['d_1c']) * 1e6) # urad
        return xpos,xinfo,ph,tm,evi,info_plus,\
            ph_cherry1,ph_cherry2,ph_scinti_desy,ph_scinti_after_magnet,ph_calo_photon,ph_calo_elect1,ph_calo_elect2,ph_apc1,ph_apc2,\
            x1,y1,x2,y2,x_cry,y_cry,theta_in_x,theta_in_y
    
def gaussian(x, A, x0, sigma):
    import numpy as np
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2))

def unique_with_global_threshold(arr, threshold):
    import numpy as np
    arr_sorted = np.sort(arr)
    unique_values = []

    # Initialize a temporary group with the first value
    group = [arr_sorted[0]]
    
    for value in arr_sorted[1:]:
        # Compare the new value with the current group
        if abs(value - group[0]) <= threshold:
            group.append(value)
        else:
            # If the value is too far, append the mean of the group to unique_values and start a new group
            unique_values.append(np.mean(group))
            group = [value]
    
    # Append the last group
    unique_values.append(np.mean(group))
    
    # # Example usage
    # arr = np.array([15.001, 15.002, 15.01, 15.1, 16.0, 16.05, 17.0])
    # threshold = 0.1
    # result = unique_with_global_threshold(arr, threshold)
    # print(result)
    return np.array(unique_values)