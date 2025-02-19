#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## Import TVB files:
from tvb_model_reference_human.simulation_file.parameter.parameter_M_Berlin_dt01s import Parameter
parameters = Parameter()

import tvb_model_reference.src.nuu_tools_simulation_EtoI_tanghiem as tools
from tvb_model_reference.view.plot import multiview, multiview_one,multiview_one_bar, prepare_surface_regions, animation_nuu

## Import tools:
import matplotlib.pyplot as plt
import numpy as np
import json
import os 
from scipy import stats
from scipy import signal
path_curr = os.path.dirname(os.path.abspath('.'))

bvals = 0 # List of values of adaptation strength which will vary the brain state
simname = ['asynchronous','synchronous']
Iext = 0.001 # External input
coupling_strength = 0.15
ratio_coupling_EE_EI = 1.4 # ratio of E to I/E to E inter-regional couplng strengths 

## Set the parameters of the stimulus (choose stimval = 0 to simulate spontaneous activity)

stimval = 0. # Hz, stimulus strength
stimdur = 50 # ms, duration of the s

# number of seeds
Nseeds = 20

## Define a location to save the files; create the folder if necessary:

folder_root = '/Volumes/groups/menon/projects/tanghiem/2022_TVB_AdEX/' # /!\ TO CHANGE TO ADAPT ON YOUR PLATFORM /!\

save_BOLD = True
try:
    os.listdir(folder_root)
except:
    os.mkdir(folder_root)

seed_start = 0

seeds_all = np.arange(seed_start, seed_start + Nseeds)

## Run simulations (takes time!)

Qi_default = [5.] # in nS, inhib conductance, whole brain #5.

PCC_nodes = [46,47]
Pcun_nodes = [50,51]
Ang_nodes = [14,15]
Insula_nodes = [18,19] # agranular insula

stim_region = np.array(Insula_nodes) # left and right insula

nodes_Qichange = PCC_nodes # np.concatenate([PCC_nodes, Pcun_nodes, Ang_nodes]) # /!\ TO CHANGE TO CHOOSE WHICH REGION /!\
node_name_Qichange = 'PCC'# '' # if empty: all nodes  # /!\ TO CHANGE TO CHOOSE WHICH REGION /!\
for Qi in Qi_default:
    Qi_change = np.arange(4.5,5.1,0.1) #  inhibitory conductance, to change in specific region
    for Qi_change_curr in Qi_change:
        for simnum in seeds_all:
            ## Set up the simulator:
            simulator = tools.init(parameters.parameter_simulation,
                                  parameters.parameter_model,
                                  parameters.parameter_connection_between_region,
                                  parameters.parameter_coupling,
                                  parameters.parameter_integrator,
                                  parameters.parameter_monitor,
                                  my_seed = int(simnum)) # seed setting for noise generator, has to be int for json
            # Set parameters
            # adap
            parameters.parameter_model['b_e'] = bvals
            
            # local I conductance
            Qi_allreg = np.ones(simulator.number_of_nodes)*Qi
            Qi_allreg[nodes_Qichange] = Qi_change_curr
            parameters.parameter_model['Q_i'] = list(Qi_allreg)
            
            # E to I inter-regional coupling strength
            K_e = parameters.parameter_model['K_ext_e']
            K_i = 0
            type_K_e = type(K_e)
            
            parameters.parameter_model['ratio_EI_EE'] = ratio_coupling_EE_EI

            parameters.parameter_model['K_ext_i'] = type_K_e(K_i)
            print('ratio EtoI/EtoE = ', parameters.parameter_model['ratio_EI_EE'])
        
            parameters.parameter_model['external_input_ex_ex']=Iext
            parameters.parameter_model['external_input_in_ex']=Iext
        
            weight = list(np.zeros(simulator.number_of_nodes))
            for reg in stim_region:
                weight[reg] = stimval # region and stimulation strength of the region 0 
                
            # random transient length for phase randomisation
            
            add_transient = 0.#1000.*np.random.rand()
            cut_transient = 5000.0 + add_transient# ms, length of the discarded initial segment
            run_sim =  100000.0 + cut_transient# for feature attribution testing to match human fMI data length 100 s 
                
            parameters.parameter_stimulus["tau"]= stimdur # stimulus duration [ms]
            parameters.parameter_stimulus["T"]= 2000.0 # interstimulus interval [ms]
            parameters.parameter_stimulus["weights"]= weight
            parameters.parameter_stimulus["variables"]=[0] #variable to kick
            
            
            # parameters.parameter_connection_between_region["path"]=connectome_path
            # attempt use default in param file?
            parameters.parameter_coupling["parameter"]["a"]=coupling_strength
        
        
            print('b_e =', bvals,', Iext = ', Iext,
                  ', Qi = ', Qi_change_curr, 'changed in ',
                  simulator.connectivity.region_labels[nodes_Qichange[0]])
        
            # random initial conditions depending on seed
            seed_curr = simnum
            np.random.seed(seed_curr)
            
            ratemax_init = 0.001 # max initial rate, in kHz
            adapmax_init = 20.*bvals # max initial E adaptation, pA
            initE = np.random.rand()*ratemax_init
            initI = initE*4  #E/I balance
            initW = initE**2*adapmax_init/ratemax_init**2 # avoid explosion when large rate 
            # randomise initial rate and adap for phase randomisation 
            
            parameters.parameter_model['initial_condition']['E'] = [initE, initE]
            parameters.parameter_model['initial_condition']['I'] = [initI, initI]
            parameters.parameter_model['initial_condition']['W_e'] = [initW, initW]

            # one stim
            parameters.parameter_stimulus['onset'] = cut_transient + 0.5*(run_sim-cut_transient)
            # repeated stim
            parameters.parameter_stimulus['onset'] = cut_transient + parameters.parameter_stimulus["T"]/2
            stim_time = parameters.parameter_stimulus['onset']
            
            stim_region_name_l = simulator.connectivity.region_labels[stim_region]
            if len(stim_region_name_l) > 1:
                name_large_regions = []
                for reg_name_curr in stim_region_name_l:
                    idx_char = 0
                    while reg_name_curr[idx_char].islower() and \
                        idx_char < len(reg_name_curr):
                            idx_char += 1
                    name_large_regions.append(reg_name_curr[:idx_char])
                stim_region_name = '-'.join(np.unique(name_large_regions))
            else:
                stim_region_name = ''.join(stim_region_name_l)
            print(stim_region_name)
                        
            # simulation saving path
            path_save = folder_root +'_b_'+str(bvals) +'_Qi_'+node_name_Qichange+str(Qi_change_curr) \
            +'_repeated'+stim_region_name+'stim_'+str(stimval)+\
            'EtoEIratio'+str(ratio_coupling_EE_EI)+'_coupling'+str(coupling_strength)+\
            'seed'+str(seed_curr)+'_noise'+str(Iext)+'/'
            parameters.parameter_simulation['path_result'] = path_save
            
            if not os.path.exists(path_save): # new folder if not yet existing
                os.makedirs(path_save)
            np.save(path_save + 'cut_transient_seed.npy', np.array(cut_transient))
            
            # running simulations
            simulator = tools.init(parameters.parameter_simulation,
                                  parameters.parameter_model,
                                  parameters.parameter_connection_between_region,
                                  parameters.parameter_coupling,
                                  parameters.parameter_integrator,
                                  parameters.parameter_monitor,
                                  parameter_stimulation=parameters.parameter_stimulus,
                                  my_seed = int(simnum))
            if stimval != False:
                print ('    Stimulating for {1} ms, {2} nS in the {0}\n'.format(simulator.connectivity.region_labels[stim_region],parameters.parameter_stimulus['tau'],stimval))
            print(run_sim)
            tools.run_simulation(simulator,
                            run_sim,                            
                            parameters.parameter_simulation,
                            parameters.parameter_monitor)

            try:
                run_sim_transient = run_sim # ms, total simulation time 
                tinterstim = parameters.parameter_stimulus["T"]
                time_after_last_stim = (run_sim_transient - cut_transient)//tinterstim*tinterstim + cut_transient
        
                time_begin_all = np.arange(cut_transient, time_after_last_stim, tinterstim)
                Esig_alltime = []
                Isig_alltime = []
                for time_begin in time_begin_all: 
                    try:# to avoid boundary error...
                        # print('loading from', time_begin, 'to', time_begin + tinterstim, ' ms')
                        result = tools.get_result(path_save,time_begin,time_begin + tinterstim)
        
                        '''fill variables'''
                        if len(result) > 0:
                            time_s = result[0][0]*1e-3 - result[0][0][0]*1e-3 #from ms to sec
                            Esig_alltime.append(result[0][1][:,0,:]*1e3)
                            Isig_alltime.append(result[0][1][:,1,:]*1e3)
                            
                    except:
                        print(time_begin, ' not found')
                Esig = np.concatenate(np.array(Esig_alltime), axis = 0)
                Isig = np.concatenate(np.array(Isig_alltime), axis = 0)
                EIsig = 0.8*Esig + 0.2*Isig_alltime
                
                # BOLD
                if save_BOLD:
                    dt_BOLD = 1. # s
                    dt = time_s[1] - time_s[0]
                    ratio_dt = int(dt_BOLD/dt)
                    kernel_hrf = np.load('kernel_hrf_dt' + str(np.round(dt,
                                                                3)) + '.npy')
                    FR_sum = result[0][1][:,0,:]*1e3*0.8 \
                        + result[0][1][:,1,:]*1e3*0.2
                    BOLD_curr = []
                    for idx_reg in range(len(FR_sum[0])):
                        conv = signal.fftconvolve(EIsig[:,idx_reg],
                                                  kernel_hrf, mode = 'same')
                        BOLD_curr.append(conv)
                        
                    # subsample
                    BOLD_subsamp = []
                    for BOLD_reg in np.array(BOLD_curr): # loop over regions
                        conv_subsamp = np.mean(np.reshape(BOLD_reg[:len(BOLD_reg)//\
                                                                   ratio_dt*ratio_dt], 
                                (int(len(BOLD_reg)/ratio_dt), ratio_dt)), axis = 1)
                        BOLD_subsamp.append(conv_subsamp)
                    BOLD_subsamp = np.array(BOLD_subsamp)
                    
                    # save file
                    folder_save = folder_root#'feature_attribution/'  
                    path_save_BOLD = '_b_'+str(bvals) +'_Qi_'+node_name_Qichange+str(Qi_change_curr) \
                    +'_repeated'+stim_region_name+'stim_'+str(stimval)+\
                    'EtoEIratio'+str(ratio_coupling_EE_EI)+'_coupling'+str(coupling_strength)+\
                    'seed'+str(seed_curr)+'_noise'+str(Iext)+'/'
                    np.save(folder_save+ 'sig_BOLD_'+path_save_BOLD[:-1],
                            BOLD_subsamp)
            except:
                print('pass seed ', path_save, ' not found')
        