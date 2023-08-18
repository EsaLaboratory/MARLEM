#import modules
import os
import copy
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import importlib.resources

##import open system modules
from oplem.Network_3ph_pf import Network_3ph
import oplem.Assets as AS  #

import gym
from gym import spaces

class OPLEM_Env(gym.Env):

    def __init__(self, network_type, vpu_bus_max = 1.05,\
                vpu_bus_min = 0.95, p_line_max = 2000):

        self.network_type = network_type
        self.vpu_bus_max = vpu_bus_max
        self.vpu_bus_min = vpu_bus_min
        self.p_line_max = p_line_max

        dt_raw = 1/60
        T_raw = int(24/dt_raw) #Number of data time intervals
        dt = 60/60 #5 minute time intervals
        T = int(24/dt) #Number of intervals
        self.dt_ems = 60/60
        T_ems = int(24/self.dt_ems)
        T0 = 0 #from 8 am to 8 am
        T0_loads = 8-T0
        
        #######################################
        ### STEP 0: Load Data
        #######################################
        #### 1) wholesale data
        parent_path = importlib.resources.files('oplem').joinpath('Data')
        prices_path = parent_path.joinpath('gbelec_apxhrdaaprice.csv')
        df = pd.read_csv(prices_path)  #2017-2021 6-12 2018-20211 1-5
        df['DateAndTime'] = pd.to_datetime(df['Unnamed: 0'], format="%d/%m/%Y %H:%M")
        df = df.set_index('DateAndTime')
        df = df[df.index.year==2021]
        prices_wsm = np.asarray(df.iloc[:,-1])/1000        
        prices_wsm = np.array(prices_wsm)  #prices £/MWh
        dt_wsm, T_wsm = 1, 24
        prices_wsm_ems = np.zeros(T_ems*365) # col0 for day ahead, col1 for intraday       
        if self.dt_ems <= dt_wsm:
            for t in range(T_wsm*365):
                prices_wsm_ems[t*int(dt_wsm/self.dt_ems) : (t+1)*int(dt_wsm/self.dt_ems)] = prices_wsm[t]
        else:
            for t in range(T_ems*365):
                prices_wsm_ems[t] = np.mean(prices_wsm[t*int(self.dt_ems/dt_wsm) : (t+1)*int(self.dt_ems/dt_wsm)], axis=0)

        self.price_0 = prices_wsm_ems
        self.interval =6

        ### 2) Load Data (365*1440,30)
        ###Thomson, M., Richardson, I. (2022). One-Minute Resolution Domestic Electricity Use Data, 2008-2009. 
        #[data collection]. UK Data Service. SN: 6583, DOI: http://doi.org/10.5255/UKDA-SN-6583-1
        #https://ukerc.rl.ac.uk/DC/cgi-bin/edc_search.pl?GoButton=Detail&WantComp=11&&RELATED=1
        Loads_data_path = os.path.join(parent_path, "Loads30min.npy")  
        Load_ems = np.load(Loads_data_path)
        self.Loads_actual = np.concatenate((Load_ems, Load_ems), axis=1)
        self.Loads_actual = self.Loads_actual[:, :55]


        #### 3) PV Data
        PV_data_path = os.path.join(parent_path, "ninja_weather_pw.csv")
        weather = pd.read_csv(PV_data_path, skiprows=3)
        ghi, temp = np.asarray(weather['radiation_surface']), np.asarray(weather['temperature'])
        ####solar output
        NOCT = 45.5 #C
        Teff = 0.045 # %
        Tcell = temp + (ghi/800*(NOCT-20))
        Ppv = ghi/1000*(1 - Teff*(Tcell - 25))
        Ppv = np.asarray(np.clip(Ppv, a_min=0, a_max=None))
        dt_pv, T_pv = 1, 24
        Ppv_ems = np.zeros(T_ems*365) # col0 for day ahead, col1 for intraday       
        if self.dt_ems <= dt_pv:
            for t in range(T_pv*365):
                Ppv_ems[t*int(dt_pv/self.dt_ems) : (t+1)*int(dt_pv/self.dt_ems)] = Ppv[t]
        else:
            for t in range(T_ems*365):
                Ppv_ems[t] = np.mean(Ppv[t*int(self.dt_ems/dt_pv) : (t+1)*int(self.dt_ems/dt_pv)], axis=0)

        self.PVpu = Ppv_ems

        self.Paggregate = np.sum(self.Loads_actual, axis=1)
        
        #######################################
        ### STEP 1: Setup the network
        #######################################
        network = Network_3ph() 
        if self.network_type == 'eulv_reduced':
            network.setup_network_eulv_reduced()
        elif self.network_type == 'ieee13':
            network.setup_network_ieee13()
        else:
            network.loadDssNetwork(self.network_type)   
        # set bus voltage limits
        network.set_pf_limits(self.vpu_bus_min*network.Vslack_ph, self.vpu_bus_max*network.Vslack_ph,
                              self.p_line_max*1e3/network.Vslack_ph)
        self.N_buses = network.N_buses
        self.N_phases = network.N_phases

        #buses that contain loads
        load_buses = np.where(np.abs(network.bus_df['Pa'])+np.abs(network.bus_df['Pb'])+np.abs(network.bus_df['Pc'])>0)[0]
        load_phases = []
        N_load_bus_phases=0
        for load_bus_idx in range(len(load_buses)):
            phase_list = []
            if np.abs(network.bus_df.iloc[load_buses[load_bus_idx]]['Pa']) > 0:
                  phase_list.append(0)
            if np.abs(network.bus_df.iloc[load_buses[load_bus_idx]]['Pb']) > 0:
                  phase_list.append(1)
            if np.abs(network.bus_df.iloc[load_buses[load_bus_idx]]['Pc']) > 0:
                  phase_list.append(2)
            load_phases.append(np.array(phase_list))  
            N_load_bus_phases += len(phase_list)
        N_loads = load_buses.size

        #######################################
        ### STEP 2: setup parameters
        ######################################
        #### 1) 
        #Community PV generation parameters
        cpv_bus_names = ['226','839','169', '391', '615', '794']
        N_cpv = len(cpv_bus_names)
        #cpv_locs = [55,56,57]
        P_cpv_nom = 500 #power rating of the PV generation
        for i in range(N_cpv):
            load_buses = np.append(load_buses,np.where(network.bus_df['name']==cpv_bus_names[i])[0][0])
            load_phases.append(np.arange(self.N_phases))
            N_load_bus_phases += self.N_phases

        #######################################
        ### STEP 3: setup the assets 
        ######################################
        inflexible_load, pv_assets =[], []
     
        #55 Homes
        for i in range(N_loads):
            Pnet = self.Loads_actual[:T_ems,i]
            Qnet=Pnet*0.05
            load_i = AS.NondispatchableAsset(Pnet, Qnet, load_buses[i], dt, T, self.dt_ems, T_ems, phases=load_phases[i])
            inflexible_load.append(load_i)

        #Community based PV systems
        for i in range(N_cpv):
            Pmax_cpv_i = np.zeros(T_ems)
            Pmin_cpv_i = -self.PVpu*P_cpv_nom
            bus_id_cbs_i = load_buses[N_loads+i]
            phases_i = load_phases[N_loads+i]
            cpv_i = AS.NondispatchableAsset(Pmin_cpv_i, np.zeros(T), bus_id_cbs_i, dt, T, self.dt_ems, T_ems, phases=phases_i, curt=True)
            cpv_i.Pnet_pred = -self.PVpu*P_cpv_nom
            pv_assets.append(cpv_i)

        self.pv_assets = pv_assets
        self.inflexible_load = inflexible_load


        self.n_agents = len(self.pv_assets)
        self.action_space = [spaces.Discrete(5)]*len(self.pv_assets)       
        
        max_bound = 0
        #obs = [t, inflexible_load, PVmax, Pagg, Eagg, price] for PV
        pv_low  = np.concatenate((np.array([0,-np.inf, -P_cpv_nom]), np.array([-np.inf]*self.interval), [0], np.array([0]*self.interval)))
        pv_high = np.concatenate((np.array([T_ems/self.dt_ems, np.inf, 0]), np.array([np.inf]*self.interval), [max_bound], np.array([np.inf]*self.interval)) )       
        
        self.observation_space = [spaces.Box(low=np.float32(pv_low) , high=np.float32(pv_high) )]*len(self.pv_assets)

        self.T_ems = T_ems                    
        self.network = network
        self.load_buses = load_buses
        self.load_phases = load_phases
        self.day_index = 0
        
        self.P_cpv_nom = P_cpv_nom

    def reset(self):
        self.episode_step= int(not(self.day_index))*(self.interval-1) #int(0/self.dt_ems)
        obs_init = []
        Eagg = 0

        #self.Loads_actual = next_day schedule
        for l, load in enumerate(self.inflexible_load):
            load.Pnet_ems = self.Loads_actual[self.day_index*self.T_ems: (self.day_index+1)*self.T_ems, l]
            load.Qnet_ems = 0.05*load.Pnet_ems

        for pv in self.pv_assets:
            P = self.Loads_actual[self.day_index*self.T_ems + self.episode_step, np.where(self.load_buses==pv.bus_id)[0][0]]
            #if more than one day, consider initialising the whole schedule:
            pv.Pnet_ems = -self.PVpu[self.day_index*self.T_ems: (self.day_index+1)*self.T_ems]*self.P_cpv_nom #Ppv_home_nom 
            PV = pv.Pnet_ems[self.episode_step]
            obs_init.append([self.episode_step, P, PV])
    

        Pagg = self.Paggregate[self.day_index*self.T_ems + self.episode_step +1 - self.interval: self.day_index*self.T_ems + self.episode_step+1] \
                 - self.P_cpv_nom*len(self.pv_assets)*\
                  self.PVpu[self.day_index*self.T_ems + self.episode_step+1 - self.interval: self.day_index*self.T_ems + self.episode_step+1]
        Price = self.price_0[self.day_index*self.T_ems + self.episode_step +1 - self.interval: self.day_index*self.T_ems + self.episode_step+1]

        
        for i in range(len(obs_init)):
            #obs_init[i].extend([Pagg, Eagg])
            obs_init[i].extend(Pagg)
            obs_init[i].extend([Eagg])
            obs_init[i].extend(Price)

        return obs_init

    def step(self, actions):
        obs_n =[]
        Eagg = 0
            
        for i,pv in enumerate(self.pv_assets):
            self._pv_update(pv, actions[len(self.storage_assets)+i])
            if self.episode_step <self.T_ems-1:
                P = self.Loads_actual[self.day_index*self.T_ems + self.episode_step+1, np.where(self.load_buses==pv.bus_id)[0][0]]
                PV = pv.Pnet_ems[self.episode_step+1]
                obs_n.append([self.episode_step+1, P, PV])
                next_idx = self.day_index*self.T_ems + self.episode_step+1

            else:
                if self.day_index<364:
                    P = self.Loads_actual[(self.day_index+1)*self.T_ems, np.where(self.load_buses==pv.bus_id)[0][0]]
                    PV = -self.PVpu[(self.day_index+1)*self.T_ems]*self.P_cpv_nom #Ppv_home_nom
                    next_idx = (self.day_index+1)*self.T_ems
                else: 
                    P = self.Loads_actual[0, np.where(self.load_buses==pv.bus_id)[0][0]]
                    PV = -self.PVpu[0]*self.P_cpv_nom #Ppv_home_nom
                    next_idx = self.interval-1 #0
                #PV = pv.Pnet_ems[0]             
                obs_n.append([0, P, PV])

        Pagg = self.Paggregate[next_idx+1 - self.interval: next_idx+1] - self.P_cpv_nom*len(self.pv_assets)*self.PVpu[next_idx+1 - self.interval: next_idx+1]

        for i in range(len(obs_n)):
            obs_n[i].extend(Pagg)
            obs_n[i].extend([Eagg])
            obs_n[i].extend(self.price_0[next_idx+1 - self.interval: next_idx+1])

        reward_n, revenue, Voltgap, buses_Vpu = self._reward(actions)

        done_n = [False]*self.n_agents
        if self.episode_step == int(24/self.dt_ems - 1): 
            done_n = [True]*self.n_agents

            if self.day_index <364:
                self.day_index += 1
            else: self.day_index=0

        self.episode_step +=1

        return obs_n, reward_n, done_n, {'revenue': revenue, 'gap': Voltgap, 'buses_Vpu': buses_Vpu}

    def _pv_update(self, pv, action):
        pv.Pnet_ems[self.episode_step] = (1-(0.25*action))*pv.Pnet_ems[self.episode_step]
            

    def _reward(self, action):
        
        network_pf = copy.deepcopy(self.network)
        network_pf.clear_loads()

        for b_idx in range(len(self.load_buses)):
            bus = self.load_buses[b_idx]
            n_phases = len(self.load_phases[b_idx])
            p_sum=0
            for asset in (self.storage_assets+self.pv_assets+self.hvac_assets):
                if asset.bus_id == bus:
                    #print('Pnet_ems in PF', asset.Pnet_ems[self.episode_step])
                    p_sum += asset.Pnet_ems[self.episode_step]

            for phase in self.load_phases[b_idx]:                                         #p_pv+p_batt
               network_pf.set_load(bus, phase, (self.Loads_actual[self.day_index*self.T_ems + self.episode_step, b_idx]+p_sum)/n_phases, \
                                                self.Loads_actual[self.day_index*self.T_ems + self.episode_step, b_idx]*0.05/n_phases)

        
        network_pf.zbus_pf()

        #### Part 1 reward: DSO revenue (G_wye shape (3, 756)) 
        p0 = np.real(network_pf.S_net_res[0:3])
        revenue= - self.price_0[self.day_index*self.T_ems + self.episode_step]*np.sum(p0)*self.dt_ems

        ##### Voltage violation penalties
        buses_Vpu = np.abs(network_pf.v_net_res)/network_pf.Vslack_ph 
        Voltgap = 0
        for b, phs in zip(self.load_buses, self.load_phases):
            for ph in phs:
                Vgap = np.maximum(np.maximum(buses_Vpu[3*b + ph] - self.vpu_bus_max, 0), self.vpu_bus_min - buses_Vpu[3*b + ph])
                Voltgap+=Vgap

        reward = revenue-100*Voltgap #[(revenue-Voltgap)/self.n_agents]*self.n_agents#-10000*drop

        return reward, revenue, Voltgap, buses_Vpu
