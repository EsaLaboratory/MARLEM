#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Asset module

Asset objects define distributed energy resources (DERs) and loads.
Attributes include network location, phase connection and real and reactive
output power profiles over the simulation time-series. 
Flexible Asset classes have an update control method, which is called by
EnergySystem simulation methods with control references to update the output
power profiles and state variables. The update control method also implements
constraints which limit the implementation of references. 

OPLEM includes the following Asset subclasses: NondispatchableAsset for
uncontrollable loads and generation sources, StorageAsset for storage systems
and BuildingAsset for buildings with flexible heating, ventilation and air
conditioning (HVAC).
"""

# import modules
import numpy as np
import picos as pic
from scipy.linalg import toeplitz

__version__ = "1.1.0"

####### will b handy to convert from/to T/T_ems
def timescale(series, t_in, t_out):
    ##### check if len(series)>1 ? ##### Tems Tout
    T_in, T_out = int(24/t_in), int(24/t_out)
    #series_out = np.zeros((T_out,np.asarray(series).shape[1])) 
    series_out = np.zeros(T_out) 
    if t_out <= t_in:
        for t in range(T_in):
            series_out[t*int(t_in/t_out) : (t+1)*int(t_in/t_out)] = series[t]
    else:
        for t in range(T_out):
            series_out[t] = np.nanmean(series[t*int(t_out/t_in) : (t+1)*int(t_out/t_in)], axis=0)

    series_out = np.nan_to_num(series_out)
    return series_out


class Asset:
    """
    An energy resource located at a particular bus in the network
    Parameters
    ----------
    bus_id : float
        id number of the bus in the network
    dt : float
        simulation time interval duration (h)
    T : int
        number of simulation time intervals
    dt_ems : float
        optimisation time interval duration (h)
    T_ems : int
        number of optimisation time intervals
    phases : list, optional, default [0,1,2].
        [0, 1, 2] indicates 3 phase connection 
        
        Wye: [0, 1] indicates an a,b connection 
        
        Delta: [0] indicates a-b, [1] b-c, [2] c-a
    Returns
    -------
    Asset
    """
    
    def __init__(self, bus_id, dt, T, dt_ems, T_ems, phases=[0, 1, 2]):
        self.bus_id = bus_id
        self.phases = np.array(phases)
        self.dt = dt
        self.T = T
        self.dt_ems = dt_ems
        self.T_ems = T_ems

class BuildingAsset(Asset):
    """
    A building asset (use for flexibility from building HVAC)

    Parameters
    ----------
    Tmax : float
        Maximum temperature inside the building (Degree C) optimisation time scale
    Tmin : float
        Minimum temperature inside the building (Degree C) optimisation time scale
    Hmax : float
        Maximum power consumed by electrical heating (kW)
    Cmax : float
        Maximum power consumed by electrical cooling (kW)
    T0 : float
        Initial temperature inside the buidling (Degree C)
    C : float
        Thermal capacitance of building (kWh/Degree C)
    R : float
        Thermal resistance of building to outside environment(Degree C/kW)
    CoP_heating : float
        Coefficient of performance of the heat pump (N/A)
    CoP_cooling : float
        Coefficient of performance of the chiller (N/A)
    Ta : numpy.ndarray (``T_ems``,)
        Ambient temperature (Degree C) in the optimisation time scale

    bus_id : float
        id number of the bus in the network
    dt : float
        simulation time interval duration (h)
    T : int
        number of simulation time intervals
    dt_ems : float
        time interval duration (optimisation time scale) (h)
    T_ems : int
        number of time intervals (optimisation time scale)
    phases : list, default [0,1,2]
        [0, 1, 2] indicates 3 phase connection 
        
        Wye: [0, 1] indicates an a,b connection 
        
        Delta: [0] indicates a-b, [1] b-c, [2] c-a
    delta : float
        deadband (Degree C)

    alpha : float
        Coefficient of previous temperature in the temperature dynamics
        equation (N/A)
    beta : float
        Coefficient of power consumed to heat/cool the building in the
        temperature dynamics equation (Degree C/kW)
    gamma : float
        Coefficient of ambient temperature in the temperature dynamics
        equation (N/A)
    Pnet : numpy.ndarray (``T``,)
        Input real power over the simulation time series (kW)
    Pnet_ems : numpy.ndarray (``T_ems``,)
        Input real power over the optimisation time series (kW)
    Qnet : numpy.ndarray (``T``,)
        Input reactive power over the simulation time series(kVAR)
    Qnet_ems : numpy.ndarray (``T_ems``,)
        Input reactive power over the optimisation time series(kVAR)
    Tin : numpy.ndarray (``T``,)
        indoor temperature in the building over the simulation time series (Degree C)
    Tin_ems: numpy.ndarray (``T_ems``,)
        indoor temperature in the building over the optimisation time series (Degree C)

    Returns
    -------
    Building Asset  
    """
    
    def __init__(self, Tmax, Tmin, Hmax, Cmax, T0, C, R, CoP_heating, CoP_cooling, Ta, bus_id, dt,
                 T, dt_ems, T_ems, phases=[0, 1, 2]):
        Asset.__init__(self, bus_id, dt, T, dt_ems, T_ems, phases=phases)
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.Hmax = Hmax
        self.Cmax = Cmax
        self.T0 = T0
        self.C = C
        self.R = R
        self.CoP_heating = CoP_heating
        self.CoP_cooling = CoP_cooling

        self.Ta_ems = Ta
        self.Ta = timescale(self.Ta_ems, self.dt_ems, self.dt)

        self.alpha = (1 - (self.dt_ems/(R*C)))
        self.beta = (self.dt_ems/C)
        self.gamma = self.dt_ems/(R*C)
        self.delta = (self.Tmax-self.Tmin)/2

        self.alphat = (1 - (self.dt/(R*C)))
        self.betat = (self.dt/C)
        self.gammat = self.dt/(R*C)

        self.Pnet = np.zeros(self.T)  
        self.Qnet = np.zeros(self.T)  
        self.Pnet_ems = np.zeros(self.T_ems)  
        self.Qnet_ems = np.zeros(self.T_ems)  

        self.Tin = self.T0*np.ones(self.T)
        self.Tin_ems = self.T0*np.ones(self.T_ems)
        
        self.type = 'building'
        self.c_deg_lin = 0

    def update_control(self, Pnet, t0=0, enforce_const=True):
        """
        Update the indoor temperature and the hvac power (if enforce_const is set to True)

        Parameters
        ----------
        Pnet : numpy.ndarray (``T``,)
            input powers over the simulation time series (kW)
        enforce_const: bool, default True
            enforce indoor temperature limits constraints or not
        t0 : int, default=0
            starting time interval (over simulation time scale ``T``) for the update
        """
       
        ##### catch errors:
        #if len(Pnet) != (self.T or 1 or self.T_ems)

        for t in range(t0, self.T):
            self._update_control_t(Pnet[t-t0], t, enforce_const)

        
    def _update_control_t(self, Pnet_t, t, enforce_const):
        """
        Update the hvac power and indoor temperature at time interval t

        Parameters
        ----------
        Pnet_t : float
            input powers over simulation time series (kW)
        t : int
            time interval over simulation time scale
        enforce_const: bool
            True: enforce indoor temperature to be in [Tmin, Tmax]
            
            False: update temperature according to Pnet_t
        """

        t_ems = int(self.dt/self.dt_ems)
        ######
        if Pnet_t < 0:
            self.Pnet[t] = - Pnet_t
            if self.Tin[t] <= self.Tmin[int(t*t_ems)]  and enforce_const==True:
                self.Pnet[t] = 0 
            if t < self.T-1:   #(1-t_ems*(self.alpha-1))
                self.Tin[t+1] = self.alphat*self.Tin[t] - self.betat*self.CoP_cooling*self.Pnet[t] + self.gammat*self.Ta[t]                
            #else:
            #    self.Tin[0] = self.alpha*self.Tin[t] + self.beta*self.CoP_cooling*self.Pnet[t] + self.gamma*self.Ta[t]   

        elif Pnet_t >= 0:
            self.Pnet[t] = Pnet_t
            if self.Tin[t] >= self.Tmax[int(t*t_ems)] and enforce_const==True:
                self.Pnet[t] = 0
            if t < self.T-1:   #(1-t_ems*(self.alpha-1))
                self.Tin[t+1] = self.alphat*self.Tin[t] + self.betat*self.CoP_heating*self.Pnet[t] + self.gammat*self.Ta[t]                
            #else:
            #    self.Tin[0] = self.alpha*self.Tin[t] + self.beta*self.CoP_heating*self.Pnet[t] + self.gamma*self.Ta[t]

        self.Tin_ems = timescale(self.Tin, self.dt, self.dt_ems)
        self.Pnet_ems = timescale(self.Pnet, self.dt, self.dt_ems)

    def update_ems(self, Pnet_ems, t0=0, enforce_const=True):
        """
        update the power schedule according to the EMS signal

        Parameters
        -------------
        Pnet_ems : numpy.ndarray (``T_ems``,)
            the EMS schedule
        t0 : int
            the time start of the update, default=0
        enforce_const : bool, default True
            enforce indoor temperature limits constraints or not
        """

        if np.isscalar(Pnet_ems):
            for t in range(int(t0*self.dt_ems/self.dt), int((t0+1)*self.dt_ems/self.dt)):
                self._update_control_t(Pnet_ems, t, enforce_const)
        else:
            for t_ems in range(t0, len(Pnet_ems)):
                for t in range(int(t_ems*self.dt_ems/self.dt), int((t_ems+1)*self.dt_ems/self.dt)):
                    self._update_control_t(Pnet_ems[t_ems-t0], t, enforce_const)

    def update_discrete(self, action, t, enforce_const=True):
        """
        Update the power schedule with a discrete EMS signal

        Parameters
        ----------
        action : {-1,0,1}
            1=heating ON, -1=cooling ON, 0=OFF
        t : int
            time interval for the update
        enforce_const : bool, default True
            enforce operational constraints on ``Tin`` ([Tmin, Tmax]) or not
        """

        if action ==2:
            Pnet_t = self.Hmax
        elif action == 0:
            Pnet_t = -self.Cmax
        else: Pnet_t =0

        for t in range(int(t0*self.dt_ems/self.dt), int((t0+1)*self.dt_ems/self.dt)):
            self._update_control_t(Pnet_t, t, enforce_const)


    def polytope(self, t0=0):
        """
        Computes the polytope representation of the asset operational constraints in the optimisation time scale
        Ax <= b, 
        
        with x=[P_h, P_c] and P_h/c is the heating/cooling power over the optimisation horizon ``T_ems``
        
        Following [1]_

        Parameters
        ---------
        t0 : int, default 0
            starting time slot for the polytope model in optimisation time scale

        Returns
        --------
        A, b : (6 (``T_ems-t_ahead_0``), 2 (``T_ems-t_ahead_0``)), numpy.ndarray (6 (``T_ems-t_ahead_0``,)
            slope and intercept
        """

        ######  Gamma version 1, Pcool>=0 Pnet= Pheat+Pcool, P_thermal =Pheat-Pcool
        Gamma = toeplitz(self.alpha**np.arange(self.T_ems-t0), np.zeros(self.T_ems-t0))

        col1 = np.concatenate((np.identity(self.T_ems-t0), 
                               -1*np.identity(self.T_ems-t0),
                               np.zeros((self.T_ems-t0,self.T_ems-t0)), 
                                np.zeros((self.T_ems-t0,self.T_ems-t0)),  
                                self.CoP_heating*Gamma, #self.dt_ems*
                                -self.CoP_heating*Gamma #self.dt_ems*
                                ), axis=0)

        col2 = np.concatenate((np.zeros((self.T_ems-t0,self.T_ems-t0)),
                               np.zeros((self.T_ems-t0,self.T_ems-t0)), 
                               -np.identity(self.T_ems-t0), 
                               1*np.identity(self.T_ems-t0),
                               -self.CoP_cooling*Gamma,#self.dt_ems*
                               self.CoP_cooling*Gamma #self.dt_ems*
                               ), axis=0)

        A = np.concatenate((col1, col2), axis=1)

        Ta_discount =np.zeros(self.T_ems-t0)
        Ta_discount[0] = self.Ta_ems[t0] #
        for t in range(1,self.T_ems-t0): #-t0 added for receding horizon
            ### line below to compute Sigma_t=1^j alpha**(j-t)*Ta[t]
            Ta_discount[t] = np.dot(Gamma[t, :t+1],self.Ta_ems[t0:t0+t+1])   
            #Ta_discount[t] = np.dot(Gamma[t-1, :t],self.Ta_ems[t0:t0+t]) 

        T0_discount = self.alpha*self.T0*np.flip(Gamma[-1]) #T0*[alpha, ...., alpha^Tems]
        T0_discount = T0_discount[t0:] #added for receiding horizon

        max_bound =  (self.Tmax[t0:] - T0_discount - self.gamma*Ta_discount)/self.beta 
        min_bound = -(self.Tmin[t0:] - T0_discount - self.gamma*Ta_discount)/self.beta 
        b= np.concatenate((self.Hmax*np.ones(self.T_ems-t0),
                           np.zeros(self.T_ems-t0), 
                           np.zeros(self.T_ems-t0),
                           self.Cmax*np.ones(self.T_ems-t0),
                           max_bound,
                           min_bound, 
                           ), axis=0)


        ## version 0, considers full horizon only
        """
        Gamma = toeplitz(self.alpha**np.arange(self.T_ems), np.zeros(self.T_ems))

        col1 = np.concatenate((np.ones((self.T_ems,self.T_ems)), 
                               -1*np.ones((self.T_ems,self.T_ems)),
                                np.zeros((self.T_ems,self.T_ems)), 
                                np.zeros((self.T_ems,self.T_ems)),
                                self.CoP_heating*Gamma,
                                -self.CoP_heating*Gamma 
                                ), axis=0)

        col2 = np.concatenate((np.zeros((self.T_ems,self.T_ems)),
                               np.zeros((self.T_ems,self.T_ems)), 
                               np.ones((self.T_ems,self.T_ems)), 
                               -1*np.ones((self.T_ems,self.T_ems)), 
                               self.CoP_cooling*Gamma,
                               -self.CoP_cooling*Gamma
                               ), axis=0)

        A = np.concatenate((col1, col2), axis=1)

        Ta_discount = np.zeros(self.T_ems)
        #Ta_ems = timescale(self.Ta, self.dt, self.dt_ems)
        for t in range(1, self.T_ems):
            ### Sigma_t=1^j alpha**(j-t)*Ta[t]
            Ta_discount[t] = np.matmul(Gamma[t-1, :t],self.Ta_ems[:t])

        #### [alpha**j * T0 fo j in 1..T_ems]
        T0_discount = self.T0*np.flip(Gamma[-1])

        max_bound =  (self.Tmax - T0_discount - self.gamma*Ta_discount)/self.beta
        min_bound = -(self.Tmin - T0_discount - self.gamma*Ta_discount)/self.beta
        b= np.concatenate((self.Hmax*np.ones(self.T_ems),
                           np.zeros(self.T_ems), 
                           np.zeros(self.T_ems), 
                           self.Cmax*np.ones(self.T_ems), 
                           max_bound,
                           min_bound), axis=0)
        """
        return (A,b)


    def maxdemand_baseline(self):
        """
        Compute the baseline consumption of the building 
        by minimizing the peak demand
        
        Returns
        --------
        P_th : numpy.ndarray (``T_ems``,)
            thermal energy schedule
        P_el : numpy.ndarray (``T_ems``,)
            daily electrical schedule
        """ 

        prob = pic.Problem()
        P_peak = pic.RealVariable('P_peak')
        """
        P_cool = pic.RealVariable('P_cool',self.T_ems)
        P_heat = pic.RealVariable('P_heat',self.T_ems)
        T_in = pic.RealVariable('T_in',self.T_ems)

        prob.add_list_of_constraints([P_peak >= P_heat[t] + P_cool[t] for t in range(self.T_ems)])

        prob.add_constraint(P_cool >= 0 )
        prob.add_constraint(P_cool <= self.Cmax )
        prob.add_constraint(P_heat >= 0 )
        prob.add_constraint(P_heat <= self.Hmax )

        prob.add_constraint(T_in[0] == self.T0)
        prob.add_list_of_constraints([T_in[t] == self.alpha*T_in[t-1] + self.beta*(self.CoP_heating*P_heat[t-1]-self.CoP_cooling*P_cool[t-1]) + self.gamma*self.Ta_ems[t-1] for t in range(1,self.T_ems)])
        prob.add_constraint(T_in >= self.Tmin )
        prob.add_constraint(T_in <= self.Tmax )
        """
        P = pic.RealVariable('P', 2*self.T_ems)
        A,b = self.polytope()
        prob.add_constraint(A*P<=b)
        prob.add_list_of_constraints([P_peak >= P[t] + P[t+self.T_ems] for t in range(self.T_ems)])  #+ ot -: both P_c and P_h >=0 so +

        prob.set_objective('min', P_peak)
        prob.solve(solver='mosek', primals=None) #, primals=None) 
        P_th = [P.value[i]-P.value[self.T_ems+i] for i in range(self.T_ems)] 
        P_el = [P.value[i]+P.value[self.T_ems+i] for i in range(self.T_ems)] 
        return P_th, P_el

    def toup_baseline(self, toup):
        """
        Compute the baseline consumption of the building

        Parameters
        ----------
        toup : numpy.ndarray (``T_ems``,) 
            time of use price
        
        Returns
        -------
        P_th : numpy.ndarray (``T_ems``,) 
            thermal energy schedule
        P_el : numpy.ndarray (``T_ems``,)
            daily electrical schedule
        """ 
        
        prob = pic.Problem()
        """
        P_cool = pic.RealVariable('P_cool',self.T_ems)
        P_heat = pic.RealVariable('P_heat',self.T_ems)
        T_in = pic.RealVariable('T_in',self.T_ems)

        prob.add_constraint(P_cool >= 0 )
        prob.add_constraint(P_cool <= self.Cmax )
        prob.add_constraint(P_heat >= 0 )
        prob.add_constraint(P_heat <= self.Hmax )

        prob.add_constraint(T_in[0] == self.T0)
        prob.add_list_of_constraints(T_in[t] == self.alpha*T_in[t-1] + self.beta*(self.CoP_heating*P_heat[t-1]-self.CoP_cooling*P_cool[t-1]) + self.gamma*self.Ta_ems[t-1] for t in range(1,self.T_ems))
        prob.add_constraint(T_in[1:] >= self.Tmin[1:] )
        prob.add_constraint(T_in[1:] <= self.Tmax[1:] )

        prob.set_objective('min', sum((P_heat[t] + P_cool[t])*toup[t] for t in range(self.T_ems))) #
        """
        P = pic.RealVariable('P', 2*self.T_ems)
        A,b = self.polytope()
        prob.add_constraint(A*P<=b)
        prob.set_objective('min', sum((P[t] + P[t+self.T_ems])*toup[t] for t in range(self.T_ems))) 

        prob.solve(solver='mosek', primals=None)  #) 

        P_th = [P.value[i]-P.value[self.T_ems+i] for i in range(self.T_ems)] 
        P_el = [P.value[i]+P.value[self.T_ems+i] for i in range(self.T_ems)]   
        return P_th, P_el

    def flexibility(self, T_flex, flex_min=None, flex_type='up'):
        """
        Compute the flexibility that can be provided by the HVAC for the flexibility period T_flex

        Parameters
        ----------
        T_flex : numpy.ndarray
            period of flexibility [t_start, t_end]
        flex_type : {'down', 'up'}, default 'up'
            the type of flexibility to provide
            
            'up'  : to decrease consumption of the HVAC in flexibility period
            
            'down': to increase consumption of the HVAC in flexibility period

        Returns
        -------
        Flex : float
            the flexibility 
        """

        prob = pic.Problem()
        Flex = pic.RealVariable('flex', 1)
        """
        P_cool = pic.RealVariable('P_cool',self.T_ems)
        P_heat = pic.RealVariable('P_heat',self.T_ems)
        T_in = pic.RealVariable('T_in',self.T_ems)
        #Soft_min = pic.RealVariable('Soft_min',self.T_ems)
        #Soft_max = pic.RealVariable('Soft_max',self.T_ems)

        
        prob.add_constraint(P_cool >= 0 )
        prob.add_constraint(P_cool <= self.Cmax )
        prob.add_constraint(P_heat >= 0 )
        prob.add_constraint(P_heat <= self.Hmax )

        prob.add_constraint(T_in[0] == self.T0)
        prob.add_list_of_constraints(T_in[t] == self.alpha*T_in[t-1] + self.beta*(self.CoP_heating*P_heat[t-1]-self.CoP_cooling*P_cool[t-1]) + self.gamma*self.Ta_ems[t-1] for t in range(1,self.T_ems))
        prob.add_constraint(T_in >= self.Tmin )
        prob.add_constraint(T_in <= self.Tmax )
        #prob.add_list_of_constraints([T_in[t] + Soft_min[t] >= Tmin for t in range(Th)])
        #prob.add_list_of_constraints([T_in[t] - Soft_max[t] <= Tmax for t in range(Th)])
        #Penalty = 1e2

        if flex_type=='up':
            prob.add_list_of_constraints([Flex <= self.Pnet_ems[t] - (P_heat[t] + P_cool[t]) for t in T_flex])           
        else:           
            prob.add_list_of_constraints([Flex <= (P_heat[t] + P_cool[t]) - self.Pnet_ems[t] for t in T_flex]) 
        """

        P = pic.RealVariable('P', 2*self.T_ems) #P_heat[:, T_ems], P_cool[T_ems:]
        A,b = self.polytope()

        prob.add_constraint(A*P<=b)
        prob.add_constraint(Flex >= 0)
        if flex_type=='up':
            prob.add_list_of_constraints([Flex <= self.Pnet_ems[t] - (P[t] + P[t+self.T_ems]) for t in T_flex])           
        else:           
            prob.add_list_of_constraints([Flex <= (P[t] + P[t+self.T_ems]) - self.Pnet_ems[t] for t in T_flex]) 
        
        prob.set_objective('max', Flex)

        prob.solve(solver='mosek', primals=None) 

        if flex_min == None:
            return Flex.value
        else:
            return Flex.value if Flex.value >= flex_min  else 0

class StorageAsset(Asset):
    """
    A storage asset (use for batteries, EVs etc.)

    Parameters
    ----------
    Emax : numpy.ndarray (``T_ems``,)
        maximum energy levels over the time series (kWh)
    Emin : numpy.ndarray (``T_ems``,)
        minimum energy levels over the time series (kWh)
    Pmax : numpy.ndarray (``T_ems``,)
        maximum input powers over the time series (kW)
    Pmin : numpy.ndarray (``T_ems``,)
        minimum input powers over the time series (kW)
    E0 : float
        initial energy level (kWh)
    ET : float
        required terminal energy level (kWh)
    bus_id : float
        id number of the bus in the network
    dt : float
        simulation time interval duration (h)
    T : int
        number of simulation time intervals
    dt_ems : float
        optimisation time interval duration (h)
    T_ems : int
        number of optimisation time intervals 
    phases : list, optional, default [0,1,2]
        [0, 1, 2] indicates 3 phase connection 
        
        Wye: [0, 1] indicates an a,b connection 
        
        Delta: [0] indicates a-b, [1] b-c, [2] c-a
    Pmax_abs : float
        max power level (kW)
    c_deg_lin : float
        battery degradation rate with energy throughput (£/kWh)
    eff_ch : float, default 1
        charging efficiency (between 0 and 1)
    eff_opt_ch : float, default 1
        charging efficiency to be used in optimiser (between 0 and 1).
    eff_dis : float, default 1
        discharging efficiency (between 0 and 1)
    eff_opt_dis : float, default 1
        discharging efficiency to be used in optimiser (between 0 and 1).
    self_dis : float, default 1
        self discharging rate

    E : numpy.ndarray (``T``,)
        Energy profile over the simulation time series (kWh)
    E_ems : numpy.ndarray (``T_ems``,)
        Energy profile over the optimisarion time series (kWh)
    Pnet : numpy.ndarray (``T``,)
        Input real power over the simulation time series (kW)
    Pnet_ems : numpy.ndarray (``T_ems``,)
        Input real power over the optimisation time series (kW)
    Qnet : numpy.ndarray (``T``,)
        Input reactive power over the simulation time series(kVAR)
    Qnet_ems : numpy.ndarray (``T_ems``,)
        Input reactive power over the optimisation time series(kVAR)

    Returns
    -------
    Storage Asset
    """
    
    def __init__(self, Emax, Emin, Pmax, Pmin, E0, ET, bus_id, dt, T, dt_ems,
                 T_ems, phases=[0, 1, 2], Pmax_abs=None, c_deg_lin=None,
                 eff_ch=1, eff_opt_ch=1, eff_dis=1, eff_opt_dis=1, self_dis =1):
        Asset.__init__(self, bus_id, dt, T, dt_ems, T_ems, phases=phases)
        self.Emax = Emax
        self.Emin = Emin
        self.Pmax = Pmax
        self.Pmin = Pmin
        if Pmax_abs is None:
            self.Pmax_abs = max(self.Pmax)
        else:
            self.Pmax_abs = Pmax_abs
        self.E0 = E0
        self.ET = ET

        self.E = E0*np.ones(self.T)
        self.E_ems = E0*np.ones(self.T_ems) 

        self.Pnet = np.zeros(self.T)
        self.Qnet = np.zeros(self.T)

        self.Pnet_ems = np.zeros(self.T_ems)
        self.Qnet_ems = np.zeros(self.T_ems)

        self.c_deg_lin = c_deg_lin or 0
        self.eff_ch = eff_ch*np.ones(100)
        self.eff_dis = eff_dis*np.ones(100)
        self.eff_opt_ch = eff_opt_ch
        self.eff_opt_dis = eff_opt_dis 
        self.self_dis = self_dis
        self.type = 'storage'

    def update_control(self, Pnet, t0=0, enforce_const=True):
        """
        Update the energy profile and the storage system power based on the 'Pnet' signal

        Parameters
        ----------
        Pnet : float or numpy.ndarray
            input powers over the simulation time series (kW)
        enforce_const: bool, default True
            True: enforce the operational constraints on ``E`` [Emin, Emax]
            
            False: update the energy profile based on ``Pnet``
        t0 : int, default=0
            time interval (over simulation time scale ``T``) for the update
        """

        ##### catch errors:
        #if len(Pnet) != (self.T or 1 or self.T_ems)-t0

        for t in range(t0, self.T):
            self._update_control_t(Pnet[t-t0], t, enforce_const)

    def _update_control_t(self, Pnet_t, t, enforce_const):
        """
        Update the storage system power and energy at time interval t

        Parameters
        ----------
        Pnet_t : float
            input powers over the time series (kW)
        t : int
            time interval (over simulation time scale T) for the update
        enforce_const : bool
            True: enforce the operational constraints on E [Emin, Emax]
            False: update the energy profile based on Pnet
        """

        self.Pnet[t] = Pnet_t
        t_ems = self.dt/self.dt_ems
        ######
        P_ratio = int(100*(abs(self.Pnet[t]/self.Pmax_abs)))
        P_eff_ch = self.eff_ch[P_ratio-1]
        P_eff_dis = self.eff_dis[P_ratio-1]

        if self.Pnet[t] < 0:
            if self.E[t] <= self.Emin[int(t*t_ems)] and enforce_const==True:
                self.E[t] = self.Emin[int(t*t_ems)]
                self.Pnet[t] = 0
            if t < self.T-1:
                self.E[t+1] = self.E[t] + (1/P_eff_dis)*self.Pnet[t]*self.dt
            #else:
            #    self.E[0] = self.E[t] + (1/P_eff_dis)*self.Pnet[t]*self.dt
        elif self.Pnet[t] >= 0:
            if self.E[t] >= self.Emax[int(t*t_ems)] and enforce_const==True:
                self.E[t] = self.Emax[int(t*t_ems)]
                self.Pnet[t] = 0
            if t < self.T-1:
                self.E[t+1] = self.E[t] + P_eff_ch*self.Pnet[t]*self.dt
            #else:
            #    self.E[0] = self.E[t] + P_eff_ch*self.Pnet[t]*self.dt
        self.E_ems = timescale(self.E, self.dt, self.dt_ems)
        self.Pnet_ems = timescale(self.Pnet, self.dt, self.dt_ems)

    def update_ems(self, Pnet_ems, t0=0, enforce_const=True):
        """
        Update the storage system energy at time interval t

        Parameters
        ----------
        Pnet_ems : float or numpy.ndarray
            input powers over the time series (kW)
        t : int, default=0
            first time interval for the update (in optimisation time scale)
        enforce_const: bool, default True
            True: enforce the operational constraints on E [Emin, Emax]
            
            False: update the energy profile based on Pnet 
        """

        if np.isscalar(Pnet_ems):
            for t in range(int(t0*self.dt_ems/self.dt), int((t0+1)*self.dt_ems/self.dt)):
                self._update_control_t(Pnet_ems, t, enforce_const)
        else:
            for t_ems in range(t0, len(Pnet_ems)):
                for t in range(int(t_ems*self.dt_ems/self.dt), int((t_ems+1)*self.dt_ems/self.dt)):
                    self._update_control_t(Pnet_ems[t_ems-t0], t, enforce_const)


    def update_discrete(self, action, t0, enforce_const=True):
        """
        Update the storage system energy at time interval t

        Parameters
        ----------
        action : {-1,0,1}
            1=charging, -1=discharging, 0=idle
        t : int
            time interval (over optimisation time scale ``T_ems``) for the update 
        enforce_const: bool, default True
            enforce the operational constraints on ``E`` ([Emin, Emax]) or not
        """

        Pnet = (action-1)*self.Pmax[t0]
        for t in range(t0*int(self.dt_ems/self.dt), (t0+1)*int(self.dt_ems/self.dt)):
            self._update_control_t(Pnet, t, enforce_const)


    def polytope(self, t0=0):
        """
        Computes the polytope representation of the asset operational constraints in optimisation time scale
        Ax <= b, 
        
        with x=[P_ch, P_dis] and P_(dis)ch is the (dis)charging power over the optimisation horizon T_ems,
        P_ch>=0 and P_dis<0
        
        Following [1]_

        Parameters
        ----------
        t0: int, default=0
            starting time slot for the polytpe model in optimisation time scale
            
        Returns
        --------
        A, b :  numpy.ndarray (6 (``T_ems-t_ahead_0``), 2 (``T_ems-t_ahead_0``)), numpy.ndarray (6 (``T_ems-t_ahead_0``,)
            slope and intercept
        """
        
        Gamma = toeplitz(self.self_dis**np.arange(self.T_ems-t0), np.zeros(self.T_ems-t0))

        col1 = np.concatenate((np.identity(self.T_ems-t0), 
                               -1*np.identity(self.T_ems-t0),
                                np.zeros((self.T_ems-t0,self.T_ems-t0)), 
                                np.zeros((self.T_ems-t0,self.T_ems-t0)), 
                                self.dt_ems*self.eff_opt_ch*Gamma, 
                                -self.dt_ems*self.eff_opt_ch*Gamma
                                ), axis=0)

        col2 = np.concatenate((np.zeros((self.T_ems-t0,self.T_ems-t0)),
                               np.zeros((self.T_ems-t0,self.T_ems-t0)), 
                               np.identity(self.T_ems-t0), 
                               -1*np.identity(self.T_ems-t0), 
                               self.dt_ems*(1/self.eff_opt_dis)*Gamma,
                               self.dt_ems*(-1/self.eff_opt_dis)*Gamma
                               ), axis=0)

        A = np.concatenate((col1, col2), axis=1)

        self_dis_pow = np.flip(Gamma[-1])
        self_dis_pow = self_dis_pow[:self.T_ems-t0] #[t0:]

        b= np.concatenate((self.Pmax[t0:], np.zeros(self.T_ems-t0), np.zeros(self.T_ems-t0), -self.Pmin[t0:], 
                        #self.Emax[t0:] - (self.E_ems[t0-1]*self_dis_pow
                        self.Emax[t0:] - (self.E0*self_dis_pow + self.E_ems[t0] - self.self_dis**t0*self.E0), 
                        #[self.ET- self.E_ems[t0-1]*self_dis_pow],   #ensure E[T_ems] == ET
                        self.E0*self_dis_pow[:-1] + self.E_ems[t0] - self.self_dis**t0*self.E0 -self.Emin[t0:-1],
                        #self.E_ems[t0-1]*self_dis_pow[:-1] -self.Emin[t0:-1],
                        #[self.E_ems[t0-1]*self_dis_pow[-1] -self.ET],
                        [self.E0*self_dis_pow[-1] + self.E_ems[t0] - self.self_dis**t0*self.E0 -self.ET]), axis=0)  #last ligne to ensure E[T_ems]>=E_T

        return (A,b)


    def EV_baseline(self, t_arr, T_avail, SOC_arr):
        """
        Compute the baseline consumption of an EV in the absence of flexibility

        Parameters
        ----------
        t_arr : int
            index of time slot correspnding to the plug-in of the EV
        T_avail : int
            number of time slots the EV remained plugged-in
        SOC_arr : float [0,1]
            SOC of EV at arrival 

        Returns
        --------
        P_ch : numpy.ndarray (``T_ems``,)
            daily charging schedule of EV
        """ 

        nbr_ts = min(T_avail, np.ceil(self.Emax*(1-SOC_arr)/(self.Pmax*self.dt_ems)))

        P_ch = np.zeros(self.T_ems)
        P_ch[int(t_arr): int(min(self.T_ems, t_arr+nbr_ts))] = self.Pmax
        
        return P_ch

    def EV_toup_baseline(self, t_arr, T_avail, SOC_arr, SOC_dep, toup):
        """
        Compute the baseline consumption of an EV in the absence of flexibility, response to TOUP signal

        Parameters
        ----------
        t_arr : int
            index of time slot correspnding to the plug-in of the EV
        T_avail : int
            number of time slots the EV remained plugged-in
        SOC_arr : float [0,1]
            SOC of EV at arrival 
        SOC_dep : float [0,1]
            desired SOC of EV at departure 
        toup : numpy.ndarray (``T_ems``,)
            time of use price

        Returns
        --------
        P_ch : numpy.ndarray (``T_ems``,)
            daily charging schedule of EV       
        """ 

        n = np.ceil((T_avail+t_arr)/self.T_ems) + int(not((T_avail+t_arr)%self.T_ems))
        T_horizon = int(self.T_ems*n)
        TOUP = np.tile(toup,(int(n),))

        T_not_avail = np.array([value for value in range(T_horizon) if value not in range(t_arr, t_arr + T_avail + 1)])

        prob = pic.Problem()
        P_ch = pic.RealVariable('P_ch',T_horizon)
        P_soft = pic.RealVariable('P_soft')

        prob.add_list_of_constraint([P_ch[t]  >= self.Pmin for t in range(T_horizon)])
        prob.add_list_of_constraints([P_ch[t]  <= self.Pmax for t in range(T_horizon)])
        prob.add_list_of_constraints([P_ch[t]  == 0 for t  in T_not_avail])
        prob.add_constraint(sum(P_ch[t]*self.dt_ems/self.Emax for t in range(T_horizon)) + SOC_arr >= SOC_dep)
        #prob.add_constraint(sum(Eff_EV*p_ch[t]*dt/Cmax for t in T_horizon) + SOC_arr + P_soft >= 0.9)

        #prob.set_objective('min', sum(P_ch[t]*TOUP[t] for t in range(T_horizon)) + P_soft*1e2)
        prob.set_objective('min', sum(P_ch[t]*TOUP[t] for t in range(T_horizon)))

        prob.solve(solver='mosek', primals=None) #, )
        if prob.status != 'optimal':
            print('non optimal solution for toup')
            P_ch_value = self.EV_baseline(t_arr, T_avail, SOC_arr)
            
        else: 
            P_ch_value = [P_ch.value[i] for i in range(self.T_ems)]
        
        return P_ch_value

    def EV_flexibility(self, t_arr, T_avail, SOC_arr, SOC_dep, T_flex, flex_type='up'):
        """
        Compute the flexibility that can be provided by the EV for the period T_flex  

        Parameters
        ----------
        t_arr : int
            index of time slot correspnding to the plug-in of the EV
        T_avail : int
            number of time slots the EV remained plugged-in
        SOC_arr : float [0,1]
            SOC of EV at arrival 
        SOC_dep : float [0,1]
            desired SOC of EV at departure 
        T_flex : 1d array
            period of flexibility [t_start, t_end]
        flex_type : {'down', 'up'}, default 'up'
            the type of flexibility to provide
            'up'   : to decrease consumption of the HVAC in flexibility period
            
            'down' : to increase consumption of the HVAC in flexibility period
        
        Returns
        --------
        Flex : float
            available flexibility
        """ 

        n = np.ceil((T_avail+t_arr)/self.T_ems) + int(not((T_avail+t_arr)%self.T_ems))
        T_horizon = int(self.T_ems*n)
        t_avail = np.arange(t_arr, T_avail+t_arr+1, dtype=int)
        T_not_avail = np.array([value for value in range(self.T_ems) if value not in t_avail])
        T_flex_avail =t_avail[np.isin(t_avail, T_flex)]

        #                  T_flex_avail interset t_avail = t_avail or T_flex_avail intersect T_flex   
        if np.array_equal(T_flex_avail, t_avail) or T_flex_avail==[]: #not(np.array_equal(T_flex_avail, T_flex)):
            #print('EV not available during the flex period')
            return 0  
        
        else:
            #T = t_dep - t_arr ################pb t_dep day after!
            prob = pic.Problem()
            Flex = pic.RealVariable('flex', 1)
            p_ch = pic.RealVariable('P_ch', T_horizon)
            P_soft = pic.RealVariable('P_soft')

            prob.add_constraint(Flex >= 0)
            prob.add_list_of_constraints([p_ch[t]  >= self.Pmin for t in t_avail])
            prob.add_list_of_constraints([p_ch[t]  <= self.Pmax for t in t_avail])
            prob.add_list_of_constraints([p_ch[t]  == 0 for t  in T_not_avail])
            #prob.add_constraint(sum(Eff_EV*p_ch[t]*dt/Cmax for t in T_avail) + SOC_arr == 1)
            #prob.add_constraint(sum(p_ch[t]*self.dt_ems/self.Emax for t in t_avail) + SOC_arr +P_soft >= 0.9)
            prob.add_constraint(sum(p_ch[t]*self.dt_ems/self.Emax for t in t_avail) + SOC_arr >= SOC_dep)
            
            if flex_type=='up':
                prob.add_list_of_constraints([Flex <= self.Pnet[t] - p_ch[t] for t in T_flex_avail])
            else: 
                prob.add_list_of_constraints([Flex <= p_ch[t] - self.Pnet[t] for t in T_flex_avail])
            
            #prob.set_objective('max', Flex - P_soft*1e2)  
            prob.set_objective('max', Flex)

            prob.solve(solver='mosek', primals=None)
            p_ch_val = [p_ch.value[i] for i in range(T_horizon)]    
            return Flex.value #if Flex >= min_flex else 0


class NondispatchableAsset(Asset):
    """
    A 3 phase nondispatchable asset class (use for inflexible loads,
    PVsources etc)

    Parameters
    ----------
    Pnet : numpy.ndarray (``T``,)
        uncontrolled real input powers over the simulation time series
    Qnet : numpy.ndarray (``T``,)
        uncontrolled reactive input powers over the simulation time series (kVar)
    Pnet_pred : numpy.ndarray (``T``,), default None
        predicted real input powers over the simulation time series (kW)
    Qnet_pred : numpy.ndarray (``T``,), default None
        predicted reactive input powers over the simulation time series (kVar)
    curt : bool, default False
        if the power can be curtailed or not

    Returns
    -------
    Non-dispachable Asset  
    """

    def __init__(self, Pnet, Qnet, bus_id, dt, T, dt_ems, T_ems, phases=[0, 1, 2], Pnet_pred=None, Qnet_pred=None, curt=False):
        Asset.__init__(self, bus_id, dt, T, dt_ems, T_ems, phases=phases)
        self.Pnet = Pnet
        self.Qnet = Qnet
        if Pnet_pred is not None:
            self.Pnet_pred = Pnet_pred
        else:
            self.Pnet_pred = Pnet
        if Qnet_pred is not None:
            self.Qnet_pred = Qnet_pred
        else:
            self.Qnet_pred = Qnet

        self.type = 'ND'
        self.Pnet_ems_pred = timescale(self.Pnet_pred, self.dt, self.dt_ems)
        self.Pnet_ems = timescale(self.Pnet, self.dt, self.dt_ems)

        self.Qnet_ems_pred = timescale(self.Pnet_pred, self.dt, self.dt_ems)
        self.Qnet_ems = timescale(self.Pnet, self.dt, self.dt_ems)

        self.curt = curt

    def mpc_demand(self, t0=0, q_val=False):
        """
        a power vector composed of the actual realisation of the current time step and the predicted values for the future time steps
        
        Parameters
        ---------------
        t0 : int, default=0
            first time slot of observation
        q_val: bool, default false
            returns reactive power values or not

        Returns
        ----------------
        demand: numpy.ndarray (``T_ems``,)
            power vector
        qdemand: numpy.ndarray (``T_ems``,)
            reactive power vector     
        """

        demand, qdemand = np.zeros(self.T_ems-t0), np.zeros(self.T_ems-t0)
        demand[0]=self.Pnet_ems[t0]
        demand[1:]=self.Pnet_ems_pred[t0+1:]

        qdemand[0]=self.Qnet_ems[t0]
        qdemand[1:]=self.Qnet_ems_pred[t0+1:]

        if q_val: 
            return demand, qdemand
        else:
            return demand

    def update_ems(self, curt, t0=0):
        """
        Update the schedule of the asset based on the curt signal from the EMS

        Parameters
        ----------
        curt : numpy.ndarray
            curtailed amount over the optimisation time series (kW)
        t0 : int, default=0
            start time interval for the update
        """

        if np.isscalar(curt):
            for t in range(int(t0*self.dt_ems/self.dt), int((t0+1)*self.dt_ems/self.dt)):
                self.Pnet[t] -= curt
        else:
            for t_ems in range(t0, len(curt)):
                for t in range(int(t_ems*self.dt_ems/self.dt), int((t_ems+1)*self.dt_ems/self.dt)):
                    self.Pnet[t] -= curt[t_ems-t0]

        self.Pnet_ems = timescale(self.Pnet, self.dt, self.dt_ems)

    def polytope(self, t0):
        """
        Computes the polytope representation of the asset operational constraints following the optimisation time scale
        Ax <= b, 
        
        with x=[P_in, P_out] and P_in (P_out) is  the absorbed (injected) power over the optimisation horizon T_ems
        
        Following [1]_

        Parameters
        ----------
        t0: int, default=0 
            starting time slot for the polytope model in optimisation time scale

        Returns
        --------
        A, b :  (6 (``T_ems-t_ahead_0``), 2 (``T_ems-t_ahead_0``)), numpy.ndarray (6 (``T_ems-t_ahead_0``,)
            slope and intercept
        """

        A = np.concatenate((np.identity(self.T_ems-t0), np.identity(self.T_ems-t0)), axis=1)
        if self.curt:
            b= np.concatenate(([self.Pnet_ems[t0]], self.Pnet_ems_pred[t0+1:], np.zeros(self.T_ems-t0)), axis=0)
        else: 
            b= np.concatenate(([self.Pnet_ems[t0]], self.Pnet_ems_pred[t0+1:], [-self.Pnet_ems[t0]], -self.Pnet_ems_pred[t0+1:]), axis=0)

        return (A,b)

if __name__ == "__main__":
    pass
