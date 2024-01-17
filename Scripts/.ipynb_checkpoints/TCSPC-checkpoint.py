import numpy as np
from numba import jit, float64
from readPTU_FLIM import PTUreader
import pandas as pd
import scipy.optimize
import lifefit as lf
import copy
from lmfit import Model, Parameters, create_params, fit_report, minimize, Minimizer


def load_store(fpath, norm = True, back = False):
    ptu_file  = PTUreader(fpath, print_header_data = False)
    decay = ptu_file.get_tcspc_decay()
    tau = np.linspace(0,decay.shape[0],decay.shape[0], dtype = int)*(ptu_file.head["MeasDesc_Resolution"]*1e9)
    timestep_ns = ptu_file.head["MeasDesc_Resolution"]*1e9
    
    if back:
        decay[:,1] -= decay[tau > 40, 1].mean()
        decay[:,3] -= decay[tau > 40, 3].mean()
    
    if norm:
        decay /= decay.max(axis=0)
       
    return decay, tau, timestep_ns  

def load_store_sub(fpath,start,fin, norm = True, back = False):
    ptu_file  = PTUreader(fpath, print_header_data = False)
    decay = ptu_file.get_tcspc_decay_sub(start,fin)
    tau = np.linspace(0,decay.shape[0],decay.shape[0], dtype = int)*(ptu_file.head["MeasDesc_Resolution"]*1e9)
    timestep_ns = ptu_file.head["MeasDesc_Resolution"]*1e9
    
    if back:
        decay[:,1] -= decay[tau > 40, 1].mean()
        decay[:,3] -= decay[tau > 40, 3].mean()
    
    if norm:
        decay /= decay.max(axis=0)
       
    return decay, tau, timestep_ns 

def loader(df,num,IRF_path,IRF_min,IRF_max,low_cut,high_cut,sub=False,sub_low = 5e5, sub_high = 10e5):
    """
    Load and process TCSPC decay and IRF data from PTU Files for use in subsequent fitting

    Parameters
    ----------
    df : data_frame
         dataframe with "paths" column that contains location of PTU files
    num : int
          index of file in df
    IRF_path : str
               path of IRF. If None it will default to IRF for main experiment
    IRF_min : int
              channel to start IRF trim   
    IRF_max : int
              channel to end IRF trim
    low_cut : int
              number of channels to exclude at the beginning of TCSPC window
    high_cut : int
              number of channels to exclude at the end  of TCSPC window
    sub : bool
          Decide whether all (False) or a subset (True) of photons to be used. Subset should result in the lifetime decay to peak around 10^4 photons
    sub_low : float
              position of starting photon for subset
    sub_high : float
               position of ending photon for subset
    
    Returns
    -------
    data_timestep,data_standard, IRF_standard, err
    data_timestep : float
                ns per channel
    data_standard : n x 2 array 
                    contains channel number and corresponding number of counts from experiment TCSPC histogram
    IRF_standard : n x 2 array 
                   contains channel number and corresponding number of counts from IRF's TCSPC histogram             
    err : ndarray
          contains Poissonian error weights (essentially just the data with any zero values replaced by 1) 
    """
    
    
    #FIXME have IRF as input
    if IRF_path == None:
        if num > 6:
            IRF_path = "../../2023/230608_clean_ibidi_ctd.sptw/IRF_531_40_1.ptu"
        else:
            IRF_path = "../../2023/230607_clean_ibidi_fcs.sptw/IRF_quick1.ptu"

#     IRF_min = 19
#     IRF_max = 44
    
    path = df.paths[num]
    if sub:
        data_decay, data_tau, data_timestep = load_store_sub(path,int(sub_low),int(sub_high),norm = False, back = False)
    else:
        data_decay, data_tau, data_timestep = load_store(path,norm = False, back = False)
    IRF_decay, IRF_tau, IRF_timestep = load_store(IRF_path,norm = False, back = False)
  
    pts_standard = data_decay[low_cut:-high_cut].shape[0]
    data_standard = np.zeros((pts_standard,2))
    data_standard[:,0] = np.arange(low_cut,data_decay.shape[0] - high_cut)
    data_standard[:,1] = data_decay[low_cut:-high_cut,1]

    IRF_standard = np.zeros((pts_standard,2))
    IRF_standard[:,0] = np.arange(0,pts_standard)
    IRF_standard[IRF_min:IRF_max,1] = IRF_decay[IRF_min:IRF_max,1]

    # err = np.sqrt(data_standard[:,1])
    err = data_standard[:,1].copy() #The lifetime class from lifefit does have a method for weights, but they use Gaussian (1/sqrt) instead of Poiss..
    err[err<1] = 1
    # err[err<10] = 10
    
    return data_timestep,data_standard, IRF_standard, err
    
    

from lifefit.tcspc import Lifetime as life

def recon(IRF,t_ns,ns_channel,tau_1,amp_1,bck,shift):
    irf_shift = life._irf_scaleshift(IRF[:,0],IRF[:,1],shift)
    decays = []
    decays.append(life.convolution(irf_shift,life.exp_decay(t_ns,tau_1/ns_channel)))
    decays.append([1] * len(t_ns))
    A = np.array(decays).T
    return np.dot(A,np.array([amp_1,bck]))

def recon_IRF_bck(IRF,t_ns,ns_channel,tau_1,amp_1,bck,shift,IRF_bck):
    irf_shift = life._irf_scaleshift(IRF[:,0],IRF[:,1],shift)
    decays = []
    decays.append(life.convolution(irf_shift-IRF_bck,life.exp_decay(t_ns,tau_1/ns_channel)))
    decays.append([1] * len(t_ns))
    A = np.array(decays).T
    return np.dot(A,np.array([amp_1,bck]))

def recon2(IRF,t_ns,ns_channel,tau_1,amp_1,tau_2,amp_2,bck,shift):
    irf_shift = life._irf_scaleshift(IRF[:,0],IRF[:,1],shift)
    decays = []
    decays.append(life.convolution(irf_shift,life.exp_decay(t_ns,tau_1/ns_channel)))
    decays.append(life.convolution(irf_shift,life.exp_decay(t_ns,tau_2/ns_channel)))
    
    decays.append([1] * len(t_ns))
    A = np.array(decays).T
    return np.dot(A,np.array([amp_1,amp_2,bck]))

def recon3(IRF,t_ns,ns_channel,tau_1,amp_1,tau_2,amp_2,tau_3,amp_3,bck,shift):
    irf_shift = life._irf_scaleshift(IRF[:,0],IRF[:,1],shift)
    decays = []
    decays.append(life.convolution(irf_shift,life.exp_decay(t_ns,tau_1/ns_channel)))
    decays.append(life.convolution(irf_shift,life.exp_decay(t_ns,tau_2/ns_channel)))
    decays.append(life.convolution(irf_shift,life.exp_decay(t_ns,tau_3/ns_channel)))

    decays.append([1] * len(t_ns))
    A = np.array(decays).T
    return np.dot(A,np.array([amp_1,amp_2,amp_3,bck]))

def recon4(IRF,t_ns,ns_channel,tau_1,amp_1,tau_2,amp_2,tau_3,amp_3,tau_4,amp_4,bck,shift):
    irf_shift = life._irf_scaleshift(IRF[:,0],IRF[:,1],shift)
    decays = []
    decays.append(life.convolution(irf_shift,life.exp_decay(t_ns,tau_1/ns_channel)))
    decays.append(life.convolution(irf_shift,life.exp_decay(t_ns,tau_2/ns_channel)))
    decays.append(life.convolution(irf_shift,life.exp_decay(t_ns,tau_3/ns_channel)))
    decays.append(life.convolution(irf_shift,life.exp_decay(t_ns,tau_4/ns_channel)))
    decays.append([1] * len(t_ns))
    A = np.array(decays).T
    return np.dot(A,np.array([amp_1,amp_2,amp_3,amp_4,bck]))

    
def paramser(n_tau,vals_dict = None):
    params = Parameters()
    if n_tau == 1:
        params.add_many(('tau_1',0.2,True,0.01,10,None,None),
                        ('amp_1',10,True,0,1e3,None,None),
                        ('bck',1,True,1,1e3,None,None),
                        ('shift',5,True,-1000,1000,None,None))
        mod = Model(recon,independent_vars=['t_ns','IRF','ns_channel'])
    elif n_tau == 2:
        params.add_many(('tau_1',0.2,True,0.01,10,None,None),
                        ('amp_1',10,True,0,1e4,None,None),
                        ('tau_2',0.05,True,0.01,10,None,None),
                        ('amp_2',1,True,0,1e3,None,None),
                        ('bck',1,True,0,1e3,None,None),
                        ('shift',5,True,-200,200,None,None))
        mod = Model(recon2,independent_vars=['t_ns','IRF','ns_channel'])
    elif n_tau == 3:
        # params.add_many(('tau_1',0.2,True,0.1,10,None,None),
        #                 ('amp_1',10,True,1e-8,1e4,None,None),
        #                 ('tau_2',0.4,True,0.1,10,None,None),
        #                 ('amp_2',10,True,1e-8,1e4,None,None),
        #                 ('tau_3',1,True,0.1,10,None,None),
        #                 ('amp_3',0.01,True,1e-8,1e4,None,None),
        #                 ('bck',1,True,0.1,1e3,None,None),
        #                 ('shift',-70,True,-200,200,None,None))
        params.add_many(('tau_1',0.2,True,0.01,10,None,None),
                    ('amp_1',10,True,1e-8,1e4,None,None),
                    ('tau_2',0.5,True,0.1,4,None,None),
                    ('amp_2',10,True,1e-8,1e4,None,None),
                    ('tau_3',1,True,0.1,4,None,None),
                    ('amp_3',0.01,True,1e-8,1e4,None,None),
                    ('bck',1,True,0.1,1e5,None,None),
                    ('shift',-5,True,-200,200,None,None))
        mod = Model(recon3,independent_vars=['t_ns','IRF','ns_channel'])
    elif n_tau == 5:
        params.add_many(('tau_1',0.2,True,0.01,10,None,None),
                        ('amp_1',10,True,0,1e3,None,None),
                        ('bck',1,True,1,1e3,None,None),
                        ('shift',5,True,-1000,1000,None,None),
                        ('IRF_bck',2,True,0,5,None,None))
        mod = Model(recon_IRF_bck,independent_vars=['t_ns','IRF','ns_channel'])
    
    else:
        params.add_many(('tau_1',0.01,True,0.001,0.1,None,None),
                        ('amp_1',10,True,0,1e4,None,None),
                        ('tau_2',0.2,True,0.05,0.3,None,None),
                        ('amp_2',1,True,0,1e3,None,None),
                        ('tau_3',1,True,0.1,4,None,None),
                        ('amp_3',0.01,True,0,1e3,None,None),
                        ('tau_4',2.7,True,0.1,4,None,None),
                        ('amp_4',0.01,True,0,1e3,None,None),
                        ('bck',1,True,0,1e3,None,None),
                        ('shift',5,True,-200,200,None,None))
        mod = Model(recon4,independent_vars=['t_ns','IRF','ns_channel'])

    if vals_dict is not None:
        for key,val in vals_dict.items():
            params[key].value = val
    return mod,params

#TODO - finish making this preassign the size of the loop. Maybe make IRF_shift faster?
# @jit
def recon1_pre(IRF,t_ns,ns_channel,tau_1,amp_1,bck,shift):
    irf_shift = irf_scaleshift(IRF[:,0],IRF[:,1],shift)
    # irf_shift = life._irf_scaleshift(IRF[:,0],IRF[:,1],shift)
    decays= np.zeros((len(t_ns),2))
    decays[:,0] = convolution(irf_shift,exp_decay(t_ns,tau_1/ns_channel))
    decays[:,1] = np.ones(len(t_ns))
    
    return np.dot(decays,np.array([amp_1,bck]))

def recon2_pre(IRF,t_ns,ns_channel,tau_1,amp_1,tau_2,amp_2,bck,shift):
    irf_shift = irf_scaleshift(IRF[:,0],IRF[:,1],shift)
    # irf_shift = life._irf_scaleshift(IRF[:,0],IRF[:,1],shift)
    decays= np.zeros((len(t_ns),3))
    decays[:,0] = convolution(irf_shift,exp_decay(t_ns,tau_1/ns_channel))
    decays[:,1] = convolution(irf_shift,exp_decay(t_ns,tau_2/ns_channel))
    decays[:,2] = np.ones(len(t_ns))
    
    return np.dot(decays,np.array([amp_1,amp_2,bck]))

def recon3_pre(IRF,t_ns,ns_channel,tau_1,amp_1,tau_2,amp_2,tau_3,amp_3,bck,shift):
    irf_shift = irf_scaleshift(IRF[:,0],IRF[:,1],shift)
    # irf_shift = life._irf_scaleshift(IRF[:,0],IRF[:,1],shift)
    decays= np.zeros((len(t_ns),4))
    decays[:,0] = convolution(irf_shift,exp_decay(t_ns,tau_1/ns_channel))
    decays[:,1] = convolution(irf_shift,exp_decay(t_ns,tau_2/ns_channel))
    decays[:,2] = convolution(irf_shift,exp_decay(t_ns,tau_3/ns_channel))
    decays[:,3] = np.ones(len(t_ns))
    
    return np.dot(decays,np.array([amp_1,amp_2,amp_3,bck]))


def recon1_MLE(pars,IRF,t_ns,ns_channel,eps):
    vals = pars.valuesdict()
    tau_1 = vals['tau_1']
    amp_1 = vals['amp_1']
    bck = vals['bck']
    shift = vals['shift']
    #NB - eps = data with data < 0 = 1
    #TODO: can I make this faster (e.g. make log faster or predefine the decays array in recon3?)
    model = recon1_pre(IRF,t_ns,ns_channel,tau_1,amp_1,bck,shift)
    # model = recon3(IRF,t_ns,ns_channel,tau_1,amp_1,tau_2,amp_2,tau_3,amp_3,bck,shift)
    # return np.sum(eps*np.log(eps/model) - (eps-model))
    return compute_sum(eps, model)

def recon2_MLE(pars,IRF,t_ns,ns_channel,eps):
    vals = pars.valuesdict()
    tau_1 = vals['tau_1']
    amp_1 = vals['amp_1']
    tau_2 = vals['tau_2']
    amp_2 = vals['amp_2']
    bck = vals['bck']
    shift = vals['shift']
    #NB - eps = data with data < 0 = 1
    #TODO: can I make this faster (e.g. make log faster or predefine the decays array in recon3?)
    model = recon2_pre(IRF,t_ns,ns_channel,tau_1,amp_1,tau_2,amp_2,bck,shift)
    # model = recon3(IRF,t_ns,ns_channel,tau_1,amp_1,tau_2,amp_2,tau_3,amp_3,bck,shift)
    # return np.sum(eps*np.log(eps/model) - (eps-model))
    return compute_sum(eps, model)

def recon3_MLE(pars,IRF,t_ns,ns_channel,eps):
    vals = pars.valuesdict()
    tau_1 = vals['tau_1']
    amp_1 = vals['amp_1']
    tau_2 = vals['tau_2']
    amp_2 = vals['amp_2']
    tau_3 = vals['tau_3']
    amp_3 = vals['amp_3']
    bck = vals['bck']
    shift = vals['shift']
    #NB - eps = data with data < 0 = 1
    #TODO: can I make this faster (e.g. make log faster or predefine the decays array in recon3?)
    model = recon3_pre(IRF,t_ns,ns_channel,tau_1,amp_1,tau_2,amp_2,tau_3,amp_3,bck,shift)
    # model = recon3(IRF,t_ns,ns_channel,tau_1,amp_1,tau_2,amp_2,tau_3,amp_3,bck,shift)
    # return np.sum(eps*np.log(eps/model) - (eps-model))
    return compute_sum(eps, model)
    # return np.sum(eps*np.log(eps/model))

    
def recon4_MLE(pars,IRF,t_ns,ns_channel,eps):
    vals = pars.valuesdict()
    tau_1 = vals['tau_1']
    amp_1 = vals['amp_1']
    tau_2 = vals['tau_2']
    amp_2 = vals['amp_2']
    tau_3 = vals['tau_3']
    amp_3 = vals['amp_3']
    tau_4 = vals['tau_4']
    amp_4 = vals['amp_4']
    bck = vals['bck']
    shift = vals['shift']
    #NB - eps = data with data < 0 = 1
    #TODO: can I make this faster (e.g. make log faster or predefine the decays array in recon3?)
   
    model = recon4_pre(IRF,t_ns,ns_channel,tau_1,amp_1,tau_2,amp_2,tau_3,amp_3,tau_4,amp_4,bck,shift)
    # return np.sum(eps*np.log(eps/model) - (eps-model))
    return compute_sum(eps, model)
    # return np.sum(eps*np.log(eps/model))
    
def recon4_pre(IRF,t_ns,ns_channel,tau_1,amp_1,tau_2,amp_2,tau_3,amp_3,tau_4,amp_4,bck,shift):
    irf_shift = irf_scaleshift(IRF[:,0],IRF[:,1],shift)
    # irf_shift = life._irf_scaleshift(IRF[:,0],IRF[:,1],shift)
    decays= np.zeros((len(t_ns),5))
    decays[:,0] = convolution(irf_shift,exp_decay(t_ns,tau_1/ns_channel))
    decays[:,1] = convolution(irf_shift,exp_decay(t_ns,tau_2/ns_channel))
    decays[:,2] = convolution(irf_shift,exp_decay(t_ns,tau_3/ns_channel))
    decays[:,3] = convolution(irf_shift,exp_decay(t_ns,tau_4/ns_channel))
    decays[:,4] = np.ones(len(t_ns))
    
    return np.dot(decays,np.array([amp_1,amp_2,amp_3,amp_4,bck]))

def pgen(parameters, flatchain, idx=None):
    # generator for all the different parameters from a flatchain.
    
    #prevent original parameters being altered
    pars = parameters.copy()
    if idx is None:
        idx = range(np.size(flatchain, 0))
    for i in idx:
        vec = flatchain.iloc[i]
        for var_name in flatchain.columns:
            pars[var_name].value = flatchain.iloc[i][var_name]
        yield pars
        
def batcher(df,n_tau,steps,burn,IRF_path,IRF_min,IRF_max,low_cut=5,high_cut=5):
    #is_weighted is true in our case...we have 
    emcee_kws = dict(steps=steps, burn=burn, thin=100, is_weighted=True,
                     progress=True)
    res_store = []
    emcee_results = []
    dat = []

    for num in range(df.shape[0]):
        data_timestep,data_standard, IRF_standard, err = loader(df,num,IRF_path,IRF_min,IRF_max,low_cut,high_cut)
        dat.append((data_timestep,data_standard,IRF_standard))
        #This could be prettier (currently returning non-optimised model)
        mod,params = paramser(n_tau)
        
#         result = mod.fit(data_standard[:,1],t_ns = data_standard[:,0],IRF = IRF_standard,
#                          ns_channel=data_timestep, params=params,nan_policy='omit',
#                          weights=1/err,method='differential evolution')
        
        result = mod.fit(data_standard[:,1],t_ns = data_standard[:,0],IRF = IRF_standard,
                         ns_channel=data_timestep, params=params,nan_policy='omit',
                         weights=1/err)
        
        result_emcee = mod.fit(data_standard[:,1],t_ns = data_standard[:,0],IRF = IRF_standard,
                               ns_channel=data_timestep, params=result.params.copy(),
                               nan_policy='omit',weights=1/err,method='emcee',fit_kws=emcee_kws)
        
     
        vals = [result_emcee.params[name].value for name in result_emcee.var_names]        
        errs = [result_emcee.params[name].stderr for name in result_emcee.var_names]
       
      
        res_store.append([df.pickle_name[num]]+[val for pair in zip(vals, errs) for val in pair] + [result_emcee.redchi])
        emcee_results.append(copy.deepcopy(result_emcee))
        
        curve = pd.DataFrame(np.array([data_standard[:,0]*data_timestep,result.data,result.best_fit]).T,columns=['t_ns','data','fit'])
        ex_name = df.pickle_name[num] + '_' + str(n_tau) + '.xlsx'
        curve.to_excel(ex_name)
        # name = f'{res_folder}{df.pickle_name[file]}_{typ}_{curve}'

    nam = result_emcee.var_names
    nam_std = [var + '_std' for var in nam]
    res = pd.DataFrame(res_store,columns=['name'] + [val for pair in zip(nam, nam_std) for val in pair] + ['red_chi'])
    data = pd.DataFrame(dat,columns=['timestep','exp','IRF'])
    # res.to_excel(f'{res_folder}{df.pickle_name[file]}_{typ}.xlsx')
    
    return res, emcee_results, data


def batcher_MLE(df,n_tau,steps,burn,IRF_path,IRF_min,IRF_max,low_cut=5,high_cut=5,sub=True,shift=None):
    #is_weighted is true in our case...we have 
    emcee_kws = dict(steps=steps, burn=burn, thin=100, is_weighted=True,
                     progress=True)
    res_store = []
    emcee_results = []
    dat = []

    for num in range(df.shape[0]):
        ns_channel,data, IRF, eps = loader(df,num,IRF_path,IRF_min=IRF_min,IRF_max=IRF_max,low_cut=5,high_cut=5,sub=sub)
        t_ns = data[:,0]

        dat.append((ns_channel,data,IRF))
        #This could be prettier (currently returning non-optimised model)
        mod,params = paramser(n_tau)
        if shift is not None:
            params['shift'].value= shift
        #could be prettier
        if n_tau == 1:
            out = minimize(recon1_MLE, params, args=(IRF,t_ns,ns_channel,eps,),method='nelder')
            fitter = Minimizer(recon1_MLE, params = out.params ,fcn_args=(IRF,t_ns,ns_channel,eps,))
        if n_tau == 2:
            out = minimize(recon2_MLE, params, args=(IRF,t_ns,ns_channel,eps,),method='nelder')
            fitter = Minimizer(recon2_MLE, params = out.params ,fcn_args=(IRF,t_ns,ns_channel,eps,))
        if n_tau == 3:
            out = minimize(recon3_MLE, params, args=(IRF,t_ns,ns_channel,eps,),method='nelder')
            fitter = Minimizer(recon3_MLE, params = out.params ,fcn_args=(IRF,t_ns,ns_channel,eps,))
        elif n_tau ==4:
            out = minimize(recon4_MLE, params, args=(IRF,t_ns,ns_channel,eps,),method='nelder')
            fitter = Minimizer(recon4_MLE, params = out.params ,fcn_args=(IRF,t_ns,ns_channel,eps,))
        fitter.emcee(steps=steps, burn=burn, thin=200, is_weighted=True,
                 progress=True,float_behavior='chi2')
        result_emcee = fitter.result
     
        vals = [result_emcee.params[name].value for name in result_emcee.var_names]        
        errs = [result_emcee.params[name].stderr for name in result_emcee.var_names]
       
        model = mod.eval(params=result_emcee.params,IRF=IRF,t_ns=t_ns,ns_channel=ns_channel)
        red_chi = red_chi_N1(data[:,1],model)
        res_store.append([df.pickle_name[num]]+[val for pair in zip(vals, errs) for val in pair] + [result_emcee.redchi] + [red_chi])
        emcee_results.append(copy.deepcopy(result_emcee))
        
        curve = pd.DataFrame(np.array([data[:,0]*ns_channel,data[:,1],model]).T,columns=['t_ns','data','fit'])
        ex_name = df.pickle_name[num] + '_' + str(n_tau) + '.xlsx'
        curve.to_excel(ex_name)
        # name = f'{res_folder}{df.pickle_name[file]}_{typ}_{curve}'

    nam = result_emcee.var_names
    nam_std = [var + '_std' for var in nam]
    res = pd.DataFrame(res_store,columns=['name'] + [val for pair in zip(nam, nam_std) for val in pair] + ['MLE', 'red_chi_N1'])
    data = pd.DataFrame(dat,columns=['timestep','exp','IRF'])
    # res.to_excel(f'{res_folder}{df.pickle_name[file]}_{typ}.xlsx')
    
    return res, emcee_results, data

def batcher_MLE_df(df,n_tau,steps,burn,low_cut=5,high_cut=5):
    #is_weighted is true in our case...we have 
    emcee_kws = dict(steps=steps, burn=burn, thin=100, is_weighted=True,
                     progress=True)
    res_store = []
    emcee_results = []
    dat = []

    for num in range(df.shape[0]):
        ns_channel,data, IRF, eps = loader(df,num,IRF_path=df['IRF'][num],IRF_min=df['IRF_min'][num],IRF_max=df['IRF_max'][num],low_cut=5,high_cut=5,sub=df['sub'][num])
        t_ns = data[:,0]

        dat.append((ns_channel,data,IRF))
        #This could be prettier (currently returning non-optimised model). Could also have a input paramter which is an array of dictionaries
        mod,params = paramser(n_tau)
        if df['shift'][num] is not None:
            params['shift'].value= df['shift'][num]
        #could be prettier
        if n_tau == 1:
            out = minimize(recon1_MLE, params, args=(IRF,t_ns,ns_channel,eps,),method='nelder')
            fitter = Minimizer(recon1_MLE, params = out.params ,fcn_args=(IRF,t_ns,ns_channel,eps,))
        if n_tau == 2:
            out = minimize(recon2_MLE, params, args=(IRF,t_ns,ns_channel,eps,),method='nelder')
            fitter = Minimizer(recon2_MLE, params = out.params ,fcn_args=(IRF,t_ns,ns_channel,eps,))
        if n_tau == 3:
            out = minimize(recon3_MLE, params, args=(IRF,t_ns,ns_channel,eps,),method='nelder')
            fitter = Minimizer(recon3_MLE, params = out.params ,fcn_args=(IRF,t_ns,ns_channel,eps,))
        elif n_tau ==4:
            out = minimize(recon4_MLE, params, args=(IRF,t_ns,ns_channel,eps,),method='nelder')
            fitter = Minimizer(recon4_MLE, params = out.params ,fcn_args=(IRF,t_ns,ns_channel,eps,))
        fitter.emcee(steps=steps, burn=burn, thin=200, is_weighted=True,
                 progress=True,float_behavior='chi2')
        result_emcee = fitter.result
     
        vals = [result_emcee.params[name].value for name in result_emcee.var_names]        
        errs = [result_emcee.params[name].stderr for name in result_emcee.var_names]
       
        model = mod.eval(params=result_emcee.params,IRF=IRF,t_ns=t_ns,ns_channel=ns_channel)
        red_chi = red_chi_N1(data[:,1],model)
        res_store.append([df.pickle_name[num]]+[val for pair in zip(vals, errs) for val in pair] + [result_emcee.redchi] + [red_chi])
        emcee_results.append(copy.deepcopy(result_emcee))
        
        curve = pd.DataFrame(np.array([data[:,0]*ns_channel,data[:,1],model]).T,columns=['t_ns','data','fit'])
        ex_name = df.pickle_name[num] + '_' + str(n_tau) + '.xlsx'
        curve.to_excel(ex_name)
        # name = f'{res_folder}{df.pickle_name[file]}_{typ}_{curve}'

    nam = result_emcee.var_names
    nam_std = [var + '_std' for var in nam]
    res = pd.DataFrame(res_store,columns=['name'] + [val for pair in zip(nam, nam_std) for val in pair] + ['MLE', 'red_chi_N1'])
    data = pd.DataFrame(dat,columns=['timestep','exp','IRF'])
    # res.to_excel(f'{res_folder}{df.pickle_name[file]}_{typ}.xlsx')
    
    return res, emcee_results, data


# Define a Numba-compiled function
@jit(nopython=False) #I need to double check this
def compute_sum(eps, model):
    return np.sum(eps * np.log(eps / model) - (eps - model))

def compute_sum_no_jit(eps, model):
    return np.sum(eps * np.log(eps / model) - (eps - model))


def red_chi_N1(data,model):
    data
    err = data.copy()
    err[err<1] = 1
    return np.sum((data - model)**2/err) / data.shape[0]


'''
Functions below pulled out of Lifefit module so I could try optimising them
'''


# @jit(nopython=True) ##Doesn't work because fft is not supported by numba, though probably wouldn't expect much performance gain
def convolution(irf, sgl_exp):
    """
    Compute convolution of irf with a single exponential decay

    Parameters
    ----------
    irf : array_like
          intensity counts of the instrument reponse function (experimental of Gaussian shaped)
    sgl_exp : array_like
              single-exponential decay

    Returns
    -------
    convolved : ndarray
                convoluted signal of IRF and exponential decay
    """
    exp_fft = np.fft.fft(sgl_exp)
    irf_fft = np.fft.fft(irf)
    convolved = np.real(np.fft.ifft(exp_fft * irf_fft))
    return convolved

@jit(nopython=False)
def irf_scaleshift(channel, irf, irf_shift):
    """
    Shift IRF by n-channels (n = irf_shift)

    Parameters
    ----------
    channel : array_like
              array of channel bins
    irf : array_like
          intensity counts of the instrument reponse function (experimental of Gaussian shaped)
    irf_shift : int
                shift of the IRF on the time axis (in channel units)

    Returns
    -------
    irf_shifted : array_like
                  time-shifted IRF

    References
    ----------
    .. [2] J. Enderlein, *Optics Communications* **1997**
    """
    n = len(irf)
    # adapted from tcspcfit (J. Enderlein)
    irf_shifted = (1 - irf_shift + np.floor(irf_shift)) * irf[np.fmod(np.fmod(channel - np.floor(irf_shift) - 1, n) + n, n).astype(np.int64)] + (irf_shift - np.floor(irf_shift)) * irf[np.fmod(np.fmod(channel - np.ceil(irf_shift) - 1, n) + n, n).astype(np.int64)]
    return irf_shifted


@jit(nopython=False)
def exp_decay(time, tau):
    """
    Create a single-exponential decay

    Parameters
    ----------
    time : array_like 
           time bins
    tau : float
          fluorescence lifetime

    Returns
    -------
    sgl_exp : array_like
              single-exponential decay
    """
    sgl_exp = np.exp(-time / tau)
    return sgl_exp


