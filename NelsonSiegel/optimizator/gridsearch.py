import os
import logging
import pandas as pd
import numpy as np
import pickle
import time
import h5py
from scipy.optimize import minimize

from datapreparation.adaptive_sampling import creating_sample
import CONFIG
from ns_func import Z, par_yield

#checking if dask is installed
try:
    import dask.multiprocessing
    from dask import compute, delayed
    use_one_worker = False
except ImportError as e:
    use_one_worker = True

##grid search over values of tau
class grid_search():
    def __init__(self, tau_grid, 
                 Loss, beta_init, 
                 loss_args, start_date, 
                 end_date, freq, 
                 toniaDF,
                 maturities=None, 
                 clean_data = None, 
                 #thresholds = [0, 370, 1825, 3600, np.inf],
                 thresholds = False,
                 several_dates = False,
                 inertia = False,
                 num_workers = 16,
                 jobid = 0,
                 calendar = None,
                 data_path = 'deals_data',
                 need_trace = False,
                 trace_path = 'trace_path',
                 min_n_deal = CONFIG.MIN_N_DEAL,
                 outlierThresh=3.5):
        
        self.Loss = Loss
        self.beta_init = beta_init
        self.loss_args = loss_args
        self.tau_grid = tau_grid
        self.maturities = maturities
        self.results = []
        self.loss_res = {}
        self.several_dates = several_dates

        #self.thresholds = thresholds
        if thresholds == False:
            #treshold = [0, 370, 1825, 3600, np.inf]
            raise Exception('we need tresholds to be set')
        else:
            self.thresholds = thresholds
        
        if self.maturities is None:
            self.maturities = np.arange(0.0001, 30, 1 / 12) 
        
        if clean_data is not None:
            self.raw_data = clean_data
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.num_workers = num_workers
        self.tonia_df = toniaDF
        self.inertia = inertia
        self.dropped_deals = {}
        self.logger = logging.getLogger(__name__)
        
        if self.several_dates or self.inertia:
            full_range = pd.date_range(start=start_date, end=end_date, normalize=True, freq='D', closed='right')
            self.logger.debug(f'full: {full_range}')
            filterd_range = pd.DatetimeIndex(list(filter(lambda d: (calendar.index.contains(d) and calendar.loc[d].daytype=='Y') or (not calendar.index.contains(d) and d.dayofweek!=5 and d.dayofweek!=6), full_range)))
            self.logger.debug(f'filterd: {filterd_range}')
            self.settle_dates = filterd_range[-2:]
            self.start_date = self.settle_dates.min()
            self.logger.debug(f'filterd14: {self.settle_dates}, min={self.start_date}')
            
        self.previous_curve = []
        self.tasks = []
        self.data_different_dates = {}
        self.data_different_dates = {}
        self.beta_best = None
        self.update_date = None
        self.iter_dates = None
        self.best_betas = None
        self.jobid = jobid
        self.data_path = data_path
        self.need_trace = need_trace
        self.trace_path = trace_path
        self.min_n_deal = min_n_deal
        self.outlierThresh=outlierThresh
        
    #actual minimizaiton
    def minimization_del(self, tau, Loss, loss_args, beta_init, **kwargs):
        '''
        Returns an array of beta parameters that minimizes loss function given value of tau
    
        Parameters:
        -----------
            tau : value of fixed tau
            Loss: loss function, by default yield loss function
            loss_agrs : a tuple of additional arguments to loss function
            beta_init: initial  guess for beta parameters that are used 
                       as optimization starting point
            
    
        Returns:
        --------
            res_ : result of optimization - [b0, b1, b2]
    
       
        '''
        logger = logging.getLogger(__name__)
        logger.debug(f'minimization_del: tau = {tau}')
        l_args = [arg for arg in loss_args]
        l_args.append(tau)
        l_args = tuple(l_args)

        res_ = minimize(Loss, beta_init, args=l_args, **kwargs, callback=lambda xk: logger.debug(f'{xk}'))
        
        if not res_.success:
            raise Exception(res_.message)
         
        return res_
    
    def is_outlier(self, points):
        '''
        Returns a boolean array with True if points are outliers and False
        otherwise.
    
        Parameters:
        -----------
            points : An numobservations by numdimensions array of observations
    
        Returns:
        --------
            mask : A numobservations-length boolean array.
    
        References:
        ----------
            Boris Iglewicz and David Hoaglin (1993), 'Volume 16: How to Detect and
            Handle Outliers', The ASQC Basic References in Quality Control:
            Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
        '''

        self.logger.debug(f'is_outlier: modified Z-score threshold = {self.outlierThresh}')
            
        if (self.inertia) & (len(self.previous_curve)!=0):
            self.logger.debug('diff to previous curve')
            diff = np.abs(points.loc[:,'ytm']- (np.exp(par_yield(points.loc[:,'span'].values/365, self.previous_curve))-1))*100
        else:
            self.logger.debug('first filtering')
            median = np.median(points.loc[:,'ytm'])
            diff = np.abs(points.loc[:,'ytm'] - median)*100
            
        sstd = np.median(diff) # med_abs_deviation
    
        z_score = 0.6745 * diff / sstd

        return (z_score, (z_score > self.outlierThresh), sstd)
    
    #filtered data generation
    def gen_subsets(self,):
        '''
        Generate a dictionary of pandas DataFrames. DataFrames represents the
        sample used for optimization for each date
    
        Parameters:
        -----------
            tau : value of fixed tau
            Loss: loss function, by default yield loss function
            loss_agrs : a tuple of additional arguments to loss function
            beta_init: initial  guess for beta parameters that are used 
                       as optimization starting point
            
    
        Returns:
        --------
            Returns nothing, 
            updates class data field: self.data_different_dates
    
       
        '''        
        self.logger.debug('gen_subsets')

        self.tasks = []
        self.data_different_dates = {}
        
        if not self.settle_dates.size:
            self.settle_dates = pd.date_range(start=self.start_date, end=self.end_date, 
                                              normalize=True, freq=self.freq, closed='right')
		
        for settle_date in self.settle_dates:
			
            self.tasks.append(delayed(creating_sample)(settle_date, self.raw_data, min_n_deal=CONFIG.MIN_N_DEAL, 
                                                       time_window=CONFIG.TIME_WINDOW, thresholds = self.thresholds))
            self.data_different_dates[settle_date] = ''
		
        self.results = compute(*self.tasks, scheduler='processes', num_workers=self.num_workers)
        
		
        for i, settle_date in enumerate(self.settle_dates):
            ind_out=[]
            for b in self.results[i].bond_maturity_type.unique():
                bsample = self.results[i].loc[self.results[i].loc[:,'bond_maturity_type']==b]
                zscores = self.is_outlier(bsample.loc[:, ['ytm']])
                self.logger.debug(f'Z-score: {zscores[2]}')
                self.results[i].loc[self.results[i].loc[:,'bond_maturity_type']==b, 'std']=zscores[2]
                
                bind_out = bsample.loc[(zscores[1])&(bsample.loc[:,'deal_type']!=1)].index.values
                if bind_out.size!= 0:
                    ind_out.append(bind_out)
            ind_out = [item for sublist in ind_out for item in sublist]
            
            self.logger.debug(f'DF shape: {self.results[i].shape} - original')
            self.logger.debug(f'Deals dropped:\n {ind_out}')
            self.dropped_deals[settle_date] = self.results[i].loc[ind_out,:]
            self.results[i].drop(ind_out, inplace = True)
            self.logger.debug(f'DF shape: {self.results[i].shape} - adjusted')
            self.logger.debug(f'Generating sample for {settle_date:%d.%m.%Y} - Done!')
            self.data_different_dates[settle_date] = self.results[i]

        self.results = []
        
    def gen_one_date(self, settle_date):
        '''
        Generates a dictionary of pandas DataFrames. DataFrames represents the
        sample used for optimization for each date
    
        Parameters:
        -----------
            tau : value of fixed tau
            Loss: loss function, by default yield loss function
            loss_agrs : a tuple of additional arguments to loss function
            beta_init: initial  guess for beta parameters that are used 
                       as optimization starting point
            
    
        Returns:
        --------
            Returns nothing, 
            updates class data field: self.data_different_dates
    
       
        '''   
        self.logger.debug(f'gen_one_date: min_n_deal={self.min_n_deal}')

        if not hasattr(self, 'data_different_dates'):
            self.data_different_dates = {}
            
        self.data_different_dates[settle_date] = creating_sample(settle_date, 
                                                                 self.raw_data, 
                                                                 min_n_deal=self.min_n_deal, 
                                                                 time_window=CONFIG.TIME_WINDOW, 
                                                                 thresholds = self.thresholds)
        
        ind_out=[]
        for b in self.data_different_dates[settle_date].bond_maturity_type.unique().sort_values():
            bsample = self.data_different_dates[settle_date].loc[self.data_different_dates[settle_date].loc[:,'bond_maturity_type']==b]
            zscores = self.is_outlier(bsample.loc[:, ['ytm', 'span']])
            self.logger.debug(f'Z-score: {zscores[0]}, {zscores[1]}, {bsample.loc[:,"ytm"]}')
            self.data_different_dates[settle_date].loc[self.data_different_dates[settle_date].loc[:,'bond_maturity_type']==b, 'std']=zscores[2]
            bind_out = bsample.loc[(zscores[1])&(bsample.loc[:,'deal_type']!=1)].index.values

            if self.need_trace:              
                bn = f'{b}'.replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '').replace('.', '_').replace(',', '_')
                self.logger.debug(f'bond_maturity_type name: {b} -> {bn}')
                zscores[0].to_excel(os.path.join(self.trace_path, f'{self.jobid}_{settle_date:%Y%m%d}_{bn}_zscore_0.xlsx'), sheet_name='zscores0', engine='xlsxwriter')
                zscores[1].to_excel(os.path.join(self.trace_path, f'{self.jobid}_{settle_date:%Y%m%d}_{bn}_zscore_1.xlsx'), sheet_name='zscores1', engine='xlsxwriter')
            
            if bind_out.size!= 0:
                ind_out.append(bind_out)
                
        ind_out = [item for sublist in ind_out for item in sublist]
        
        self.logger.debug(f'DF shape: {self.data_different_dates[settle_date].shape} - original')
        self.logger.debug(f'Deals dropped: {ind_out}')
        self.dropped_deals[settle_date] = self.data_different_dates[settle_date].loc[ind_out,:]
        self.data_different_dates[settle_date].drop(ind_out, inplace = True)
        self.logger.debug(f'DF shape: {self.data_different_dates[settle_date].shape} - adjusted')

        self.logger.debug(f'Generating sample for {settle_date:%d.%m.%Y} - Done!')
        
        if self.need_trace and not self.dropped_deals[settle_date].empty:              
            self.dropped_deals[settle_date].to_excel(os.path.join(self.trace_path, f'{self.jobid}_{settle_date:%Y%m%d}_deals_dropped_by_zscore.xlsx'), sheet_name='dropped_deals', engine='xlsxwriter')
    
    def new_dates(self, new_end_date = None):
        
        if new_end_date == None:
            self.update_date = [self.settle_dates[-1]+1]
            self.settle_dates = self.settle_dates.union([self.settle_dates[-1]+1])
            
    
    
    def dump(self):
        
        best_betas = {}
        for date in self.settle_dates:
            idx = self.loss_res[date].loc[:, 'loss'].idxmin()
            best_betas[date] = self.loss_res[pd.to_datetime(date)].loc[idx, ['b0','b1','b2','teta']].values
        best_betas = pd.DataFrame.from_dict(best_betas, orient='index', columns = ['b0','b1','b2','teta'])
        best_betas.sort_index(inplace=True)
        
        attributes = ['several_dates', 
                      'thresholds', 
                      'start_date', 
                      'end_date', 
                      'freq', 
                      'num_workers', 
                      'inertia', 
                      'settle_dates',
                      'loss_res']
        
        params = {k:self.__getattribute__(k) for k in attributes}
                  
       
        with h5py.File('grid_data.hdf5', 'w') as f:
            g = f.create_group('curveData')
            betas = g.create_dataset('betas', data = [pickle.dumps(best_betas)])
            samples = g.create_dataset('samples', data = [pickle.dumps(self.data_different_dates)])
            dropped_deals = g.create_dataset('dropped', data = [pickle.dumps(self.dropped_deals)])
            raw_data = g.create_dataset('raw_data', data = [pickle.dumps(self.raw_data)])
            params = g.create_dataset('params', data = [pickle.dumps(params)])
            
            meta = {'save date': f'{pd.datetime.now():%Y-%m-%d %H:%M:%S}',
                    'frequency': self.freq,
                    'start_date': self.start_date,
                    'end_date':self.end_date,
                   
                    }
            g.attrs.update(meta)
        
            self.logger.debug('saving data:')
            self.logger.debug('-'*10)
            for m in g.attrs.keys():
                self.logger.debug(f'{m}: {g.attrs[m]}')
            self.logger.debug('-'*10)
                
    def load(self):
        
        with h5py.File('grid_data.hdf5', 'r') as f:
            g = f['curveData']
            self.logger.debug('loading stored data:')
            self.logger.debug('-'*10)
            for m in g.attrs.keys():
                self.logger.debug(f'{m}: {g.attrs[m]}')
            self.logger.debug('-'*10, '\n')
            best_betas = pickle.loads(g['betas'][()])
            params = pickle.loads(g['params'][()])
            samples = pickle.loads(g['samples'][()])
            dropped = pickle.loads(g['dropped'][()])
            
            
        self.previous_curve = best_betas.iloc[-1].copy()
        self.beta_init = best_betas.iloc[-1].copy()
        self.data_different_dates = samples
        self.dropped_deals = dropped
        
        # self.logger.debug('Following parameters were used:') #uncomment for diagnostics
        # self.logger.debug('-'*10) #uncomment for diagnostics
        for k,v in params.items():
            # self.logger.debug(f'{k}: {v}') #uncomment for diagnostics
            self.__dict__[k] = v
    
    #creation of loss frame grid
    def loss_grid(self, **kwargs):
        self.logger.debug('loss_grid')

        #if num_worker == 1 dask will not be used at all to avoid overhead expenses
        if self.num_workers == 1:
            self.logger.debug('start: num_workers == 1')

            res_ = []
            for i, tau in enumerate(self.tau_grid):
                res = self.minimization_del(tau, self.Loss, 
                          self.loss_args, self.beta_init, **kwargs)
                res_.append(res)
        elif self.several_dates:
            self.logger.debug('start: several_dates')

            loss_args = self.loss_args
            
            if not hasattr(self, 'data_different_dates'):
                self.data_different_dates = {}
                self.gen_subsets()
            
            for date, dataset in self.data_different_dates.items():
                
                l_args = [arg for arg in loss_args]

                l_args[0] = dataset
                l_args = tuple(l_args)
                
                constr = ({'type':'eq',
                           'fun': lambda x: np.array(x[0] + x[1]- np.log(1 + self.tonia_df.loc[date][0]))},)
    
                #parallelization of loop via dask multiprocessing
                values = [delayed(self.minimization_del)(tau, self.Loss, 
                          l_args, self.beta_init, constraints = constr, **kwargs) for tau in self.tau_grid]
    
                res_ = compute(*values, scheduler='processes', num_workers=self.num_workers)
            #parallelization of loop via dask multiprocessing
            values = [delayed(self.minimization_del)(tau, self.Loss, 
                      self.loss_args, self.beta_init, **kwargs) for tau in self.tau_grid]
            res_ = compute(*values, get=dask.multiprocessing.get, num_workers=self.num_workers)
            
            #putting betas and Loss value in Pandas DataFrame
            loss_frame = pd.DataFrame([], columns=['b0', 'b1', 'b2', 'teta', 'loss'])
            loss_frame['b0'] = [res.x[0] for res in res_]
            loss_frame['b1'] = [res.x[1] for res in res_]
            loss_frame['b2'] = [res.x[2] for res in res_]
            loss_frame['teta'] = [t for t in self.tau_grid]
            loss_frame['loss'] = [res.fun for res in res_]
            
            self.loss_res[date] = loss_frame
            self.logger.info(f'Optimization for {date:%d.%m.%Y} - Done!')
        
        elif self.inertia:
            self.logger.debug('start: inertia')

            loss_args = self.loss_args
            
            if self.update_date != None:
                self.iter_dates = self.update_date
            else:
                self.iter_dates = self.settle_dates
            
            i = 0
            lastind = len(self.iter_dates)
            for settle_date in self.iter_dates:
                i = i + 1
                
                self.gen_one_date(settle_date)
                
                l_args = [arg for arg in loss_args]
    
                l_args[0] = self.data_different_dates[settle_date]
                l_args = tuple(l_args)
                
                constr = ({'type':'eq',
                           'fun': lambda x: np.array(x[0] + x[1]- np.log(1 + self.tonia_df.loc[settle_date][0]))},)

                if self.need_trace:              
                    binit = pd.DataFrame(self.beta_init)
                    binit.to_excel(os.path.join(self.trace_path, f'{self.jobid}_{settle_date:%Y%m%d}_beta_init.xlsx'), sheet_name='beta_init', engine='xlsxwriter')
                    self.data_different_dates[settle_date].to_excel(os.path.join(self.trace_path, f'{self.jobid}_{settle_date:%Y%m%d}_settle_date_deals.xlsx'), sheet_name='deals', engine='xlsxwriter')
                    self.raw_data.to_excel(os.path.join(self.trace_path, f'{self.jobid}_{settle_date:%Y%m%d}_raw_data.xlsx'), sheet_name='raw_data', engine='xlsxwriter')

                if i == lastind:
                    self.logger.info(f'store deals to xlsx for {settle_date:%Y%m%d}')
                    self.data_different_dates[settle_date].to_excel(os.path.join(self.data_path, f'{self.jobid}_settle_date_deals.xlsx'), sheet_name='deals', engine='xlsxwriter')

                self.logger.debug('populating distributed tasks')
                #parallelization of loop via dask multiprocessing
                values = [delayed(self.minimization_del)(tau, self.Loss, 
                          l_args, self.beta_init, constraints = constr, **kwargs) for tau in self.tau_grid]
                
                self.logger.info('start minimizing')
                res_ = compute(*values, scheduler='processes', num_workers=self.num_workers)
                
                #putting betas and Loss value in Pandas DataFrame
                loss_frame = pd.DataFrame([], columns=['b0', 'b1', 'b2', 'teta', 'loss'])
                loss_frame['b0'] = [res.x[0] for res in res_]
                loss_frame['b1'] = [res.x[1] for res in res_]
                loss_frame['b2'] = [res.x[2] for res in res_]
                loss_frame['teta'] = [t for t in self.tau_grid]
                loss_frame['loss'] = [res.fun for res in res_]
        
                self.loss_res[settle_date] = loss_frame
                self.beta_best = loss_frame.loc[loss_frame['loss'].idxmin(), :].values[:-1]
                self.beta_init = self.beta_best[:-1].copy()
                self.previous_curve = self.beta_best.copy()

                self.logger.info(f'Optimization for {settle_date:%d.%m.%Y} - Done!')
                self.logger.info(f'Beta best: {self.beta_best}')
                self.logger.info(f'Previous beta set to {self.previous_curve}')
                
                if self.need_trace:              
                    loss_frame.to_excel(os.path.join(self.trace_path, f'{self.jobid}_{settle_date:%Y%m%d}_loss_frame.xlsx'), sheet_name='loss_frame', engine='xlsxwriter')
                
            self.update_date = None          
        return loss_frame
    
    #filtering frame from unacceptable data (spot rates < 0)
    def filter_frame(self, loss_frame):
        accepted_ind = []
        for ind in loss_frame.index:
            beta = loss_frame.loc[ind, loss_frame.columns[:-1]]
            spot_rate_curve = Z(self.maturities, beta) 
            if (spot_rate_curve >= 0).all():
                accepted_ind.append(ind)
        loss_frame_filtered = loss_frame.loc[accepted_ind, :]
        #printing info about № of dropped rows
        n_rows = loss_frame.shape[0]
        n_dropped_rows = n_rows - loss_frame_filtered.shape[0]
        self.logger.debug(f'{n_dropped_rows} out of {n_rows} of rows were dropped')
        return loss_frame_filtered
    
    #actual fitting of data
    def fit(self, return_frame=False, **kwargs):
        if use_one_worker:
            self.logger.warning('Multiprocessing is not enabled as dask is not installed. Install dask to enbale multiprocessing.')
            self.num_workers = 1
        else:
            self.num_workers = self.num_workers
        self.loss_frame = self.loss_grid(**kwargs)
        #loss_frame = self.filter_frame(loss_frame)
        self.beta_best = self.loss_frame.loc[self.loss_frame['loss'].argmin(), :].values[:-1]
        best_betas = {}
        for date in self.settle_dates:
            idx = self.loss_res[date].loc[:, 'loss'].idxmin()
            best_betas[date] = self.loss_res[pd.to_datetime(date)].loc[idx, ['b0','b1','b2','teta']].values
        self.best_betas = best_betas

        if return_frame:
            return self.beta_best, self.loss_frame
        else:
            return self.beta_best
