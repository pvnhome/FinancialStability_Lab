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
from payments_calendar import creating_coupons

#checking if dask is installed
try:
    import dask.multiprocessing
    from dask import compute, delayed
    no_dask = False
except ImportError as e:
    no_dask = True

##grid search over values of tau
class grid_search():
    def __init__(self, tau_grid, 
                 Loss, beta_init, 
                 start_date, end_date, 
                 freq, 
                 toniaDF,
                 thresholds,
                 loss_args,
                 loss_args_auct, 
                 raw_data, 
                 raw_data_auct, 
                 parMsi = False,
                 maturities = None, 
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

        self.logger = logging.getLogger(__name__)
        
        self.Loss = Loss
        self.beta_init = beta_init
        self.loss_args = loss_args
        self.loss_args_auct = loss_args_auct
        self.parMsi = parMsi
        self.tau_grid = tau_grid
        self.maturities = maturities
        self.results = []
        self.loss_res = {}
        self.several_dates = several_dates

        if thresholds is None:
            raise Exception('we need tresholds to be set')
        else:
            self.thresholds = thresholds
        
        if self.maturities is None:
            self.maturities = np.arange(0.0001, 30, 1 / 12) 
        
        self.raw_data = raw_data
        self.raw_data_auct = raw_data_auct
            
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        self.num_workers = num_workers
        self.tonia_df = toniaDF
        self.inertia = inertia
        self.dropped_deals = {}
        
        if self.several_dates or self.inertia:
            self.settle_dates, self.start_date = self.calculateSettleDates(calendar)
            
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

    def calculateSettleDates(self, calendar):
        '''
        Метод для определения расчетной даты и даты прогрева с учетом календаря торговых дней.
        '''
        
        full_range = pd.date_range(start=self.start_date, end=self.end_date, normalize=True, freq='D', closed='right')
        self.logger.debug(f'full: {full_range}')
        
        filterd_range = pd.DatetimeIndex(list(filter(lambda d: self.isTradeDay(d, calendar), full_range)))
        self.logger.debug(f'filterd: {filterd_range}')
        
        settleDates = filterd_range[-2:]
        
        startDate = settleDates.min()
        
        self.logger.debug(f'filterd14: {settleDates}, min={startDate}')
        
        return settleDates, startDate
        
    def isTradeDay(self, day, calendar):
        '''
        Метод для определения торгового дня с учетом календаря.
        '''
        return (calendar.index.contains(day) and calendar.loc[day].daytype=='Y') or (not calendar.index.contains(day) and day.dayofweek!=5 and day.dayofweek!=6)
        
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

    def simple_outlier(self, points):
        '''
        Returns a boolean array with True if points are outliers and False
        otherwise.
    
        Parameters:
        -----------
            points : An numobservations by numdimensions array of observations
    
        Returns:
        --------
            mask : A numobservations-length boolean array.
        '''

        self.logger.debug('simple_outlier: threshold -0.01 and +0.01')

#        points.loc[:,'par'] = par_yield(points.loc[:,'span'].values / 365, self.previous_curve)
#        points.loc[:,'par_min'] = points.loc[:,'par'].values - 0.01
#        points.loc[:,'par_max'] = points.loc[:,'par'].values + 0.01
#        points.loc[:,'good'] = points.loc[:,'ytm'].values > points.loc[:,'par_min'].values and points.loc[:,'ytm'].values < points.loc[:,'par_max'].values
        
        #points['par'] = points.apply(lambda row: par_yield(row.span / 365, self.previous_curve), axis=1)
        points['par'] = points.apply(lambda row: np.exp(par_yield(row.span / 365, self.previous_curve)) - 1, axis=1)
#        points['par_min'] = points.apply(lambda row: row.par - 0.01, axis=1)
#        points['par_max'] = points.apply(lambda row: row.par + 0.01, axis=1)
        points['par_min'] = points.apply(lambda row: row.par - 0.005, axis=1)
        points['par_max'] = points.apply(lambda row: row.par + 0.05, axis=1)
        #points['par_ytm'] = points.apply(lambda row: row.ytm_kase if row.ytm_kase>0.0 else row.ytm, axis=1)
        points['bad_deals'] = points.apply(lambda row: row.ytm_fixed<row.par_min or row.ytm_fixed>row.par_max, axis=1)

        #return points
    
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

        idDatePrefix = ''
        if self.need_trace:              
            idDatePrefix = f'{self.jobid}_{settle_date:%Y%m%d}'

        if not hasattr(self, 'data_different_dates'):
            self.data_different_dates = {}

        parMsiActive = self.parMsi & self.inertia & (len(self.previous_curve) == 0) 

        rawData = self.raw_data_auct if parMsiActive else self.raw_data 

        self.logger.debug(f'before drop outliers: rawData: {rawData.shape[0]}, self.raw_data: {self.raw_data.shape[0]}, self.raw_data_auct: {self.raw_data_auct.shape[0]}, parMsiActive: {parMsiActive}')
        
        #rawData.loc[:,'bond_maturity_type'] = pd.cut(rawData.span, bins=self.thresholds)
            
        #sample = creating_sample(settle_date, rawData, min_n_deal=self.min_n_deal, time_window=CONFIG.TIME_WINDOW, thresholds = self.thresholds)
        
        if self.need_trace:              
            zsRawDataFileName = f'{idDatePrefix}_zscore_raw_data_parmsi.xlsx' if parMsiActive else f'{idDatePrefix}_zscore_raw_data.xlsx'
            rawData.to_excel(os.path.join(self.trace_path, zsRawDataFileName), sheet_name='raw_data', engine='xlsxwriter')
            
        #ind_out=[]
        clearedData = None 
        
        if not parMsiActive:
            # Чистим массив по Z-Score только если мы не в режиме parMsi и это не прогрев.     
            self.logger.debug(f'try to drop outlies')

            #bsampleYtmSpan = rawData.loc[:, ['ytm', 'span']]

            #if self.need_trace:              
            #    bsampleYtmSpan.to_excel(os.path.join(self.trace_path, f'{idDatePrefix}_bsample_ytm_span.xlsx'), sheet_name='bsample_ytm_span', engine='xlsxwriter')

            #zscores = self.simple_outlier(bsampleYtmSpan)
            self.simple_outlier(rawData)

            if self.need_trace:              
                rawData.to_excel(os.path.join(self.trace_path, f'{idDatePrefix}_par.xlsx'), sheet_name='par', engine='xlsxwriter')

            #bind_out = rawData.loc[(zscores.bad_deals) & (rawData.loc[:,'deal_type'] != 1)]
            #ind_out = bind_out.index.values
        
#            for b in rawData.bond_maturity_type.unique().sort_values():
#                xlsxPrefix = '' 
#                if self.need_trace:              
#                    bn = f'{b}'.replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '').replace('.', '_').replace(',', '_')
#                    self.logger.debug(f'bond_maturity_type name: {b} -> {bn}')
#                    xlsxPrefix = f'{idDatePrefix}_{bn}'
#                
#                bsample = rawData.loc[rawData.loc[:,'bond_maturity_type']==b]
#                bsampleYtmSpan = bsample.loc[:, ['ytm', 'span']]
#    
#                if self.need_trace:              
#                    bsample.to_excel(os.path.join(self.trace_path, f'{xlsxPrefix}_bsample.xlsx'), sheet_name='bsample', engine='xlsxwriter')
#                    bsampleYtmSpan.to_excel(os.path.join(self.trace_path, f'{xlsxPrefix}_bsample_ytm_span.xlsx'), sheet_name='bsample_ytm_span', engine='xlsxwriter')
#                
#                zscores = self.is_outlier(bsampleYtmSpan)
#                
#                # self.logger.debug(f'Z-score: {zscores[0]}, {zscores[1]}, {bsample.loc[:,"ytm"]}')
#                
#                rawData.loc[rawData.loc[:,'bond_maturity_type']==b, 'std']=zscores[2]
#                
#                bind_out = bsample.loc[(zscores[1]) & (bsample.loc[:,'deal_type'] != 1)]
#                bind_out_values = bind_out.index.values
#    
#                if self.need_trace:              
#                    zscores[0].to_excel(os.path.join(self.trace_path, f'{xlsxPrefix}_zscore_0.xlsx'), sheet_name='zscores0', engine='xlsxwriter')
#                    zscores[1].to_excel(os.path.join(self.trace_path, f'{xlsxPrefix}_zscore_1.xlsx'), sheet_name='zscores1', engine='xlsxwriter')
#    
#                    if not bind_out.empty:              
#                        bind_out.to_excel(os.path.join(self.trace_path, f'{xlsxPrefix}_bind_out.xlsx'), sheet_name='bind_out_values-', engine='xlsxwriter')
#                
#                if bind_out_values.size != 0:
#                    ind_out.append(bind_out_values)
                
            # Преобразует [[a,b],[c,d],[e,f]] в [a,b,c,d,e,f]         
            # ind_out = [item for sublist in ind_out for item in sublist]
            
            self.logger.debug(f'DF shape: {rawData.shape} - original')
            #self.logger.debug(f'Deals dropped: {ind_out}')
            #self.dropped_deals[settle_date] = rawData.loc[ind_out,:]
            
            group_ind_cols = ['deal_date', 'symbol', 'deal_type', 'deal_price']
 
            rawDataNoIndex = rawData.reset_index();
            
            dd = rawDataNoIndex.loc[(rawDataNoIndex.deal_type != 1) & (rawDataNoIndex.bad_deals)]
            self.dropped_deals[settle_date] = dd.set_index(group_ind_cols) 
            # rawData.drop(ind_out, inplace = True)
            cd = rawDataNoIndex.loc[(rawDataNoIndex.deal_type == 1) | (~rawDataNoIndex.bad_deals)]
            clearedData = cd.set_index(group_ind_cols)
            self.logger.debug(f'DF shape: {clearedData.shape} - adjusted')
        
        else:
            self.dropped_deals[settle_date] = None
            clearedData = rawData
            
        groupedData = self.group_by_code_date_type(clearedData)        
        
        sample = creating_sample(settle_date, groupedData, min_n_deal=self.min_n_deal, time_window=CONFIG.TIME_WINDOW, thresholds = self.thresholds)

        if self.need_trace:              
            zsSampeFileName = f'{idDatePrefix}_zscore_sample_parmsi.xlsx' if parMsiActive else f'{idDatePrefix}_zscore_sample.xlsx'
            sample.to_excel(os.path.join(self.trace_path, zsSampeFileName), sheet_name='sample', engine='xlsxwriter')
            clearedData.to_excel(os.path.join(self.trace_path, f'{idDatePrefix}_deals_cleared_by_par.xlsx'), sheet_name='cleared_deals', engine='xlsxwriter')
            groupedData.to_excel(os.path.join(self.trace_path, f'{idDatePrefix}_deals_grouped.xlsx'), sheet_name='cleared_deals', engine='xlsxwriter')
            # clearedData.to_pickle(os.path.join(self.trace_path, f'{idDatePrefix}_deals_cleared_by_par.pkl.compress'), compression="gzip")
            # clearedDataCopy = pd.read_pickle(os.path.join(self.trace_path, "data.pkl.compress"), compression="gzip")
            if (self.dropped_deals[settle_date] is not None) and (not self.dropped_deals[settle_date].empty):              
                self.dropped_deals[settle_date].to_excel(os.path.join(self.trace_path, f'{idDatePrefix}_deals_dropped_by_par.xlsx'), sheet_name='dropped_deals', engine='xlsxwriter')
        
        self.data_different_dates[settle_date] = sample 

        self.logger.debug(f'Generating sample for {settle_date:%d.%m.%Y} - Done!')
        
    def group_by_code_date_type(self, df):
        ind_col = ['deal_date', 'symbol', 'deal_price']
        #group_ind_cols = ['deal_date', 'symbol', 'deal_type']
        group_ind_cols = ['deal_date', 'symbol']
        
        dfni = df.reset_index();
        
        #dfni['deal_date'] = dfni['deal_date'].dt.floor('d')
    
        grouped = dfni.groupby(group_ind_cols)
        
        df_agg = grouped.agg({
            "volume_kzt": "sum", 
            "span": "first",
            "coupon_rate": "first",
            "annual_freq": "first",
            "base_time": "first",
            "bond_symb": "first"
        })
        
        df_price = grouped.apply(lambda x: np.average(x.deal_price, weights=x.volume_kzt))
    
        df_ytm = grouped.apply(lambda x: np.average(x.ytm_fixed, weights=x.volume_kzt))
        
        df_agg['deal_price'] = df_price 
        df_agg['ytm'] = df_ytm 
    
        df_final = df_agg.reset_index().set_index(ind_col)
        
        return df_final     
    
    def new_dates(self, new_end_date = None):
        
        if new_end_date == None:
            self.update_date = [self.settle_dates[-1]+1]
            self.settle_dates = self.settle_dates.union([self.settle_dates[-1]+1])
            
    
    
#    def dump(self):
#        
#        best_betas = {}
#        for date in self.settle_dates:
#            idx = self.loss_res[date].loc[:, 'loss'].idxmin()
#            best_betas[date] = self.loss_res[pd.to_datetime(date)].loc[idx, ['b0','b1','b2','teta']].values
#        best_betas = pd.DataFrame.from_dict(best_betas, orient='index', columns = ['b0','b1','b2','teta'])
#        best_betas.sort_index(inplace=True)
#        
#        attributes = ['several_dates', 
#                      'thresholds', 
#                      'start_date', 
#                      'end_date', 
#                      'freq', 
#                      'num_workers', 
#                      'inertia', 
#                      'settle_dates',
#                      'loss_res']
#        
#        params = {k:self.__getattribute__(k) for k in attributes}
#                  
#       
#        with h5py.File('grid_data.hdf5', 'w') as f:
#            g = f.create_group('curveData')
#            betas = g.create_dataset('betas', data = [pickle.dumps(best_betas)])
#            samples = g.create_dataset('samples', data = [pickle.dumps(self.data_different_dates)])
#            dropped_deals = g.create_dataset('dropped', data = [pickle.dumps(self.dropped_deals)])
#            raw_data = g.create_dataset('raw_data', data = [pickle.dumps(self.raw_data)])
#            params = g.create_dataset('params', data = [pickle.dumps(params)])
#            
#            meta = {'save date': f'{pd.datetime.now():%Y-%m-%d %H:%M:%S}',
#                    'frequency': self.freq,
#                    'start_date': self.start_date,
#                    'end_date':self.end_date,
#                   
#                    }
#            g.attrs.update(meta)
#        
#            self.logger.debug('saving data:')
#            self.logger.debug('-'*10)
#            for m in g.attrs.keys():
#                self.logger.debug(f'{m}: {g.attrs[m]}')
#            self.logger.debug('-'*10)
                
#    def load(self):
#        
#        with h5py.File('grid_data.hdf5', 'r') as f:
#            g = f['curveData']
#            self.logger.debug('loading stored data:')
#            self.logger.debug('-'*10)
#            for m in g.attrs.keys():
#                self.logger.debug(f'{m}: {g.attrs[m]}')
#            self.logger.debug('-'*10, '\n')
#            best_betas = pickle.loads(g['betas'][()])
#            params = pickle.loads(g['params'][()])
#            samples = pickle.loads(g['samples'][()])
#            dropped = pickle.loads(g['dropped'][()])
#            
#            
#        self.previous_curve = best_betas.iloc[-1].copy()
#        self.beta_init = best_betas.iloc[-1].copy()
#        self.data_different_dates = samples
#        self.dropped_deals = dropped
#        
#        # self.logger.debug('Following parameters were used:') #uncomment for diagnostics
#        # self.logger.debug('-'*10) #uncomment for diagnostics
#        for k,v in params.items():
#            # self.logger.debug(f'{k}: {v}') #uncomment for diagnostics
#            self.__dict__[k] = v
    
    #creation of loss frame grid
    def loss_grid(self, **kwargs):
        self.logger.debug(f'loss_grid: num_workers = {self.num_workers}')

        #Ветка для работы без dask больше не поддерживается.
        # TODO Удалить код
        #if self.num_workers == 1:
        #    self.logger.debug('start: num_workers == 1')
        #
        #    res_ = []
        #    for i, tau in enumerate(self.tau_grid):
        #        res = self.minimization_del(tau, self.Loss, 
        #                  self.loss_args, self.beta_init, **kwargs)
        #        res_.append(res)
        #elif self.several_dates:
        if self.several_dates:
            self.logger.debug('start: several_dates')

            # Поддерживается только ветка "inertia" 
            # Временно выдаем Exception 
            raise Exception('Case with several_dates not supported')

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
            self.logger.debug(f'start: inertia, num_workers = {self.num_workers}')

            if self.update_date != None:
                self.logger.debug('iter_dates = update_date')
                self.iter_dates = self.update_date
            else:
                self.logger.debug('iter_dates = settle_dates')
                self.iter_dates = self.settle_dates
            
            i = 0
            lastind = len(self.iter_dates)
            for settle_date in self.iter_dates:
                i = i + 1
                
                self.gen_one_date(settle_date)

                parMsiActive = self.parMsi & (len(self.previous_curve) == 0) 
                loss_args = self.loss_args_auct if parMsiActive else self.loss_args
                
                l_args = [arg for arg in loss_args]
    
                #l_args[0] = self.data_different_dates[settle_date]
                df = self.data_different_dates[settle_date]
                l_args[0] = df
                # Вроде бы уже время убрли в creating_new_columns?
                ## Заново перерасчитываем (без времени)
                coupons_cf, streak_data = creating_coupons(df)
                ## и заменяем
                l_args[1] = coupons_cf
                l_args[2] = streak_data
                l_args = tuple(l_args)
                
                constr = ({'type':'eq',
                           'fun': lambda x: np.array(x[0] + x[1]- np.log(1 + self.tonia_df.loc[settle_date][0]))},)

                if self.need_trace:              
                    binit = pd.DataFrame(self.beta_init)
                    binit.to_excel(os.path.join(self.trace_path, f'{self.jobid}_{settle_date:%Y%m%d}_beta_init.xlsx'), sheet_name='beta_init', engine='xlsxwriter')
                    #self.raw_data.to_excel(os.path.join(self.trace_path, f'{self.jobid}_{settle_date:%Y%m%d}_raw_data.xlsx'), sheet_name='raw_data', engine='xlsxwriter')
                    coupons_cf.to_excel(os.path.join(self.trace_path, f'{self.jobid}_{settle_date:%Y%m%d}_coupons_cf_new.xlsx'), sheet_name='data', engine='xlsxwriter')
                    streak_data.to_excel(os.path.join(self.trace_path, f'{self.jobid}_{settle_date:%Y%m%d}_streak_data_new.xlsx'), sheet_name='data', engine='xlsxwriter')

                    df_copy = df.copy();
                    df_copy = df_copy.reset_index()
                    self.logger.info(f'df_copy: {df_copy.columns.tolist()}')

                    # Добавить в settle_date_deals поле YTC которое будет рассчитываться как span/base_time, а поле YTM умножить на 100
                    df_copy['ytm'] = df_copy['ytm'] * 100
                    df_copy['ytc'] = df_copy['span'] / df_copy['base_time']
                    
                    cols = ['deal_date', 'symbol', 'deal_price', 'volume_kzt', 'span', 'ytc', 'ytm', 'deal_type', 'reverse_span', 'bond_maturity_type', 'coupon_rate', 'annual_freq', 'base_time', 'bond_symb', 'settle_date', 'deal_only_date']
                    df_copy = df_copy.reindex(columns=cols)
                    
                    df_copy.to_excel(os.path.join(self.trace_path, f'{self.jobid}_{settle_date:%Y%m%d}_settle_date_deals.xlsx'), sheet_name='deals_trace', engine='xlsxwriter')

                if i == lastind:
                    self.logger.info(f'store deals to xlsx for {settle_date:%Y%m%d}')
                    df.to_excel(os.path.join(self.data_path, f'{self.jobid}_settle_date_deals.xlsx'), sheet_name='deals', engine='xlsxwriter')

                self.logger.debug('populating distributed tasks')
                #parallelization of loop via dask multiprocessing
                values = [delayed(self.minimization_del)(tau, self.Loss, 
                          l_args, self.beta_init, constraints = constr, **kwargs) for tau in self.tau_grid]
                
                self.logger.info(f'start minimizing: num_workers = {self.num_workers}')
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
        if no_dask:
            raise Exception('Multiprocessing is not enabled as dask is not installed. Install dask to enbale multiprocessing')

        self.logger.debug(f'fit: num_workers = {self.num_workers}')
            
        self.loss_frame = self.loss_grid(**kwargs)
        
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
