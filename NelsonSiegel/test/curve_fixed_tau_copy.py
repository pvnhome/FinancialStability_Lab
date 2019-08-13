import os
import sys
import warnings
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath('..'))
try:
    import matplotlib.pyplot as plt
    from dataextraction import draw
    to_draw_graphs = True
    plt.style.use('ggplot')
except ImportError as e:
    to_draw_graphs = False
    print(e)

import CONFIG
from datapreparation.preparation_functions import processing_data, read_download_preprocessed_data
from datapreparation.adaptive_sampling import creating_sample
from optimizator.fixed_minimization import fixed_tau_minimizer
from estimation_ytm.estimation_ytm import new_ytm, filtering_ytm
from Loss import yield_Loss, price_Loss, naive_yield_Loss
from payments_calendar import download_calendar, creating_coupons
from weight_scheme import weight

PATH = os.path.join('.', 'extracted_data')
datasets_path = os.path.join('.', 'datasets')
clean_data_path = os.path.join(datasets_path, 'clean_data.hdf')
calendar_data_path = os.path.join(datasets_path, 'coupons_data.hdf')
save_data = True
warnings.simplefilter('ignore')

step = 9

### Initialization
path = os.path.join(datasets_path, '20160101_20190531_deals_gov.xlsx')
#path = os.path.join(datasets_path, 'bonds_for_Viktor.xlsx')
#path = os.path.join(datasets_path, 'gzb_one_year.xlsx')
print(f'XLSX-file with deals: {path}\n')
df = pd.read_excel(path, skiprows=2).rename(columns=CONFIG.NAME_MASK)
df.to_excel(os.path.join(PATH, f'step{step}_deals.xlsx'), sheet_name='deals')

### data mungling
if save_data:
    clean_data = processing_data(df,
                  mask_face_value=CONFIG.MASK_FACE_VALUE, mask_base_time=CONFIG.MASK_BASE_TIME,
                  needed_bonds=CONFIG.INSTRUMENTS, use_otc=CONFIG.USE_OTC, deal_market=CONFIG.DEAL_MARKET,
                  notes_in_otc=CONFIG.NOTES_IN_OTC, maturity_filter=CONFIG.MATURITY_FILTER,
                  specific_deals=CONFIG.SPECIFIC_DEALS)

    #calendar payments data: saving and loading
    coupons_cf, streak_data = download_calendar(clean_data, hdf_coupons_path=calendar_data_path)
    print('DOWNLOADED CALENDAR\n')
    #coupons_cf, streak_data = creating_coupons(clean_data)
    #print('NEW CALENDAR\n')
    #d1 = pd.DataFrame(coupons_cf)
    coupons_cf.to_excel(os.path.join(PATH, f'step{step}_coupons_cf.xlsx'), sheet_name='coupons_cf')
    #d1 = pd.DataFrame(streak_data)
    streak_data.to_excel(os.path.join(PATH, f'step{step}_streak_data.xlsx'), sheet_name='streak_data')

    #Estimating correct yield for data
    clean_data = (clean_data.pipe(new_ytm, coupons_cf, streak_data)
                            .pipe(filtering_ytm, max_yield=CONFIG.MAX_YIELD,
                                  min_yield=CONFIG.MIN_YIELD))

    clean_data['bond_symb'] = clean_data.index.get_level_values(1).str.extract(r'([A-Z]+)', expand=False)
    clean_data = read_download_preprocessed_data(save_data, clean_data_path=clean_data_path, clean_data=clean_data)
else:
    clean_data = read_download_preprocessed_data(save_data, clean_data_path=clean_data_path,)
    #Coupon Data: saving and loading
    coupons_cf, streak_data = download_calendar(clean_data, hdf_coupons_path=calendar_data_path)


print('starting to read filtered data for {} settle date'.format(CONFIG.SETTLE_DATE))
###Creating sample
filtered_data = creating_sample(CONFIG.SETTLE_DATE, clean_data, min_n_deal=CONFIG.MIN_N_DEAL,
                                time_window=CONFIG.TIME_WINDOW)
print('filtered data shape {}'.format(filtered_data.shape[0]))

###Setting Loss arguments and optimization paramters
#Initial guess vector(for optimization)
x0 = [0.09, -0.01, 0]
#Parameters constraints
constr = ({'type':'ineq',
           'fun': lambda x: np.array(x[0] + x[1])})
#Longest matuiry year of deals in data
max_deal_span = (filtered_data.span / 365).round().max()
#Parameters bounds for constraint optimization
bounds = ((0, 1), (None, None), (None, None))
print(bounds)
#Maturity limit for Zero-curve plot
longest_maturity_year = max([max_deal_span, 20])
np_ls_num = 10000 # Так было изначально
#np_ls_num = 200 # Двадцать лет по 10 точек
print(f'num in np.linspace is {np_ls_num}')
theor_maturities = np.linspace(0.001, longest_maturity_year, np_ls_num)
options = {'maxiter': 150, 'eps': 9e-5, 'disp': True}
#Tuple of arguments for loss function
loss_args = (filtered_data, coupons_cf, streak_data, CONFIG.RHO, CONFIG.WEIGHT_SCHEME, CONFIG.TAU)

# debug output
filtered_data.to_excel(os.path.join(PATH, f'step{step}_filtered_data.xlsx'), sheet_name='filtered_data')

#defining loss -- Crucial
loss = yield_Loss

filtered_data['weight'] = weight([1, 1, 1, 1], filtered_data, CONFIG.WEIGHT_SCHEME)

print('start optimization\n')
###### OPTIMIZATION
res_ = fixed_tau_minimizer(Loss=loss, beta_init=x0,
                loss_args=loss_args, method='SLSQP',  bounds=bounds,
                #constraints=constr,
                max_deal_span=max_deal_span, options=options)
beta_best = np.append(res_.x, CONFIG.TAU)
print(f'end optimization: b1={beta_best[0]}, b2={beta_best[1]}, b3={beta_best[2]}, tau={beta_best[3]}')


### Showing results of work
##plotting and saving Spot rate curve
#draw(beta_best, filtered_data, theor_maturities, CONFIG.SETTLE_DATE,
#     longest_maturity_year, draw_points=True,
#     weight_scheme=CONFIG.WEIGHT_SCHEME, label='Spot rate curve',
#     alpha=0.8, shift=True)
#plt.ylim(0, filtered_data.ytm.max() + 0.01)
#plt.savefig(os.path.join(PATH, f'zero_curve_fix_tau_{CONFIG.SETTLE_DATE}_{loss.__name__}.png'), dpi=400);

betaBestData = pd.DataFrame({'name': np.array(['A','B','C','TAU']), 'beta_init' : np.append(x0, CONFIG.TAU), 'beta_best' : beta_best})
betaBestData.to_excel(os.path.join(PATH, f'beta_best_from_test_{CONFIG.SETTLE_DATE}_{loss.__name__}.xlsx'), sheet_name='beta_best')

