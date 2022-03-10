import warnings;
warnings.simplefilter('ignore')
import pandas as pd
import streamlit as st
import numpy as np
import warnings
import itertools
import matplotlib
import matplotlib.pyplot as plt
from pylab import rcParams
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.stattools import adfuller
import plotly.figure_factory as ff
import time

st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings("ignore")

plt.style.use('seaborn-bright')
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

import pandas as pd
# Import the data
df = pd.read_csv("data_new_test1.csv")
df['Date'] = pd.to_datetime(df['Date'])
# Set the date as index 
df = df.set_index('Date')
# Select the proper time period for weekly aggreagation
df = df['2013-01-01':'2017-12-31'].resample('W').sum()
df.head()  
#st.sidebar.selectbox("Choose the forecasting method to continue", ['Simple Exponential Smoothing','HOLTs Forecasting Method', 'HOLT-WINTERs Forecasting Method'])
rad = st.sidebar.radio("Navigation - Choose a forecasting method to continue",["Visualising Your Data","Simple Exponential Smoothing","HOLTs Forecasting Method","HOLT-WINTERs Forecasting Method"])#"Sarima"])

if rad == "Visualising Your Data":
    header = st.container() 
    from PIL import Image
    
    with header:
        image = Image.open('cslp_logo_1.png')
        st.image(image)
        st.text('INDU 6990 - INDUSTRIAL ENGINEERING CAPSTONE')
        st.text('GROUP 16')
        st.title('FORECASTING')    
else:
    pass
        
###############################################################################

import warnings
import matplotlib.pyplot as plt
# Orders Graph
y = df['Orders']
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(y,marker='.', linestyle='-', linewidth=0.5, label='Weekly')
ax.plot(y.resample('M').mean(),marker='o', markersize=8, linestyle='-', label='Monthly Mean Resample')
ax.set_ylabel('Orders')
ax.legend();
orders = st.container()

if rad == "Visualising Your Data":
    with orders:
        st.header('Graph of your Orders')
        st.pyplot(fig)

###############################################################################

#Seasonal Decomposed Graph
import statsmodels.api as sm

# graphs to show seasonal_decompose
def seasonal_decompose (y):
    decomposition = sm.tsa.seasonal_decompose(y, model='additive',extrapolate_trend='freq')
    fig = decomposition.plot()
    fig.set_size_inches(14,7)
    plt.show()
decompose = st.container()
seasonal_decompose(y) 

if rad == "Visualising Your Data":
    with decompose:
        st.header('Here you can see individual elements of your orders data. TREND AND SEASONALITY')
        st.pyplot(y.empty)

###############################################################################

### plot for Rolling Statistic for testing Stationarity
def test_stationarity(timeseries, title):
    
    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=12).mean() 
    rolstd = pd.Series(timeseries).rolling(window=12).std()
    
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(timeseries, label= title)
    ax.plot(rolmean, label='rolling mean');
    ax.plot(rolstd, label='rolling std (x10)');
    ax.legend()

pd.options.display.float_format = '{:.8f}'.format
test_stationarity(y,'raw data')

# Augmented Dickey-Fuller Test
from statsmodels.tsa.stattools import adfuller

def ADF_test(timeseries, dataDesc):
    print(' > Is the {} stationary ?'.format(dataDesc))
    dftest = adfuller(timeseries.dropna(), autolag='AIC')
    print('Test statistic = {:.3f}'.format(dftest[0]))
    print('P-value = {:.3f}'.format(dftest[1]))
    print('Critical values :')
    for k, v in dftest[4].items():
        print('\t{}: {} - The data is {} stationary with {}% confidence'.format(k, v, 'not' if v<dftest[0] else '', 100-int(k[:-1])))

ADF_test(y,'raw data')

# Detrending
y_detrend =  (y - y.rolling(window=12).mean())/y.rolling(window=12).std()

test_stationarity(y_detrend,'de-trended data')
ADF_test(y_detrend,'de-trended data')

# Differencing
y_12lag =  y - y.shift(12)

test_stationarity(y_12lag,'12 lag differenced data')
ADF_test(y_12lag,'12 lag differenced data')

# Detrending + Differencing

y_12lag_detrend =  y_detrend - y_detrend.shift(12)

test_stationarity(y_12lag_detrend,'12 lag differenced de-trended data')
ADF_test(y_12lag_detrend,'12 lag differenced de-trended data')

y_to_train = y[:'2017-07-02'] # dataset to train
y_to_val = y['2016-07-03':] # last X months for test  
predict_date = len(y) - len(y[:'2016-07-03']) # the number of data points for the test set

###############################################################################

if rad == "Simple Exponential Smoothing": 
# =============================================================================
#     progress = st.progress(0)
#     for i in range(100):
#         time.sleep(0.1)
#         progress.progress(i+1)
# =============================================================================
       
    import numpy as np
    from statsmodels.tsa.api import SimpleExpSmoothing

    def ses(y, y_to_train,y_to_test,smoothing_level,predict_date):
        y.plot(marker='o', color='black', legend=True, figsize=(14, 7))
        
        fit1 = SimpleExpSmoothing(y_to_train).fit(smoothing_level=smoothing_level,optimized=False)
        fcast1 = fit1.forecast(predict_date).rename(r'$\alpha={}$'.format(smoothing_level))
        # specific smoothing level
        fcast1.plot(marker='o', color='blue', legend=True)
        fit1.fittedvalues.plot(marker='o',  color='blue')
        mse1 = ((fcast1 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of our forecasts with smoothing level of {} is {}'.format(smoothing_level,round(np.sqrt(mse1), 2)))
        
        ## auto optimization
        fit2 = SimpleExpSmoothing(y_to_train).fit()
        fcast2 = fit2.forecast(predict_date).rename(r'$\alpha=%s$'%fit2.model.params['smoothing_level'])
        # plot
        fcast2.plot(marker='o', color='green', legend=True)
        fit2.fittedvalues.plot(marker='o', color='green')
        
        mse2 = ((fcast2 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of our forecasts with auto optimization is {}'.format(round(np.sqrt(mse2), 2)))
        
        plt.show()
        
    sesgraph = ses(y, y_to_train,y_to_val,0.8,predict_date)  


    simple_exponintial_smoothing = st.container()
    with simple_exponintial_smoothing:
        st.header('Simple exponential smoothing')
        st.text('This graph shows forecast using simple exponential smoothing.')
        st.text('NOTE : This is a basic forecasting method and does not include any trend or')
        st.text('seasonality factor')
        st.pyplot(sesgraph)

###############################################################################

if rad == "HOLTs Forecasting Method":
# =============================================================================
#     progress = st.progress(0)
#     for i in range(100):
#         time.sleep(0.1)
#         progress.progress(i+1)
#     from statsmodels.tsa.api import Holt
# =============================================================================
    
    def holt(y,y_to_train,y_to_test,smoothing_level,smoothing_slope, predict_date):
        y.plot(marker='o', color='black', legend=True, figsize=(14, 7))
        
        fit1 = Holt(y_to_train).fit(smoothing_level, smoothing_slope, optimized=False)
        fcast1 = fit1.forecast(predict_date).rename("Holt's linear trend")
        mse1 = ((fcast1 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of Holt''s Linear trend {}'.format(round(np.sqrt(mse1), 2)))
    
        fit2 = Holt(y_to_train, exponential=True).fit(smoothing_level, smoothing_slope, optimized=False)
        fcast2 = fit2.forecast(predict_date).rename("Exponential trend")
        mse2 = ((fcast2 - y_to_test) ** 2).mean()
        print('The Root Mean Squared Error of Holt''s Exponential trend {}'.format(round(np.sqrt(mse2), 2)))
        
        fit1.fittedvalues.plot(marker="o", color='blue')
        fcast1.plot(color='blue', marker="o", legend=True)
        fit2.fittedvalues.plot(marker="o", color='red')
        fcast2.plot(color='red', marker="o", legend=True)
    
        plt.show()
        
    fig = holt(y, y_to_train,y_to_val,0.6,0.2,predict_date)
    
    holt_graph = st.container()
    
    #if rad == "HOLTs Forecasting Method":
    with holt_graph:
        st.header('HOLTs Forecasting Method')
        st.text('This graph shows forecast using HOLTs Forecasting Method')
        st.text('NOTE : This forecasting method considers trend factor and is more')
        st.text('accurate than Simple exponential Smoothing')
        st.pyplot(fig)
        plt.close(fig)

###############################################################################

if rad == "HOLT-WINTERs Forecasting Method":
# =============================================================================
#     progress = st.progress(0)
#     for i in range(100):
#         time.sleep(0.1)
#         progress.progress(i+1)
# =============================================================================

    from statsmodels.tsa.api import ExponentialSmoothing
    
    def holt_win_sea(y,y_to_train,y_to_test,seasonal_type,seasonal_period,predict_date):
        
        y.plot(marker='o', color='black', legend=True, figsize=(14, 7))
        
        if seasonal_type == 'additive':
            fit1 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='add').fit(use_boxcox=True)
            fcast1 = fit1.forecast(predict_date).rename('Additive')
            mse1 = ((fcast1 - y_to_test) ** 2).mean()
            print('The Root Mean Squared Error of additive trend, additive seasonal of '+ 
                  'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse1), 2)))
            
            fit2 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='add', damped=True).fit(use_boxcox=True)
            fcast2 = fit2.forecast(predict_date).rename('Additive+damped')
            mse2 = ((fcast2 - y_to_test) ** 2).mean()
            print('The Root Mean Squared Error of additive damped trend, additive seasonal of '+ 
                  'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse2), 2)))
            
            fit1.fittedvalues.plot(style='--', color='red')
            fcast1.plot(style='--', marker='o', color='red', legend=True)
            fit2.fittedvalues.plot(style='--', color='green')
            fcast2.plot(style='--', marker='o', color='green', legend=True)
        
        elif seasonal_type == 'multiplicative':  
            fit3 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='mul').fit(use_boxcox=True)
            fcast3 = fit3.forecast(predict_date).rename('Multiplicative')
            mse3 = ((fcast3 - y_to_test) ** 2).mean()
            print('The Root Mean Squared Error of additive trend, multiplicative seasonal of '+ 
                  'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse3), 2)))
            
            fit4 = ExponentialSmoothing(y_to_train, seasonal_periods = seasonal_period, trend='add', seasonal='mul', damped=True).fit(use_boxcox=True)
            fcast4 = fit4.forecast(predict_date).rename('Multiplicative+damped')
            mse4 = ((fcast3 - y_to_test) ** 2).mean()
            print('The Root Mean Squared Error of additive damped trend, multiplicative seasonal of '+ 
                  'period season_length={} and a Box-Cox transformation {}'.format(seasonal_period,round(np.sqrt(mse4), 2)))
            
            fit3.fittedvalues.plot(style='--', color='red')
            fcast3.plot(style='--', marker='o', color='red', legend=True)
            fit4.fittedvalues.plot(style='--', color='green')
            fcast4.plot(style='--', marker='o', color='green', legend=True)
            
        else:
            print('Wrong Seasonal Type. Please choose between additive and multiplicative')
    
        plt.show()
        
    fig = holt_win_sea(y, y_to_train,y_to_val,'additive',52, predict_date)
    
    holt_winter = st.container()
    
    
    with holt_winter:
        st.header('HOLT-WINTERs Forecasting Method')
        st.text('This graph shows forecast using HOLT-WINTERs Forecasting Method')
        st.text('NOTE : This forecasting method considers trend and seasonality factor')
        st.text('and is more accurate than simple exponential smoothing and HOLTs')
        st.text('forecasting method')
        st.pyplot(fig)
        plt.close(fig)
else:
    pass

###############################################################################

# =============================================================================
# if rad =="Sarima":
# # =============================================================================
# #     progress = st.progress(0)
# #     for i in range(100):
# #         time.sleep(0.1)
# #         progress.progress(i+1)
# # =============================================================================
#         
#     import itertools
#     
#     def sarima_grid_search(y,seasonal_period):
#         p = d = q = range(0, 2)
#         pdq = list(itertools.product(p, d, q))
#         seasonal_pdq = [(x[0], x[1], x[2],seasonal_period) for x in list(itertools.product(p, d, q))]
#     
#         mini = float('+inf')
#     
#     
#         for param in pdq:
#             for param_seasonal in seasonal_pdq:
#                 try:
#                     mod = sm.tsa.statespace.SARIMAX(y,
#                                                 order=param,
#                                                 seasonal_order=param_seasonal,
#                                                 enforce_stationarity=False,
#                                                 enforce_invertibility=False)
# 
#                     results = mod.fit()
#                 
#                     if results.aic < mini:
#                         mini = results.aic
#                         param_mini = param
#                         param_seasonal_mini = param_seasonal
# 
# #                   print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
#                 except:
#                     continue
#                 print('The set of parameters with the minimum AIC is: SARIMA{}x{} - AIC:{}'.format(param_mini, param_seasonal_mini, mini))
#         
#     sarima_grid_search(y,52)
#     
#     # Call this function after pick the right(p,d,q) for SARIMA based on AIC               
#     def sarima_eva(y,order,seasonal_order,seasonal_period,pred_date,y_to_test):
#         # fit the model 
#         mod = sm.tsa.statespace.SARIMAX(y,
#                                     order=order,
#                                     seasonal_order=seasonal_order,
#                                     enforce_stationarity=False,
#                                     enforce_invertibility=False)
#     
#         results = mod.fit()
#         print(results.summary().tables[1])
#         
#         results.plot_diagnostics(figsize=(16, 8))
#         plt.show()
#         
#         # The dynamic=False argument ensures that we produce one-step ahead forecasts, 
#         # meaning that forecasts at each point are generated using the full history up to that point.
#         pred = results.get_prediction(start=pd.to_datetime(pred_date), dynamic=False)
#         pred_ci = pred.conf_int()
#         y_forecasted = pred.predicted_mean
#         mse = ((y_forecasted - y_to_test) ** 2).mean()
#         print('The Root Mean Squared Error of SARIMA with season_length={} and dynamic = False {}'.format(seasonal_period,round(np.sqrt(mse), 2)))
#     
#         ax = y.plot(label='observed')
#         y_forecasted.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
#         ax.fill_between(pred_ci.index,
#                         pred_ci.iloc[:, 0],
#                         pred_ci.iloc[:, 1], color='k', alpha=.2)
#     
#         ax.set_xlabel('Date')
#         ax.set_ylabel('Sessions')
#         plt.legend()
#         plt.show()
#     
#         # A better representation of our true predictive power can be obtained using dynamic forecasts. 
#         # In this case, we only use information from the time series up to a certain point, 
#         # and after that, forecasts are generated using values from previous forecasted time points.
#         pred_dynamic = results.get_prediction(start=pd.to_datetime(pred_date), dynamic=True, full_results=True)
#         pred_dynamic_ci = pred_dynamic.conf_int()
#         y_forecasted_dynamic = pred_dynamic.predicted_mean
#         mse_dynamic = ((y_forecasted_dynamic - y_to_test) ** 2).mean()
#         print('The Root Mean Squared Error of SARIMA with season_length={} and dynamic = True {}'.format(seasonal_period,round(np.sqrt(mse_dynamic), 2)))
#     
#         ax = y.plot(label='observed')
#         y_forecasted_dynamic.plot(label='Dynamic Forecast', ax=ax,figsize=(14, 7))
#         ax.fill_between(pred_dynamic_ci.index,
#                         pred_dynamic_ci.iloc[:, 0],
#                         pred_dynamic_ci.iloc[:, 1], color='k', alpha=.2)
#     
#         ax.set_xlabel('Date')
#         ax.set_ylabel('Sessions')
#     
#         plt.legend()
#         plt.show()
#         
#         return (results)
# 
#     model = sarima_eva(y,(1, 0, 1),(0, 1, 1, 52),52,'2016-07-03',y_to_val)
#     plt.close(fig)
#     
#     def forecast(model,predict_steps,y):
#     
#         pred_uc = model.get_forecast(steps=predict_steps)
#     
#         #SARIMAXResults.conf_int, can change alpha,the default alpha = .05 returns a 95% confidence interval.
#         pred_ci = pred_uc.conf_int()
#     
#         ax = y.plot(label='observed', figsize=(14, 7))
#     #     print(pred_uc.predicted_mean)
#         pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
#         ax.fill_between(pred_ci.index,
#                         pred_ci.iloc[:, 0],
#                         pred_ci.iloc[:, 1], color='k', alpha=.25)
#         ax.set_xlabel('Date')
#         ax.set_ylabel(y.name)
#     
#         plt.legend()
#         plt.show()
#         #ax.figure.savefig('sarima.png')
#         
#         # Produce the forcasted tables 
#         pm = pred_uc.predicted_mean.reset_index()
#         pm.columns = ['Date','Predicted_Mean']
#         pci = pred_ci.reset_index()
#         pci.columns = ['Date','Lower Bound','Upper Bound']
#         final_table = pm.join(pci.set_index('Date'), on='Date')
#         
#         return (final_table)    
#     
#     final_table = forecast(model,15,y)
#     final_table.head()
#     #ax.figure.savefig(final_table)
#         
#     
#     sarima_sl = st.container()
#     with sarima_sl:
#         st.header("SARIMA")
#         st.text("SARIMA stands for Seasonal Autoregressive Integrated Moving Average,")
#         st.text("which is accurate and gives a range of probable forecast")
#         st.table(final_table)
# =============================================================================

###############################################################################


    


    
    


