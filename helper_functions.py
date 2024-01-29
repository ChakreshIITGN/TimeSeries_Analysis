from prophet import Prophet
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, month_plot

import matplotlib.pyplot as plt

import pandas as pd 
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import time



### Function to standardise the data

def standardiseData(dataFrame,minDate,maxDate):
    """
        Restircts the data betweeen the chosen dates for uniformity. 
        Then Resamples using freq='B' i.s considering only the business days

        All the missing values are filled using 'bfill' 
        ============================================
        arguments : 

        minDate: minimum date for the analysis
        maxDate: maximum date for the analysis

    """
    new_index = pd.date_range(minDate,maxDate,freq='B') # dates between which we obderve the time Series

    dataFrame['Date'] = pd.to_datetime(dataFrame['Date'],yearfirst=True)
    dataFrame.set_index('Date',inplace=True)
    dataFrame = dataFrame[minDate:maxDate]

    newDF = dataFrame.reindex(new_index,method='bfill')

    newDF = newDF[['Close']]
    return newDF

## test function to check the data 

def checkData(df,minDate,maxDate):
    """
        Sanity check on the dataframe before any analysis

        checks for  - length, datatypes

        ==============================
        arguments: 

        df: dataframe to be checked
        minDate: minimum date for the analysis
        maxDate: maximum date for the analysis

    """
    days = len(pd.date_range(minDate,maxDate,freq='B'))

    assert len(df) == days
    assert df['Close'].dtypes == 'float64'
    assert df.index.dtype == '<M8[ns]'

    return None


### Creating the Data for analysis combining from different 
### .csv files 
def create_AnalysisData(companyNames,minDate,maxDate,logData=False):

    """
        takes the company names in the existing data folder 
        Creates the standardised data set for the analysis

        =================================================
        arguments:
        
        if logData == True : converts the Numeric data to log10

        minDate: minimum date for the analysis
        maxDate: maximum date for the analysis
    """

    listDF = []
    final_cols = []
    for i in range(len(companyNames)):
        final_cols.append(companyNames[i])
        df_i = pd.read_csv(f'data/{companyNames[i]}_daily.csv', header=0)
        df_i = standardiseData(df_i,minDate,maxDate)
        checkData(df_i,minDate,maxDate)
        
        listDF.append(df_i)

    finalDF = pd.concat(listDF,ignore_index=False,axis=1)
    finalDF.columns = final_cols

    if logData:

        finalDF = finalDF.apply(np.log10,axis=0)

    return finalDF

def smoothing(smoothing_type,ts_decompose_model, train, test):
    """
        This function creates the smoothing for the time series based on
        the user selected options. 

        =================================================
        arguments:

        smoothing_type     : list of smoothing types 'Single', 'Double', 'Triple' 
        ts_decompose_model : str, defines the decomposition b/w additive or multiplicative
        train              : training data
        test               : test data
        returns - list of predictions for each chosen smoothing   
    """
    predictions = []

    if 'Single' in smoothing_type:

        ## Single Exponential Smoothing 
        singleExp = SimpleExpSmoothing(train).fit(optimized=True)
        singleExp_pred = singleExp.forecast(len(test))
        predictions.append(singleExp_pred)
    
    if 'Double' in smoothing_type:

        # Double Exponential Smoothing 
        doubleExp = Holt(train).fit(optimized=True)
        doubleExp_pred = doubleExp.forecast(len(test))
        predictions.append(doubleExp_pred)

    if 'Triple' in smoothing_type:

        # Triple Exponential Smoothing
        tripleExp = ExponentialSmoothing(train,
                                        trend=ts_decompose_model,
                                        seasonal=ts_decompose_model,
                                        seasonal_periods=100).fit(optimized=True)
        tripleExp_pred = tripleExp.forecast(len(test))
        predictions.append(tripleExp_pred)
    

    return predictions


def plot_smoothing(predictions,timeSeriesData_rolling,split_at,train,test):

    """
        Plots the smoothing for the selected types as a matplotlib figure
    """
    
    train_time,test_time = timeSeriesData_rolling.index.values[:-split_at], timeSeriesData_rolling.index.values[-split_at:]

    fig2 = plt.figure(figsize=(12,4))
    ax = fig2.add_subplot(111)

    ax.set_title(f'Exponential smoothing on time series')

    ax.plot(train_time,train,'--b',label='Train')
    ax.plot(test_time,test,'--', color='gray',label='Test')

    for i,val in enumerate(predictions):

        if i == 0:
            label='Single Exp. Predictions'
            color='red'
        if i == 1:
            label='Double Exp. Predictions'
            color='green'
        if i == 2:
            label='Triple Exp. Predictions'
            color='darkorange'

        ax.plot(test_time,val,'--', color=color,label=label)
        
    ax.legend()
    st.pyplot(fig2)
    return None

def generate_smoothing(smoothing_type,ts_decompose_model,train,test,timeSeriesData_rolling,split_at):
    
    with st.spinner('Fitting Exponential'):
        time.sleep(3)
        predictions = smoothing(smoothing_type,ts_decompose_model,train,test)
        plot_smoothing(predictions,timeSeriesData_rolling,split_at,train,test)
        
    st.success('Done')
    return None

def show_monthly_sale(timeSeriesData):

    timeSeriesData_monthly = timeSeriesData.resample('M').sum()
    fig1 = plt.figure(figsize=(12,4))
    ax = fig1.add_subplot(111)
    _ = month_plot(timeSeriesData_monthly[:-1],ax=ax)
    st.pyplot(fig1)

    return None 

def check_stationarity(train,diff=0):

    statistic, p_value, n_lags, critical_values = kpss(train)

    stationarity_message = f"""

    :red[DIFF = {diff}]

    :blue[KPSS Statistic] : {np.round(statistic,4)}, 

    :blue[p-value]        : {p_value}, 
    
    :blue[num lage]       : {n_lags}, 


    :blue[Critical Values] - 
    {critical_values}

    :orange[RESULT : The series is {"not " if p_value < 0.05 else ""}stationary]

    

                         """

    st.write(stationarity_message)
    
    if p_value < 0.05:
        diff += 1
        train = train.diff().dropna()
        check_stationarity(train,diff=diff)


    return None

def get_arima_forecast(arima_resid,split_at):

    arima_pred = arima_resid.forecast(split_at,alpha=0.05)
    fitted = arima_resid.get_forecast(split_at)

    conf_int_95 = fitted.conf_int(alpha=0.05)
    conf_int_90 = fitted.conf_int(alpha=0.10)

    return arima_pred,conf_int_90,conf_int_95

def plot_arima_forecast(pred,conf_90,conf_95,p,d,q,company,train,test,train_time,test_time):

    arima_pred = pred
    conf_int_90 = conf_90
    conf_int_95 = conf_95

    fig4 = plt.figure(figsize=(12,6))
    ax = fig4.add_subplot(111)

    ax.set_title(f'ARIMA {p,d,q} model predictions for {company}')

    ax.plot(train_time,train,'--b',label='Train')
    # ax.plot(train_time[1:],arima_resid.fittedvalues[1:],'-k',label='Model Fitted Values')
    ax.plot(test_time,test,'--', color='gray',label='Test')
    ax.plot(test_time,arima_pred,'--', color='red',label='ARMA predictions')
    ax.fill_between(
        conf_int_95.index,
        conf_int_95[f"lower {company}"],
        conf_int_95[f"upper {company}"],
        color="b",
        alpha=0.1,
        label="95% CI"
    )

    ax.fill_between(
        conf_int_90.index,
        conf_int_90[f"lower {company}"],
        conf_int_90[f"upper {company}"],
        color="b",
        alpha=0.2,
        label="90% CI"
    )
    ax.legend(loc="upper left")

    st.pyplot(fig4)

    return None


def prophet_prediction (train,company):

    ##### put a test here to see that the train data is the data for only 1 company and has an index and a value column

    prophet_model = Prophet()

    df_modify = pd.DataFrame()
    df_modify['ds'] = train.index
    df_modify['y'] = train.values

    prophet_model.fit(df_modify)

    # preparing prediction dataframe 

    future = prophet_model.make_future_dataframe(periods = 100,freq = 'B') 

    prophet_forecast = prophet_model.predict(future)

    st.write(prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    
    fig5 = plt.figure(figsize=(12,4))
    ax = plt.subplot(111)
    _ = prophet_model.plot(prophet_forecast,ax=ax,xlabel='Days',ylabel='Closing value',include_legend=True)
    # ax.plot(test,'r--', label='Observed Data')
    ax.set_title(f'{company} stock forecast using Facebook Prophet')
    ax.legend()

    st.pyplot(fig5)

    return None

def autoregression_plots(data, lags):

    """
        This uses the statsmodels pacf and acf plots functions to create the 
        Autoregression plots for the selected companies time series data

        =====================================
        arguments:

        lags : number of lags steps to be considered for the ACF and PACF plots
    """

    fig6, (ax1,ax2) = plt.subplots(2,1,figsize=(10,8))
    plot_acf(data,lags=lags,ax=ax1)
    plot_pacf(data,lags=lags,ax = ax2)

    st.pyplot(fig6)

    return None
    

