from helper_functions import *
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA as arima_model


import streamlit as st 
import datetime

######### initial setup 

companyNames = ['AMAZON', 'APPLE', 'GOOGLE', 'META', 'NETFLIX']
st.set_page_config(page_title='TSA@streamlit', layout='wide')
st.title(':orange[Time Series Analysis]')


######### About the app ###############
markdown_about_msg = """
        
        ## Introduction

        The app implements the fundamental steps in time series analysis and forecasting. 
        It lets you play around with key paramters and visualise their effect in the statistics and prediction. Currently the app
        is using a model data set (source below)

        Data source :  KAGGLE : [link](https://www.kaggle.com/datasets/nikhil1e9/netflix-stock-price) to the data set 
        Companies used in the dataset MAANG = Meta, Apple, Amazon, Netflix and, Google

        
        :blue[KeyWords] :  **Time Series Analysis, ARMA, SeriesDecomposition, Forecasting**

    """



############ SIDEBAR for setting the parameters ##########################
with st.sidebar:
    st.header(':red[Chose your Parameters]')
    st.write(" :violet[Running on Streamlit version] -- " + st.__version__)


    # dates = st.date_input(label=':orange[Enter the date range for the analysis]',
    #                       value = (datetime.date(2019,1,1), datetime.date(2024,1,10)),
    #                       min_value=, 
    #                       max_value=datetime.date(2024,1,10),
    #                       format="YYYY-MM-DD")
    
    minDate = st.date_input(label='Enter :orange[minimum] date for analysis', value=datetime.date(2019,1,1),
                             min_value=datetime.date(2018,1,1),
                             max_value=datetime.date(2023,1,1),format="YYYY-MM-DD")
    
    maxDate = st.date_input(label='Enter :orange[maximum] date for analysis', value=datetime.date(2024,1,10),
                             min_value=datetime.date(2022,1,1),
                             max_value=datetime.date(2024,1,10))
    
    if minDate > maxDate:

        st.warning('Minimum Date should be earlier than maximum Date')
    
    # minDate,maxDate = str(dates[0]), str(dates[1])
    logData = st.radio(':orange[Logged Values]', options = [True, False],index=None)
    company = st.radio(':orange[Chose the company to analyse]', options=companyNames)
    window_size = st.slider(':orange[Chose the rolling window size]',min_value=5,max_value=50,value=28)
    monthly_plot = st.button('Show Monthly Plot')


    st.subheader('Lags for ACF and PACF plots')

    lags = st.slider("Set the Lag", min_value=5,max_value=100,value=50)

    ts_decompose_model = st.radio('# :orange[Choose Decomposition Model]', options=['additive', 'multiplicative'])

    split_at = st.number_input(':orange[Split data into Train-Test starting from the end]', 
                               min_value=100,max_value=500,value=250,step=50)
    
    ## ARIMA model parameters
    st.subheader(':green[ARIMA model parameters]')
    p = st.number_input(':green[p]', min_value=0,max_value=30,value=1)
    d = st.number_input(':green[d]', min_value=0,max_value=30,value=1)
    q = st.number_input(':green[q]', min_value=0,max_value=30,value=0)
    


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["About", "Exploration", "ARIMA models",
                                              "Prophet", "DeepLearning Models", "Next Steps"])


with tab1:

    st.markdown(markdown_about_msg)

    def load_data(logData):
        df = create_AnalysisData(companyNames,minDate,maxDate,logData)
        return df

    DF = load_data(logData)

    col1,col2 = st.columns(2,gap="medium")

    with col1:
        st.subheader(':orange[Data frame head]')
        st.dataframe(DF.head(7))

    with col2:

        st.subheader(f':orange[Time Series View]',)
        st.line_chart(DF)

    with st.expander(':orange[Expand to see the summary statistics]'):

        st.write((DF.describe()))


    st.divider()

    ##### End Message
    st.markdown(':orange[:heart: and 🕊️ for all - Chakresh]')

with tab2:

    st.subheader(f'Analysis for :orange[{company}] time series Data: ')

    timeSeriesData = DF[company]
    timeSeriesData_rolling = timeSeriesData.rolling(window_size).mean().dropna()
    st.line_chart(timeSeriesData_rolling)


    if monthly_plot:
        st.subheader(f'Monthly price plot for :orange[{company}] data')
        show_monthly_sale(timeSeriesData)

    st.divider()


    st.subheader('Time Series Decomposition')

    st.write(f'with decomposition model as :blue[{ts_decompose_model}]')

    ts_decomposition = seasonal_decompose(x=timeSeriesData[:-1],model=ts_decompose_model,period=30)

    T,S,R = ts_decomposition.trend, ts_decomposition.seasonal, ts_decomposition.resid

    with st.expander("See the Trend, Seasonality and Residual Plots"):

        st.subheader('Trend')
        st.line_chart(T)
        st.subheader('Seasonality')
        st.line_chart(S)
        st.subheader('Residual')
        st.line_chart(R,width=1)


    # st.header()
        
    st.subheader('Exponential Smoothing - (Done on the rolling average series)')
    st.markdown(r"""
                
            Using `statsmodels.tsa.api` to generate different exponential smoothing. 
                
            Single Exponential Smoothing - Trend - ❌, Seasonality - ❌
                
            $$
                s_t = \alpha x_t + (1 - \alpha) s_{t-1} \\
                0 \leq \alpha \leq 1
            $$

            Double Exponential Smoothing - Trend - ✅, Seasonality - ❌
            
            $$
                s_t = \alpha x_t + (1 - \alpha) (s_{t-1} + b_{t-1}) \\
                b_t = \beta (s_t - s_{t-1}) + (1 - \beta) b_{t-1} \\ 

                0 \leq \alpha,\beta \leq 1
            $$

            :orange[$\alpha$] : Data Smoothing Factor
            :orange[$\beta$] : Trend Smoothing Factor
                
            Triple Exploential Smoothing - Trend - ✅, Seasonality - ✅
                
            $$
                s_0 = x_0 \\
                s_t = \alpha \frac{x_t}{c_t - L} + (1 - \alpha) (s_{t-1} + b_{t-1}) \\
                b_t = \beta (s_t - s_{t-1}) + (1 - \beta) b_{t-1} \\ 
                c_t = \gamma \frac{x_t}{s_t} + (1 - \gamma) c_{t-L} \\
                
                0 \leq \alpha, \beta, \gamma \leq 1
            $$
                
            :orange[$\gamma$] : Seasonal change smoothing factor, :orange[$\alpha, \beta$] same as above
            
    """)


    smoothing_type = st.multiselect(':orange[Chose Smoothing Type]',options=['Single', 'Double', 'Triple'], default=['Single'])

    # Creating the training and the test data. We do it outside the functions for increased scope. 
    training_Data,test_Data = timeSeriesData_rolling[:-split_at], timeSeriesData_rolling[-split_at:]


    gen_smooth = st.button('Generate Smoothing ')

    if gen_smooth:
        generate_smoothing(smoothing_type,ts_decompose_model,training_Data,test_Data,timeSeriesData_rolling,split_at)

    
    st.subheader('Autoregression Plots')

    show_arplots = st.button("Show ACF and PACF plots")

    if show_arplots:
    
        autoregression_plots(timeSeriesData_rolling,lags=lags)
    
    
    
    st.divider()



with tab3:

    st.subheader('Time Series Models - ARIMA (p,d,q)')

    st.markdown(""" ##### Before we fit ARIMA models we run stationarity tests on the time series""")

    with st.expander('Stationarity Test (KPSS) Results'):

        check_stationarity(training_Data)

    arima_model = arima_model(training_Data,order=(p,d,q))
    arima_resid = arima_model.fit()



    with st.expander('Model fit summary and diagnostics plots for arima model'):

        st.write(arima_resid.summary())
        fig3 = plt.figure()
        ax = fig3.add_subplot(111)


        st.pyplot(arima_resid.plot_diagnostics(figsize=(12,10)))


    arima_forecast = st.button('Get ARIMA forecast')

    if arima_forecast:
        arima_preds,conf_90,conf_95 = get_arima_forecast(arima_resid,split_at)
        train_time,test_time = timeSeriesData_rolling.index.values[:-split_at], timeSeriesData_rolling.index.values[-split_at:]
        plot_arima_forecast(arima_preds,conf_90,conf_95,p,d,q,company,training_Data,test_Data,train_time,test_time)


    st.divider()

with tab4:

    st.subheader('Facebook - Prophet')

    prophet_prediction(timeSeriesData_rolling,company)

    st.divider()


with tab5:

    # st.header('Deep Learning')

    st.write("""
        Coming soon .......... 

""")

    st.divider()


with tab6:
    
    next_steps = """

        ### :red[what to expect in future versions]

        1. `Load Data` option to use the app for analysing any time-series. 

        2. `User defined` arcitecture for the RNN and LSTM models
        
        3. `SARIMA models`
        
        4. `Theoretical` explanantion for the parameters search in statistical models
        
        5. `Scraping` data from online 

        6. Implementation of `quantum algorithms` in forecasting.  
    """

    st.markdown(next_steps)

    st.divider()