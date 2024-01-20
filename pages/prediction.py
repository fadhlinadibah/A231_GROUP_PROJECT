import dash 
from dash import html, Input, Output, callback, dcc
import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn import metrics
from math import sqrt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from ThymeBoost import ThymeBoost as tb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta


dash.register_page(__name__, path='/prediction', name="Prediction 📝")

# Load Dataset
url = 'https://storage.data.gov.my/healthcare/covid_cases.csv'
data = pd.read_csv(url)

# Filtering records containing state = 'Malaysia' 
df = data.loc[data['state'] == 'Malaysia']

# Reset index of data frame
df.reset_index(drop=True,inplace=True)

# Dropping columns 
df = df.drop(['cases_import','cases_recovered', 'cases_active', 'cases_cluster'],axis=1)

# Checking null values and dropping that records
df = df.dropna()

# Converting data type of column 'date' to datetime data type
df['date'] = df['date'].astype('datetime64[ns]')
# New Data Types of Columns : 
df.dtypes

# Creating datetime object with date 25th January, 2020 and storing it in variable x
x = datetime.datetime(2020,1,25)
# Subtracting x from dates stored in date column of data frame and storing the result in new column 'days'
# Inserting column 'days' as first column
df.insert(0,'days','')
df['days'] = (df['date']-x).dt.days

# Days stores values in days columns of data frame df
# Cases stores cases values in cases columns of data frame df
days = np.asarray(df['days'],dtype=int)
cases = np.asarray(df['cases_new'],dtype=int)

# Plotting Line plot of number of cases vs days
fig=plt.figure(figsize=(12,5), dpi= 100, facecolor='w', edgecolor='k')
plt.plot(days,cases)
plt.xlabel('Number of Days since 25th January, 2020')
plt.ylabel('Number of Cases reported in Malaysia')
plt.title('Covid Cases in Malaysia')
#plt.show()

# Visualizing last 10 days covid cases 
fig=plt.figure(figsize=(12,5), dpi= 100, facecolor='w', edgecolor='k')
plt.plot(days[-10:],cases[-10:])
plt.xlabel('Number of Days since 25th January, 2020')
plt.ylabel('Number of Cases reported in Malaysia in last 10 days')
plt.title('Covid Cases in Malaysia')
#plt.show()

# Accuracy metrics used for prediction
def forecast_accuracy(forecast,actual):
    mape = np.mean(np.abs(forecast-actual)/np.abs(actual+1))  # MAPE
    me = np.mean(forecast-actual)             # ME
    mae = np.mean(np.abs(forecast-actual))    # MAE
    mpe = np.mean((forecast-actual)/(actual+1))   # MPE
    rmse = np.mean((forecast-actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast,actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast.values[:, None], actual.values[:, None]]), axis=1)
    maxs = np.amax(np.hstack([forecast.values[:, None], actual.values[:, None]]), axis=1)
    minmax = 1-np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})

# Adfuller is used for ADF Test to determine the stationary of data
result = adfuller(df['cases_new'])
print('ADF Statistic: ',result[0])
print('p-value: %f',result[1])

# ACF plot to determine q value in Sarima model
#Showing first 20 lags.
plot_acf(df['cases_new'],lags=20)
#plt.show()

# PACF plot to determine p value in Sarima model
plot_pacf(df['cases_new'],lags=20)
#plt.show()

# Fitting ARIMA model
model = SARIMAX(df['cases_new'],order=(3,0,9))  #order = (p,d,q)
model_fit = model.fit(disp=0)

# Forecast of covid cases for next 15 days
fc = model_fit.forecast(15)
dates = np.zeros(15,dtype=object)
i = 0
curr_date = df['date'][df.shape[0]-1]
while(i < 15):
  dat = curr_date+timedelta(i+1)
  dates[i] = dat
  i += 1
res_dates = pd.DataFrame(data=dates,columns=['date'])
res_days = pd.DataFrame(data=fc.index,columns=['Days'])
res_cases = pd.DataFrame(data=fc.values,columns=['cases_new'])
res_df = pd.concat([res_dates,res_days,res_cases],axis=1)
res_df

# Plot training and forecast 
fcast = model_fit.get_forecast(15)
conf = fcast.conf_int(alpha=0.05)
lower_series = pd.Series(conf.iloc[:,0])
upper_series = pd.Series(conf.iloc[:,1])
fc_series = pd.Series(fc,index=fc.index)

# Plot
plt.figure(figsize=(12,5),dpi=100)
plt.plot(df['cases_new'][-50:],label='training',c='b')
plt.plot(fc_series,label='forecast',c='k')
plt.fill_between(lower_series.index, lower_series, upper_series, color='r', alpha=.05)
plt.legend(loc='upper left',fontsize=8)
plt.title('Forecast')
#plt.show()

# Creating training and testing sets
Xtrain = df['cases_new'][:-15]    #Training set
Xtest = df['cases_new'][-15:]     #Test set

# Fitting SARIMA model for forecasting 
model_1 = SARIMAX(Xtrain,order=(20,0,3))  #order = (p,d,q)
model_fit_1 = model_1.fit(disp=0)
     
# Forecast of covid cases for next 15 days.
fc_1 = model_fit_1.forecast(15)
error = forecast_accuracy(fc_1,Xtest)
print("Errors are : ",error)

# Plotting training and forecast 
fcast = model_fit_1.get_forecast(15)
conf = fcast.conf_int(alpha=0.05)
lower_series = pd.Series(conf.iloc[:,0])
upper_series = pd.Series(conf.iloc[:,1])
fc_series = pd.Series(fc_1,index=fc_1.index)

# Plot
plt.figure(figsize=(12,5),dpi=100)
plt.plot(Xtrain[-50:],label='training',c='b')
plt.plot(Xtest,label='test',c='r')
plt.plot(fc_series,label='forecast',c='k')
plt.fill_between(lower_series.index, lower_series, upper_series, color='r', alpha=.05)
plt.legend(loc='upper left',fontsize=8)
plt.title('Forecast vs Actuals')
#plt.show()

# Layout of the Dash app
layout = html.Div([
    # Chart 1: Line plot of number of cases vs days
    dcc.Graph(
        id='line-plot',
        figure={
            'data': [
                {'x': days, 'y': cases, 'type': 'line', 'name': 'Number of Cases'},
            ],
            'layout': {
                'title': 'Covid Cases in Malaysia',
                'xaxis': {'title': 'Number of Days since 25th January, 2020'},
                'yaxis': {'title': 'Number of Cases reported in Malaysia'},
            }
        }
    ),

    # Chart 2: Line plot of last 10 days covid cases
    dcc.Graph(
        id='last-10-days-plot',
        figure={
            'data': [
                {'x': days[-10:], 'y': cases[-10:], 'type': 'line', 'name': 'Number of Cases (Last 10 days)'},
            ],
            'layout': {
                'title': 'Covid Cases in Malaysia (Last 10 days)',
                'xaxis': {'title': 'Number of Days since 25th January, 2020'},
                'yaxis': {'title': 'Number of Cases reported in Malaysia'},
            }
        }
    ),

    # Chart 3: Forecast vs Actuals
    dcc.Graph(
        id='forecast-vs-actuals',
        figure={
            'data': [
                {'x': Xtrain.index, 'y': Xtrain, 'type': 'line', 'name': 'Training'},
                {'x': Xtest.index, 'y': Xtest, 'type': 'line', 'name': 'Test'},
                {'x': fc_series.index, 'y': fc_series, 'type': 'line', 'name': 'Forecast'},
            ],
            'layout': {
                'title': 'Forecast vs Actuals',
                'xaxis': {'title': 'Number of Days since 25th January, 2020'},
                'yaxis': {'title': 'Number of Cases'},
            }
        }
    ),

    # Display MAE value
    html.Div([
        html.Label('Mean Absolute Error (MAE):'),
        html.Div(id='mae-value', style={'font-weight': 'bold'}),
    ]),
])

# Callback to update the MAE value dynamically
@callback(
    Output('mae-value', 'children'),
    [Input('forecast-vs-actuals', 'figure')]
)
def update_mae_value(figure):
    # Calculate MAE from the forecast vs actuals chart
    forecast_values = figure['data'][2]['y']
    actual_values = Xtest
    mae_value = metrics.mean_absolute_error(actual_values, forecast_values)
    return f'MAE: {mae_value:.2f}'