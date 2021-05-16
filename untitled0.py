
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
from itertools import cycle
from random import shuffle
from plotly.validators.scatter.marker import SymbolValidator
from plotly.validators.scatter.line import DashValidator


#Import Libraries and csv files
hol=pd.read_csv('holiday_17_18_19.csv')
ist_ene_17=pd.read_csv('IST_Civil_Pav_2017_Ene_Cons.csv')
ist_ene_18=pd.read_csv('IST_Civil_Pav_2018_Ene_Cons.csv')
ist_meteo=pd.read_csv('IST_meteo_data_2017_2018_2019.csv')

# Data preparation 

#### Set the date and meteo dataframe 

ist_meteo=ist_meteo.rename(columns={'yyyy-mm-dd hh:mm:ss':'Date'})
ist_ene_17=ist_ene_17.rename(columns={'Date_start':'Date'})
ist_ene_18=ist_ene_18.rename(columns={'Date_start':'Date'})

ist_meteo_1=ist_meteo[~ist_meteo.Date.str.contains('2019')]
ist_meteo_17=ist_meteo_1[~ist_meteo_1.Date.str.contains('2018')].copy()
ist_meteo_18=ist_meteo_1[~ist_meteo_1.Date.str.contains('2017')].copy()
ist_meteo_18.index = np.arange(0, len(ist_meteo_18))

ist_ene_17['Date'] = pd.to_datetime(ist_ene_17['Date'])
ist_ene_18['Date'] = pd.to_datetime(ist_ene_18['Date'])
ist_meteo_17['Date'] = pd.to_datetime(ist_meteo_17['Date'])
ist_meteo_18['Date'] = pd.to_datetime(ist_meteo_18['Date'])

ist_ene_17 = ist_ene_17.set_index ('Date', drop = True)
ist_ene_18 = ist_ene_18.set_index ('Date', drop = True)
ist_meteo_17 = ist_meteo_17.set_index ('Date', drop = True).resample('H').mean().bfill()
ist_meteo_18 = ist_meteo_18.set_index ('Date', drop = True).resample('H').mean().bfill()

ist_ene_17.index = pd.to_datetime(ist_ene_17.index, format='%y/%m/%d %H:%M:%S').strftime('%d/%m/%y %H:%M:%S')
ist_ene_18.index = pd.to_datetime(ist_ene_18.index, format='%y/%m/%d %H:%M:%S').strftime('%d/%m/%y %H:%M:%S')
ist_meteo_17.index = pd.to_datetime(ist_meteo_17.index, format='%m/%d/%y %H:%M:%S').strftime('%d/%m/%y %H:%M:%S')
ist_meteo_18.index = pd.to_datetime(ist_meteo_18.index, format='%m/%d/%y %H:%M:%S').strftime('%d/%m/%y %H:%M:%S')

# Merge the data in one table for each year

tot_17=pd.merge(ist_meteo_17,ist_ene_17,on='Date')
tot_18=pd.merge(ist_meteo_18,ist_ene_18,on='Date')

tot_18['day'] = pd.to_datetime(tot_18.index,format='%d/%m/%y %H:%M:%S').weekday
tot_17['day'] = pd.to_datetime(tot_17.index,format='%d/%m/%y %H:%M:%S').weekday

df_data=tot_17.append(tot_18)

fig1 =px.line(x=df_data.index, y=df_data['Power_kW'])
fig1.update_traces(mode='markers+lines', line = dict(color='royalblue', width=1),
                                                     marker=dict(color='firebrick',symbol='octagon',size=3))
fig1.update_layout(margin_pad=20)
fig1.update_yaxes(title_text='Power kW')


fig22 =px.line(x=df_data.index, y=df_data['temp_C'])
fig22.update_traces(mode='markers+lines', line = dict(color='royalblue', width=1),
                                                     marker=dict(color='firebrick',symbol='octagon',size=3))
fig22.update_layout(margin_pad=20)
fig22.update_yaxes(title_text='temperature')

hol['Date'] = pd.to_datetime(hol['Date'])
hol = hol.set_index('Date',drop= True).resample('H').mean()
hol['Datetime']=hol.index
hol.index = pd.to_datetime(hol.index, format='%y/%m/%d  %H:%M:%S').strftime('%d/%m/%y')
hol['Holiday'] = hol.groupby([hol.index])['Holiday'].ffill()

hol['Holiday']=hol['Holiday'].fillna(0)
hol['Datetime'] = pd.to_datetime(hol['Datetime'])
hol=hol.set_index('Datetime',drop= True)
hol.index = pd.to_datetime(hol.index, format='%y/%m/%d %H:%M:%S').strftime('%d/%m/%y %H:%M:%S')

hol=hol[~hol.index.str.contains('/19')]
df_data['Holiday']=hol['Holiday']
df_data['Holiday'] = np.where(df_data['Holiday'] == 0, 1,0)

df_data['time_hour'] = pd.to_datetime(df_data.index).hour

##### Shift the Day in order to have sunday as 0

df_data['day']=df_data['day'].shift(-24)
df_data['day'].fillna(1,inplace=True)
df_data

#Remove data below quantile 0.25

df_dataclean3 = df_data[df_data['Power_kW'] >df_data['Power_kW'].quantile(0.25)]
fig2 =px.line(x=df_dataclean3.index, y=df_dataclean3['Power_kW'])
fig2.update_traces(mode='markers+lines', line = dict(color='royalblue', width=1),
                                                     marker=dict(color='firebrick',symbol='octagon',size=3))

fig2.update_layout(margin_pad=20)
fig2.update_yaxes(title_text='Power kW')


fig23 =px.line(x=df_dataclean3.index, y=df_dataclean3['temp_C'])
fig23.update_traces(mode='markers+lines', line = dict(color='royalblue', width=1),
                                                     marker=dict(color='firebrick',symbol='octagon',size=3))
fig23.update_layout(margin_pad=20)
fig23.update_yaxes(title_text='temperature')
# Clustering

# import KMeans that is a clustering method 
from sklearn.cluster import KMeans
from pandas import DataFrame #to manipulate data frame

# create kmeans object
model = KMeans(n_clusters=3).fit(df_dataclean3) #fit means to develop the model on the given data
pred = model.labels_
df_dataclean3['Cluster']=pred
df1 = df_dataclean3[df_dataclean3.isna().any(axis=1)]


##### Set number of cluster equal to three as it's shown in the figure as the trade off between ccuracy and computational time

## Graphical clustering analysis 

cluster_0=df_dataclean3[pred==0]
cluster_1=df_dataclean3[pred==1]
cluster_2=df_dataclean3[pred==2]


fig3=px.scatter(x=df_dataclean3['temp_C'], y=df_dataclean3['Power_kW'], color=df_dataclean3['Cluster'])
fig4=px.scatter(x=df_dataclean3['time_hour'], y=df_dataclean3['Power_kW'], color=df_dataclean3['Cluster'])
fig5=px.scatter(x=df_dataclean3['day'], y=df_dataclean3['Power_kW'], color=df_dataclean3['Cluster'])

fig3.update_yaxes(title_text='Power kW')
fig3.update_xaxes(title_text='temp_C')
fig4.update_yaxes(title_text='Power kW')
fig4.update_xaxes(title_text='time_hour')
fig5.update_yaxes(title_text='Power kW')
fig5.update_xaxes(title_text='day')
#Silhouettes 

df=df_dataclean3
df=df.drop(columns=['temp_C','HR','windSpeed_m/s','windGust_m/s','pres_mbar','solarRad_W/m2','rain_mm/h','rain_day','Holiday','day'])
df=df.rename(columns = {'Power_kW':'Power'})
df.index = pd.to_datetime(df.index, format='%d/%m/%y %H:%M:%S').strftime('%d/%m/%y')
#Create a pivot table creating a table where une column becomes a pivot  
df_pivot = df.pivot_table(values='Power', index=[df.index],
                    columns=['time_hour'])
df_pivot = df_pivot.dropna()


from sklearn.preprocessing import MinMaxScaler # it normalize the data to find the euclidian distance or it doesn't make sense 
from sklearn.metrics import silhouette_score

sillhoute_scores = []
n_cluster_list = np.arange(2,10).astype(int)

X = df_pivot.values.copy()
    
# Very important to scale!
sc = MinMaxScaler()
X = sc.fit_transform(X)

for n_cluster in n_cluster_list:
    
    kmeans = KMeans(n_clusters=n_cluster)
    cluster_found = kmeans.fit_predict(X)
    sillhoute_scores.append(silhouette_score(X, kmeans.labels_))

#when graph is equal two one is better 
kmeans = KMeans(n_clusters=3)
cluster_found = kmeans.fit_predict(X)
cluster_found_sr = pd.Series(cluster_found, name='cluster')
df_pivot = df_pivot.set_index(cluster_found_sr, append=True )

df_pivot
#two index in the table 

fig6, ax= plt.subplots(1,1, figsize=(18,10))
color_list = ['blue','red','green']
cluster_values = sorted(df_pivot.index.get_level_values('cluster').unique())
for cluster, color in zip(cluster_values, color_list):
    df_pivot.xs(cluster, level=1).T.plot(    #xs prende solo cluster leve=1 prende i numeri 1 Tplot serve per plottare dopo aver fatto operazioni
        ax=ax, legend=False, alpha=0.01, color=color, label= f'Cluster {cluster}')
    df_pivot.xs(cluster, level=1).median().plot(
        ax=ax, color=color, alpha=0.9, ls='--'    )

ax.set_xticks(np.arange(1,25))
ax.set_ylabel('kilowatts')
ax.set_xlabel('hour')

fig6.savefig('fig6.png')

#ax.legend() the difference between green and red is the chang between solare e legale ora 
df_dataclean3=df_dataclean3.drop(columns='Cluster')

# FEATURE SELECTION
df_dataclean3['Power-1_kW']=df_dataclean3['Power_kW'].shift(1) # Previous hour consumption
df_dataclean3=df_dataclean3.dropna()
#Log of temperature
df_dataclean3['logtemp']=np.log(df_dataclean3['temp_C']) 

#Logaritmic temperature allows a better distinction between low and high temperature

# Weekday square
df_dataclean3['day2']=np.square(df_dataclean3['day'])
df_dataclean3.head()

#square of the number of the week day emphasizes the differents between the various day

#Hour parabolic shape
df_dataclean3['Hpshape']=-((np.square(df_dataclean3['time_hour'])/2)-14*df_dataclean3['time_hour']+26)

#Enanche the difference between midday's hour (often characterized by high consumption) with the night's hour (often characterized by high consumption)

df_dataclean3['Holtimesweek']=df_dataclean3['day']*df_dataclean3['Holiday']

#holiday and sundy 0 week days 1

df_dataclean3=df_dataclean3.drop(columns=['HR', 'windSpeed_m/s','windGust_m/s', 'pres_mbar','solarRad_W/m2',
       'rain_mm/h', 'rain_day','day2'])


fig8=go.Figure()
fig8.add_scatter(name="Power",x=df_dataclean3.index, y=df_dataclean3['Power_kW'], mode='lines')
fig8.add_scatter(name="temp_C",x=df_dataclean3.index, y=df_dataclean3['temp_C']*50, mode='lines')
fig8.add_scatter(name="logtemp",x=df_dataclean3.index, y=df_dataclean3['logtemp']*30, mode='lines')

fig8.update_layout(xaxis_range=[4100,4200],showlegend=True)
fig8.update_layout(legend=dict(
    orientation="h",
   yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig9=go.Figure()
fig9.add_scatter(name="Power",x=df_dataclean3.index, y=df_dataclean3['Power_kW'], mode='lines')
fig9.add_scatter(name="day",x=df_dataclean3.index, y=df_dataclean3['day']*100, mode='lines')
fig9.add_scatter(name="Holiday",x=df_dataclean3.index, y=df_dataclean3['Holiday']*100, mode='lines')
fig9.add_scatter(name="timehour",x=df_dataclean3.index, y=df_dataclean3['time_hour']*10, mode='lines')
fig9.add_scatter(name="Holtimesweek",x=df_dataclean3.index, y=df_dataclean3['Holtimesweek']*8, mode='lines')
fig9.add_scatter(name="Hpshape",x=df_dataclean3.index, y=df_dataclean3['Hpshape']*20, mode='lines')
fig9.update_layout(xaxis_range=[4100,4200],showlegend=True)  
fig9.update_layout(legend=dict(
    orientation="h",
 yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))
df_dataclean3=df_dataclean3.drop(columns=['Holiday','day'])

# Clustering after feature selection

# import KMeans that is a clustering method 
from sklearn.cluster import KMeans
from pandas import DataFrame #to manipulate data frame

# create kmeans object
model = KMeans(n_clusters=3).fit(df_dataclean3) #fit means to develop the model on the given data
pred = model.labels_
df_dataclean3['Cluster']=pred
df1 = df_dataclean3[df_dataclean3.isna().any(axis=1)]


##### Set number of cluster equal to three as it's shown in the figure as the trade off between ccuracy and computational time

## Graphical clustering analysis 

cluster_0=df_dataclean3[pred==0]
cluster_1=df_dataclean3[pred==1]
cluster_2=df_dataclean3[pred==2]

fig7=px.scatter(x=df_dataclean3['temp_C'], y=df_dataclean3['Power_kW'], color=df_dataclean3['Cluster'])
fig10=px.scatter(x=df_dataclean3['logtemp'], y=df_dataclean3['Power_kW'], color=df_dataclean3['Cluster'])
fig11=px.scatter(x=df_dataclean3['Hpshape'], y=df_dataclean3['Power_kW'], color=df_dataclean3['Cluster'])
fig12=px.scatter(x=df_dataclean3['time_hour'], y=df_dataclean3['Power_kW'], color=df_dataclean3['Cluster'])
fig13=px.scatter(x=df_dataclean3['Holtimesweek'], y=df_dataclean3['Power_kW'], color=df_dataclean3['Cluster'])
fig10.update_yaxes(title_text='Power kW')
fig10.update_xaxes(title_text='logtemp')
fig11.update_yaxes(title_text='Power kW')
fig11.update_xaxes(title_text='Hpshape')
fig12.update_yaxes(title_text='Power kW')
fig12.update_xaxes(title_text='time_hour')
fig13.update_yaxes(title_text='Power kW')
fig13.update_xaxes(title_text='Holtimesweek')
fig7.update_yaxes(title_text='Power kW')
fig7.update_xaxes(title_text='temp_C')


#Silhouettes 

df=df_dataclean3
df=df.drop(columns=[ 'Power-1_kW','Holtimesweek','logtemp', 'Hpshape'])
df=df.rename(columns = {'Power_kW':'Power'})
df.index = pd.to_datetime(df.index, format='%d/%m/%y %H:%M:%S').strftime('%d/%m/%y')
#Create a pivot table creating a table where une column becomes a pivot  
df_pivot = df.pivot_table(values='Power', index=[df.index],
                    columns=['time_hour'])
df_pivot = df_pivot.dropna()
df_pivot.T.plot(figsize=(13,8), legend=False, color='blue', alpha=0.02).set_ylabel('kilowatts')


from sklearn.preprocessing import MinMaxScaler # it normalize the data to find the euclidian distance or it doesn't make sense 
from sklearn.metrics import silhouette_score

sillhoute_scores = []
n_cluster_list = np.arange(2,10).astype(int)

X = df_pivot.values.copy()
    
# Very important to scale!
sc = MinMaxScaler()
X = sc.fit_transform(X)

for n_cluster in n_cluster_list:
    
    kmeans = KMeans(n_clusters=n_cluster)
    cluster_found = kmeans.fit_predict(X)
    sillhoute_scores.append(silhouette_score(X, kmeans.labels_))

#when graph is equal two one is better 
kmeans = KMeans(n_clusters=3)
cluster_found = kmeans.fit_predict(X)
cluster_found_sr = pd.Series(cluster_found, name='cluster')
df_pivot = df_pivot.set_index(cluster_found_sr, append=True )

df_pivot
#two index in the table 

fig14, ax= plt.subplots(1,1, figsize=(18,10))
color_list = ['blue','red','green']
cluster_values = sorted(df_pivot.index.get_level_values('cluster').unique())
for cluster, color in zip(cluster_values, color_list):
    df_pivot.xs(cluster, level=1).T.plot(    #xs prende solo cluster leve=1 prende i numeri 1 Tplot serve per plottare dopo aver fatto operazioni
        ax=ax, legend=False, alpha=0.01, color=color, label= f'Cluster {cluster}')
    df_pivot.xs(cluster, level=1).median().plot(
        ax=ax, color=color, alpha=0.9, ls='--'    )

ax.set_xticks(np.arange(1,25))
ax.set_ylabel('kilowatts')
ax.set_xlabel('hour')

fig14.savefig('fig14.png')

#ax.legend() the difference between green and red is the chang between solare e legale ora 
df_dataclean3=df_dataclean3.drop(columns='Cluster')

#REGRESSION
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import  metrics

# recurrent
X=df_dataclean3.values
Y=X[:,1]
X=X[:,[2,3,4,5,6]] 

#by default, it chooses randomly 75% of the data for training and 25% for testing
X_train, X_test, y_train, y_test = train_test_split(X,Y)

#Linear Regression
from sklearn import  linear_model

regr = linear_model.LinearRegression()

regr.fit(X_train,y_train)

y_pred_LR = regr.predict(X_test)

MAE_LR=metrics.mean_absolute_error(y_test,y_pred_LR) 
MSE_LR=metrics.mean_squared_error(y_test,y_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y_test)


err= {'Methods':['LR'], 'Error':[cvRMSE_LR]}
err=pd.DataFrame(err)

fig15=go.Figure()
fig15.add_scatter(name="Ytest",y=y_test, mode='lines')
fig15.add_scatter(name="y_pred_LR",y=y_pred_LR, mode='lines')
fig16=px.scatter(x=y_test, y=y_pred_LR)
fig15.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig16.update_yaxes(title_text='y_pred_LR')
fig16.update_xaxes(title_text='y_test')
#Random forest

from sklearn.ensemble import RandomForestRegressor

parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 200, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 20,
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)
#RF_model = RandomForestRegressor()
RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_test)
#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)

new_row = {'Methods':'RF', 'Error':cvRMSE_RF}
err = err.append(new_row, ignore_index=True)
fig17=go.Figure()
fig17.add_scatter(name="Ytest",y=y_test, mode='lines')
fig17.add_scatter(name="y_pred_RF",y=y_pred_RF, mode='lines')
fig18=px.scatter(x=y_test, y=y_pred_RF)

fig17.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig18.update_yaxes(title_text='y_pred_RF')
fig18.update_xaxes(title_text='y_test')



#Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor

#params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
#          'learning_rate': 0.01, 'loss': 'ls'}
#GB_model = GradientBoostingRegressor(**params)
GB_model = GradientBoostingRegressor()
GB_model.fit(X_train, y_train)
y_pred_GB =GB_model.predict(X_test)

MAE_GB=metrics.mean_absolute_error(y_test,y_pred_GB) 
MSE_GB=metrics.mean_squared_error(y_test,y_pred_GB)  
RMSE_GB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(y_test)


new_row = {'Methods':'GB', 'Error':cvRMSE_GB}
err = err.append(new_row, ignore_index=True)

fig19=go.Figure()
fig19.add_scatter(name="Ytest",y=y_test, mode='lines')
fig19.add_scatter(name="y_pred_GB",y=y_pred_GB, mode='lines')
fig20=px.scatter(x=y_test, y=y_pred_GB)

fig20.update_yaxes(title_text='y_pred_GB')
fig20.update_xaxes(title_text='y_test')
fig19.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))
err=err.set_index('Methods',drop = False)
fig21=px.bar(err,x="Methods",y="Error")


import dash
import dash_core_components as dcc
import dash_html_components as html
import base64
from dash.dependencies import Input, Output
import pandas as pd


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#external_stylesheets = ['mystyle.css']

image_filename ='fig6.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

image_filename2 ='fig14.png'
encoded_image2 = base64.b64encode(open(image_filename2, 'rb').read())
#html.Div([html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
   #     ])
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Img(src=app.get_asset_url('IST_logo.png')),
    html.H2('Edoardo Rossi ist1100824'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Exploratory Data Analysis', value='tab-1'),
        dcc.Tab(label='Clustering', value='tab-2'),
        dcc.Tab(label='Feature Selection', value='tab-3'),
        dcc.Tab(label='Regression', value='tab-4'),
        
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
             

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Exploratory Data Analysis'),
            dcc.RadioItems(
        id='radio',
        options=[
            {'label': 'Raw Data', 'value': 1},
            {'label': 'Clean Data', 'value': 2}
        ], 
        value=1,
        labelStyle={'display': 'inline-block'}
        ),
        html.Div(id='EDA_html'),
                    ],style={'textAlign': 'center'}) 
    
    elif tab == 'tab-2':
        return html.Div([
            html.H2('Clustering'),
            dcc.RadioItems( 
        id='radio2',
        options=[
            {'label': 'Power vs Temperature', 'value': 1},
            {'label': 'Power vs Hour', 'value': 2},
            {'label': 'Power vs Day', 'value': 3},
            {'label': 'Silhouettes Score', 'value': 4},
        ], 
        value=1,
        labelStyle={'display': 'inline-block'}
        ),
        html.Div([
            html.Div([
                html.H3('Clustering performed before Feature Selection'),
                html.Div(id='cluster1')
                ], 
            className="six columns"
            ),
           
            html.Div([
                 html.H3('Clustering performed after Feature Selection'),
                 html.Div(id='cluster2')
                ], 
            className="six columns"
            ),
    ], 
        className="row"
        )
],style={'textAlign': 'center',})
            
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Feature selection'),
             dcc.RadioItems( 
        id='radio3',
        options=[
            {'label': 'Power vs Temperature', 'value': 1},
            {'label': 'Power vs Day Features', 'value': 2},
        ], 
        value=1,
        labelStyle={'display': 'inline-block'}
        ),
        html.Div(id='Featureselection_html'),
            
        ],style={'textAlign': 'center'})
    
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Regression'),
                dcc.RadioItems( 
        id='radio4',
        options=[
            {'label': 'Linear', 'value': 1},
            {'label': 'Random Forest', 'value': 2},
            {'label': 'Gradient Boosting', 'value': 3},
            {'label': 'Errors of the Forcasting Models', 'value': 4},
        ], 
        value=1,
                labelStyle={'display': 'inline-block'}
        ),
          html.Div([
            html.Div([
                html.Div(id='Regression1')
                ], 
            className="six columns"
            ),
            html.Div([
                 html.Div(id='Regression2')
                ], 
            className="six columns"
            ),
    ], 
        className="row"
        )
],style={'textAlign': 'center'})

@app.callback(Output('EDA_html', 'children'), 
              Input('radio', 'value'))

def render_figure_html(EDA_RI):
    
    if EDA_RI == 1:
        return html.Div([html.H5('Raw Data'),
            html.Div([dcc.Slider(
        min=1,
        max=2,
        step=None,
        marks={1: 'Power kW', 2: 'Temperature'},
        value=1,
        
        id='slider4'
    ),
],style={'marginLeft': 100, 'marginRight': 100}),
    html.Div(id='slider5')
    ],style={'textAlign': 'center'},)
    elif EDA_RI == 2:
        return  html.Div([html.H5('Elimination of Data below Quantile 0.25'), 
            html.Div([dcc.Slider(
        min=1,
        max=2,
        step=None,
        marks={1: 'Power kW', 2: 'Temperature'},
        value=1,
        id='slider6'
    ),
    html.Div(id='slider7')
    ],style={'marginLeft': 100, 'marginRight': 100})
])             
@app.callback(Output('cluster1', 'children'), 
              Input('radio2', 'value'))


def render_figure_html(cluster_RI):
    
    if cluster_RI == 1:
        return html.Div([dcc.Graph(figure=fig3),])
    elif cluster_RI == 2:
        return html.Div([dcc.Graph(figure=fig4),])
    elif cluster_RI == 3:
        return html.Div([dcc.Graph(figure=fig5),])       
    elif cluster_RI == 4:
        return html.Div([html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),style={'height':'100%','width':'100%'})
        ])
    
@app.callback(Output('cluster2', 'children'), 
              Input('radio2', 'value'))


def render_figure_html(cluster_RI2):
    
   if cluster_RI2 == 1:
            return html.Div([
                html.Div([dcc.Slider(
        min=1,
        max=2,
        step=None,
        marks={1: 'temp_C', 2: 'logtemp'},
        value=1,
         id='slider'
    )],style={'marginLeft': 100, 'marginRight': 100}),
    html.Div(id='slider1')
    ])
        
   elif cluster_RI2 == 2:
            return html.Div([
                html.Div([dcc.Slider(
        min=1,
        max=2,
        step=None,
        marks={1: 'timehour', 2: 'Hpshape'},
        value=1,
        id='slider2'
    )],style={'marginLeft': 100, 'marginRight': 100}),
    html.Div(id='slider3')
    ])
   elif cluster_RI2 == 3:
            return html.Div([dcc.Graph(figure=fig13),])  
   elif cluster_RI2 == 4:
            return html.Div([html.Img(src='data:image/png;base64,{}'.format(encoded_image2.decode()),style={'height':'100%','width':'100%'})
        ])
   
@app.callback(Output('Featureselection_html', 'children'), 
              Input('radio3', 'value'))

def render_figure_html(FS_RI):
    
    if FS_RI == 1:
        return html.Div([dcc.Graph(figure=fig8),])
    elif FS_RI == 2:
        return html.Div([dcc.Graph(figure=fig9),])
    
@app.callback(Output('Regression1', 'children'), 
              Input('radio4', 'value'))

def render_figure_html(Regression_RI1):
    
    if Regression_RI1 == 1:
        return html.Div([dcc.Graph(figure=fig15),])
    elif Regression_RI1 == 2:
        return html.Div([dcc.Graph(figure=fig17),])
    elif Regression_RI1 == 3:
        return html.Div([dcc.Graph(figure=fig19),])
    elif Regression_RI1 == 4:
        return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in err.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(err.iloc[i][col]) for col in err.columns
            ]) for i in range(len(err))
        ])
    ],style={'marginLeft': 'auto', 'marginRight': 'auto', 'marginTop': '8vw'})
    
@app.callback(Output('Regression2', 'children'), 
              Input('radio4', 'value'))

def render_figure_html(Regression_RI2):
    
    if Regression_RI2 == 1:
        return html.Div([dcc.Graph(figure=fig16),])
    elif Regression_RI2 == 2:
        return html.Div([dcc.Graph(figure=fig18),])
    elif Regression_RI2 == 3:
        return html.Div([dcc.Graph(figure=fig20),])
    elif Regression_RI2 == 4:
        return html.Div([dcc.Graph(figure=fig21),])
    
@app.callback(Output('slider1', 'children'), 
              Input('slider', 'value')) 
  
def render_figure_html(Slider_clusterT):
      if Slider_clusterT == 1:
        return html.Div([dcc.Graph(figure=fig7),])
      elif Slider_clusterT == 2:
        return html.Div([dcc.Graph(figure=fig10),])

@app.callback(Output('slider3', 'children'), 
              Input('slider2', 'value')) 
  
def render_figure_html(Slider_clusterH):
      if Slider_clusterH == 1:
        return html.Div([dcc.Graph(figure=fig12),])
      elif Slider_clusterH == 2:
        return html.Div([dcc.Graph(figure=fig11),])
    
@app.callback(Output('slider5', 'children'), 
              Input('slider4', 'value')) 
    
def render_figure_html(Slider_dataraw):
      if Slider_dataraw == 1:
        return html.Div([dcc.Graph(figure=fig1, config={
                             'displayModeBar': False})
                        ])
    
      elif Slider_dataraw == 2:
        return html.Div([dcc.Graph(figure=fig22,config={
                             'displayModeBar': False})])
    
@app.callback(Output('slider7', 'children'), 
              Input('slider6', 'value')) 

def render_figure_html(Slider_dataclean):
      if Slider_dataclean == 1:
        return html.Div([dcc.Graph(figure=fig2),])
      elif Slider_dataclean == 2:
        return html.Div([dcc.Graph(figure=fig23),])
    

if __name__ == '__main__':
    app.run_server(debug=True) 


