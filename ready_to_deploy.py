# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 12:17:27 2023

@author: sagar
"""
import pandas as pd
import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
import holidays
import xgboost as xgb
import plotly.express as px
import base64
import datetime
from sklearn.preprocessing import LabelEncoder

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("power.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://miro.medium.com/max/1280/0*AN8suioCkeRkugES.gif");
background-size: 120%;
background-position: top left;
background-repeat: repeat;
background-attachment: local;
}}
[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
html_temp = """
 <div style ="background-color:lightseagreen;padding:13px">
 <h1 style ="color:black;text-align:center;">Model Deployment:Streamlit Time Series Forcasting on power conspution using XG boosting </h1>
 </div>
 """
st.title(":bar_chart: Time Series Forcasting")
st.markdown('##')
st.markdown(page_bg_img,unsafe_allow_html=True)
st.markdown(html_temp, unsafe_allow_html = True)

def user_input_features():
    Day = st.sidebar.selectbox('please enter the date ie.day',(range(1,32,1)))
    year = st.sidebar.selectbox('which year data you need to see',(range(2002,2019,1)))
    month = st.sidebar.selectbox('please enter the month',(1,2,3,4,5,6,7,8,9,10,11,12))
    hour = st.sidebar.selectbox("Insert the time",(range(0,24,1)))
    name_of_week = st.sidebar.selectbox("Insert name of week",(range(0,853,1)))
    st.sidebar.header('*Enter 1 if month starts from march 20 and ends at june 20')
    st.sidebar.header('*Enter 2 if month starts from june 21 and ends at sept 22')
    st.sidebar.header('*Enter 0 if month starts from sept 23 and ends at Dec 21')
    st.sidebar.header('*Enter 3 if month starts from dec 22 and ends at march19')
    season=st.sidebar.selectbox("Insert the time",(0,1,2,3))
    st.sidebar.header('*Enter 1 if there is holiday other wise enter 0')
    holiday=st.sidebar.selectbox('select 1 if there is holiday other wise 0',(0,1))
        
    data = {'Day':Day,
            'year':year,
            'month':month,
            'hour':hour,
            'name_of_week':name_of_week,
            'season':season,
            'holiday':holiday
            }
    
    features = pd.DataFrame(data,index = [0])
    return features 

df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

st.sidebar.subheader('Uploading the Dataset')
power_conspution = pd.read_excel('PJMW_MW_Hourly.xlsx')
power_conspution=power_conspution.sort_values('Datetime')
power_conspution['Day']=power_conspution['Datetime'].dt.day
power_conspution['dayofyear']=power_conspution['Datetime'].dt.dayofyear
power_conspution['year']=power_conspution['Datetime'].dt.year
power_conspution['month']=power_conspution['Datetime'].dt.month
power_conspution['hour']=power_conspution['Datetime'].dt.hour
power_conspution['name_of_week']=power_conspution['Datetime'].dt.isocalendar().week
power_conspution['season']=power_conspution['Datetime'].apply(lambda x: 'Winter' if x.month == 12 or x.month == 1 or x.month == 2  else 'Spring' if x.month==3 or x.month==4 or x.month==5 else 'Summer' if x.month==6 or x.month==7 or x.month==8 else 'Autumn' if x.month==9 or x.month==10 or x.month==11 else "")
power_conspution['name_of_week']=power_conspution['name_of_week'].astype(np.int32)
holidays=holidays.US()
power_conspution['holiday']=power_conspution.Datetime.map(lambda x:x in holidays)
power_conspution=power_conspution.reset_index()
power_conspution=power_conspution.drop('index',axis=1)
power_conspution['season']=LabelEncoder().fit_transform(power_conspution['season'])
power_conspution['holiday']=LabelEncoder().fit_transform(power_conspution['holiday'])
st.sidebar.header('User Input Parameters')


#data imputation
power_conspution.iloc[10148,1]=4560
new_values=power_conspution.iloc[87578:87607,1]
power_conspution.iloc[87650:87679,1]=new_values
power_conspution=power_conspution.set_index('Datetime')
values=power_conspution.iloc[103040:103051,1]
power_conspution.iloc[103064:103075,1]=values
new_values=power_conspution.iloc[102602:102723,1]
power_conspution.iloc[103130:103251,1]=new_values
val=power_conspution.iloc[112847:112892,1]
power_conspution.iloc[112943:112988,1]=val
q=power_conspution.iloc[112903:112975,1]
power_conspution.iloc[112999:113071,1]=q
power_conspution = power_conspution.dropna()

X=power_conspution[['Day','year','month','hour', 'name_of_week', 'season','holiday']]
Y=power_conspution[['PJMW_MW']]
model=xgb.XGBRegressor(base_score=1, booster='gbtree',    
                       n_estimators=1000,
                       objective='reg:squarederror',
                       max_depth=10,
                       learning_rate=0.1,gamma=1)
model.fit(X, Y,verbose=100)

  
prediction = model.predict(df)
st.subheader('Predicted Result')
st.success('The output is {}MW'.format(np.round(prediction,2)))

st.subheader('Power consuption data from 2002 to 2018')
fig=px.line(power_conspution,x=power_conspution.index,y=power_conspution['PJMW_MW'])
fig.update_xaxes(rangeslider_visible=True,rangeselector=dict(buttons=list([dict(count=1,label='one year',step='year',stepmode='backward')
                                                                          ,dict(count=2,label='two year',step='year',stepmode='backward'),
                                                                          dict(count=3,label='three year',step='year',stepmode='backward'),dict(step='all')])))
fig.update_traces(line_color='firebrick')
st.plotly_chart(fig)


#ploting pie plots for visual understandings 
    
new_d=pd.read_excel('PJMW_MW_Hourly.xlsx')
new_d=new_d.sort_values('Datetime')
new_d['Day']=new_d['Datetime'].dt.day
new_d['dayofyear']=new_d['Datetime'].dt.dayofyear
new_d['year']=new_d['Datetime'].dt.year
new_d['month']=new_d['Datetime'].dt.month
new_d['hour']=new_d['Datetime'].dt.hour
new_d['name_of_week']=new_d['Datetime'].dt.isocalendar().week
new_d['season']=new_d['Datetime'].apply(lambda x: 'Winter' if x.month == 12 or x.month == 1 or x.month == 2  else 'Spring' if x.month==3 or x.month==4 or x.month==5 else 'Summer' if x.month==6 or x.month==7 or x.month==8 else 'Autumn' if x.month==9 or x.month==10 or x.month==11 else "")

fig,axs=plt.subplots(3,1,figsize=(2,6))
year=new_d.year.unique()
value=[new_d[new_d['year']==2002]['PJMW_MW'].mean(),new_d[new_d['year']==2003]['PJMW_MW'].mean(),new_d[new_d['year']==2004]['PJMW_MW'].mean(),new_d[new_d['year']==2005]['PJMW_MW'].mean(),new_d[new_d['year']==2006]['PJMW_MW'].mean(),new_d[new_d['year']==2007]['PJMW_MW'].mean(),new_d[new_d['year']==2008]['PJMW_MW'].mean(),new_d[new_d['year']==2009]['PJMW_MW'].mean(),new_d[new_d['year']==2010]['PJMW_MW'].mean(),new_d[new_d['year']==2011]['PJMW_MW'].mean(),new_d[new_d['year']==2012]['PJMW_MW'].mean(),new_d[new_d['year']==2013]['PJMW_MW'].mean(),new_d[new_d['year']==2014]['PJMW_MW'].mean(),new_d[new_d['year']==2015]['PJMW_MW'].mean(),new_d[new_d['year']==2016]['PJMW_MW'].mean(),new_d[new_d['year']==2017]['PJMW_MW'].mean(),new_d[new_d['year']==2018]['PJMW_MW'].mean()]
wp = { 'linewidth' : 0.5, 'edgecolor' : "black" }
def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%".format(pct, absolute)
st.subheader('Average power consuption in years from 2002 to 2018')
axs[0].pie(value,autopct = lambda pct: func(pct, new_d['PJMW_MW']),labels=year,wedgeprops = wp,textprops={'fontsize':3});
axs[0].set_title('Average power consuption in years from 2002 to 2018',fontsize=3)

new_d['month']=new_d['month'].apply(lambda x: 'Jan' if x==1 else 'Feb' if x==2 else 'March'if x==3 else 'April' if x==4 else 'May' if x==5 else 'June' if x==6 else 'July' if x==7 else 'Aug' if x==8 else 'Sep' if x==9 else 'Oct' if x==10 else 'Nov'  if x==11 else 'Dec' if x==12 else '') 
new_d.month.unique()
months=[new_d[new_d['month']=='Jan']['PJMW_MW'].mean(),new_d[new_d['month']=='Feb']['PJMW_MW'].mean(),new_d[new_d['month']=='March']['PJMW_MW'].mean(),new_d[new_d['month']=='April']['PJMW_MW'].mean(),new_d[new_d['month']=='May']['PJMW_MW'].mean(),new_d[new_d['month']=='June']['PJMW_MW'].mean(),new_d[new_d['month']=='July']['PJMW_MW'].mean(),new_d[new_d['month']=='Aug']['PJMW_MW'].mean(),new_d[new_d['month']=='Sep']['PJMW_MW'].mean(),new_d[new_d['month']=='Oct']['PJMW_MW'].mean(),new_d[new_d['month']=='Nov']['PJMW_MW'].mean(),new_d[new_d['month']=='Dec']['PJMW_MW'].mean()]
label=new_d.month.unique()

axs[1].pie(months,autopct = lambda pct: func(pct, new_d['PJMW_MW']),labels=label,wedgeprops = wp,textprops={'fontsize':3});
axs[1].set_title('Average power consuption in months from 2002 to 2018',fontsize=3)
season=[new_d[new_d['season']=='Winter']['PJMW_MW'].mean(),new_d[new_d['season']=='Autumn']['PJMW_MW'].mean(),new_d[new_d['season']=='Summer']['PJMW_MW'].mean(),new_d[new_d['season']=='Spring']['PJMW_MW'].mean()]
label=['Winter', 'Autumn', 'Summer', 'Spring']

axs[2].pie(season,autopct = lambda pct: func(pct, new_d['PJMW_MW']),labels=label,wedgeprops = wp,textprops={'fontsize':3});
axs[2].set_title('Average power consuption in  different Seasons of years from 2002 to 2018',fontsize=3)
st.pyplot(fig)

#creating the dataframe to predict the future
start_date=datetime.datetime(2018,8,3)
end_date=start_date+datetime.timedelta(days=30)
dates=[]
for i in range(30):
    for j in range(24):
        date=start_date+datetime.timedelta(days=i,hours=j)
        dates.append(date)
        
for date in dates:
    print(date)
    
new_data=pd.DataFrame({'Datetime':dates})
new_data['Day']=new_data['Datetime'].dt.day
new_data['dayofyear']=new_data['Datetime'].dt.dayofyear
new_data['year']=new_data['Datetime'].dt.year
new_data['month']=new_data['Datetime'].dt.month
new_data['hour']=new_data['Datetime'].dt.hour
new_data['name_of_week']=new_data['Datetime'].dt.isocalendar().week
new_data['name_of_week']=new_data['name_of_week'].astype(np.int32)
new_data['season']=new_data['Datetime'].apply(lambda x: 'Winter' if x.month == 12 or x.month == 1 or x.month == 2  else 'Spring' if x.month==3 or x.month==4 or x.month==5 else 'Summer' if x.month==6 or x.month==7 or x.month==8 else 'Autumn' if x.month==9 or x.month==10 or x.month==11 else "")
holidays=holidays
new_data['holiday']=new_data.index.map(lambda x:x in holidays)
new_data['season']=LabelEncoder().fit_transform(new_data['season'])
new_data['holiday']=LabelEncoder().fit_transform(new_data['holiday'])
new_data=new_data.set_index('Datetime')
x=new_data[['Day','year','month','hour', 'name_of_week', 'season','holiday']]
new_data['prediction'] = model.predict(new_data[['Day','year','month','hour', 'name_of_week', 'season','holiday']])
st.header('Next 30 days with 24hours prediction is as fallows')  
st.write(new_data['prediction'].head(15))
#let me plot the prediction
st.header('30 days prediction using xg boost')
fig=px.line(new_data,x=new_data.index,y=new_data['prediction'])
fig.update_traces(line_color='firebrick')
st.plotly_chart(fig)
hide_st_style = """
             <style>
             mainmenu {visibility: hidden;}
             footer {visibility: hidden;}
             header {visibility: hidden;}
             </style>"""
             
st.markdown(hide_st_style, unsafe_allow_html=True)