#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import glob


# In[5]:


file_pattern = 'D:\\Tarun\\UTS\\Subjects\\ADV ML\\output\\*.csv'


# In[6]:


file_list = glob.glob(file_pattern)


# In[7]:


dfs = []


# In[8]:


for file in file_list:
    df = pd.read_csv(file)
    dfs.append(df)


# In[9]:


combined_df = pd.concat(dfs, ignore_index=True)


# In[7]:


combined_df.shape


# In[10]:


df = combined_df[['store_id', 'item_id', 'date', 'sales_revenue']]


# In[9]:


df.head(5)


# In[11]:


df = df.copy()
df['date'] = pd.to_datetime(df['date'])


# In[11]:


# if prophet not installed, install by using the below command
#pip install prophet


# In[12]:


from prophet import Prophet


# In[13]:


df_prophet = df[['date', 'sales_revenue']].rename(columns={'date': 'ds', 'sales_revenue': 'y'})


# In[14]:


df_prophet.head(5)


# In[14]:


model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)


# In[15]:


model.fit(df_prophet)


# In[17]:


future = model.make_future_dataframe(periods=7)


# In[18]:


forecast = model.predict(future)


# In[19]:


forecast_next_7_days = forecast.tail(7)


# In[ ]:


forecast = forecast_next_7_days[['ds', 'yhat']]


# In[20]:


print(forecast)


# In[1]:


from fastapi import FastAPI


# In[2]:


app = FastAPI()


# In[3]:


@app.get("/")
def read_root():
    return {"Hello": "World"}


# In[ ]:


@app.get("/forecast_revenue_next_7_days")
def forecast_revenue_next_7_days():
    return forecast

