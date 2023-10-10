#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import glob


# In[12]:


file_pattern = 'D:\\Tarun\\UTS\\Subjects\\ADV ML\\output\\*.csv'


# In[13]:


file_list = glob.glob(file_pattern)


# In[14]:


dfs = []


# In[15]:


for file in file_list:
    df = pd.read_csv(file)
    dfs.append(df)


# In[16]:


combined_df = pd.concat(dfs, ignore_index=True)


# In[17]:


combined_df.shape


# In[9]:


combined_df.head(5)


# In[10]:


combined_df.dtypes


# In[11]:


combined_df = combined_df[combined_df['d'] <= 1541]


# In[12]:


combined_df.shape


# In[49]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[45]:


combined_df_temp = combined_df[combined_df['date'] > '2015-04-18']


# In[47]:


combined_df_temp.shape


# In[18]:


df = combined_df[['store_id', 'item_id', 'date', 'sales_revenue']]


# In[32]:


df.head(5)


# In[33]:


df = df.copy()
df['date'] = pd.to_datetime(df['date'])


# In[11]:


df.isna().sum()


# In[34]:


df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day


# In[35]:


from sklearn.preprocessing import LabelEncoder
le_store_id = LabelEncoder()
le_item_id = LabelEncoder()


# In[36]:


df['le_store_id'] = le_store_id.fit_transform(df['store_id'])
df['le_item_id'] = le_item_id.fit_transform(df['item_id'])


# In[55]:


df.head(5)


# In[38]:


encoding_map_store_id = dict(zip(df['store_id'], df['le_store_id']))


# In[41]:


encoding_map_store_id['CA_3']


# In[42]:


encoding_map_item_id = dict(zip(df['item_id'], df['le_item_id']))


# In[43]:


encoding_map_item_id['HOBBIES_1_001']


# In[44]:


df = df.drop(['store_id', 'item_id', 'date'], axis='columns')


# In[45]:


df_X_train = df.drop(['sales_revenue'], axis='columns')
df_Y_train = df.sales_revenue


# In[46]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df_X_train, df_Y_train, train_size=0.8)
X_test.head(5)


# In[22]:


from sklearn.ensemble import RandomForestRegressor


# In[23]:


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)


# In[24]:


rf_model.fit(X_train, Y_train)


# In[25]:


predictions = rf_model.predict(X_test)


# In[26]:


from sklearn.metrics import mean_squared_error


# In[27]:


mse = mean_squared_error(Y_test, predictions)


# In[28]:


mse


# In[29]:


rf_model.score(X_test, Y_test)


# In[83]:


# pip install fastapi


# In[85]:


# pip install "uvicorn[standard]"


# In[87]:


# pip install uvicorn


# In[1]:


from fastapi import FastAPI


# In[2]:


app = FastAPI()


# In[3]:


@app.get("/")
def read_root():
    return {"Hello": "World"}


# In[22]:


import datetime


# In[58]:


@app.get("/predict_revenue")
def predict_revenue(
    store_id: str,
    item_id: str,
    date: str
):
    datetime_object = datetime.datetime.strptime(date, '%Y-%m-%d')
    
    pred = rf_model.predict([[
        datetime_object.year, 
        datetime_object.month, 
        datetime_object.day, 
        encoding_map_store_id[store_id], 
        encoding_map_item_id[item_id]]])
    return pred


# In[63]:


print(predict_revenue('WI_3', 'HOUSEHOLD_1_201', '2015-04-12'))


# In[ ]:




