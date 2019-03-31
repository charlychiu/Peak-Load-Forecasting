
# coding: utf-8

# ## DSAI HW1 Peak Load Forecasting

# - Description  
# 請根據台電歷史資料，預測未來七天的"電力尖峰負載"(MW)。
# 
# - Evaluation  
# a. Goal
# 預測 2019/4/2 ~ 2019/4/8 的每日"電力尖峰負載"(MW)  
# b. Metric
# 作業將以 “尖峰負載預測值” 與 "尖峰負載實際數值"之 Root-Mean-Squared-Error (RMSE) 作為評估分數。

# In[1]:


import numpy as np
import pandas as pd
from keras import backend as K
import keras.models as kModels
import keras.layers as kLayers
from keras.callbacks import EarlyStopping, ModelCheckpoint


# ### import data (using 2017[12month], 2018[12month], 2019[1month])

# In[2]:


db_data_2017 = pd.read_csv('taipower_2017.csv')[:365]      # 20170101 ~ 20171231  365 records
db_data_2018 = pd.read_csv('taipower_2018.csv')[:365]      # 20180101 ~ 20181231  365 records
db_data_2019 = pd.read_csv('taipower_2019.csv')              # 20190101 ~ 20190330  89 records  ***had been processed


# ### Data Preprocessing

# In[3]:


train_df = db_data_2017.append(db_data_2018)


# - Augment Features

# In[4]:


def augFeatures(df):
    df["日期"] = pd.to_datetime(df["日期"], format="%Y%m%d")
    df["year"] = df["日期"].dt.year
    df["month"] = df["日期"].dt.month
    df["date"] = df["日期"].dt.day
    df["day"] = df["日期"].dt.dayofweek
    return df


# In[5]:


train_df = augFeatures(train_df)
train_df = train_df.append(db_data_2019)


# - Normalization

# In[7]:


def normalize(df):
    df = df[['淨尖峰供電能力(MW)','尖峰負載(MW)', '備轉容量(MW)', '備轉容量率(%)', 'month', 'date', 'day']]
    df_norm = df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    return df_norm

def normalize_with_ref(df, ref_df):
    # tmp_df['尖峰負載(MW)'].describe()['mean']
    df = df[['淨尖峰供電能力(MW)','尖峰負載(MW)', '備轉容量(MW)', '備轉容量率(%)', 'month', 'date', 'day']]
    col_list = ['淨尖峰供電能力(MW)','尖峰負載(MW)', '備轉容量(MW)', '備轉容量率(%)', 'month', 'date', 'day']
    for col in col_list:
        df[col] = (df[col] - ref_df[col].describe()['mean']) / (ref_df[col].describe()['max'] - ref_df[col].describe()['min'])
#     df_norm = df.apply(lambda x: (x - v_mean) / (v_max - v_min))
    return df


# In[8]:


tmp_df = train_df
train_df_norm = normalize(tmp_df)[:len(train_df)]


# In[9]:


def buildTrain(train, pastDay=7, futureDay=7):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-futureDay-pastDay):
        X_train.append(np.array(train.iloc[i:i+pastDay]))
        Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["尖峰負載(MW)"]))
    return np.array(X_train), np.array(Y_train)

def buildPredict(train, pastDay=7, futureDay=7):
    X_train = []
    for i in range(train.shape[0]-futureDay+1):
        X_train.append(np.array(train.iloc[i:i+pastDay]))
    return np.array(X_train)


# In[10]:


# build Data, use last 7 days to predict next 7 days
X_train, Y_train = buildTrain(train_df_norm, 7, 7)


# In[11]:


def splitData(X,Y,rate):
    X_train = X[int(X.shape[0]*rate):]
    Y_train = Y[int(Y.shape[0]*rate):]
    X_val = X[:int(X.shape[0]*rate)]
    Y_val = Y[:int(Y.shape[0]*rate)]
    return X_train, Y_train, X_val, Y_val


# In[12]:


# split training data and validation data
X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)


# In[13]:


# from 2 dimmension to 3 dimension
Y_train = Y_train[:,:,np.newaxis]
Y_val = Y_val[:,:,np.newaxis]


# In[14]:


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 


# In[15]:


def buildManyToManyModel(shape):
    model = kModels.Sequential()
    model.add(kLayers.LSTM(10, input_length=shape[1], input_dim=shape[2], return_sequences=True))
    # output shape: (7, 1)
    model.add(kLayers.TimeDistributed(kLayers.Dense(1)))
    model.compile(loss=root_mean_squared_error, optimizer="RMSprop")
    model.summary()
    return model


# In[16]:


model = buildManyToManyModel(X_train.shape)
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
train_history = model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback])


# In[19]:


def denormalized(data):
    return data * (tmp_df['尖峰負載(MW)'].describe()['max'] - tmp_df['尖峰負載(MW)'].describe()['min']) + tmp_df['尖峰負載(MW)'].describe()['mean']


# ### Predict
# - Using 7 pastday predict 7 futureday

# 2019/03/26~2019/04/01 predict 2019/04/02~2019/04/08

# In[24]:


db_data_20190326_20190401 = pd.read_csv('taipower_predictData.csv')


# In[26]:


db_data_20190326_20190401 = normalize_with_ref(db_data_20190326_20190401, tmp_df)


# In[28]:


X_predict = buildPredict(db_data_20190326_20190401)


# In[29]:


Y_predict = model.predict(X_predict)


# In[30]:


Y_predict = denormalized(Y_predict)


# *** 四捨五入

# ### Save to submission.csv

# In[37]:


date = ['20190402', '20190403', '20190404', '20190405', '20190406', '20190407', '20190408']
data = []
for idx, v in enumerate(Y_predict[0]):
    print(int(round(v[0], 0)))
    data.append([date[idx], int(round(v[0], 0))])


# In[39]:


df = pd.DataFrame(data, columns = ['date', 'peak_load(MW)'])


# In[43]:


df.to_csv('submission.csv', index=0)

