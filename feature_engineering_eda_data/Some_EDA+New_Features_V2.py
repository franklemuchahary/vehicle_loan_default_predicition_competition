
# coding: utf-8

# In[47]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from collections import Counter
from datetime import datetime
import re

#%matplotlib inline
pd.set_option('display.max_columns', 100)


# In[48]:

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score


# #### Helper Functions

# In[49]:

def label_encoding_func(df_name, df_col_name):
    '''
    usage: dataframe[column_name] = label_encoding_function(dataframe, column_name)
    '''
    le = preprocessing.LabelEncoder()
    le.fit(df_name[df_col_name])
    return le.transform(df_name[df_col_name])


# In[50]:

def do_one_hot_encoding(df_name, df_column_name, suffix=''):
    '''
    usage: dataframe[column_name] = do_one_hot_encoding(dataframe, column_name, suffix_for_column_name)
    '''
    x = pd.get_dummies(df_name[df_column_name])
    df_name = df_name.join(x, lsuffix=suffix)
    df_name = df_name.drop(df_column_name, axis=1) 
    return df_name


# #### Load and Process Data

# In[51]:

train_df = pd.read_csv('train_aox2Jxw/train.csv')
test_df = pd.read_csv('test_bqCt9Pv.csv')


# In[52]:

train_df.head()


# In[53]:

test_df.head()


# In[54]:

X_train = train_df.copy()
X_test = test_df.copy()
X_test['loan_default'] = 0
Y = train_df['loan_default']


# In[55]:

concat_df = pd.concat([X_train, X_test], keys=['train', 'test'])


# ### Feature Engineering

# In[56]:

concat_df.head()


# #### Generating new features and extracting information from existing ones
# Might Contain a lot of nulls and inf -> fix after all new features are created -> replace with 0 <br>
# `Track of Columns to Drop` <br>
# - Date.of.Birth_dt / Date.of.Birth
# - DisbursalDate_dt / DisbursalDate
# - AVERAGE.ACCT.AGE
# - CREDIT.HISTORY.LENGTH

# In[57]:

### manually calculate ltv by ratio of disbursed amount to asset cost
concat_df['F1_Manual_LTV'] = concat_df['disbursed_amount']/concat_df['asset_cost']*100

### difference between disbursed amount and asset cost
concat_df['F2_difference_asset_disbursed'] = concat_df['asset_cost'] - concat_df['disbursed_amount']

### average sanctioned amount primary
concat_df['F3_avg_primary_sanctioned_amt'] = concat_df['PRI.SANCTIONED.AMOUNT']/concat_df['PRI.NO.OF.ACCTS']

### average disbursed amount primary 
concat_df['F4_avg_primary_disbursed_amt'] = concat_df['PRI.DISBURSED.AMOUNT']/concat_df['PRI.NO.OF.ACCTS']

### primary overdue ratio
concat_df['F5_ratio_primary_active_overdue'] = concat_df['PRI.OVERDUE.ACCTS']/concat_df['PRI.ACTIVE.ACCTS']


# <b> Dealing with DOB and Disbursed Date

# In[58]:

def string_to_date(dt_str_value):
    return datetime.strptime(dt_str_value, "%d-%m-%y").date()


# In[59]:

concat_df['Date.of.Birth_dt'] = concat_df['Date.of.Birth'].apply(string_to_date)


# In[60]:

concat_df['DisbursalDate_dt'] = concat_df['DisbursalDate'].apply(string_to_date)


# In[61]:

def get_age_at_disbursal(df):
    delta = df['DisbursalDate_dt'] - df['Date.of.Birth_dt']
    return abs(delta.days/365)

concat_df['F6_age_at_disbursal'] = concat_df.apply(get_age_at_disbursal, axis=1)


# In[62]:

def extract_year_month_date(date_value):
    return date_value.year, date_value.month, date_value.day


# In[63]:

concat_df['F7.1_DOB_Y'], concat_df['F7.2_DOB_M'], concat_df['F7.3_DOB_D'] = zip(*concat_df['Date.of.Birth_dt'].map(
                                                                                extract_year_month_date))


# In[64]:

concat_df['F8.1_DisDate_Y'], concat_df['F8.2_DisDate_M'], concat_df['F8.3_DisDate_D'] = zip(*concat_df['DisbursalDate_dt'].map(extract_year_month_date))


# <b> Dealing with CreditHistoryLength and AVGAccountAge

# In[65]:

def extract_num_from_string(string_with_num):
    string_list = string_with_num.split(" ")
    y = re.search(r'(\d+)', string_list[0]).groups()[0]
    m = re.search(r'(\d+)', string_list[1]).groups()[0]
    
    return y, m


# In[66]:

concat_df['F9.1_AVG.ACCT.AGE_Y'],concat_df['F9.2_AVG.ACCT.AGE_M'] =             zip(*concat_df['AVERAGE.ACCT.AGE'].map(extract_num_from_string))
    

concat_df['F10.1_CREDIT.HIST_Y'],concat_df['F10.2_CREDIT.HIST_M'] =             zip(*concat_df['CREDIT.HISTORY.LENGTH'].map(extract_num_from_string))


#  

# In[67]:

concat_df = concat_df.reset_index().drop('level_1', axis=1)


# #### View Sample of all newly created variables

# In[68]:

concat_df.head()


# #### Fill NANs and inf generated with 0

# In[69]:

concat_df['Employment.Type'].fillna('NotAvail', inplace=True)


# In[70]:

concat_df.fillna(0, inplace=True)


# In[71]:

#filling infs
for i in concat_df.columns.values:
    if (len(concat_df.loc[concat_df[i] == np.inf, i]) != 0)or(len(concat_df.loc[concat_df[i] == -np.inf, i]) != 0):
        print(i)
        concat_df.loc[concat_df[i] == np.inf, i] = 0
        concat_df.loc[concat_df[i] == -np.inf, i] = 0


# In[72]:

concat_df.isnull().sum().reset_index().T


# #### Drop Some Unnecessary columns

# In[73]:

concat_df.columns.values


# In[74]:

cols_to_drop = ['Date.of.Birth_dt', 'DisbursalDate_dt', 'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH']
concat_df.drop(cols_to_drop, axis=1, inplace=True)


# In[75]:

concat_df.head()


# In[76]:

train_engineered = concat_df[concat_df['level_0']=='train'].drop('level_0', axis=1)
test_engineered = concat_df[concat_df['level_0']=='test'].drop(['level_0', 'loan_default'], axis=1)


# In[77]:

train_engineered.head()


# In[78]:

test_engineered.head()


# In[79]:

train_engineered.to_csv('train_feature_engineered_V2.csv', index=False)
test_engineered.to_csv('test_feature_engineered_V2.csv', index=False)


# In[1]:

print("----------"*5)
print("...Complete...")

