#!/usr/bin/env python
# coding: utf-8

# In[45]:


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


# In[46]:


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score


# #### Helper Functions

# In[47]:


def label_encoding_func(df_name, df_col_name):
    '''
    usage: dataframe[column_name] = label_encoding_function(dataframe, column_name)
    '''
    le = preprocessing.LabelEncoder()
    le.fit(df_name[df_col_name])
    return le.transform(df_name[df_col_name])


# In[48]:


def do_one_hot_encoding(df_name, df_column_name, suffix=''):
    '''
    usage: dataframe[column_name] = do_one_hot_encoding(dataframe, column_name, suffix_for_column_name)
    '''
    x = pd.get_dummies(df_name[df_column_name])
    df_name = df_name.join(x, lsuffix=suffix)
    df_name = df_name.drop(df_column_name, axis=1) 
    return df_name


# #### Load and Process Data

# In[49]:


train_df = pd.read_csv('train_aox2Jxw/train.csv')
test_df = pd.read_csv('test_bqCt9Pv.csv')


# In[50]:


train_df.head()


# In[51]:


test_df.head()


# In[52]:


X_train = train_df.copy()
X_test = test_df.copy()
X_test['loan_default'] = 0
Y = train_df['loan_default']


# In[53]:


concat_df = pd.concat([X_train, X_test], keys=['train', 'test'])


# ### Data Distributions

# ##### Disbursed Amount

# In[54]:


sns.distplot(train_df[train_df['loan_default']==1]['disbursed_amount'], color='green')
sns.distplot(train_df[train_df['loan_default']==0]['disbursed_amount'], color='blue')
plt.rcParams["figure.figsize"] = (12,8)
plt.xlim([0, 200000])
plt.show()


# ##### LTV

# In[55]:


sns.distplot(train_df[train_df['loan_default']==1]['ltv'], color='green')
sns.distplot(train_df[train_df['loan_default']==0]['ltv'], color='blue')
plt.rcParams["figure.figsize"] = (12,8)
plt.show()


# ##### Distinct value counts for all variables

# In[56]:


unique_value_counts = []

for i in concat_df.columns.values:
    unique_value_counts.append(len(concat_df[i].value_counts()))
    
    
unique_values_df = pd.DataFrame({
    'cols': concat_df.columns.values,
    'unique_values': unique_value_counts
}).sort_values('unique_values')

unique_values_df


# ##### Comparison between means of values of variables between target = 1 and target = 0 and between test and train

# In[57]:


### prepare arrays containing specific column types

dtypes_df = train_df.dtypes.reset_index()

numeric_cols = dtypes_df[(dtypes_df[0] != object) & (dtypes_df['index'] != 'UniqueID') & 
          (dtypes_df['index'] != 'loan_default')]['index'].values

greater_than_2_distinct_values = unique_values_df[unique_values_df['unique_values'] > 2]['cols'].values
non_categorical_numeric_columns = [i for i in numeric_cols if i in greater_than_2_distinct_values]


# In[58]:


train_df[train_df['loan_default']==0][non_categorical_numeric_columns].mean(axis=0).plot(kind='bar', 
                                                                        label='0', color='blue',
                                                                        linewidth=2, edgecolor='k') 
                                                         
train_df[train_df['loan_default']==1][non_categorical_numeric_columns].mean(axis=0).plot(kind='bar', 
                                                                label='1', color='red', alpha=0.3) 
plt.rcParams["figure.figsize"] = (16,7)
plt.legend()
plt.show()


# In[59]:


train_df[non_categorical_numeric_columns].mean(axis=0).plot(kind='bar', label='train', color='blue', 
                                                           linewidth=2, edgecolor='k') 
test_df[non_categorical_numeric_columns].mean(axis=0).plot(kind='bar', label='test', color='red', alpha=0.3) 
#plt.ylim(0,100000)
plt.rcParams["figure.figsize"] = (16,7)
plt.legend()
plt.show()


# ##### Correlation Heatmap

# In[60]:


sns.heatmap(concat_df[non_categorical_numeric_columns].corr(), annot=True, cbar=False, linewidths=2, linecolor='k',
           robust=True)
plt.rcParams["figure.figsize"] = (20,18)
plt.show()
#plt.savefig('CorrelationPlot.png', dpi=300)


# ##### CNS Score

# In[61]:


print("CNS Score is zero: ", sum(concat_df['PERFORM_CNS.SCORE'] == 0))

concat_df[concat_df['PERFORM_CNS.SCORE'] == 0]["PERFORM_CNS.SCORE.DESCRIPTION"].value_counts()


# In[62]:


sns.distplot(train_df[train_df['PRI.OVERDUE.ACCTS']>0]['PERFORM_CNS.SCORE'])
sns.distplot(train_df[train_df['PRI.OVERDUE.ACCTS']==0]['PERFORM_CNS.SCORE'], label='0')
plt.legend()
plt.rcParams["figure.figsize"] = (16,8)
plt.show()

cns_not_zero = concat_df[concat_df['PERFORM_CNS.SCORE'] != 0]
cns_train = ['NO.OF_INQUIRIES', 'PRI.NO.OF.ACCTS', 'PRI.CURRENT.BALANCE', 
             'PRI.DISBURSED.AMOUNT', 'PRI.OVERDUE.ACCTS', 'PRI.ACTIVE.ACCTS',
            'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS', 'NEW.ACCTS.IN.LAST.SIX.MONTHS']

cns_x = cns_not_zero[cns_train]
cns_y = cns_not_zero['PERFORM_CNS.SCORE']

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

cns_x_scaled = StandardScaler().fit_transform(cns_x)

cns_lr = LinearRegression()

cross_val_score(cns_lr, cns_x_scaled, cns_y, cv=5, scoring='neg_mean_absolute_error')
#cns_lr.fit(cns_x_scaled, cns_y)
#  

#  

#  

# ### Feature Engineering

# In[63]:


concat_df.head()


# #### Generating new features and extracting information from existing ones
# Might Contain a lot of nulls and inf -> fix after all new features are created -> replace with 0 <br>
# `Track of Columns to Drop` <br>
# - Date.of.Birth_dt / Date.of.Birth
# - DisbursalDate_dt / DisbursalDate
# - AVERAGE.ACCT.AGE
# - CREDIT.HISTORY.LENGTH

# In[64]:


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

# In[65]:


def string_to_date(dt_str_value):
    return datetime.strptime(dt_str_value, "%d-%m-%y").date()


# In[66]:


concat_df['Date.of.Birth_dt'] = concat_df['Date.of.Birth'].apply(string_to_date)


# In[67]:


concat_df['DisbursalDate_dt'] = concat_df['DisbursalDate'].apply(string_to_date)


# In[68]:


def get_age_at_disbursal(df):
    delta = df['DisbursalDate_dt'] - df['Date.of.Birth_dt']
    return abs(delta.days/365)

concat_df['F6_age_at_disbursal'] = concat_df.apply(get_age_at_disbursal, axis=1)


# In[69]:


def extract_year_month_date(date_value):
    return date_value.year, date_value.month, date_value.day


# In[70]:


concat_df['F7.1_DOB_Y'], concat_df['F7.2_DOB_M'], concat_df['F7.3_DOB_D'] = zip(*concat_df['Date.of.Birth_dt'].map(
                                                                                extract_year_month_date))


# In[71]:


concat_df['F8.1_DisDate_Y'], concat_df['F8.2_DisDate_M'], concat_df['F8.3_DisDate_D'] = zip(*concat_df['DisbursalDate_dt'].map(extract_year_month_date))


# <b> Dealing with CreditHistoryLength and AVGAccountAge

# In[72]:


def extract_num_from_string(string_with_num):
    string_list = string_with_num.split(" ")
    y = re.search(r'(\d+)', string_list[0]).groups()[0]
    m = re.search(r'(\d+)', string_list[1]).groups()[0]
    
    return y, m


# In[73]:


concat_df['F9.1_AVG.ACCT.AGE_Y'],concat_df['F9.2_AVG.ACCT.AGE_M'] =             zip(*concat_df['AVERAGE.ACCT.AGE'].map(extract_num_from_string))
    

concat_df['F10.1_CREDIT.HIST_Y'],concat_df['F10.2_CREDIT.HIST_M'] =             zip(*concat_df['CREDIT.HISTORY.LENGTH'].map(extract_num_from_string))


#  

# In[74]:


concat_df = concat_df.reset_index().drop('level_1', axis=1)


# <b> Primary Overdue Accounts Grouped Features

# In[75]:


pincode_overdue_primary = concat_df.groupby('Current_pincode_ID')['PRI.OVERDUE.ACCTS'].sum().reset_index()
state_overdue_primary = concat_df.groupby('State_ID')['PRI.OVERDUE.ACCTS'].sum().reset_index()
supplier_overdue_primary = concat_df.groupby('supplier_id')['PRI.OVERDUE.ACCTS'].sum().reset_index()
branch_overdue_primary = concat_df.groupby('branch_id')['PRI.OVERDUE.ACCTS'].sum().reset_index()


# In[76]:


concat_df = pd.merge(concat_df, pincode_overdue_primary, how='left', on='Current_pincode_ID', 
         suffixes=('', '_sum_pincode_F11.1'))
concat_df = pd.merge(concat_df, state_overdue_primary, how='left', on='State_ID', 
         suffixes=('', '_sum_state_F11.2'))
concat_df = pd.merge(concat_df, supplier_overdue_primary, how='left', on='supplier_id', 
         suffixes=('', '_sum_supplier_F11.3'))
concat_df = pd.merge(concat_df, branch_overdue_primary, how='left', on='branch_id', 
         suffixes=('', '_sum_branch_F11.4'))


# <b> Primary Active Accounts Grouped Features

# In[77]:


pincode_active_primary = concat_df.groupby('Current_pincode_ID')['PRI.ACTIVE.ACCTS'].sum().reset_index()
state_active_primary = concat_df.groupby('State_ID')['PRI.ACTIVE.ACCTS'].sum().reset_index()
supplier_active_primary = concat_df.groupby('supplier_id')['PRI.ACTIVE.ACCTS'].sum().reset_index()
branch_active_primary = concat_df.groupby('branch_id')['PRI.ACTIVE.ACCTS'].sum().reset_index()


# In[78]:


concat_df = pd.merge(concat_df, pincode_active_primary, how='left', on='Current_pincode_ID', 
         suffixes=('', '_sum_pincode_F12.1'))
concat_df = pd.merge(concat_df, state_active_primary, how='left', on='State_ID', 
         suffixes=('', '_sum_state_F12.2'))
concat_df = pd.merge(concat_df, supplier_active_primary, how='left', on='supplier_id', 
         suffixes=('', '_sum_supplier_F12.3'))
concat_df = pd.merge(concat_df, branch_active_primary, how='left', on='branch_id', 
         suffixes=('', '_sum_branch_F12.4'))


# #### Avg Ltv branch and pincode

# In[79]:


pincode_ltv_primary = concat_df.groupby('Current_pincode_ID')['ltv'].mean().reset_index()
branch_ltv_primary = concat_df.groupby('branch_id')['ltv'].mean().reset_index()

concat_df = pd.merge(concat_df, pincode_ltv_primary, how='left', on='Current_pincode_ID', 
         suffixes=('', '_mean_pincode_F13.1'))
concat_df = pd.merge(concat_df, branch_ltv_primary, how='left', on='branch_id', 
         suffixes=('', '_mean_branch_F13.2'))


# In[80]:


concat_df['ltv_mean_pincode_F13.1'] = round(concat_df['ltv_mean_pincode_F13.1'], 2)
concat_df['ltv_mean_branch_F13.2'] = round(concat_df['ltv_mean_branch_F13.2'], 2)


# <b> Average PRI.CURRENT.BALANCE based on pincode and branch

# In[81]:


pincode_pri_current_bal = concat_df.groupby('Current_pincode_ID')['PRI.CURRENT.BALANCE'].mean().reset_index()
branch_pri_current_bal = concat_df.groupby('branch_id')['PRI.CURRENT.BALANCE'].mean().reset_index()

concat_df = pd.merge(concat_df, pincode_pri_current_bal, how='left', on='Current_pincode_ID', 
         suffixes=('', '_mean_pincode_F14.1'))
concat_df = pd.merge(concat_df, branch_pri_current_bal, how='left', on='branch_id', 
         suffixes=('', '_mean_branch_F14.2'))


# In[82]:


concat_df['PRI.CURRENT.BALANCE_mean_pincode_F14.1'] = concat_df['PRI.CURRENT.BALANCE_mean_pincode_F14.1'].apply(int)
concat_df['PRI.CURRENT.BALANCE_mean_branch_F14.2'] = concat_df['PRI.CURRENT.BALANCE_mean_branch_F14.2'].apply(int)


# #### View Sample of all newly created variables

# In[83]:


concat_df.head()


# #### Fill NANs and inf generated with 0

# In[84]:


concat_df['Employment.Type'].fillna('NotAvail', inplace=True)


# In[85]:


concat_df.fillna(0, inplace=True)


# In[86]:


#filling infs
for i in concat_df.columns.values:
    if (len(concat_df.loc[concat_df[i] == np.inf, i]) != 0)or(len(concat_df.loc[concat_df[i] == -np.inf, i]) != 0):
        print(i)
        concat_df.loc[concat_df[i] == np.inf, i] = 0
        concat_df.loc[concat_df[i] == -np.inf, i] = 0


# In[87]:


concat_df.isnull().sum().reset_index().T


# #### Drop Some Unnecessary columns

# In[88]:


concat_df.columns.values


# In[89]:


cols_to_drop = ['Date.of.Birth_dt', 'DisbursalDate_dt', 'AVERAGE.ACCT.AGE', 'CREDIT.HISTORY.LENGTH']
concat_df.drop(cols_to_drop, axis=1, inplace=True)


# In[90]:


concat_df.head()


# In[91]:


train_engineered = concat_df[concat_df['level_0']=='train'].drop('level_0', axis=1)
test_engineered = concat_df[concat_df['level_0']=='test'].drop(['level_0', 'loan_default'], axis=1)


# In[92]:


train_engineered.head()


# In[93]:


test_engineered.head()


# In[94]:


train_engineered.to_csv('train_feature_engineered_V1.csv', index=False)
test_engineered.to_csv('test_feature_engineered_V1.csv', index=False)


# In[1]:


print("----------"*5)
print("...Complete...")

