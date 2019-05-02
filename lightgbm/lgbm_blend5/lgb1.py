#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from collections import Counter
import math
import random

#%matplotlib inline


# In[2]:


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from imblearn.over_sampling import RandomOverSampler, SMOTE

#!ls 
# #### Helper Functions

# In[4]:


def label_encoding_func(df_name, df_col_name):
    '''
    usage: dataframe[column_name] = label_encoding_function(dataframe, column_name)
    '''
    le = preprocessing.LabelEncoder()
    le.fit(df_name[df_col_name])
    return le.transform(df_name[df_col_name])


# In[5]:


def do_one_hot_encoding(df_name, df_column_name, suffix=''):
    '''
    usage: dataframe[column_name] = do_one_hot_encoding(dataframe, column_name, suffix_for_column_name)
    '''
    x = pd.get_dummies(df_name[df_column_name])
    df_name = df_name.join(x, lsuffix=suffix)
    df_name = df_name.drop(df_column_name, axis=1) 
    return df_name


# #### Load Data

# In[2]:


path = '../../feature_engineering_eda_data/' 
train_file = 'train_feature_engineered_V2.csv'
test_file = 'test_feature_engineered_V2.csv'


# In[3]:


train_df = pd.read_csv(path+train_file)
test_df = pd.read_csv(path+test_file)
sample_submission = pd.read_csv(path+'sample_submission_24jSKY6.csv')


# In[8]:


print(train_df.shape)
print(test_df.shape)


# #### Sample Data

# In[9]:


pd.set_option('display.max_columns', 100)


# In[10]:


train_df.head()


# In[11]:


test_df.head()


# ## Data Cleaning / Exploration

# #### Distribution of Target

# In[12]:


print(train_df['loan_default'].value_counts()/sum(train_df['loan_default'].value_counts()))
(train_df['loan_default'].value_counts()/sum(train_df['loan_default'].value_counts())).plot(kind='bar')
plt.show()


# #### Remove cols from train and test and separate target

# In[13]:


cols_to_exclude = ['loan_default', 'UniqueID', 'SEC.ACTIVE.ACCTS', 'SEC.OVERDUE.ACCTS',
                  'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT', 'SEC.DISBURSED.AMOUNT', 
                  'MobileNo_Avl_Flag', 'Passport_flag', 'Driving_flag']

X_train = train_df[train_df.columns.difference(cols_to_exclude)]
X_test = test_df[test_df.columns.difference(cols_to_exclude)]
Y = train_df['loan_default']


# In[14]:


### Concat train and test for common preprocessing
concat_df = pd.concat([X_train, X_test], keys=['train', 'test'])


# In[15]:


concat_df.head()


# #### Check for nulls

# In[16]:


concat_df.isna().sum(axis=0).reset_index().T


# In[17]:


#### replace nulls as a new category
concat_df['Employment.Type'].fillna('NA', inplace=True)


# #### Label encode strings

# In[18]:


def label_encode_apply(df):
    if df[0] == object:
        concat_df[df['index']] = label_encoding_func(concat_df, df['index'])
        
_ = concat_df.dtypes.reset_index().apply(label_encode_apply, axis=1)
print('Done')


# In[19]:


concat_df.head()


# #### Separate train and test

# In[20]:


X_train = concat_df.loc['train']
X_test = concat_df.loc['test']


# In[21]:


print(X_train.shape)
print(X_test.shape)


# #### FE Functions

# In[23]:


### this function is for running inside cv

def generate_summed_features(train, val, test, variable=''):
    '''
    function to generate new features inside cv
    '''
    pincode = train.groupby('Current_pincode_ID')[variable].sum().reset_index()
    state = train.groupby('State_ID')[variable].sum().reset_index()
    supplier = train.groupby('supplier_id')[variable].sum().reset_index()
    branch = train.groupby('branch_id')[variable].sum().reset_index()
    
    list_of_dfs = [train, val, test]
    
    for i in range(len(list_of_dfs)):
        list_of_dfs[i] = pd.merge(list_of_dfs[i], pincode, how='left', on='Current_pincode_ID', 
         suffixes=('', '_sum_pincode_F11.1'))
        list_of_dfs[i] = pd.merge(list_of_dfs[i], state, how='left', on='State_ID', 
         suffixes=('', '_sum_state_F11.2'))
        list_of_dfs[i] = pd.merge(list_of_dfs[i], supplier, how='left', on='supplier_id', 
         suffixes=('', '_sum_supplier_F11.3'))
        list_of_dfs[i] = pd.merge(list_of_dfs[i], branch, how='left', on='branch_id', 
         suffixes=('', '_sum_branch_F11.4'))
        
        list_of_dfs[i].fillna(0, inplace=True)
        
    train, val, test = list_of_dfs[0], list_of_dfs[1], list_of_dfs[2]    
    return train, val, test  


#### this function is for running inside cv

def generate_averaged_features(train, val, test, variable=''):
    '''
    function to generate new features inside cv
    '''
    pincode = train.groupby('Current_pincode_ID')[variable].mean().reset_index()
    branch = train.groupby('branch_id')[variable].mean().reset_index()
    employee_code_id = train.groupby('Employee_code_ID')[variable].mean().reset_index()

    list_of_dfs = [train, val, test]
    
    for i in range(len(list_of_dfs)):
        list_of_dfs[i] = pd.merge(list_of_dfs[i], pincode, how='left', on='Current_pincode_ID', 
         suffixes=('', '_mean_pincode_F13.1'))
        list_of_dfs[i] = pd.merge(list_of_dfs[i], branch, how='left', on='branch_id', 
         suffixes=('', '_mean_branch_F13.2'))
        list_of_dfs[i] = pd.merge(list_of_dfs[i], employee_code_id, how='left', on='Employee_code_ID', 
         suffixes=('', '_mean_employeeid_F13.3'))
        
        list_of_dfs[i].fillna(0, inplace=True)
        
    train, val, test = list_of_dfs[0], list_of_dfs[1], list_of_dfs[2]    
    return train, val, test  


# ## Baseline Model Building

# In[24]:


strf_split = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


# In[25]:


val_auc_scores = []
test_preds_list = []

cv_counter = 1

for train_idx, val_idx in strf_split.split(X_train, Y):
    print("***************** ", cv_counter, " *****************", end="\n\n")
    
    t_x = X_train.iloc[train_idx]
    v_x = X_train.iloc[val_idx]
    
    t_y = Y[train_idx]
    v_y = Y[val_idx]
    
    
    test_x = X_test.copy()

    #generate new features
    t_x, v_x, test_x = generate_averaged_features(t_x, v_x, test_x, 'ltv')
    t_x, v_x, test_x = generate_averaged_features(t_x, v_x, test_x, 'PRI.CURRENT.BALANCE')
    t_x, v_x, test_x = generate_averaged_features(t_x, v_x, test_x, 'disbursed_amount')
    t_x, v_x, test_x = generate_averaged_features(t_x, v_x, test_x, 'asset_cost')
    
    t_x, v_x, test_x = generate_summed_features(t_x, v_x, test_x, 'PRI.ACTIVE.ACCTS')
    t_x, v_x, test_x = generate_summed_features(t_x, v_x, test_x, 'PRI.OVERDUE.ACCTS')
    
    print("Train Shape: ", t_x.shape)
    print("Val Shape: ", v_x.shape)
    print("Test Shape: ", test_x.shape, end="\n\n")
    
    
    params = {}
    params['learning_rate'] = 0.009
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'binary'
    params['metric'] = 'auc'
    params['feature_fraction'] = 0.35
    params['bagging_freq'] = 5
    params['bagging_fraction'] = 0.38
    params['num_leaves'] = 1000
    params['min_data_in_leaf'] = 80
    params['max_depth'] = 25
    params['num_threads'] = 20
    params['tree_learner'] = 'serial'
    params['min_sum_hessian_in_leaf'] = 0.001
    params['boost_from_average'] = False
    params['lambda_l1'] = 3
    params['lambda_l2'] = 5
    
    
    d_train = lgb.Dataset(t_x, label=t_y)
    d_valid = lgb.Dataset(v_x, label=v_y)
    
    
    model = lgb.train(params, d_train, 3000, early_stopping_rounds=300, valid_sets=[d_train, d_valid], 
                         verbose_eval=50)
    
   
    
    val_preds = model.predict(v_x)
    val_score = roc_auc_score(v_y, val_preds)
    
    print(val_score)
    
    val_auc_scores.append(val_score)
    
        
    test_preds = model.predict(test_x)
    test_preds_list.append(test_preds)
    
    cv_counter+=1
    
    print("============"*8, end="\n\n")


# In[27]:


print("CV Score: ", np.mean(val_auc_scores))

#pd.DataFrame({
#    'cols': t_x.columns.values,
#    'imp_values': model.feature_importance()
#}).sort_values('imp_values', ascending=False).T
# In[29]:


### Combine all CV preds for test
test_preds_cv = pd.DataFrame(np.asarray(test_preds_list).T).mean(axis=1).values


# #### Prep Submission

# In[31]:


sample_submission['loan_default'] = test_preds_cv

sample_submission.head()


# In[32]:


sample_submission.to_csv('lgb1.csv', index=False)


# In[4]:


print("---------"*5)
print("...Completed...")

