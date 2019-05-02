#!/usr/bin/env python
# coding: utf-8

# In[7]:


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

# In[8]:


path = '../../feature_engineering_eda_data/' 
train_file = 'train_feature_engineered_V1.csv'
test_file = 'test_feature_engineered_V1.csv'


# In[9]:


train_df = pd.read_csv(path+train_file)
test_df = pd.read_csv(path+test_file)
sample_submission = pd.read_csv(path+'sample_submission_24jSKY6.csv')


# In[12]:


print(train_df.shape)
print(test_df.shape)


# #### Sample Data

# In[70]:


pd.set_option('display.max_columns', 100)


# In[71]:


train_df.head()


# In[72]:


test_df.head()


# ## Data Cleaning / Exploration

# #### Distribution of Target

# In[73]:


print(train_df['loan_default'].value_counts()/sum(train_df['loan_default'].value_counts()))
(train_df['loan_default'].value_counts()/sum(train_df['loan_default'].value_counts())).plot(kind='bar')
plt.show()


# #### Remove cols from train and test and separate target

# In[75]:


cols_to_exclude = ['loan_default', 'UniqueID', 'SEC.ACTIVE.ACCTS', 'SEC.OVERDUE.ACCTS',
                  'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT', 'SEC.DISBURSED.AMOUNT', 
                  'MobileNo_Avl_Flag', 'Passport_flag', 'Driving_flag']

X_train = train_df[train_df.columns.difference(cols_to_exclude)]
X_test = test_df[test_df.columns.difference(cols_to_exclude)]
Y = train_df['loan_default']


# In[76]:


### Concat train and test for common preprocessing
concat_df = pd.concat([X_train, X_test], keys=['train', 'test'])


# In[77]:


concat_df.head()


# #### Check for nulls

# In[78]:


concat_df.isna().sum(axis=0).reset_index().T


# In[79]:


#### replace nulls as a new category
concat_df['Employment.Type'].fillna('NA', inplace=True)


# #### Label encode strings

# In[80]:


def label_encode_apply(df):
    if df[0] == object:
        concat_df[df['index']] = label_encoding_func(concat_df, df['index'])
        
_ = concat_df.dtypes.reset_index().apply(label_encode_apply, axis=1)
print('Done')


# In[81]:


concat_df.head()


# #### Separate train and test

# In[82]:


X_train = concat_df.loc['train']
X_test = concat_df.loc['test']


# In[88]:


print(X_train.shape)
print(X_test.shape)


# In[89]:


15/X_train.shape[1]


# ## Baseline Model Building

# In[90]:


strf_split = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


# In[94]:


val_auc_scores = []
test_preds_list = []

cv_counter = 1

for train_idx, val_idx in strf_split.split(X_train, Y):
    print("***************** ", cv_counter, " *****************", end="\n\n")
    
    t_x = X_train.iloc[train_idx]
    v_x = X_train.iloc[val_idx]
    
    t_y = Y[train_idx]
    v_y = Y[val_idx]
    
    ros = RandomOverSampler(sampling_strategy=0.6, random_state=42)
    t_x, t_y = ros.fit_resample(t_x, t_y)
    
    params = {}
    params['learning_rate'] = 0.009
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'binary'
    params['metric'] = 'auc'
    params['feature_fraction'] = 0.26
    params['bagging_freq'] = 5
    params['bagging_fraction'] = 0.38
    params['num_leaves'] = 10000
    params['min_data_in_leaf'] = 80
    params['max_depth'] = 25
    params['num_threads'] = 20
    params['tree_learner'] = 'feature'
    params['min_sum_hessian_in_leaf'] = 0.001
    params['boost_from_average'] = False
    
    d_train = lgb.Dataset(t_x, label=t_y)
    d_valid = lgb.Dataset(v_x, label=v_y)
    
    
    model = lgb.train(params, d_train, 3000, early_stopping_rounds=300, valid_sets=[d_train, d_valid], 
                         verbose_eval=50)
    
   
    
    val_preds = model.predict(v_x)
    val_score = roc_auc_score(v_y, val_preds)
    
    print(val_score)
    
    val_auc_scores.append(val_score)
    
        
    test_preds = model.predict(X_test)
    test_preds_list.append(test_preds)
    
    cv_counter+=1
    
    print("============"*8, end="\n\n")


# In[95]:


print("CV Score: ", np.mean(val_auc_scores))

#pd.DataFrame({
#    'cols': X_train.columns.values,
#    'imp_values': model.feature_importance()
#}).sort_values('imp_values', ascending=False)
# In[97]:


### Combine all CV preds for test
test_preds_cv = pd.DataFrame(np.asarray(test_preds_list).T).mean(axis=1).values


# #### Prep Submission

# In[100]:


sample_submission['loan_default'] = test_preds_cv

sample_submission.head()


# In[101]:


sample_submission.to_csv('lgb5_feV1_oversampling.csv', index=False)


# In[13]:


print("---------"*5)
print("...Completed...")

