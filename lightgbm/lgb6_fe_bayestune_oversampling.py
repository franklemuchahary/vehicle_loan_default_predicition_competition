#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from collections import Counter
import math
import random

#%matplotlib inline


# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from imblearn.over_sampling import RandomOverSampler, SMOTE
from skopt import BayesSearchCV


# #### Helper Functions

# In[ ]:


def label_encoding_func(df_name, df_col_name):
    '''
    usage: dataframe[column_name] = label_encoding_function(dataframe, column_name)
    '''
    le = preprocessing.LabelEncoder()
    le.fit(df_name[df_col_name])
    return le.transform(df_name[df_col_name])


# In[ ]:


def do_one_hot_encoding(df_name, df_column_name, suffix=''):
    '''
    usage: dataframe[column_name] = do_one_hot_encoding(dataframe, column_name, suffix_for_column_name)
    '''
    x = pd.get_dummies(df_name[df_column_name])
    df_name = df_name.join(x, lsuffix=suffix)
    df_name = df_name.drop(df_column_name, axis=1) 
    return df_name


# #### Load Data

# In[4]:


path = '../../feature_engineering_eda_data/' 
train_file = 'train_feature_engineered_V1.csv'
test_file = 'test_feature_engineered_V1.csv'


# In[5]:


train_df = pd.read_csv(path+train_file)
test_df = pd.read_csv(path+test_file)
sample_submission = pd.read_csv(path+'sample_submission_24jSKY6.csv')


# #### Sample Data

# In[ ]:


pd.set_option('display.max_columns', 100)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# ## Data Cleaning / Exploration

# #### Distribution of Target

# In[ ]:


print(train_df['loan_default'].value_counts()/sum(train_df['loan_default'].value_counts()))
(train_df['loan_default'].value_counts()/sum(train_df['loan_default'].value_counts())).plot(kind='bar')
plt.show()


# #### Remove cols from train and test and separate target

# In[ ]:


cols_to_exclude = ['loan_default', 'UniqueID', 'SEC.ACTIVE.ACCTS', 'SEC.OVERDUE.ACCTS',
                  'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT', 'SEC.DISBURSED.AMOUNT', 
                  'MobileNo_Avl_Flag', 'Passport_flag', 'Driving_flag']

X_train = train_df[train_df.columns.difference(cols_to_exclude)]
X_test = test_df[train_df.columns.difference(cols_to_exclude)]
Y = train_df['loan_default']


# In[ ]:


### Concat train and test for common preprocessing
concat_df = pd.concat([X_train, X_test], keys=['train', 'test'])


# In[ ]:


concat_df.head()


# #### Check for nulls

# In[ ]:


concat_df.isna().sum(axis=0).reset_index().T


# In[ ]:


#### replace nulls as a new category
concat_df['Employment.Type'].fillna('NA', inplace=True)


# #### Label encode strings

# In[ ]:


def label_encode_apply(df):
    if df[0] == object:
        concat_df[df['index']] = label_encoding_func(concat_df, df['index'])
        
_ = concat_df.dtypes.reset_index().apply(label_encode_apply, axis=1)
print('Done')


# In[ ]:


concat_df.head()


# #### Separate train and test

# In[ ]:


X_train = concat_df.loc['train']
X_test = concat_df.loc['test']
print(X_train.shape)
print(X_test.shape)


# ## Baseline Model Building

# In[ ]:


strf_split = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


# In[ ]:


val_auc_scores = []
test_preds_list = []

cv_counter = 1

for train_idx, val_idx in strf_split.split(X_train, Y):
    print("***************** ", cv_counter, " *****************", end="\n\n")
    
    t_x = X_train.iloc[train_idx]
    v_x = X_train.iloc[val_idx]
    
    t_y = Y[train_idx]
    v_y = Y[val_idx]
    
    ros = RandomOverSampler(sampling_strategy=0.6)
    t_x, t_y = ros.fit_resample(t_x, t_y)
    
    params = {'bagging_freq': 1, 'bagging_fraction': 0.9, 'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.25385461407805393, 'learning_rate': 0.05901911792677764, 
               'max_bin': 100, 'max_depth': 5, 'min_child_samples': 100, 'min_child_weight': 1.165370768368568, 'n_estimators': 1500, 
               'num_leaves': 185, 'reg_alpha': 1e-09, 'reg_lambda': 1000.0, 'subsample': 0.40936459175143447, 'subsample_for_bin': 174333, 'subsample_freq': 10}

     
    model = lgb.LGBMClassifier(**params, n_jobs=-1, silent=False)
    
    model = model.fit(t_x, t_y)
    
    
    val_preds = model.predict_proba(v_x)[:,1]
    val_score = roc_auc_score(v_y, val_preds)
    
    print(val_score)
    
    val_auc_scores.append(val_score)
    
        
    test_preds = model.predict_proba(X_test)[:,1]
    test_preds_list.append(test_preds)
    
    cv_counter+=1
    
    print("============"*8, end="\n\n")


# In[ ]:


print("CV Score: ", np.mean(val_auc_scores))


# In[ ]:


### Combine all CV preds for test
test_preds_cv = pd.DataFrame(np.asarray(test_preds_list).T).mean(axis=1).values


# #### Prep Submission

# In[ ]:


sample_submission['loan_default'] = test_preds_cv

sample_submission.head()


# In[ ]:


sample_submission.to_csv('lgb6_fe_bayestune_oversampling.csv', index=False)

