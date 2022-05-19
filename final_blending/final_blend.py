#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import gmean


# In[366]:


cb1 = pd.read_csv('../catboost/cb1.csv')
cb2 = pd.read_csv('../catboost/cb2.csv')
cb3 = pd.read_csv('../catboost/cb3.csv')

lgb2 = pd.read_csv('../lightgbm/lgb2_base_features_oversampled.csv')
lgb4 = pd.read_csv('../lightgbm/lgb4_base_features_goss_ov.csv')
lgb5 = pd.read_csv('../lightgbm/lgb5_feV1_oversampling.csv')
lgb6 = pd.read_csv('../lightgbm/lgb6_fe_bayestune_oversampling.csv')


# In[3]:


path = "../feature_engineering_eda_data/"

sample_submission = pd.read_csv(path+'sample_submission_24jSKY6.csv')


# ### Blending - Weighted and Geometric Mean

# In[386]:


gmean_df = pd.concat([cb1['loan_default'],
                    cb2['loan_default'], 
                    cb3['loan_default'],
                    lgb2['loan_default'],
                    lgb4['loan_default'],
                    lgb5['loan_default'],
                    lgb6['loan_default']], axis=1)

simple_gmean = gmean(gmean_df, axis=1)
simple_gmean


# In[ ]:


simple_ar_mean = gmean_df.mean(axis=1)
simple_ar_mean.head()


# Perp Preds

# In[383]:


sample_submission['loan_default'] = simple_ar_mean
sample_submission.head()


# In[384]:


sample_submission.to_csv('final_gmean_blend.csv', index=False)


# In[1]:


print("----------"*5)
print("...Complete...")

