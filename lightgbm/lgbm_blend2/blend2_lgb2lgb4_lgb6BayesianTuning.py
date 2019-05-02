#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[6]:


lgb2_ovsampled = pd.read_csv('../lgbm_blend1/lgb2_base_features_oversampled.csv')
lgb4_goss_ov = pd.read_csv('../lgbm_blend1/lgb4_base_features_goss_ov.csv')

lgb6_fe_ov_bayes = pd.read_csv('lgb6_fe_bayestune_oversampling.csv')


# In[7]:


path = "../../feature_engineering_eda_data/"

sample_submission = pd.read_csv(path+'sample_submission_24jSKY6.csv')


# In[9]:


lgb2_ovsampled.head()


# In[10]:


lgb4_goss_ov.head()


# In[11]:


lgb6_fe_ov_bayes.head()


# In[12]:


blend1 = (lgb2_ovsampled['loan_default'] * 0.2) + (lgb4_goss_ov['loan_default'] * 0.2) + (lgb6_fe_ov_bayes['loan_default']*0.6)


# In[13]:


sample_submission['loan_default'] = blend1
sample_submission.head()


# In[14]:


sample_submission.to_csv('blend2.csv', index=False)


# In[ ]:


print("----------"*5)
print("...Complete...")

