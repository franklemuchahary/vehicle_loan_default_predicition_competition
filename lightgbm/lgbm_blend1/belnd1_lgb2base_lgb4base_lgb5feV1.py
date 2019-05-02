#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


lgb2_ovsampled = pd.read_csv('lgb2_base_features_oversampled.csv')
lgb4_goss_ov = pd.read_csv('lgb4_base_features_goss_ov.csv')
lgb5_fe_ov = pd.read_csv('lgb5_feV1_oversampling.csv')


# In[4]:


path = "../../feature_engineering_eda_data/"

sample_submission = pd.read_csv(path+'sample_submission_24jSKY6.csv')


# In[6]:


lgb2_ovsampled.head()


# In[16]:


lgb4_goss_ov.head()


# In[17]:


lgb5_fe_ov.head()


# In[18]:


blend1 = (lgb2_ovsampled['loan_default'] * 0.25)  + (lgb4_goss_ov['loan_default'] * 0.25) + (lgb5_fe_ov['loan_default'] * 0.5)


# In[19]:


sample_submission['loan_default'] = blend1
sample_submission.head()


# In[20]:


sample_submission.to_csv('blend1.csv', index=False)


# In[21]:


print("---------"*5)
print("...Completed...")

