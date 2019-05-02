#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from scipy.stats import gmean


# In[366]:


cb1 = pd.read_csv('cb1.csv')
cb2 = pd.read_csv('cb2.csv')
cb3 = pd.read_csv('cb3.csv')

blend2_df = pd.read_csv('../../lightgbm/lgbm_blend2/blend2.csv')
blend7_df = pd.read_csv('../../lightgbm/lgbm_blend7/blend7.csv')


# In[3]:


path = "../../feature_engineering_eda_data/"

sample_submission = pd.read_csv(path+'sample_submission_24jSKY6.csv')


# In[368]:


cb1.head()


# In[369]:


cb2.head()


# In[370]:


cb3.head()


# In[325]:


blend7_df.head()


# In[ ]:


blend2_df.head()


# ### Blending - Weighted and Geometric Mean

# In[386]:


gmean_df = pd.concat([cb2['loan_default'],
                    cb1['loan_default'], 
                    cb3['loan_default'],
                    blend7_df['loan_default'], 
                    blend2_df['loan_default']], axis=1)

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


sample_submission.to_csv('blend9.csv', index=False)


# In[1]:


print("----------"*5)
print("...Complete...")

