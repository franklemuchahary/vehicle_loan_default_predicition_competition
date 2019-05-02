#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import gmean


# In[155]:


lgb2 = pd.read_csv('lgb2.csv')
lgb5 = pd.read_csv('lgb5.csv')
lgb6 = pd.read_csv('lgb6.csv')

blend1_df = pd.read_csv('../lgbm_blend1/blend1.csv')
blend2_df = pd.read_csv('../lgbm_blend2/blend2.csv')
blend5_df = pd.read_csv('../lgbm_blend5/blend5.csv')


# In[2]:


path = "../../feature_engineering_eda_data/"

sample_submission = pd.read_csv(path+'sample_submission_24jSKY6.csv')


# In[144]:


lgb2.head()


# In[146]:


lgb6.head()


# In[147]:


lgb5.head()


# In[148]:


blend2_df.head()


# In[149]:


blend1_df.head()


# In[151]:


blend5_df.head()


# ### Blending - Weighted and Geometric Mean

# In[203]:


gmean_df = pd.concat([lgb2['loan_default'], lgb6['loan_default'], lgb5['loan_default'], blend5_df['loan_default'],
                    blend2_df['loan_default'], blend1_df['loan_default']], 
                     axis=1)

simple_gmean = gmean(gmean_df, axis=1)
simple_gmean

#blend_weighted = (blend3_df['loan_default'] * 0.15) + (lgb1['loan_default'] * 0.15) + \
#(blend2_df['loan_default'] * 0.15) + (lgb2['loan_default'] * 0.15) + (lgb3['loan_default']*0.2) +\
#(lgb4['loan_default']*0.2)

#blend_weighted.values
# Perp Preds

# In[204]:


sample_submission['loan_default'] = simple_gmean
sample_submission.head()


# In[205]:


sample_submission.to_csv('blend7.csv', index=False)


# In[2]:


print('----------'*5)
print("...Completed...")

