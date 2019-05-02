#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import gmean


# In[3]:


lgb1 = pd.read_csv('lgb1.csv')
#lgb2 = pd.read_csv('19thApril/LGB_FEInsideCV_10foldCV_19thApril_3.csv')
lgb4 = pd.read_csv('lgb4.csv')

blend1_df = pd.read_csv('../lgbm_blend1/blend1.csv')
blend2_df = pd.read_csv('../lgbm_blend2/blend2.csv')


# In[2]:


path = "../../feature_engineering_eda_data/"

sample_submission = pd.read_csv(path+'sample_submission_24jSKY6.csv')


# In[50]:


lgb1.head()


# In[51]:


#lgb2.head()


# In[53]:


lgb4.head()


# In[54]:


blend1_df.head()


# In[56]:


blend2_df.head()


# In[5]:


gmean_df = pd.concat([lgb1['loan_default'], #lgb2['loan_default'], 
                      #lgb3['loan_default'], 
                      lgb4['loan_default'],
                    blend2_df['loan_default'], blend1_df['loan_default']], axis=1)

simple_gmean = gmean(gmean_df, axis=1)
simple_gmean

#blend_weighted = (blend3_df['loan_default'] * 0.15) + (lgb1['loan_default'] * 0.15) + \
#(blend2_df['loan_default'] * 0.15) + (lgb2['loan_default'] * 0.15) + (lgb3['loan_default']*0.2) +\
#(lgb4['loan_default']*0.2)

#blend_weighted.values
# Perp Preds

# In[6]:


sample_submission['loan_default'] = simple_gmean
sample_submission.head()


# In[66]:


sample_submission.to_csv('blend5.csv', index=False)


# In[ ]:


print('-----------'*5)
print("...Completed...")

