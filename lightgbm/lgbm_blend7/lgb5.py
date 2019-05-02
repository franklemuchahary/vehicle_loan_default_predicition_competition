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


# In[3]:


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from imblearn.over_sampling import RandomOverSampler, SMOTE
from category_encoders import TargetEncoder

#!ls 
# #### Helper Functions

# In[5]:


def label_encoding_func(df_name, df_col_name):
    '''
    usage: dataframe[column_name] = label_encoding_function(dataframe, column_name)
    '''
    le = preprocessing.LabelEncoder()
    le.fit(df_name[df_col_name])
    return le.transform(df_name[df_col_name])


# In[6]:


def do_one_hot_encoding(df_name, df_column_name, suffix=''):
    '''
    usage: dataframe[column_name] = do_one_hot_encoding(dataframe, column_name, suffix_for_column_name)
    '''
    x = pd.get_dummies(df_name[df_column_name])
    df_name = df_name.join(x, lsuffix=suffix)
    df_name = df_name.drop(df_column_name, axis=1) 
    return df_name


# In[8]:


#function for perform target encoding later on
def perform_target_encoding(columns, X, Y, X_Val, X_Test):
    for i in tqdm(columns):
        target_enc = TargetEncoder(cols=[i], smoothing=3)
        target_enc_fit = target_enc.fit(X, Y)
        X[i] = target_enc.transform(X, Y)[i]
        X_Val[i] = target_enc.transform(X_Val)[i]
        X_Test[i] = target_enc.transform(X_Test)[i]
        
    return X, X_Val, X_Test


# #### Load Data

# In[2]:


path = '../../feature_engineering_eda_data/' 
train_file = 'train_feature_engineered_V2.csv'
test_file = 'test_feature_engineered_V2.csv'


# In[3]:


train_df = pd.read_csv(path+train_file)
test_df = pd.read_csv(path+test_file)
sample_submission = pd.read_csv(path+'sample_submission_24jSKY6.csv')


# In[4]:


print(train_df.shape)
print(test_df.shape)


# #### Sample Data

# In[12]:


pd.set_option('display.max_columns', 100)


# In[13]:


train_df.head()


# In[14]:


test_df.head()


# ## Data Cleaning / Exploration

# #### Distribution of Target

# In[15]:


print(train_df['loan_default'].value_counts()/sum(train_df['loan_default'].value_counts()))
(train_df['loan_default'].value_counts()/sum(train_df['loan_default'].value_counts())).plot(kind='bar')
plt.show()


# #### Remove cols from train and test and separate target

# In[16]:


cols_to_exclude = ['loan_default', 'UniqueID', 'SEC.ACTIVE.ACCTS', 'SEC.OVERDUE.ACCTS',
                  'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT', 'SEC.DISBURSED.AMOUNT', 
                  'MobileNo_Avl_Flag', 'Passport_flag', 'Driving_flag']

X_train = train_df[train_df.columns.difference(cols_to_exclude)]
X_test = test_df[test_df.columns.difference(cols_to_exclude)]
Y = train_df['loan_default']


# In[17]:


### Concat train and test for common preprocessing
concat_df = pd.concat([X_train, X_test], keys=['train', 'test'])


# In[18]:


concat_df.head()


# #### Check for nulls

# In[19]:


concat_df.isna().sum(axis=0).reset_index().T


# In[20]:


#### replace nulls as a new category
concat_df['Employment.Type'].fillna('NA', inplace=True)


# #### Some New Features

# In[201]:


concat_df['employee_id_branch_id'] = concat_df['branch_id'].apply(str)+"-"+concat_df['Employee_code_ID'].apply(str)


# In[202]:


concat_df['avg_employment_length'] = concat_df['F6_age_at_disbursal'] - 25


# In[221]:


bins = [-np.inf, 20, 25, 30, 35, 40, 45, 50, np.inf]
labels = [1,2,3,4,5,6,7,8]

concat_df['F6_age_bins'] = np.asarray(pd.cut(concat_df['F6_age_at_disbursal'], bins=bins, labels=labels).values)


# #### Label encode strings

# In[222]:


def label_encode_apply(df):
    if df[0] == object:
        print(df['index'])
        concat_df[df['index']] = label_encoding_func(concat_df, df['index'])
        
_ = concat_df.dtypes.reset_index().apply(label_encode_apply, axis=1)
print('Done')


# In[223]:


concat_df.head()


# #### Separate train and test

# In[224]:


X_train = concat_df.loc['train']
X_test = concat_df.loc['test']


# In[225]:


print(X_train.shape)
print(X_test.shape)


# In[226]:


45/127


# #### FE Functions

# In[236]:


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
    #employee_code_id_branch_id = train.groupby('employee_id_branch_id')[variable].mean().reset_index()
    supplier_id = train.groupby('supplier_id')[variable].mean().reset_index()
    age_bin = train.groupby('F6_age_bins')[variable].mean().reset_index()

    list_of_dfs = [train, val, test]
    
    for i in range(len(list_of_dfs)):
        list_of_dfs[i] = pd.merge(list_of_dfs[i], pincode, how='left', on='Current_pincode_ID', 
         suffixes=('', '_mean_pincode_F13.1'))
        list_of_dfs[i] = pd.merge(list_of_dfs[i], branch, how='left', on='branch_id', 
         suffixes=('', '_mean_branch_F13.2'))
        list_of_dfs[i] = pd.merge(list_of_dfs[i], employee_code_id, how='left', on='Employee_code_ID', 
         suffixes=('', '_mean_employeeid_F13.3'))
        #list_of_dfs[i] = pd.merge(list_of_dfs[i], employee_code_id_branch_id, how='left',on='employee_id_branch_id', 
        # suffixes=('', '_mean_employee_id_branch_id_F13.4'))
        list_of_dfs[i] = pd.merge(list_of_dfs[i], supplier_id, how='left', on='supplier_id', 
         suffixes=('', '_mean_supplier_id_F13.5'))
        list_of_dfs[i] = pd.merge(list_of_dfs[i], age_bin, how='left', on='F6_age_bins', 
         suffixes=('', '_mean_age_bins_F13.6'))
        
        
        list_of_dfs[i].fillna(0, inplace=True)
        
    train, val, test = list_of_dfs[0], list_of_dfs[1], list_of_dfs[2]    
    return train, val, test  


# In[228]:


X_train.columns.values


# ## Baseline Model Building

# In[229]:


strf_split = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


# In[239]:


val_auc_scores = []
test_preds_list = []
all_train_predictions = np.zeros([X_train.shape[0]])

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
    t_x, v_x, test_x = generate_averaged_features(t_x, v_x, test_x, 'PRIMARY.INSTAL.AMT')
    t_x, v_x, test_x = generate_averaged_features(t_x, v_x, test_x, 'PERFORM_CNS.SCORE')
    t_x, v_x, test_x = generate_averaged_features(t_x, v_x, test_x, 'F4_avg_primary_disbursed_amt')
    t_x, v_x, test_x = generate_averaged_features(t_x, v_x, test_x, 'F2_difference_asset_disbursed')
    t_x, v_x, test_x = generate_averaged_features(t_x, v_x, test_x, 'PRI.NO.OF.ACCTS')
    t_x, v_x, test_x = generate_averaged_features(t_x, v_x, test_x, 'PRI.OVERDUE.ACCTS')
    t_x, v_x, test_x = generate_averaged_features(t_x, v_x, test_x, 'F6_age_at_disbursal')
    
    
    t_x, v_x, test_x = generate_summed_features(t_x, v_x, test_x, 'PRI.ACTIVE.ACCTS')
    t_x, v_x, test_x = generate_summed_features(t_x, v_x, test_x, 'PRI.OVERDUE.ACCTS')
    t_x, v_x, test_x = generate_summed_features(t_x, v_x, test_x, 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS')
    
    
    print("Train Shape: ", t_x.shape)
    print("Val Shape: ", v_x.shape)
    print("Test Shape: ", test_x.shape, end="\n\n")
    
    
    
    
    params = {}
    params['learning_rate'] = 0.009
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'binary'
    params['metric'] = 'auc'
    params['feature_fraction'] = 0.45
    params['bagging_freq'] = 5
    params['bagging_fraction'] = 0.38
    params['num_leaves'] = 1000
    params['min_data_in_leaf'] = 80
    params['max_depth'] = 35
    params['num_threads'] = 20
    params['tree_learner'] = 'serial'
    params['min_sum_hessian_in_leaf'] = 0.001
    params['boost_from_average'] = False
    params['lambda_l1'] = 9
    params['lambda_l2'] = 2
    
    
    d_train = lgb.Dataset(t_x, label=t_y)
    d_valid = lgb.Dataset(v_x, label=v_y)
    
    
    model = lgb.train(params, d_train, 3000, early_stopping_rounds=300, valid_sets=[d_train, d_valid], 
                         verbose_eval=50)
    
   
    
    val_preds = model.predict(v_x)
    val_score = roc_auc_score(v_y, val_preds)
    
    all_train_predictions[val_idx] = val_preds
    
    print(val_score)
    
    val_auc_scores.append(val_score)
    
        
    test_preds = model.predict(test_x)
    test_preds_list.append(test_preds)
    
    cv_counter+=1
    
    print("============"*8, end="\n\n")


# In[240]:


print("CV Score: ", np.mean(val_auc_scores))

#pd.DataFrame({
#    'cols': t_x.columns.values,
#    'imp_values': model.feature_importance()
#}).sort_values('imp_values', ascending=False).T
# In[242]:


### Combine all CV preds for test
test_preds_cv = pd.DataFrame(np.asarray(test_preds_list).T).mean(axis=1).values


# In[243]:


### train predictions

train_preds_cv = pd.DataFrame({
    'UniqueID': train_df['UniqueID'].values,
    'TrainPreds': all_train_predictions
})

train_preds_cv.head()


# #### Prep Submission

# In[244]:


sample_submission['loan_default'] = test_preds_cv

sample_submission.head()


# In[170]:


sample_submission.to_csv('lgb5.csv', index=False)
#train_preds_cv.to_csv('LGB_FE_3_InsideCV_10foldCV_19thApril_1_TRAIN.csv', index=False)

