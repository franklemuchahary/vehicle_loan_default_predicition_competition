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
from tqdm import tqdm

#%matplotlib inline


# In[2]:


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import preprocessing
from category_encoders import target_encoder, TargetEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from imblearn.over_sampling import RandomOverSampler, SMOTE
import catboost as cb

pd.set_option('display.max_columns', 200)


# In[3]:


def label_encoding_func(df_name, df_col_name):
    '''
    usage: dataframe[column_name] = label_encoding_function(dataframe, column_name)
    '''
    le = preprocessing.LabelEncoder()
    le.fit(df_name[df_col_name])
    return le.transform(df_name[df_col_name])

def do_one_hot_encoding(df_name, df_column_name, suffix=''):
    '''
    usage: dataframe[column_name] = do_one_hot_encoding(dataframe, column_name, suffix_for_column_name)
    '''
    x = pd.get_dummies(df_name[df_column_name])
    df_name = df_name.join(x, lsuffix=suffix)
    df_name = df_name.drop(df_column_name, axis=1) 
    return df_name

#function for perform target encoding later on
def perform_target_encoding(columns, X, Y, X_Val, X_Test):
    for i in tqdm(columns):
        target_enc = TargetEncoder(cols=[i], smoothing=3)
        target_enc_fit = target_enc.fit(X, Y)
        X[i] = target_enc.transform(X, Y)[i]
        X_Val[i] = target_enc.transform(X_Val)[i]
        X_Test[i] = target_enc.transform(X_Test)[i]
        
    return X, X_Val, X_Test


# ### Load Feature Engineered Datasets

# In[2]:


path = '../../feature_engineering_eda_data/' 
train_file = 'train_feature_engineered_V2.csv'
test_file = 'test_feature_engineered_V2.csv'

train_df = pd.read_csv(path+train_file)
test_df = pd.read_csv(path+test_file)
sample_submission = pd.read_csv(path+'sample_submission_24jSKY6.csv')


# In[5]:


train_df.head()


# In[6]:


test_df.head()


# ### Combine Train and Test

# In[7]:


X_train = train_df[train_df.columns.difference(['loan_default', 'UniqueID'])]
X_test = test_df[train_df.columns.difference(['loan_default', 'UniqueID'])]
Y = train_df['loan_default']

### Concat train and test for common preprocessing
concat_df = pd.concat([X_train, X_test], keys=['train', 'test'])


# In[8]:


concat_df.head()


# In[9]:


#filling NAs with 0
concat_df.isna().sum(axis=0).reset_index().T


# ### More Features

# In[10]:


concat_df['employee_id_branch_id'] = concat_df['branch_id'].apply(str)+"-"+concat_df['Employee_code_ID'].apply(str)

bins = [-np.inf, 20, 25, 30, 35, 40, 45, 50, np.inf]
labels = [1,2,3,4,5,6,7,8]

concat_df['F6_age_bins'] = np.asarray(pd.cut(concat_df['F6_age_at_disbursal'], bins=bins, labels=labels).values)


# In[11]:


concat_df['F10.3_CREDIT.HIST_DAYS'] = (concat_df['F10.1_CREDIT.HIST_Y'] * 365) +                                        (concat_df['F10.2_CREDIT.HIST_M'] * 30)

concat_df['F9.3_AVG.ACCT.AGE_DAYS'] = (concat_df['F9.1_AVG.ACCT.AGE_Y'] * 365) +                                        (concat_df['F9.2_AVG.ACCT.AGE_M'] * 30)

concat_df['BalancePerActiveAccount'] = concat_df['PRI.CURRENT.BALANCE']/concat_df['PRI.ACTIVE.ACCTS']

concat_df['PRI.NoOfInstallmentsLeft'] = concat_df['PRI.CURRENT.BALANCE']/concat_df['PRIMARY.INSTAL.AMT']

concat_df['Disbursed_CurrentBalance_Diff'] = concat_df['disbursed_amount'] - concat_df['PRI.CURRENT.BALANCE']

#combine primary and secondary values
concat_df['TotalInstallAmt'] = concat_df['PRIMARY.INSTAL.AMT'] + concat_df['SEC.INSTAL.AMT']
concat_df['TotalDisbAmt'] = concat_df['PRI.DISBURSED.AMOUNT'] + concat_df['SEC.DISBURSED.AMOUNT']
concat_df['TotalCurrentBalance'] = concat_df['PRI.CURRENT.BALANCE'] + concat_df['SEC.CURRENT.BALANCE']
concat_df['TotalActiveAccts'] = concat_df['PRI.ACTIVE.ACCTS'] + concat_df['SEC.ACTIVE.ACCTS']
concat_df['TotalOverdueAccts'] = concat_df['PRI.OVERDUE.ACCTS'] + concat_df['SEC.OVERDUE.ACCTS']
concat_df['TotalAccts'] = concat_df['PRI.NO.OF.ACCTS'] + concat_df['SEC.NO.OF.ACCTS']
concat_df['TotalSancAmt'] = concat_df['PRI.SANCTIONED.AMOUNT'] + concat_df['SEC.SANCTIONED.AMOUNT']


# ### Dealing with NAs and Label Encoding Categorical features

# In[12]:


#### replace nulls as a new category
concat_df['Employment.Type'].fillna('NA', inplace=True)

concat_df.fillna(0, inplace=True)

### replace -inf and +inf with 0

#filling infs
for i in concat_df.columns.values:
    if (len(concat_df.loc[concat_df[i] == np.inf, i]) != 0)or(len(concat_df.loc[concat_df[i] == -np.inf, i]) != 0):
        print(i)
        concat_df.loc[concat_df[i] == np.inf, i] = 0
        concat_df.loc[concat_df[i] == -np.inf, i] = 0


# In[13]:


def label_encode_apply(df):
    if df[0] == object:
        concat_df[df['index']] = label_encoding_func(concat_df, df['index'])
        
_ = concat_df.dtypes.reset_index().apply(label_encode_apply, axis=1)
print('Done')


# ### Split Train Test

# In[14]:


X_train = concat_df.loc['train']
X_test = concat_df.loc['test']


# ### Yet More Features

# In[15]:


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
    employee_code_id_branch_id = train.groupby('employee_id_branch_id')[variable].mean().reset_index()
    supplier_id = train.groupby('supplier_id')[variable].mean().reset_index()

    list_of_dfs = [train, val, test]
    
    for i in range(len(list_of_dfs)):
        list_of_dfs[i] = pd.merge(list_of_dfs[i], pincode, how='left', on='Current_pincode_ID', 
         suffixes=('', '_mean_pincode_F13.1'))
        list_of_dfs[i] = pd.merge(list_of_dfs[i], branch, how='left', on='branch_id', 
         suffixes=('', '_mean_branch_F13.2'))
        list_of_dfs[i] = pd.merge(list_of_dfs[i], employee_code_id, how='left', on='Employee_code_ID', 
         suffixes=('', '_mean_employeeid_F13.3'))
        list_of_dfs[i] = pd.merge(list_of_dfs[i], employee_code_id_branch_id, how='left',on='employee_id_branch_id', 
         suffixes=('', '_mean_employee_id_branch_id_F13.4'))
        list_of_dfs[i] = pd.merge(list_of_dfs[i], supplier_id, how='left', on='supplier_id', 
         suffixes=('', '_mean_supplier_id_F13.5'))
        
        list_of_dfs[i].fillna(0, inplace=True)
        
    train, val, test = list_of_dfs[0], list_of_dfs[1], list_of_dfs[2]    
    return train, val, test  


# ### Specify cols to TE inside CV and categorical cols for catboost

# In[16]:


cols_to_target_encode = ['DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS', 'F6_age_bins',
                        'manufacturer_id', 'State_ID', 'Employment.Type', 'PRI.OVERDUE.ACCTS',
                        'PRI.ACTIVE.ACCTS', 'F7.1_DOB_Y', 'PERFORM_CNS.SCORE.DESCRIPTION', 'SEC.NO.OF.ACCTS',
                        'NO.OF_INQUIRIES', 'NEW.ACCTS.IN.LAST.SIX.MONTHS']


# In[17]:


for i,j in enumerate(X_train.columns):
    print(i," --> ", j, end=" || ")


# In[18]:


categorical_col_indices = [0, 1, 17, 18, 5, 8, 9, 37, 46, 6, 52, 48, 64]


# ### Model with 10 Fold CV

# In[19]:


strf_split = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


# In[20]:


train_preds_list_oof_semi_stacking = []
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

    print('Generating New Features: ')
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
    t_x, v_x, test_x = generate_averaged_features(t_x, v_x, test_x, 'F6_age_at_disbursal')
    t_x, v_x, test_x = generate_averaged_features(t_x, v_x, test_x, 'TotalCurrentBalance')
    t_x, v_x, test_x = generate_averaged_features(t_x, v_x, test_x, 'TotalInstallAmt')
    
    t_x, v_x, test_x = generate_summed_features(t_x, v_x, test_x, 'PRI.ACTIVE.ACCTS')
    t_x, v_x, test_x = generate_summed_features(t_x, v_x, test_x, 'PRI.OVERDUE.ACCTS')
    t_x, v_x, test_x = generate_summed_features(t_x, v_x, test_x, 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS')
    t_x, v_x, test_x = generate_summed_features(t_x, v_x, test_x, 'NO.OF_INQUIRIES')
    t_x, v_x, test_x = generate_summed_features(t_x, v_x, test_x, 'TotalActiveAccts')
    t_x, v_x, test_x = generate_summed_features(t_x, v_x, test_x, 'TotalOverdueAccts')
    
    
    print('Target Encoding: ')
    t_x, v_x, test_x = perform_target_encoding(cols_to_target_encode, t_x, t_y, v_x, test_x)
    
    
    print("Train Shape: ", t_x.shape)
    print("Val Shape: ", v_x.shape)
    print("Test Shape: ", test_x.shape, end="\n\n")
    
    params = {
            'eval_metric': 'AUC',
            'learning_rate': 0.01,
            'random_seed': 12321,
            'l2_leaf_reg': 15,
            'bootstrap_type': 'Bernoulli',
            #'bagging_temperature': 0.3,
            'subsample': 0.5,
            'max_depth': 8,
            'feature_border_type': 'MinEntropy',
            'thread_count': 4, 
            'objective': 'CrossEntropy',
            #'min_data_in_leaf': 100,
            'task_type': 'GPU',
            'od_type': 'Iter',
            'allow_writing_files': False,
            'boosting_type': 'Plain'
        }

    #print(t_x.iloc[:,categorical_col_indices])
    
    dtrain = cb.Pool(t_x, label=t_y, cat_features=categorical_col_indices)
    dvalid = cb.Pool(v_x, label=v_y, cat_features=categorical_col_indices)
    dtest = cb.Pool(test_x, cat_features=categorical_col_indices)
        
    model = cb.train(dtrain=dtrain, params = params, num_boost_round=8000, eval_set=[dvalid], early_stopping_rounds=500, 
        verbose_eval=200) 
    
    val_preds = model.predict(dvalid, prediction_type='Probability')
    val_score = roc_auc_score(v_y, val_preds[:,1])
    
    print(val_score)
    
    val_auc_scores.append(val_score)
    
        
    test_preds = model.predict(dtest, prediction_type='Probability')
    test_preds_list.append(test_preds[:,1])
    
    all_train_predictions[val_idx] = val_preds[:,1]
    
    cv_counter+=1
    
    print("============"*8, end="\n\n")
    del t_x, v_x, test_x, model
    gc.collect()


# In[21]:


print("CV Score: ", np.mean(val_auc_scores))


# In[22]:


### Combine all CV preds for test
test_preds_cv = pd.DataFrame(np.asarray(test_preds_list).T).mean(axis=1).values


# In[23]:


sample_submission['loan_default'] = test_preds_cv
#sample_submission['loan_default'] = sample_submission['loan_default'].rank(pct=True)
sample_submission.head()


# In[24]:


train_oof_preds = train_df[['UniqueID', 'loan_default']].copy()
train_oof_preds['loan_default'] = all_train_predictions
train_oof_preds.head()


# In[25]:


sample_submission.to_csv('cb3.csv', index=False)
#train_oof_preds.to_csv('cb3_TRAIN.csv', index=False)


# Download Directly Without Commiting
