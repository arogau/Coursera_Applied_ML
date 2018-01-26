
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# ## Assignment 4 - Understanding and Predicting Property Maintenance Fines
# 
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 
# 
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
# 
# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.
# 
# All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:
# 
# * [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
# * [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
# * [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
# * [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
# * [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)
# 
# ___
# 
# We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
# 
# Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
# 
# <br>
# 
# **File descriptions** (Use only this data for training your model!)
# 
#     train.csv - the training set (all tickets issued 2004-2011)
#     test.csv - the test set (all tickets issued 2012-2016)
#     addresses.csv & latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#      Note: misspelled addresses may be incorrectly geolocated.
# 
# <br>
# 
# **Data fields**
# 
# train.csv & test.csv
# 
#     ticket_id - unique identifier for tickets
#     agency_name - Agency that issued the ticket
#     inspector_name - Name of inspector that issued the ticket
#     violator_name - Name of the person/organization that the ticket was issued to
#     violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
#     mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
#     ticket_issued_date - Date and time the ticket was issued
#     hearing_date - Date and time the violator's hearing was scheduled
#     violation_code, violation_description - Type of violation
#     disposition - Judgment and judgement type
#     fine_amount - Violation fine amount, excluding fees
#     admin_fee - $20 fee assigned to responsible judgments
# state_fee - $10 fee assigned to responsible judgments
#     late_fee - 10% fee assigned to responsible judgments
#     discount_amount - discount applied, if any
#     clean_up_cost - DPW clean-up or graffiti removal cost
#     judgment_amount - Sum of all fines and fees
#     grafitti_status - Flag for graffiti violations
#     
# train.csv only
# 
#     payment_amount - Amount paid, if any
#     payment_date - Date payment was made, if it was received
#     payment_status - Current payment status as of Feb 1 2017
#     balance_due - Fines and fees still owed
#     collection_status - Flag for payments in collections
#     compliance [target variable for prediction] 
#      Null = Not responsible
#      0 = Responsible, non-compliant
#      1 = Responsible, compliant
#     compliance_detail - More information on why each ticket was marked compliant or non-compliant
# 
# 
# ___
# 
# ## Evaluation
# 
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC). 
# 
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.
# ___
# 
# For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using `train.csv`. Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from `test.csv` will be paid, and the index being the ticket_id.
# 
# Example:
# 
#     ticket_id
#        284932    0.531842
#        285362    0.401958
#        285361    0.105928
#        285338    0.018572
#                  ...
#        376499    0.208567
#        376500    0.818759
#        369851    0.018528
#        Name: compliance, dtype: float32

# In[1]:

import pandas as pd
import numpy as np


# In[2]:

train = pd.read_csv("train.csv", engine="c", sep=',', encoding='ISO-8859-1')
train.head()


# In[3]:

train = train[train['compliance']<=1]


# In[4]:

test = pd.read_csv("test.csv", engine="c", sep=',', encoding='ISO-8859-1')
test.head()


# In[5]:

train.info()


# In[6]:

train['ticket_issued_date'] = pd.to_datetime(train['ticket_issued_date'])
train['hearing_date'] = pd.to_datetime(train['hearing_date'])


# In[7]:

test['ticket_issued_date'] = pd.to_datetime(test['ticket_issued_date'])
test['hearing_date'] = pd.to_datetime(test['hearing_date'])


# In[8]:

print("train df is from: ", train['ticket_issued_date'].min()," to ", train['ticket_issued_date'].max())


# In[9]:

train = train[train['ticket_issued_date']>='2004-01-01']


# In[10]:

train.describe()


# In[11]:

print("test df is from: ", test['ticket_issued_date'].min()," to ", test['ticket_issued_date'].max())


# In[12]:

test.describe()


# In[13]:

train.shape


# In[14]:

test.shape


# In[15]:

train_only_col = ['payment_amount', 'payment_date', 'payment_status', 'balance_due', 'collection_status', 'compliance_detail']


# In[16]:

train.drop(train_only_col, axis=1, inplace=1)


# Percentage of missing values

# In[17]:

(np.sum(train.isnull())/len(train['ticket_id']))


# In[18]:

missing_data_col = ['grafitti_status','violation_zip_code']


# In[19]:

train.drop(missing_data_col, inplace=True, axis=1)


# In[20]:

test.drop(missing_data_col, inplace=True, axis=1)


# ## Data Preparation

# In[21]:

del train['non_us_str_code']
del test['non_us_str_code']


# In[22]:

def agency_name(x):
    if 'buildings' in x.lower():
        return 'Buildings'
    elif 'health' in x.lower():
        return 'Health'
    elif 'public' in x.lower():
        return 'Public Work'
    elif 'police' in x.lower():
        return 'Police'
    elif 'halls' in x.lower():
        return 'Neighborhood'
train['agency_name'] = train['agency_name'].map(agency_name)


# In[23]:

test['agency_name'] = test['agency_name'].map(agency_name)


# In[24]:

''' to create a dummy variable for common values in the same columns of each df'''
def dummy_common(train_df, test_df, column):
    #column: is the columns for which dummy is to be created
    col_names = list(set(train_df[column]) & set(test_df[column]));
    for col in col_names:
        train_df[col] = train_df[column].map(lambda x: 1 if x==col else 0)
        test_df[col] = test_df[column].map(lambda x: 1 if x==col else 0)
    del train_df[column]
    del test_df[column]
    return (train_df,test_df);


# In[25]:

train, test = dummy_common(train, test, 'agency_name')


# In[26]:

def disposition(x):
    if 'fine waved' in x.lower():
        return 'fine_waved'
    elif 'determination' in x.lower():
        return 'determination'
    elif 'default' in x.lower():
        return 'default'
    elif 'admission' in x.lower():
        return 'admission'
    else:
        return 'other'


# In[27]:

train['disposition'] = train['disposition'].map(disposition)


# In[28]:

test['disposition'] = test['disposition'].map(disposition)


# In[29]:

train, test = dummy_common(train, test, 'disposition')


# In[30]:

train.columns


# In[31]:

train['violation_code'] = train['violation_code'].map(lambda x: "-".join(x.split('-')[0:2]).split('.')[0]) # since there are many dates we will have to leave this columns. This dates could be probable mistake in entry


# In[32]:

test['violation_code'] = test['violation_code'].map(lambda x: "-".join(x.split('-')[0:2]).split('.')[0]) # since there are many dates we will have to leave this columns. This dates could be probable mistake in entry


# In[33]:

test['violation_code'] = test['violation_code'].map(lambda x: 'other' if len(x) > 4 else x.split('-')[0])


# In[34]:

train['violation_code'] = train['violation_code'].map(lambda x: 'other' if len(x) > 4 else x.split('-')[0])


# In[35]:

train['time_gap'] = (train['hearing_date']-train['ticket_issued_date']).dt.days+1;
train['time_gap'] 


# In[36]:

train[train['time_gap']<0][['ticket_issued_date','hearing_date']] #deleting these values since hearing date can't be before issue date


# In[37]:

test['time_gap'] = (test['hearing_date']-test['ticket_issued_date']).dt.days+1;


# In[38]:

test['time_gap'].fillna(0, inplace=True)
train['time_gap'].fillna(0,inplace=True)


# In[39]:

np.sum(train.isnull())


# In[40]:

np.sum(test.isnull())


# ##### trying to get some meaningful out of  inspector name, since outcome can be based on Inspector judgement. 

# In[41]:

train['inspector_name'] = train['inspector_name'].map(lambda x: x.lower())
test['inspector_name'] = test['inspector_name'].map(lambda x: x.lower())


# In[42]:

avg_compliance = train.groupby('compliance').count()['ticket_id']/len(train['compliance'])
x = pd.DataFrame(train.groupby(['inspector_name','compliance']).count()['ticket_id']/train.groupby(['inspector_name']).count()['ticket_id']).reset_index()
x['compliance_by_inspector'] = None
for i in [0,1]:
    x.loc[(x['compliance']==i)&((x['ticket_id']>=avg_compliance[i]*1.3)|(x['ticket_id']<=avg_compliance[i]*0.7)), 'compliance_by_inspector']=i
x = x.join(pd.get_dummies(x['compliance_by_inspector']))
for i in [0,1]:
    x.rename(columns={i:'compliance_by_inspector_{}'.format(i)}, inplace=True)
x.drop(['compliance_by_inspector','compliance','ticket_id'], axis=1, inplace=True)
x = x.groupby('inspector_name').sum().reset_index()
train=train.merge(x, on='inspector_name', how='inner')


# In[43]:

test=test.merge(x, on='inspector_name', how='left')


# In[44]:

train.drop('inspector_name',axis=1, inplace=True)
test.drop('inspector_name',axis=1, inplace=True)


# In[45]:

train.groupby('compliance').mean()['fine_amount']


# In[46]:

train.loc[train['fine_amount'].isnull(), 'fine_amount'] = 0
test.loc[test['fine_amount'].isnull(), 'fine_amount'] = 0


# In[47]:

train.drop('country', axis=1, inplace=True)
test.drop('country', axis=1, inplace=True)


# In[48]:

train['city'] = train['city'].map(lambda x: x.lower())
test['city'] = test['city'].map(lambda x: str(x).lower())


# In[49]:

train['from_detroit'] = train['city'].map(lambda x: 1 if x=='detroit' else 0)
test['from_detroit'] = test['city'].map(lambda x: 1 if x=='detroit' else 0)


# In[50]:

#train.drop(['admin_fee', 'state_fee', 'late_fee', 'discount_amount', 'clean_up_cost', 'judgment_amount'], axis=1, inplace=True)


# In[51]:

train = train[['ticket_id','fine_amount', 'admin_fee', 'state_fee',
       'late_fee', 'discount_amount', 'clean_up_cost', 'judgment_amount',
       'compliance','Public Work', 'Buildings', 'Police', 'admission',
       'default', 'other', 'determination', 'time_gap',
       'compliance_by_inspector_0', 'compliance_by_inspector_1',
       'from_detroit']].set_index('ticket_id')


# In[52]:

submission = test[['ticket_id','fine_amount', 'admin_fee', 'state_fee',
       'late_fee', 'discount_amount', 'clean_up_cost', 'judgment_amount',
       'Public Work', 'Buildings', 'Police', 'admission',
       'default', 'other', 'determination', 'time_gap',
       'compliance_by_inspector_0', 'compliance_by_inspector_1',
       'from_detroit']].set_index('ticket_id')


# In[53]:

np.sum(train.isnull())


# In[54]:

np.sum(submission.isnull())


# In[55]:

submission.fillna(0, inplace=True)


# In[56]:

np.sum(submission.isnull())


# In[57]:

from sklearn.linear_model import LogisticRegression


# In[58]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('compliance', axis=1), train['compliance'])


# In[59]:

model = LogisticRegression(C= 0.01, n_jobs= -1, solver= 'newton-cg', tol= 0.0001)
model.fit(X_train, y_train)


# In[60]:

from sklearn.metrics import auc, roc_curve


# In[61]:

y_true = model.predict_proba(X_test)[:,1]


# In[62]:

fpr, tpr, thresholds = roc_curve(y_test, y_true)


# In[63]:

auc = auc(fpr, tpr)


# In[64]:

auc


# In[65]:

pro = model.predict_proba(submission)[:,1]


# In[75]:

sub_df = pd.DataFrame(pro, submission.index)


# In[76]:

sub_df.head()


# In[77]:

sub_df.columns=['1']


# In[80]:



def blight_model():
    
    # Your code here
    
    return sub_df


# In[ ]:



