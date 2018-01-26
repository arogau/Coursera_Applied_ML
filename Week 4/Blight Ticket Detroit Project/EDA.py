
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
get_ipython().magic('matplotlib inline')


# In[2]:

train = pd.read_csv("train.csv", engine="c", sep=',', encoding='ISO-8859-1')
train.head()


# In[3]:

test = pd.read_csv("test.csv", engine="c", sep=',', encoding='ISO-8859-1')
test.head()


# In[4]:

train.info()


# In[5]:

train['ticket_issued_date'] = pd.to_datetime(train['ticket_issued_date'])
train['hearing_date'] = pd.to_datetime(train['hearing_date'])


# In[6]:

pd.Series.sort_values


# In[7]:

train['ticket_issued_date'].sort_values()


# In[8]:

train = train[train['ticket_issued_date']>='2004-01-01']


# In[9]:

train.describe()


# In[134]:

print("test df is from: ", test['ticket_issued_date'].min()," to ", test['ticket_issued_date'].max())


# In[10]:

test.describe()


# In[11]:

train.shape


# In[12]:

test.shape


# Percentage of missing values

# In[13]:

(np.sum(train.isnull())/len(train['ticket_id']))


# In[14]:

train['grafitti_status'].unique()


# In[15]:

train['grafitti_status'] = train['grafitti_status'].map(lambda x: 1 if x=='GRAFFITI TICKET'  else 0)


# In[16]:

train['fine_amount'].dropna(inplace=True)


# In[17]:

del train['violation_zip_code'] # 100% of the data is missing


# In[18]:

del train['non_us_str_code'] # 100% of data is missing


# In[19]:

del train['compliance_detail'] # as this can lead to data leakage. If this info becomes available we will come to know what complaince was.


# ## EDA

# In[20]:

train['compliance'].fillna(2, inplace=True) #replacing the Nan values with 2 for better manipulation 


# In[21]:

train['agency_name'].unique()


# In[136]:

test['agency_name'].unique()


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

train['agency_name'].unique()


# In[24]:

train.pivot_table(index='agency_name', columns='compliance', aggfunc='count', values='ticket_id').truediv(train.groupby('agency_name')['ticket_id'].count(), axis=0).plot.bar()


# In[25]:

train['compliance'] = train['compliance'].astype('int8')


# In[26]:

fig, ax = plt.subplots(1,3, figsize=(15,4), sharey=True)
train.pivot_table(index = 'agency_name', columns='compliance', values='fine_amount', aggfunc='mean').plot.bar(ax=ax[0]);
train.groupby(['agency_name'])['fine_amount'].mean().plot.bar(ax=ax[1]);
train.groupby('compliance')['fine_amount'].mean().plot.bar(ax=ax[2]);

ax[0].set_ylabel("Mean fine amount");
fig.legend(('0','1','2'),('Responsible, non-compliant','Responsible, compliant', 'Not responsible'));
#ax[1,0].set_xlabel("Count of Fines");


# In[27]:

fig, ax = plt.subplots(1,3, figsize=(15,4), sharey=True)
train.pivot_table(index = 'agency_name', columns='compliance', values='fine_amount', aggfunc='count').plot.bar(ax=ax[0]);
train.groupby(['agency_name'])['fine_amount'].count().plot.bar(ax=ax[1]);
train.groupby('compliance')['fine_amount'].count().plot.bar(ax=ax[2]);
ax[0].set_ylabel("Total tickets");
#fig.legend(('0','1','2'),('Responsible, non-compliant','Responsible, compliant', 'Not responsible'));


# In[28]:

train.head()


# In[29]:

train['disposition'].unique()   # but this would also come after judgement so we delete this as well.


# In[30]:

train.groupby('disposition')['ticket_id'].count().plot.bar();


# In[31]:

train.pivot_table(index='ticket_issued_date', columns='compliance', values='fine_amount', aggfunc='mean').resample('A').plot()
ax = plt.gca()
ax.set_title('Variation of Fine amount by Year')


# In[32]:

train.pivot_table(index='ticket_issued_date', columns='compliance', values='judgment_amount', aggfunc='mean').resample('A').plot()
ax = plt.gca()
ax.set_title('Variation of Judgment Amount by Year')


# In[33]:

train.pivot_table(index='ticket_issued_date', columns='compliance', values='discount_amount', aggfunc='mean').resample('A').plot()
ax = plt.gca()
ax.set_title('Variation of mean Discount Amount by Year')


# In[34]:

train.drop('disposition',axis=1, inplace=True)


# In[35]:

train.pivot_table(index='ticket_issued_date', columns='agency_name', values='fine_amount', aggfunc='mean').resample('A').plot()
ax = plt.gca()
ax.set_title('Variation of Fine amount by Year for each agency')
#ax.set_yticks(np.linspace(0,1100, 12))


# In[36]:

train[train['agency_name']=='Public Work'].pivot_table(index='ticket_issued_date', columns='compliance', values='judgment_amount', aggfunc='mean').resample('A').plot()
ax = plt.gca()
ax.set_title('Variation of Judgment Amount by Year issued by Public Work department')


# In[37]:

train.columns


# In[38]:

train[['agency_name','violation_code', 'violation_description']].head() # we can remove the digits after last - or '.',
# as the violation code are section and article wise and then sub articles. 


# In[39]:

train['violation_code'] = train['violation_code'].map(lambda x: "-".join(x.split('-')[0:2]).split('.')[0]) # since there are many dates we will have to leave this columns. This dates could be probable mistake in entry


# In[40]:

train['violation_description'].nunique()


# In[41]:



table  = train.pivot_table(index='violation_code', columns='compliance', values='ticket_id',aggfunc='count').truediv(train.groupby('violation_code')['ticket_id'].count(), axis=0).plot()


# In[42]:

train['time_gap'] = (train['hearing_date']-train['ticket_issued_date']).dt.days+1;#/np.timedelta64(1,'D')


# In[43]:

year_data = train.resample('A',on='ticket_issued_date')


# In[44]:

year_data.sum()


# In[45]:

sns.set_style('white')
fig = plt.figure();
ax= fig.gca();
plt.plot(year_data.mean().index, year_data.mean()['fine_amount'],c='r',label='Mean Amount', ls='--' );
ax.set_yticks(np.linspace(0,1000,6));
ax2= ax.twinx();
plt.plot(year_data.mean().index, year_data.sum()['fine_amount']/1e3, c='b', label='Total Amount', ls='--', axes=ax2);
ax2.set_yticks(np.linspace(0,45000,10));
ax.grid(b=False, which='both');
ax.set_xlabel('Year');
ax.set_ylabel('Mean Fine Amount');
ax2.set_ylabel('Total Fine Amount in Thousands \n Count of tickets');

ax.legend(bbox_to_anchor=(0,.1,1,1));
ax1 = fig.add_axes();
plt.plot(year_data.count().index, year_data.count()['fine_amount'], c='g',label='Count of tickets', ls='-', axes=ax1 );
ax2.legend(bbox_to_anchor=(0,.1,.6,1));
#plt.legend(bbox_to_anchor=(0,.2,.9,1));
#ax1.set_ylabel('Count of Tickets');


# In[46]:

del year_data


# ##### trying to get some meaningful out of  inspector name, since outcome can be based on Inspector judgement. 

# In[47]:

avg_compliance = train.groupby('compliance').count()['ticket_id']/len(train['compliance'])


# In[48]:

avg_compliance


# In[49]:

x = pd.DataFrame(train.groupby(['inspector_name','compliance']).count()['ticket_id']/train.groupby(['inspector_name']).count()['ticket_id']).reset_index()


# In[50]:

x['compliance_by_inspector'] = None


# 

# In[51]:

for i in [0,1,2]:
    x.loc[(x['compliance']==i)&((x['ticket_id']>=avg_compliance[i]*1.3)|(x['ticket_id']<=avg_compliance[i]*0.7)), 'compliance_by_inspector']=i


# In[52]:

x = x.join(pd.get_dummies(x['compliance_by_inspector']))
for i in [0,1,2]:
    x.rename(columns={i:'compliance_by_inspector_{}'.format(i)}, inplace=True)
    


# In[53]:

x.drop(['compliance_by_inspector','compliance','ticket_id'], axis=1, inplace=True)


# In[54]:

x = x.groupby('inspector_name').sum().reset_index()


# In[55]:

train=train.merge(x, on='inspector_name', how='inner')


# In[56]:

del x


# In[57]:

train.drop('inspector_name',axis=1, inplace=True)


# In[58]:

train.groupby('compliance').mean()['fine_amount']


# In[59]:

train.loc[train['fine_amount'].isnull(), 'fine_amount'] = 0


# In[60]:

plt.hist(train['fine_amount'],20);


# In[61]:

#this 10000 looks like an outlier but it is not. There are total of 357 count of 10000
np.sum(train['fine_amount']==10000)


# In[62]:

train[train['fine_amount']==10000]['violation_description'].unique()


# In[63]:

train.columns


# In[64]:

train['country'].unique()


# In[65]:

test['country'].unique() # we can safely remove the column as all the violators are from USA


# In[66]:

train.drop('country', axis=1, inplace=True)


# In[67]:

train['city'] = train['city'].map(lambda x: x.lower())


# In[68]:

train.groupby('city').count().nlargest(20,'ticket_id')['ticket_id'].plot() #as the decrease in the number of tickets is nowhere comparable to detroit we can create dummy for detroit


# In[69]:

train['from_detroit'] = train['city'].map(lambda x: 1 if x=='detroit' else 0)


# In[70]:

train.resample('A', on='ticket_issued_date').count()['ticket_id']


# In[71]:

train.columns


# In[72]:

train['violation_code'] = train['violation_code'].map(lambda x: x if '-' in x else 'date_in_violation_code')


# In[82]:

train.pivot_table(index='violation_code', columns='compliance', aggfunc='count', values='ticket_id').truediv(train.groupby('violation_code')['ticket_id'].count(), axis=0).plot.bar();
fig = plt.gcf()
fig.set_size_inches(13, 5)
fig.set_dpi(100)
plt.gca().set_title('avg compliance in each section')


# In[85]:

add = pd.read_csv('addresses.csv')
latlon = pd.read_csv('latlons.csv')


# In[86]:

add.head()


# In[87]:

latlon.head()


# In[89]:

np.sum(add.isnull())


# In[90]:

np.sum(latlon.isnull())


# In[92]:

add = latlon.merge(add, how='right', on='address').set_index('ticket_id')


# In[93]:

add.describe()


# In[102]:

train_add = train.merge(add.reset_index(), how='left', on = 'ticket_id')


# In[103]:

train.groupby


# In[ ]:

train_add[]


# In[130]:

for ((dd, data),label) in zip(train_add[train_add['compliance']<2].groupby('compliance'), train_add[train_add['compliance']<2]['compliance'].unique()):
    plt.scatter(data['lon'], data['lat'], alpha=0.5, label=label)

fig = plt.gcf();
fig.set_size_inches(13,7);
fig.set_dpi(100);
plt.legend();


# In[ ]:

train.drop(['admin_fee', 'state_fee', 'late_fee', 'discount_amount', 'clean_up_cost', 'judgment_amount'], axis=1, inplace=True)


# In[ ]:

del train['payment_amount']
del train['payment_date']
del train['payment_status']
del train['balance_due']
del train['collection_status']
#these variables also are linked directly to the output variable
# Also, these variables are not available in test data


# In[ ]:



def blight_model():
    
    # Your code here
    
    return # Your answer here

