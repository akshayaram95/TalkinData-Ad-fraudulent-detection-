
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns


# In[ ]:


train_sample=pd.read_csv("/Users/akshaya/Desktop/train_sample.csv")
train_sample.head()
variables=['ip','app','device','os','channel']
train_sample.info()


# In[58]:


plt.figure(figsize=(10, 6))
cols = ['ip', 'app', 'device', 'os', 'channel']
uniques = [len(train_sample[col].unique()) for col in cols]
sns.set(font_scale=1.2)
ax = sns.barplot(cols, uniques, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature (from 10,000,000 samples)')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 


# In[19]:


for v in variables:
    train_sample[v]=train_sample[v].astype('category')
train_sample['click_time']=pd.to_datetime(train_sample['click_time'])
train_sample['attributed_time']=pd.to_datetime(train_sample['attributed_time'])

train_sample['is_attributed']=train_sample['is_attributed'].astype('category')
train_sample.describe()


# In[20]:


train_sample['is_attributed'].value_counts().plot.bar()


# In[34]:


train_sample.describe()


# In[38]:


temp=train_sample['ip'].value_counts().reset_index(name='count')
temp.columns=['ip','count']
temp[:10]


# In[39]:


train_sample['app'].value_counts().sort_index().plot()


# In[40]:


train_sample['ip'].value_counts().sort_index().plot()


# In[41]:


train_sample['attributed_time'].value_counts().sort_index().plot()


# In[49]:


plt.figure(figsize=(6,6))
#sns.set(font_scale=1.2)
mean = (train_sample.is_attributed.values == 1).mean()
ax = sns.barplot(['App Downloaded (1)', 'Not Downloaded (0)'], [mean, 1-mean])
ax.set(ylabel='Proportion', title='App Downloaded vs Not Downloaded')
for p, uniq in zip(ax.patches, [mean, 1-mean]):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height+0.01,
            '{}%'.format(round(uniq * 100, 2)),
            ha="center")


# In[46]:


train_sample[['attributed_time', 'is_attributed']][train_sample['is_attributed']==1].describe()


# In[52]:


train_sample['is_attributed']=train_sample['is_attributed'].astype(int)

proportion = train_sample[['ip', 'is_attributed']].groupby('ip', as_index=False).mean().sort_values('is_attributed', ascending=False)
counts = train_sample[['ip', 'is_attributed']].groupby('ip', as_index=False).count().sort_values('is_attributed', ascending=False)
merge = counts.merge(proportion, on='ip', how='left')
merge.columns = ['ip', 'click_count', 'prop_downloaded']

ax = merge[:300].plot(secondary_y='prop_downloaded')
plt.title('Conversion Rates over Counts of 300 Most Popular IPs')
ax.set(ylabel='Count of clicks')
plt.ylabel('Proportion Downloaded')
plt.show()
print('Counversion Rates over Counts of Most Popular IPs')
print(merge[:20])


# In[53]:




proportion = train_sample[['app', 'is_attributed']].groupby('app', as_index=False).mean().sort_values('is_attributed', ascending=False)
counts = train_sample[['app', 'is_attributed']].groupby('app', as_index=False).count().sort_values('is_attributed', ascending=False)
merge = counts.merge(proportion, on='app', how='left')
merge.columns = ['app', 'click_count', 'prop_downloaded']

ax = merge[:100].plot(secondary_y='prop_downloaded')
plt.title('Conversion Rates over Counts of 100 Most Popular Apps')
ax.set(ylabel='Count of clicks')
plt.ylabel('Proportion Downloaded')
plt.show()
print('Counversion Rates over Counts of Most Popular Apps')
print(merge[:20])


# In[55]:


proportion = train_sample[['os', 'is_attributed']].groupby('os', as_index=False).mean().sort_values('is_attributed', ascending=False)
counts = train_sample[['os', 'is_attributed']].groupby('os', as_index=False).count().sort_values('is_attributed', ascending=False)
merge = counts.merge(proportion, on='os', how='left')
merge.columns = ['os', 'click_count', 'prop_downloaded']

ax = merge[:100].plot(secondary_y='prop_downloaded')
plt.title('Conversion Rates over Counts of 100 Most Popular Operating Systems')
ax.set(ylabel='Count of clicks')
plt.ylabel('Proportion Downloaded')
plt.show()

print('Counversion Rates over Counts of Most Popular Operating Systems')
print(merge[:20])


# In[56]:


proportion = train_sample[['device', 'is_attributed']].groupby('device', as_index=False).mean().sort_values('is_attributed', ascending=False)
counts = train_sample[['device', 'is_attributed']].groupby('device', as_index=False).count().sort_values('is_attributed', ascending=False)
merge = counts.merge(proportion, on='device', how='left')
merge.columns = ['device', 'click_count', 'prop_downloaded']

print('Count of clicks and proportion of downloads by device:')
print(merge)


# In[57]:


proportion = train_sample[['channel', 'is_attributed']].groupby('channel', as_index=False).mean().sort_values('is_attributed', ascending=False)
counts = train_sample[['channel', 'is_attributed']].groupby('channel', as_index=False).count().sort_values('is_attributed', ascending=False)
merge = counts.merge(proportion, on='channel', how='left')
merge.columns = ['channel', 'click_count', 'prop_downloaded']

ax = merge[:100].plot(secondary_y='prop_downloaded')
plt.title('Conversion Rates over Counts of 100 Most Popular Apps')
ax.set(ylabel='Count of clicks')
plt.ylabel('Proportion Downloaded')
plt.show()

print('Counversion Rates over Counts of Most Popular Channels')
print(merge[:20])


# In[66]:


train_sample['click_time'] = pd.to_datetime(train_sample['click_time'])
train_sample['attributed_time'] = pd.to_datetime(train_sample['attributed_time'])
#round the time to nearest hour
train_sample['click_rnd']=train_sample['click_time'].dt.round('H')  

#check for hourly patterns
train_sample[['click_rnd','is_attributed']].groupby(['click_rnd'], as_index=True).count().plot()
plt.title('HOURLY CLICK FREQUENCY');
plt.ylabel('Number of Clicks');

train_sample[['click_rnd','is_attributed']].groupby(['click_rnd'], as_index=True).mean().plot()
plt.title('HOURLY CONVERSION RATIO');
plt.ylabel('Converted Ratio');



#extract hour as a feature
train_sample['click_hour']=train_sample['click_time'].dt.hour
train_sample.head(7)


train_sample['timePass']= train_sample['attributed_time']-train_sample['click_time']
#check:
train_sample[train_sample['is_attributed']==1][:15]

train_sample['timePass'].describe()


# In[68]:


#check first 10,000,000 of actual train data
train_sample['timePass']= train_sample['attributed_time']-train_sample['click_time']
train_sample['timePass'].describe()

